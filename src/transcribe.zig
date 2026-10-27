//! Greedy long-form decoding with whisper.cpp's rules, window by window, with segments
//! handed to a callback the moment their closing timestamp is decoded.
//!
//! Times are in centiseconds from the start of the audio, like whisper.cpp. A 30 s window
//! is encoded, the decoder is primed with the previous window's text (after a `prev`
//! marker) and `sot`, then tokens are sampled one at a time under the timestamp rules:
//! timestamps come in pairs, never go backwards, and win whenever their total probability
//! beats the best text token. The last timestamp decides where the next window starts.
//!
//! Each finished segment is also tagged with a voice (see voice.zig) when the raw samples
//! are supplied: pitch from the audio, timbre from the encoder rows it spans.
const std = @import("std");
const ops = @import("ops.zig");
const model = @import("model.zig");
const mel = @import("mel.zig");
const voice = @import("voice.zig");
const clock = @import("clock.zig");
const Encoder = @import("encoder.zig").Encoder;
const Decoder = @import("decoder.zig").Decoder;

pub const Segment = struct {
    /// Start and end, centiseconds.
    t0: u32,
    t1: u32,
    /// UTF-8, valid only during the callback.
    text: []const u8,
    /// Token ids, timestamps included, valid only during the callback.
    tokens: []const u32,
    /// Voice label, 1-based in order of first appearance; 0 when voices are off.
    voice: u32 = 0,
    /// Median pitch in Hz over the segment, 0 if unvoiced or voices are off.
    pitch_hz: f32 = 0,
    /// Standardized distance to the voice it joined (0 = started a new voice or voices off).
    voice_dist: f32 = 0,
    /// True when the span is inside quotation marks (dialogue).
    quoted: bool = false,
};

pub const Options = struct {
    /// Keep the previous window's text as context (whisper.cpp's default).
    condition_on_previous: bool = true,
    /// Largest first timestamp, seconds (whisper.cpp max_initial_ts).
    max_initial_ts: f32 = 1.0,
    /// Tag segments with voices. Needs `samples` in `transcribe`.
    voices: bool = true,
    /// Split segments at quotation marks so dialogue and narration get their own spans
    /// (times inside a segment are interpolated by character count).
    split_quotes: bool = true,
    /// Stop early; checked between tokens.
    stop: ?*const std.atomic.Value(bool) = null,
};

pub const Stats = struct {
    windows: usize = 0,
    tokens: usize = 0,
    segments: usize = 0,
    voices: usize = 0,
    encode_ns: u64 = 0,
    decode_ns: u64 = 0,
    voice_ns: u64 = 0,
};

pub const Engine = struct {
    gpa: std.mem.Allocator,
    m: *const model.Model,
    enc: *Encoder,
    dec: *Decoder,
    /// All threads: the encoder and prompt batches.
    pool: *ops.Pool,
    /// A few threads: the token-at-a-time decoder, whose ops are too small to share wider
    /// without the barrier cost and preemption stalls outweighing the work.
    pool_dec: *ops.Pool,
    logits: []f32,
    logprobs: []f32,
    tok_space: u32,
    /// Voice clusters persist across calls so ids stay stable through a book.
    voices: voice.Voices,

    pub fn init(gpa: std.mem.Allocator, m: *const model.Model, n_threads: usize) !*Engine {
        const e = try gpa.create(Engine);
        errdefer gpa.destroy(e);
        e.gpa = gpa;
        e.m = m;
        e.pool = try ops.Pool.init(gpa, n_threads);
        errdefer e.pool.deinit(gpa);
        e.pool_dec = try ops.Pool.init(gpa, @min(n_threads, 4));
        errdefer e.pool_dec.deinit(gpa);
        e.enc = try Encoder.load(gpa, m);
        errdefer e.enc.deinit(gpa);
        e.dec = try Decoder.load(gpa, m);
        errdefer e.dec.deinit(gpa);
        e.logits = try gpa.alloc(f32, m.hp.n_vocab);
        errdefer gpa.free(e.logits);
        e.logprobs = try gpa.alloc(f32, m.hp.n_vocab);
        errdefer gpa.free(e.logprobs);
        e.tok_space = m.tokenId(" ") orelse 220;
        e.voices = try voice.Voices.init(gpa, voice.n_bands);
        return e;
    }

    /// Forget every voice seen so far (a new book).
    pub fn resetVoices(e: *Engine) !void {
        e.voices.deinit();
        e.voices = try voice.Voices.init(e.gpa, voice.n_bands);
    }

    pub fn deinit(e: *Engine) void {
        const gpa = e.gpa;
        e.voices.deinit();
        gpa.free(e.logprobs);
        gpa.free(e.logits);
        e.dec.deinit(gpa);
        e.enc.deinit(gpa);
        e.pool_dec.deinit(gpa);
        e.pool.deinit(gpa);
        gpa.destroy(e);
    }

    /// Transcribe a whole spectrogram. `samples` (16 kHz mono, the audio the spectrogram
    /// came from) enables voice tags; pass null to skip them. `emit(ctx, segment)` is
    /// called as segments finish.
    pub fn transcribe(e: *Engine, spec: *const mel.Mel, samples: ?[]const f32, opts: Options, ctx: anytype, comptime emit: fn (@TypeOf(ctx), Segment) void) !Stats {
        const tok = e.m.tok;
        const n_ctx_text = e.m.hp.n_text_ctx;
        const max_prompt_ctx = n_ctx_text / 2;
        const n_max: usize = n_ctx_text / 2 - 4;
        const window: u32 = @intCast(2 * e.m.hp.n_audio_ctx); // 3000 frames = 30 s
        const n_state = e.m.hp.n_audio_state;
        var stats = Stats{};

        var prompt_past: std.ArrayList(u32) = .empty;
        defer prompt_past.deinit(e.gpa);
        var prompt: std.ArrayList(u32) = .empty;
        defer prompt.deinit(e.gpa);
        var tokens: std.ArrayList(u32) = .empty; // sampled this window
        defer tokens.deinit(e.gpa);
        var tids: std.ArrayList(u32) = .empty; // most likely timestamp at each step
        defer tids.deinit(e.gpa);
        var text: std.ArrayList(u8) = .empty;
        defer text.deinit(e.gpa);

        const tag_voices = opts.voices and samples != null;
        const voices = &e.voices;
        const feat = try e.gpa.alloc(f32, voice.n_bands + 1);
        defer e.gpa.free(feat);
        _ = n_state;
        const pitch_scratch = try e.gpa.alloc(f32, 30 * 16000 / 512 + 2);
        defer e.gpa.free(pitch_scratch);

        const seek_end: u32 = @intCast(spec.n_len_org);
        var seek: u32 = 0;
        while (seek < seek_end) {
            if (seek + 100 >= seek_end) break;
            if (opts.stop) |s| if (s.load(.acquire)) return error.Stopped;

            var timer = clock.Timer.start();
            const enc_out = e.enc.encode(e.pool, spec, seek);
            e.dec.setAudio(e.pool, enc_out);
            stats.encode_ns += timer.lap();
            stats.windows += 1;

            // prompt: [prev, past text...] + [sot]
            prompt.clearRetainingCapacity();
            if (opts.condition_on_previous and prompt_past.items.len > 0) {
                try prompt.append(e.gpa, tok.prev);
                const n_take = @min(max_prompt_ctx - 1, prompt_past.items.len);
                try prompt.appendSlice(e.gpa, prompt_past.items[prompt_past.items.len - n_take ..]);
            }
            try prompt.append(e.gpa, tok.sot);

            tokens.clearRetainingCapacity();
            tids.clearRetainingCapacity();
            text.clearRetainingCapacity();
            e.dec.reset();
            e.dec.forward(if (prompt.items.len > 8) e.pool else e.pool_dec, prompt.items, e.logits);

            var seek_delta: u32 = window;
            var has_ts = false;
            var result_len: usize = 0;
            var failed = false;
            // streaming segment state
            var seg_start: usize = 0;
            var seg_t0: u32 = seek;

            var i: usize = 0;
            while (i < n_max) : (i += 1) {
                const is_initial = tokens.items.len == 0;
                const s = e.sample(tokens.items, is_initial, has_ts, seek_delta, opts);
                try tokens.append(e.gpa, s.id);
                try tids.append(e.gpa, s.tid);
                stats.tokens += 1;
                if (is_initial) seg_t0 = seek + 2 * (s.tid -| tok.beg);

                if (s.id > tok.beg) {
                    const seek_delta_new = 2 * (s.id - tok.beg);
                    if (has_ts and seek_delta > seek_delta_new and result_len < i) {
                        failed = true;
                        break;
                    }
                    seek_delta = seek_delta_new;
                    result_len = i + 1;
                    has_ts = true;
                    // a timestamp after text closes a segment; emit it now
                    const t1 = seek + 2 * (s.tid -| tok.beg);
                    if (text.items.len > 0) {
                        try e.finish(&stats, voices, tag_voices, opts.split_quotes, samples, spec, feat, pitch_scratch, .{ .t0 = seg_t0, .t1 = t1, .text = text.items, .tokens = tokens.items[seg_start..] }, ctx, emit);
                    }
                    text.clearRetainingCapacity();
                    seg_t0 = t1;
                    seg_start = i + 1;
                } else if (s.id < tok.eot) {
                    try text.appendSlice(e.gpa, e.m.tokenText(s.id));
                }

                if (s.id == tok.eot or (has_ts and seek + seek_delta + 100 >= seek_end)) {
                    if (result_len == 0) {
                        if (seek + seek_delta + 100 >= seek_end) {
                            result_len = i + 1;
                        } else {
                            failed = true;
                        }
                    }
                    break;
                }
                if (i == n_max - 1 and (result_len == 0 or seek_delta < window / 2)) {
                    failed = true;
                    break;
                }
                if (opts.stop) |st| if (st.load(.acquire)) return error.Stopped;
                e.dec.forward(e.pool_dec, tokens.items[tokens.items.len - 1 ..], e.logits);
            }
            stats.decode_ns += timer.lap();

            // whisper.cpp would retry a failed window at a higher temperature; we keep what
            // we have and move on by a whole window if nothing usable came out.
            if (failed and result_len == 0) {
                seek += window;
                prompt_past.clearRetainingCapacity();
                continue;
            }
            const used = tokens.items[0..result_len];

            // trailing text inside the result (only when the window ran to the end of audio)
            if (seg_start < used.len) {
                text.clearRetainingCapacity();
                for (used[seg_start..]) |id| if (id < tok.eot) try text.appendSlice(e.gpa, e.m.tokenText(id));
                if (text.items.len > 0) {
                    try e.finish(&stats, voices, tag_voices, opts.split_quotes, samples, spec, feat, pitch_scratch, .{ .t0 = seg_t0, .t1 = seek + seek_delta, .text = text.items, .tokens = used[seg_start..] }, ctx, emit);
                }
            }

            // rolling context for the next window: what was fed as history this time (after
            // `prev`, before `sot`) plus what was kept from this window
            if (opts.condition_on_previous) {
                const old: []const u32 = if (prompt.items.len > 1) prompt.items[1 .. prompt.items.len - 1] else &.{};
                const merged = try e.gpa.alloc(u32, old.len + used.len);
                defer e.gpa.free(merged);
                @memcpy(merged[0..old.len], old);
                @memcpy(merged[old.len..], used);
                prompt_past.clearRetainingCapacity();
                try prompt_past.appendSlice(e.gpa, merged);
            }
            seek += seek_delta;
        }
        stats.voices = voices.count();
        return stats;
    }

    /// Split a segment at quotes (optionally), tag each span with its voice (when enabled)
    /// and hand the spans to the callback in order.
    fn finish(e: *Engine, stats: *Stats, voices: *voice.Voices, tag: bool, split: bool, samples: ?[]const f32, spec: *const mel.Mel, feat: []f32, pitch_scratch: []f32, seg: Segment, ctx: anytype, comptime emit: fn (@TypeOf(ctx), Segment) void) !void {
        _ = e;
        const text = seg.text;
        var start: usize = 0;
        var quoted = false;
        // An opening quote ends the span before it; a closing quote ends the span after it,
        // so each span carries its own quote marks and no span is just a quote.
        if (split and text.len > 0) {
            var i: usize = 0;
            while (i < text.len) : (i += 1) {
                if (text[i] != '"') continue;
                const end = if (quoted) i + 1 else i;
                if (hasWord(text[start..end])) {
                    try finishSpan(stats, voices, tag, samples, spec, feat, pitch_scratch, seg, start, end, quoted, ctx, emit);
                    start = end;
                }
                quoted = !quoted;
            }
        }
        if (start < text.len and (hasWord(text[start..]) or start == 0)) {
            try finishSpan(stats, voices, tag, samples, spec, feat, pitch_scratch, seg, start, text.len, quoted, ctx, emit);
        }
    }

    fn hasWord(s: []const u8) bool {
        for (s) |c| if (std.ascii.isAlphanumeric(c)) return true;
        return false;
    }

    fn finishSpan(stats: *Stats, voices: *voice.Voices, tag: bool, samples: ?[]const f32, spec: *const mel.Mel, feat: []f32, pitch_scratch: []f32, seg_raw: Segment, c0: usize, c1: usize, quoted: bool, ctx: anytype, comptime emit: fn (@TypeOf(ctx), Segment) void) !void {
        var seg = seg_raw;
        // the first token's "most likely timestamp" can sit past the real start; never let
        // a span run backwards
        if (seg.t1 < seg.t0) seg.t0 = seg.t1;
        const seg_in = seg;
        const n: f32 = @floatFromInt(@max(seg_in.text.len, 1));
        const span_ms: f32 = @floatFromInt(seg_in.t1 - seg_in.t0);
        seg.t0 = seg_in.t0 + @as(u32, @intFromFloat(span_ms * @as(f32, @floatFromInt(c0)) / n));
        seg.t1 = seg_in.t0 + @as(u32, @intFromFloat(span_ms * @as(f32, @floatFromInt(c1)) / n));
        if (c1 >= seg_in.text.len) seg.t1 = seg_in.t1;
        seg.text = seg_in.text[c0..c1];
        seg.quoted = quoted;
        stats.segments += 1;
        if (tag) {
            var t = clock.Timer.start();
            const s = samples.?;
            const a = @min(@as(usize, seg.t0) * 160, s.len);
            const b = @min(@as(usize, seg.t1) * 160, s.len);
            const p = voice.pitch(s[a..b], pitch_scratch);
            voice.envelope(spec, seg.t0, seg.t1, feat[0..voice.n_bands]);
            feat[voice.n_bands] = if (p.hz > 0) std.math.log2(p.hz) else 0;
            const asg = try voices.assign(feat);
            seg.voice = asg.id;
            seg.pitch_hz = p.hz;
            seg.voice_dist = asg.distance;
            stats.voice_ns += t.lap();
        }
        emit(ctx, seg);
    }

    const Sampled = struct { id: u32, tid: u32 };

    /// whisper.cpp's logit rules, then argmax. Also returns the most likely timestamp.
    fn sample(e: *Engine, cur: []const u32, is_initial: bool, has_ts: bool, seek_delta: u32, opts: Options) Sampled {
        const tok = e.m.tok;
        const lg = e.logits;
        const n = lg.len;
        const inf = std.math.inf(f32);

        if (is_initial) {
            lg[tok.eot] = -inf;
            lg[e.tok_space] = -inf;
        }
        lg[tok.not] = -inf;
        lg[tok.sot] = -inf;
        lg[tok.nosp] = -inf;
        lg[tok.solm] = -inf;
        lg[tok.translate] = -inf;
        lg[tok.transcribe] = -inf;
        lg[tok.prev] = -inf;

        const last_was_ts = cur.len > 0 and cur[cur.len - 1] >= tok.beg;
        const penultimate_was_ts = cur.len < 2 or cur[cur.len - 2] >= tok.beg;
        if (last_was_ts) {
            if (penultimate_was_ts) {
                for (lg[tok.beg..]) |*v| v.* = -inf;
            } else {
                for (lg[0..tok.eot]) |*v| v.* = -inf;
            }
        }
        if (is_initial and opts.max_initial_ts > 0) {
            const precision: f32 = 30.0 / @as(f32, @floatFromInt(e.m.hp.n_audio_ctx));
            const tid0: usize = @intFromFloat(@round(opts.max_initial_ts / precision));
            if (tok.beg + tid0 + 1 < n) for (lg[tok.beg + tid0 + 1 ..]) |*v| {
                v.* = -inf;
            };
        }
        if (has_ts) {
            const tid0 = seek_delta / 2;
            for (lg[tok.beg .. tok.beg + tid0]) |*v| v.* = -inf;
        }

        // log-probs (-inf entries contribute 0 through the vector exp's flush)
        const lp = e.logprobs;
        var mx: f32 = -inf;
        for (lg) |v| mx = @max(mx, v);
        const mv: ops.Vec = @splat(mx);
        var sv: ops.Vec = @splat(0);
        var vi: usize = 0;
        while (vi + ops.V <= n) : (vi += ops.V) sv += ops.expVec(@as(ops.Vec, lg[vi..][0..ops.V].*) - mv);
        var sum = @reduce(.Add, sv);
        while (vi < n) : (vi += 1) if (lg[vi] > -inf) {
            sum += @exp(lg[vi] - mx);
        };
        const lse = @log(sum) + mx;
        for (lp, lg) |*p, v| p.* = if (v > -inf) v - lse else -inf;

        // timestamps as a block vs the best text token
        var ts_max: f32 = -inf;
        for (lp[tok.beg..]) |v| ts_max = @max(ts_max, v);
        var ts_lp: f32 = -inf;
        if (ts_max > -inf) {
            var acc: f32 = 0;
            for (lp[tok.beg..]) |v| if (v > -inf) {
                acc += @exp(v - ts_max);
            };
            ts_lp = @log(acc) + ts_max;
        }
        var text_max: f32 = -inf;
        for (lp[0..tok.beg]) |v| text_max = @max(text_max, v);
        if (ts_lp > text_max) {
            for (lg[0..tok.beg]) |*v| v.* = -inf;
            for (lp[0..tok.beg]) |*v| v.* = -inf;
        }

        var best: u32 = 0;
        var best_v: f32 = -inf;
        var tid: u32 = tok.beg;
        var tid_v: f32 = -inf;
        for (lg, 0..) |v, i| {
            if (v > best_v) {
                best_v = v;
                best = @intCast(i);
            }
            if (i >= tok.beg and v > tid_v) {
                tid_v = v;
                tid = @intCast(i);
            }
        }
        return .{ .id = best, .tid = tid };
    }
};
