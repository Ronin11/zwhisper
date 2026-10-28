//! A decoder-only language model (Qwen2 family) from a GGUF file, for the scene and
//! character steps that used to need a server: pre-norm blocks with grouped-query
//! attention, rotary positions, SwiGLU, and tied or separate output weights. Greedy
//! generation with a KV cache; the prompt goes through in batches.
//!
//! Weights come in as q8_0 and stay int8 (our Q8 layout); norms and biases are f32.
const std = @import("std");
const ops = @import("ops.zig");
const gemm = @import("gemm.zig");
const gguf = @import("gguf.zig");
const bpe = @import("bpe.zig");
const clock = @import("clock.zig");

pub const Layer = struct {
    attn_norm: []f32,
    wq: gemm.Q8,
    bq: ?[]f32,
    wk: gemm.Q8,
    bk: ?[]f32,
    wv: gemm.Q8,
    bv: ?[]f32,
    wo: gemm.Q8,
    ffn_norm: []f32,
    w_gate: gemm.Q8,
    w_up: gemm.Q8,
    w_down: gemm.Q8,
    /// Keys transposed: [n_head_kv][d][n_ctx], so attention reads them as is.
    cache_kt: []f32,
    cache_v: []f32, // [n_ctx][n_head_kv * d]
};

pub const Model = struct {
    gpa: std.mem.Allocator,
    arena: std.heap.ArenaAllocator,
    tok: *bpe.Tokenizer,
    n_vocab: usize,
    n_embd: usize,
    n_head: usize,
    n_head_kv: usize,
    d: usize,
    n_ff: usize,
    n_layer: usize,
    n_ctx: usize,
    eps: f32,
    rope_base: f32,
    embd: gemm.Q8, // [n_vocab][n_embd]
    output: gemm.Q8, // same as embd when tied
    output_norm: []f32,
    layers: []Layer,
    cos: []f32, // [n_ctx][d/2]
    sin: []f32,
    n_past: usize = 0,
    batch: usize,
    // scratch for a batch
    x: []f32,
    h: []f32,
    q: []f32,
    kb: []f32,
    vb: []f32,
    attn: []f32,
    gate: []f32,
    up: []f32,
    logits: []f32,
    attn_scratch: []f32,
    pool: *ops.Pool,
    pool_dec: *ops.Pool,

    pub const Options = struct {
        /// Context length to allocate (KV cache); clipped to the model's maximum.
        n_ctx: usize = 4096,
        /// Prompt tokens per forward pass.
        batch: usize = 256,
        threads: usize = 4,
    };

    pub fn load(gpa: std.mem.Allocator, f: *const gguf.File, opts: Options) !*Model {
        const arch = f.getString("general.architecture") orelse return error.Unsupported;
        if (!std.mem.eql(u8, arch, "qwen2")) {
            std.debug.print("zwhisper: unsupported architecture {s} (qwen2 only)\n", .{arch});
            return error.Unsupported;
        }
        const m = try gpa.create(Model);
        errdefer gpa.destroy(m);
        m.* = undefined;
        m.gpa = gpa;
        m.arena = std.heap.ArenaAllocator.init(gpa);
        errdefer m.arena.deinit();
        const a = m.arena.allocator();
        m.n_past = 0;
        m.tok = try bpe.Tokenizer.init(gpa, f);
        errdefer m.tok.deinit();
        m.n_embd = f.getU32("qwen2.embedding_length") orelse return error.Unsupported;
        m.n_head = f.getU32("qwen2.attention.head_count") orelse return error.Unsupported;
        m.n_head_kv = f.getU32("qwen2.attention.head_count_kv") orelse m.n_head;
        m.d = m.n_embd / m.n_head;
        m.n_ff = f.getU32("qwen2.feed_forward_length") orelse return error.Unsupported;
        m.n_layer = f.getU32("qwen2.block_count") orelse return error.Unsupported;
        const max_ctx = f.getU32("qwen2.context_length") orelse 32768;
        m.n_ctx = @min(opts.n_ctx, max_ctx) / ops.V * ops.V;
        m.eps = f.getF32("qwen2.attention.layer_norm_rms_epsilon") orelse 1e-6;
        m.rope_base = f.getF32("qwen2.rope.freq_base") orelse 1_000_000.0;
        m.batch = opts.batch;

        const embd_t = f.tensor("token_embd.weight") orelse return error.MissingTensor;
        m.n_vocab = embd_t.ne[1];
        m.embd = try loadQ(a, embd_t);
        m.output = if (f.tensor("output.weight")) |t| try loadQ(a, t) else m.embd;
        m.output_norm = try loadF(a, f.tensor("output_norm.weight") orelse return error.MissingTensor);
        m.layers = try a.alloc(Layer, m.n_layer);
        const kvw = m.n_head_kv * m.d;
        for (m.layers, 0..) |*l, i| {
            l.attn_norm = try loadF(a, try need(f, "blk.{d}.attn_norm.weight", i));
            l.wq = try loadQ(a, try need(f, "blk.{d}.attn_q.weight", i));
            l.bq = try loadOptF(a, f, "blk.{d}.attn_q.bias", i);
            l.wk = try loadQ(a, try need(f, "blk.{d}.attn_k.weight", i));
            l.bk = try loadOptF(a, f, "blk.{d}.attn_k.bias", i);
            l.wv = try loadQ(a, try need(f, "blk.{d}.attn_v.weight", i));
            l.bv = try loadOptF(a, f, "blk.{d}.attn_v.bias", i);
            l.wo = try loadQ(a, try need(f, "blk.{d}.attn_output.weight", i));
            l.ffn_norm = try loadF(a, try need(f, "blk.{d}.ffn_norm.weight", i));
            l.w_gate = try loadQ(a, try need(f, "blk.{d}.ffn_gate.weight", i));
            l.w_up = try loadQ(a, try need(f, "blk.{d}.ffn_up.weight", i));
            l.w_down = try loadQ(a, try need(f, "blk.{d}.ffn_down.weight", i));
            l.cache_kt = try a.alloc(f32, m.n_ctx * kvw);
            @memset(l.cache_kt, 0);
            l.cache_v = try a.alloc(f32, m.n_ctx * kvw);
        }
        // rotary tables
        const half = m.d / 2;
        m.cos = try a.alloc(f32, m.n_ctx * half);
        m.sin = try a.alloc(f32, m.n_ctx * half);
        for (0..m.n_ctx) |p| for (0..half) |i| {
            const inv = 1.0 / std.math.pow(f64, m.rope_base, @as(f64, @floatFromInt(2 * i)) / @as(f64, @floatFromInt(m.d)));
            const ang = @as(f64, @floatFromInt(p)) * inv;
            m.cos[p * half + i] = @floatCast(@cos(ang));
            m.sin[p * half + i] = @floatCast(@sin(ang));
        };
        const b = m.batch;
        m.x = try a.alloc(f32, b * m.n_embd);
        m.h = try a.alloc(f32, b * m.n_embd);
        m.q = try a.alloc(f32, b * m.n_embd);
        m.kb = try a.alloc(f32, b * kvw);
        m.vb = try a.alloc(f32, b * kvw);
        m.attn = try a.alloc(f32, b * m.n_embd);
        m.gate = try a.alloc(f32, b * m.n_ff);
        m.up = try a.alloc(f32, b * m.n_ff);
        m.logits = try a.alloc(f32, m.n_vocab);
        m.attn_scratch = try a.alloc(f32, ops.attentionScratchKT(b, m.n_ctx, m.n_head));
        m.pool = try ops.Pool.init(gpa, opts.threads);
        errdefer m.pool.deinit(gpa);
        m.pool_dec = try ops.Pool.init(gpa, @min(opts.threads, 4));
        return m;
    }

    pub fn deinit(m: *Model) void {
        const gpa = m.gpa;
        m.pool_dec.deinit(gpa);
        m.pool.deinit(gpa);
        m.tok.deinit();
        m.arena.deinit();
        gpa.destroy(m);
    }

    pub fn reset(m: *Model) void {
        m.n_past = 0;
    }

    /// Feed `tokens` (<= batch) at positions n_past..; logits of the last token land in m.logits.
    pub fn forward(m: *Model, tokens: []const u32) void {
        const n = tokens.len;
        std.debug.assert(n > 0 and n <= m.batch and m.n_past + n <= m.n_ctx);
        const pool = if (n > 8) m.pool else m.pool_dec;
        const ne = m.n_embd;
        const kvw = m.n_head_kv * m.d;
        for (tokens, 0..) |t, i| m.embd.row(t, m.x[i * ne ..][0..ne]);
        const n_kv = m.n_past + n;
        for (m.layers) |*l| {
            ops.rmsNorm(m.x, n, ne, l.attn_norm, m.eps, m.h);
            ops.linear(pool, m.h, n, ne, l.wq, l.bq, ne, m.q);
            ops.linear(pool, m.h, n, ne, l.wk, l.bk, kvw, m.kb);
            ops.linear(pool, m.h, n, ne, l.wv, l.bv, kvw, m.vb);
            m.rope(m.q, n, m.n_head);
            m.rope(m.kb, n, m.n_head_kv);
            // scatter the new keys into the transposed cache
            for (0..n) |i| for (0..m.n_head_kv) |h| {
                const src = m.kb[i * kvw + h * m.d ..][0..m.d];
                const col = m.n_past + i;
                for (0..m.d) |p| l.cache_kt[(h * m.d + p) * m.n_ctx + col] = src[p];
            };
            @memcpy(l.cache_v[m.n_past * kvw ..][0 .. n * kvw], m.vb[0 .. n * kvw]);
            ops.attentionEx(pool, m.q, l.cache_kt, l.cache_v[0 .. n_kv * kvw], .{
                .n_q = n,
                .n_kv = n_kv,
                .n_head = m.n_head,
                .n_head_kv = m.n_head_kv,
                .d = m.d,
                .q_stride = ne,
                .kv_stride = kvw,
                .out_stride = ne,
                .causal = true,
                .kt_stride = m.n_ctx,
            }, m.attn_scratch, m.attn);
            ops.linear(pool, m.attn, n, ne, l.wo, null, ne, m.h);
            ops.add(m.x[0 .. n * ne], m.h[0 .. n * ne]);

            ops.rmsNorm(m.x, n, ne, l.ffn_norm, m.eps, m.h);
            ops.linear(pool, m.h, n, ne, l.w_gate, null, m.n_ff, m.gate);
            ops.linear(pool, m.h, n, ne, l.w_up, null, m.n_ff, m.up);
            ops.siluMul(m.gate[0 .. n * m.n_ff], m.up[0 .. n * m.n_ff]);
            ops.linear(pool, m.gate, n, m.n_ff, l.w_down, null, ne, m.h);
            ops.add(m.x[0 .. n * ne], m.h[0 .. n * ne]);
        }
        m.n_past += n;
        const last = m.x[(n - 1) * ne ..][0..ne];
        ops.rmsNorm(last, 1, ne, m.output_norm, m.eps, m.h[0..ne]);
        ops.linear(pool, m.h[0..ne], 1, ne, m.output, null, m.n_vocab, m.logits);
    }

    /// Rotary embedding, NeoX style (first half paired with second half), in place on
    /// `n` rows of `heads` heads starting at position n_past.
    fn rope(m: *Model, x: []f32, n: usize, heads: usize) void {
        const half = m.d / 2;
        const stride = heads * m.d;
        for (0..n) |i| {
            const p = m.n_past + i;
            const c = m.cos[p * half ..][0..half];
            const s = m.sin[p * half ..][0..half];
            for (0..heads) |h| {
                const v = x[i * stride + h * m.d ..][0..m.d];
                for (0..half) |k| {
                    const a = v[k];
                    const b = v[k + half];
                    v[k] = a * c[k] - b * s[k];
                    v[k + half] = a * s[k] + b * c[k];
                }
            }
        }
    }

    /// Mean of the final hidden state (before the output norm) over all of `tokens`,
    /// fed in batches from a fresh context. `out` has n_embd floats. A frozen feature
    /// vector for a window of text, for the trained tagger.
    pub fn features(m: *Model, tokens: []const u32, out: []f32) !void {
        if (tokens.len > m.n_ctx) return error.ContextFull;
        @memset(out, 0);
        m.reset();
        var i: usize = 0;
        while (i < tokens.len) {
            const n = @min(m.batch, tokens.len - i);
            m.forward(tokens[i .. i + n]);
            for (0..n) |r| ops.add(out, m.x[r * m.n_embd ..][0..m.n_embd]);
            i += n;
        }
        ops.scale(out, 1.0 / @as(f32, @floatFromInt(tokens.len)));
    }

    /// Feed a prompt in batches; returns nothing, logits are for its last token.
    pub fn prompt(m: *Model, tokens: []const u32) !void {
        if (m.n_past + tokens.len > m.n_ctx) return error.ContextFull;
        var i: usize = 0;
        while (i < tokens.len) {
            const n = @min(m.batch, tokens.len - i);
            m.forward(tokens[i .. i + n]);
            i += n;
        }
    }

    pub fn argmax(m: *const Model) u32 {
        var best: usize = 0;
        for (m.logits, 0..) |v, i| if (v > m.logits[best]) {
            best = i;
        };
        return @intCast(best);
    }

    /// Qwen chat template around `user`, with an optional system message.
    pub fn chatPrompt(m: *Model, gpa: std.mem.Allocator, system: ?[]const u8, user: []const u8) ![]u32 {
        var out: std.ArrayList(u32) = .empty;
        errdefer out.deinit(gpa);
        const t = m.tok;
        const im_start = t.im_start orelse return error.NoChatTokens;
        const im_end = t.im_end orelse return error.NoChatTokens;
        const nl = try t.encode(gpa, "\n");
        defer gpa.free(nl);
        const parts = [_]struct { role: []const u8, text: ?[]const u8 }{
            .{ .role = "system", .text = system },
            .{ .role = "user", .text = user },
        };
        for (parts) |p| {
            const text = p.text orelse continue;
            try out.append(gpa, im_start);
            const role = try t.encode(gpa, p.role);
            defer gpa.free(role);
            try out.appendSlice(gpa, role);
            try out.appendSlice(gpa, nl);
            const body = try t.encode(gpa, text);
            defer gpa.free(body);
            try out.appendSlice(gpa, body);
            try out.append(gpa, im_end);
            try out.appendSlice(gpa, nl);
        }
        try out.append(gpa, im_start);
        const asst = try t.encode(gpa, "assistant");
        defer gpa.free(asst);
        try out.appendSlice(gpa, asst);
        try out.appendSlice(gpa, nl);
        return out.toOwnedSlice(gpa);
    }

    pub const GenStats = struct { prompt_tokens: usize = 0, new_tokens: usize = 0, prompt_ns: u64 = 0, gen_ns: u64 = 0 };

    /// Divide (or multiply, when negative) the logits of tokens generated in the last
    /// `window` steps by `penalty`, the usual guard against a small model looping.
    pub var repeat_penalty: f32 = 1.1;
    pub var repeat_window: usize = 64;

    fn penalize(m: *Model, recent: []const u32) void {
        if (repeat_penalty <= 1.0) return;
        const from = recent.len -| repeat_window;
        for (recent[from..]) |t| {
            const v = m.logits[t];
            m.logits[t] = if (v > 0) v / repeat_penalty else v * repeat_penalty;
        }
    }

    /// Greedy generation after `prompt_tokens`; each new token's text goes to `emit`.
    pub fn generate(m: *Model, prompt_tokens: []const u32, max_new: usize, ctx: anytype, comptime emit: fn (@TypeOf(ctx), []const u8) void) !GenStats {
        var st = GenStats{ .prompt_tokens = prompt_tokens.len };
        var timer = clock.Timer.start();
        m.reset();
        try m.prompt(prompt_tokens);
        st.prompt_ns = timer.lap();
        var recent: std.ArrayList(u32) = .empty;
        defer recent.deinit(m.gpa);
        var i: usize = 0;
        while (i < max_new) : (i += 1) {
            m.penalize(recent.items);
            const t = m.argmax();
            try recent.append(m.gpa, t);
            if (t == m.tok.eos or (m.tok.im_end != null and t == m.tok.im_end.?)) break;
            emit(ctx, m.tok.text(t));
            st.new_tokens += 1;
            if (m.n_past + 1 > m.n_ctx) break;
            m.forward(&.{t});
        }
        st.gen_ns = timer.lap();
        return st;
    }
};

fn need(f: *const gguf.File, comptime fmt: []const u8, i: usize) !gguf.Tensor {
    var buf: [64]u8 = undefined;
    const name = try std.fmt.bufPrint(&buf, fmt, .{i});
    return f.tensor(name) orelse {
        std.debug.print("zwhisper: model is missing tensor {s}\n", .{name});
        return error.MissingTensor;
    };
}

fn loadQ(a: std.mem.Allocator, t: gguf.Tensor) !gemm.Q8 {
    const k = t.ne[0];
    const n = t.ne[1];
    return switch (t.ty) {
        .q8_0 => gemm.Q8.fromQ8_0(a, t.data, n, k),
        .f16, .f32 => blk: {
            const tmp = try gguf.toF32(t, std.heap.page_allocator);
            defer std.heap.page_allocator.free(tmp);
            break :blk gemm.Q8.quantize(a, tmp, n, k);
        },
        else => error.Unsupported,
    };
}

fn loadF(a: std.mem.Allocator, t: gguf.Tensor) ![]f32 {
    return gguf.toF32(t, a);
}

fn loadOptF(a: std.mem.Allocator, f: *const gguf.File, comptime fmt: []const u8, i: usize) !?[]f32 {
    var buf: [64]u8 = undefined;
    const name = try std.fmt.bufPrint(&buf, fmt, .{i});
    const t = f.tensor(name) orelse return null;
    return try loadF(a, t);
}

// ---------------------------------------------------------------------------------------
// Template-driven scene extraction. The model never writes JSON syntax: we feed the
// scaffolding as forced tokens and let it fill the values, choosing enums and digits by
// comparing logits, so a small model cannot hand back placeholders or comments.
// ---------------------------------------------------------------------------------------

pub const Scene = struct {
    start_s: f32,
    summary: []u8,
    music: enum { score, silence, song },
    mood: []u8,
    intensity: u8,
    song_artist: []u8,
    song_title: []u8,
};

pub const SceneOptions = struct {
    max_scenes: usize = 4,
    min_scenes: usize = 1,
    /// Longest free-text field, in tokens.
    max_field_tokens: usize = 40,
};

/// The moods a scene can ask for. A fixed vocabulary, each starting with a different
/// word so the model can pick one by its first token, and so the music side can score
/// every track against every mood once, at scan time, instead of per book.
pub const moods = [_][]const u8{
    "warm solo piano, slow, tender",
    "driving percussion, fast, urgent",
    "tense low strings, slow pulse, ominous",
    "soft ambient pad, still, curious",
    "bright acoustic guitar, light, hopeful",
    "dark drones, sparse, dread",
    "playful pizzicato, quick, mischievous",
    "epic brass and drums, big, triumphant",
    "gentle harp and flute, calm, wonder",
    "pulsing synth, steady, focused",
    "melancholy cello, slow, grief",
    "shimmering strings, rising, awe",
    "quiet guitar, unhurried, nostalgic",
    "rhythmic marching drums, determined",
    "eerie high strings, uneasy, mystery",
    "lively folk fiddle, upbeat, festive",
};

fn freeSlice(gpa: std.mem.Allocator, list: []Scene) void {
    for (list) |s| {
        gpa.free(s.summary);
        gpa.free(s.mood);
        gpa.free(s.song_artist);
        gpa.free(s.song_title);
    }
    gpa.free(list);
}

const music_options = [_][]const u8{ "score", "silence", "song" };
const digit_options = [_][]const u8{ "1", "2", "3", "4", "5" };
const more_options = [_][]const u8{ ",", "]" };

pub const Filler = struct {
    m: *Model,
    gpa: std.mem.Allocator,
    recent: std.ArrayList(u32) = .empty,
    /// Option scores under a neutral prompt, subtracted when choosing so an option's
    /// own likelihood ("warm solo piano" is just a likely phrase) doesn't decide.
    prior_mood: [moods.len]f32 = @splat(0),
    prior_digit: [5]f32 = @splat(0),
    prior_more: [2]f32 = @splat(0),
    prior_music: [3]f32 = @splat(0),

    fn feed(f: *Filler, text: []const u8) !void {
        const toks = try f.m.tok.encode(f.gpa, text);
        defer f.gpa.free(toks);
        try f.m.prompt(toks);
    }

    /// Generate free text until a stop character shows up; returns it without the stop.
    fn freeText(f: *Filler, max_tokens: usize, stops: []const u8) ![]u8 {
        var out: std.ArrayList(u8) = .empty;
        errdefer out.deinit(f.gpa);
        const quote = f.m.tok.ids.get("\"") orelse 0;
        var i: usize = 0;
        while (i < max_tokens) : (i += 1) {
            f.m.penalize(f.recent.items);
            // the field must not be empty: no closing quote as the first token
            if (i == 0) f.m.logits[quote] = -std.math.inf(f32);
            if (f.m.tok.im_end) |e| f.m.logits[e] = -std.math.inf(f32);
            f.m.logits[f.m.tok.eos] = -std.math.inf(f32);
            const t = f.m.argmax();
            const text = f.m.tok.text(t);
            var cut: ?usize = null;
            for (text, 0..) |c, k| if (std.mem.indexOfScalar(u8, stops, c) != null) {
                cut = k;
                break;
            };
            if (cut) |k| {
                try out.appendSlice(f.gpa, text[0..k]);
                break;
            }
            try out.appendSlice(f.gpa, text);
            try f.recent.append(f.gpa, t);
            if (f.m.n_past + 1 > f.m.n_ctx) break;
            f.m.forward(&.{t});
        }
        // the text in the context must match what we keep: re-feed the stop ourselves
        const trimmed = std.mem.trim(u8, out.items, " \n\t");
        const copy = try f.gpa.dupe(u8, trimmed);
        out.deinit(f.gpa);
        return copy;
    }

    /// Score each option: its first token's log-probability for short lists, the mean
    /// log-probability of its whole token sequence (teacher-forced, cache rolled back
    /// between trials) for long ones. `out` receives one score per option.
    fn score(f: *Filler, options: []const []const u8, out: []f32) !void {
        if (options.len <= 3) {
            for (options, 0..) |o, i| {
                const toks = try f.m.tok.encode(f.gpa, o);
                defer f.gpa.free(toks);
                out[i] = if (toks.len == 0) -std.math.inf(f32) else logProb(f.m.logits, toks[0]);
            }
            return;
        }
        const mark = f.m.n_past;
        const saved = try f.gpa.dupe(f32, f.m.logits);
        defer f.gpa.free(saved);
        for (options, 0..) |o, i| {
            const toks = try f.m.tok.encode(f.gpa, o);
            defer f.gpa.free(toks);
            if (toks.len == 0) {
                out[i] = -std.math.inf(f32);
                continue;
            }
            @memcpy(f.m.logits, saved);
            f.m.n_past = mark;
            var sum: f32 = 0;
            for (toks) |t| {
                sum += logProb(f.m.logits, t);
                f.m.forward(&.{t});
            }
            out[i] = sum / @as(f32, @floatFromInt(toks.len));
        }
        @memcpy(f.m.logits, saved);
        f.m.n_past = mark;
    }

    /// Pick the option whose score beats its neutral-prompt prior by the most.
    fn choose(f: *Filler, options: []const []const u8, prior: []const f32) !usize {
        var sc: [moods.len]f32 = undefined;
        try f.score(options, sc[0..options.len]);
        var best: usize = 0;
        var best_v: f32 = -std.math.inf(f32);
        for (0..options.len) |i| {
            const v = sc[i] - prior[i];
            if (v > best_v) {
                best_v = v;
                best = i;
            }
        }
        return best;
    }

    /// Measure every option set under a neutral prompt so `choose` can discount it.
    fn calibrate(f: *Filler) !void {
        const toks = try f.m.chatPrompt(f.gpa, "You are a careful assistant that answers with JSON only.", "Describe one scene from a book as JSON with start, summary, music, mood and intensity.");
        defer f.gpa.free(toks);
        f.m.reset();
        try f.m.prompt(toks);
        try f.feed("{\"scenes\": [{\"start\": 0, \"summary\": \"A scene.\", \"music\": \"");
        try f.score(&music_options, &f.prior_music);
        try f.feed("score\", \"mood\": \"");
        try f.score(&moods, &f.prior_mood);
        try f.feed("soft ambient pad, still, curious\", \"intensity\": ");
        try f.score(&digit_options, &f.prior_digit);
        try f.feed("2}");
        try f.score(&more_options, &f.prior_more);
    }
};

/// Run the scene prompt for one window of `lines` (already formatted "[m:ss = s] text"
/// lines) and return the scenes. `first` is the time the first scene must start at.
pub fn scenes(m: *Model, gpa: std.mem.Allocator, chapter: []const u8, title: []const u8, first: u32, lines: []const u8, opts: SceneOptions) ![]Scene {
    var mood_buf: [2048]u8 = undefined;
    var mood_list: []u8 = mood_buf[0..0];
    for (moods, 0..) |mo, i| {
        const w = try std.fmt.bufPrint(mood_buf[mood_list.len..], "{s}\"{s}\"", .{ if (i == 0) "" else ", ", mo });
        mood_list = mood_buf[0 .. mood_list.len + w.len];
    }
    const head = try std.fmt.allocPrint(gpa,
        \\You are scoring an audiobook with background music, like a film composer.
        \\Below is part of chapter {s} ("{s}"), as timed lines. Times are seconds from the start of the chapter.
        \\Split it into scenes, in order. A scene is usually 2 to 8 minutes and changes when the mood, place, or action changes. This part is a few minutes long, so it has between 1 and 4 scenes.
        \\For each scene give:
        \\- start: the time of the line where it begins (the first scene starts at {d})
        \\- summary: one short sentence of what happens
        \\- music: "score" for background music, "silence" for moments that need none (a quiet, intimate, or emotionally raw exchange, a pause for effect), or "song" only when the text says a specific real song is playing
        \\- mood: for "score", the music to play, one of these exactly: {s}
        \\- intensity: 1 (barely there) to 5 (full action). Conversation and explanation are 1 or 2; danger or action is 4 or 5.
        \\- song_artist and song_title: only for "song", the song the text names
        \\Lines:
        \\{s}
    , .{ chapter, title, first, mood_list, lines });
    defer gpa.free(head);
    var f = Filler{ .m = m, .gpa = gpa };
    defer f.recent.deinit(gpa);
    try f.calibrate();

    const toks = try m.chatPrompt(gpa, "You are a careful assistant that answers with JSON only.", head);
    defer gpa.free(toks);
    m.reset();
    try m.prompt(toks);
    var list: std.ArrayList(Scene) = .empty;
    errdefer freeSlice(gpa, list.toOwnedSlice(gpa) catch &.{});

    try f.feed("{\"scenes\": [");
    var n: usize = 0;
    while (n < opts.max_scenes) : (n += 1) {
        var sc: Scene = .{ .start_s = @floatFromInt(first), .summary = &.{}, .music = .score, .mood = &.{}, .intensity = 2, .song_artist = &.{}, .song_title = &.{} };
        if (n == 0) {
            try f.feed(try std.fmt.bufPrint(&scratch_buf, "{{\"start\": {d}, \"summary\": \"", .{first}));
        } else {
            try f.feed("{\"start\": ");
            const num = try f.freeText(6, ",} \n");
            defer gpa.free(num);
            sc.start_s = std.fmt.parseFloat(f32, num) catch @floatFromInt(first);
            try f.feed(try std.fmt.bufPrint(&scratch_buf, "{d}, \"summary\": \"", .{@as(u32, @intFromFloat(@max(sc.start_s, 0)))}));
        }
        sc.summary = try f.freeText(opts.max_field_tokens, "\"\n");
        try f.feed(try std.fmt.bufPrint(&scratch_buf, "{s}\", \"music\": \"", .{sc.summary}));
        const music_opts = music_options;
        const mi = try f.choose(&music_opts, &f.prior_music);
        sc.music = switch (mi) {
            0 => .score,
            1 => .silence,
            else => .song,
        };
        try f.feed(try std.fmt.bufPrint(&scratch_buf, "{s}\", ", .{music_opts[mi]}));
        if (sc.music == .score) {
            try f.feed("\"mood\": \"");
            const mi2 = try f.choose(&moods, &f.prior_mood);
            sc.mood = try gpa.dupe(u8, moods[mi2]);
            try f.feed(try std.fmt.bufPrint(&scratch_buf, "{s}\", ", .{sc.mood}));
        } else if (sc.music == .song) {
            try f.feed("\"song_artist\": \"");
            sc.song_artist = try f.freeText(16, "\"\n");
            try f.feed(try std.fmt.bufPrint(&scratch_buf, "{s}\", \"song_title\": \"", .{sc.song_artist}));
            sc.song_title = try f.freeText(16, "\"\n");
            try f.feed(try std.fmt.bufPrint(&scratch_buf, "{s}\", ", .{sc.song_title}));
        }
        try f.feed("\"intensity\": ");
        const digits = digit_options;
        const di = try f.choose(&digits, &f.prior_digit);
        sc.intensity = @intCast(di + 1);
        try f.feed(try std.fmt.bufPrint(&scratch_buf, "{s}}}", .{digits[di]}));
        try list.append(gpa, sc);
        // another scene, or the end of the list? let the model say, within the bounds
        if (n + 1 >= opts.max_scenes) break;
        const more_opts = more_options;
        const more = if (n + 1 < opts.min_scenes) 0 else try f.choose(&more_opts, &f.prior_more);
        if (more == 1) break;
        try f.feed(", ");
    }
    return list.toOwnedSlice(gpa);
}

var scratch_buf: [4096]u8 = undefined;

fn logProb(logits: []const f32, t: u32) f32 {
    var mx: f32 = -std.math.inf(f32);
    for (logits) |v| mx = @max(mx, v);
    var sum: f32 = 0;
    for (logits) |v| sum += @exp(v - mx);
    return logits[t] - mx - @log(sum);
}
