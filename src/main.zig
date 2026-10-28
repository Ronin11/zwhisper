const std = @import("std");
const Io = std.Io;
const zw = @import("zwhisper");
const clock = zw.clock;

const usage =
    \\usage: zwhisper <command> ...
    \\
    \\  info <model>                        hyperparameters and tensor summary
    \\  tokens <model> <text>...            how the prompt tokenizer splits text
    \\  mel <model> <wav> [out.f32]         log-mel spectrogram stats, optionally dumped [n_mel][n_len]
    \\  logits <model> <wav> <out.f32>      logits after [sot] on the first window (compare with ref)
    \\  bench <threads>                     kernel micro-benchmarks
    \\  ggufinfo <model.gguf>               metadata and tensors of a GGUF file
    \\  llmtokens <model.gguf> <text>       BPE token ids for text
    \\  llm <model.gguf> [-t N] [--max N] [--ctx N] [--system S] [--rep P] [--raw] <prompt>
    \\                                      chat with a Qwen2 model, streaming
    \\  embed <minilm.gguf> <transcript.jsonl>... | --text T
    \\                                      sentence embedding per transcript (BERT/MiniLM)
    \\  features <model.gguf> <transcript.jsonl>... [-t N]
    \\                                      mean-pooled hidden state per transcript, one line each
    \\  scenes <model.gguf> <transcript.jsonl> [-t N] [--ctx N]
    \\                                      scenes for a transcript, template-driven
    \\  transcribe <model> <wav> [-t N] [--json] [--no-context] [--no-voices]
    \\                                      [--prompt TEXT] [--hotwords a,b,c] [--no-learn] [--profile]
    \\                                      stream segments as text or JSON lines
    \\
;

pub fn main(init: std.process.Init) !void {
    const arena = init.arena.allocator();
    const io = init.io;
    const args = try init.minimal.args.toSlice(arena);
    var buf: [64 * 1024]u8 = undefined;
    var stdout_file: Io.File.Writer = .init(.stdout(), io, &buf);
    const out = &stdout_file.interface;
    defer out.flush() catch {};

    if (args.len < 3) {
        std.debug.print("{s}", .{usage});
        std.process.exit(2);
    }
    const gpa = std.heap.smp_allocator;
    const cmd = args[1];

    if (std.mem.eql(u8, cmd, "embed")) {
        return embedMain(io, out, gpa, args);
    }
    if (std.mem.eql(u8, cmd, "llm") or std.mem.eql(u8, cmd, "ggufinfo") or std.mem.eql(u8, cmd, "llmtokens") or std.mem.eql(u8, cmd, "scenes") or std.mem.eql(u8, cmd, "features")) {
        return llmMain(io, out, gpa, cmd, args);
    }

    if (std.mem.eql(u8, cmd, "bench")) {
        const n = try std.fmt.parseInt(usize, args[2], 10);
        return zw.bench.run(gpa, out, n);
    }

    var mapped = try MappedFile.open(io, args[2]);
    defer mapped.close();
    var m = zw.Model.parse(gpa, mapped.data) catch |err| {
        std.debug.print("zwhisper: {s}: {t}\n", .{ args[2], err });
        std.process.exit(1);
    };
    defer m.deinit();

    if (std.mem.eql(u8, cmd, "tokens")) {
        for (args[3..]) |text| {
            const toks = try zw.tokenizer.encode(gpa, &m, text);
            defer gpa.free(toks);
            try out.print("{s} ->", .{text});
            for (toks) |t| try out.print(" {d}'{s}'", .{ t, m.tokenText(t) });
            try out.writeAll("\n");
        }
        return;
    }

    if (std.mem.eql(u8, cmd, "info")) {
        const hp = m.hp;
        try out.print("n_vocab={d} n_audio_ctx={d} n_audio_state={d} n_audio_head={d} n_audio_layer={d}\n", .{ hp.n_vocab, hp.n_audio_ctx, hp.n_audio_state, hp.n_audio_head, hp.n_audio_layer });
        try out.print("n_text_ctx={d} n_text_state={d} n_text_head={d} n_text_layer={d} n_mels={d} ftype={d}\n", .{ hp.n_text_ctx, hp.n_text_state, hp.n_text_head, hp.n_text_layer, hp.n_mels, hp.ftype });
        try out.print("filters {d}x{d}, vocab {d}, tensors {d}\n", .{ m.filters_n_mel, m.filters_n_fft, m.vocab.len, m.tensors.count() });
        var params: usize = 0;
        var it = m.tensors.iterator();
        while (it.next()) |kv| params += kv.value_ptr.count();
        try out.print("parameters {d:.1} M\n", .{@as(f64, @floatFromInt(params)) / 1e6});
        try out.print("token 50257 = {s}, token 220 = '{s}', token 329 = '{s}'\n", .{ m.tokenText(50257), m.tokenText(220), m.tokenText(329) });
        return;
    }

    if (args.len < 4) {
        std.debug.print("{s}", .{usage});
        std.process.exit(2);
    }
    var wav_file = try MappedFile.open(io, args[3]);
    defer wav_file.close();
    const samples = zw.wav.decode(gpa, wav_file.data) catch |err| {
        std.debug.print("zwhisper: {s}: {t} (need 16 kHz mono 16-bit PCM)\n", .{ args[3], err });
        std.process.exit(1);
    };
    defer gpa.free(samples);

    var timer = clock.Timer.start();
    var spec = try zw.mel.logMel(gpa, samples, m.filters, m.hp.n_mels);
    defer spec.deinit(gpa);
    const mel_ns = timer.lap();

    if (std.mem.eql(u8, cmd, "mel")) {
        var mn: f32 = std.math.inf(f32);
        var mx: f32 = -std.math.inf(f32);
        var sum: f64 = 0;
        for (spec.data) |v| {
            mn = @min(mn, v);
            mx = @max(mx, v);
            sum += v;
        }
        try out.print("samples={d} n_mel={d} n_len={d} n_len_org={d} min={d:.4} max={d:.4} mean={d:.5} in {d:.1} ms\n", .{ samples.len, spec.n_mel, spec.n_len, spec.n_len_org, mn, mx, sum / @as(f64, @floatFromInt(spec.data.len)), @as(f64, @floatFromInt(mel_ns)) / 1e6 });
        if (args.len > 4) try writeFile(io, args[4], std.mem.sliceAsBytes(spec.data));
        return;
    }

    var threads: usize = @max(1, (std.Thread.getCpuCount() catch 4) - 1);
    var json = false;
    var context = true;
    var voices = true;
    var learn = true;
    var prompt_text: ?[]const u8 = null;
    var hotwords: ?[]const u8 = null;
    var hotword_bias: ?f32 = null;
    var i: usize = 4;
    while (i < args.len) : (i += 1) {
        if (std.mem.eql(u8, args[i], "-t") and i + 1 < args.len) {
            i += 1;
            threads = try std.fmt.parseInt(usize, args[i], 10);
        } else if (std.mem.eql(u8, args[i], "--json")) {
            json = true;
        } else if (std.mem.eql(u8, args[i], "--no-context")) {
            context = false;
        } else if (std.mem.eql(u8, args[i], "--no-voices")) {
            voices = false;
        } else if (std.mem.eql(u8, args[i], "--no-learn")) {
            learn = false;
        } else if (std.mem.eql(u8, args[i], "--prompt") and i + 1 < args.len) {
            i += 1;
            prompt_text = args[i];
        } else if (std.mem.eql(u8, args[i], "--hotwords") and i + 1 < args.len) {
            i += 1;
            hotwords = args[i];
        } else if (std.mem.eql(u8, args[i], "--hotword-bias") and i + 1 < args.len) {
            i += 1;
            hotword_bias = try std.fmt.parseFloat(f32, args[i]);
        } else if (std.mem.eql(u8, args[i], "--profile")) {
            zw.ops.profile.enabled = true;
        } else if (std.mem.eql(u8, cmd, "logits") and i == 4) {
            // output path, handled below
        } else {
            std.debug.print("zwhisper: unknown option {s}\n", .{args[i]});
            std.process.exit(2);
        }
    }

    const engine = try zw.Engine.init(gpa, &m, threads);
    defer engine.deinit();
    if (prompt_text) |t| try engine.setPrompt(t);
    if (hotwords) |h| try engine.setHotwords(h);
    if (hotword_bias) |b| engine.hotword_bias = b;
    const load_ns = timer.lap();

    if (std.mem.eql(u8, cmd, "logits")) {
        if (args.len < 5) {
            std.debug.print("{s}", .{usage});
            std.process.exit(2);
        }
        const enc_out = engine.enc.encode(engine.pool, &spec, 0);
        const enc_ns = timer.lap();
        engine.dec.setAudio(engine.pool, enc_out);
        engine.dec.reset();
        engine.dec.forward(engine.pool, &.{m.tok.sot}, engine.logits);
        const dec_ns = timer.lap();
        try writeFile(io, args[4], std.mem.sliceAsBytes(engine.logits));
        var best: usize = 0;
        for (engine.logits, 0..) |v, k| if (v > engine.logits[best]) {
            best = k;
        };
        try out.print("wrote {d} logits; argmax {d} '{s}' = {d:.6}; mel {d:.1} ms, load {d:.1} ms, encode {d:.1} ms, decode {d:.1} ms\n", .{ engine.logits.len, best, m.tokenText(@intCast(best)), engine.logits[best], ms(mel_ns), ms(load_ns), ms(enc_ns), ms(dec_ns) });
        return;
    }

    if (std.mem.eql(u8, cmd, "transcribe")) {
        const Sink = struct {
            out: *Io.Writer,
            json: bool,
            fn emit(s: *const @This(), seg: zw.Segment) void {
                if (s.json) {
                    s.out.print("{{\"t0\":{d},\"t1\":{d},\"voice\":{d},\"quoted\":{},\"pitch\":{d:.0},\"dist\":{d:.3},\"text\":", .{ seg.t0, seg.t1, seg.voice, seg.quoted, seg.pitch_hz, seg.voice_dist }) catch return;
                    jsonString(s.out, seg.text) catch return;
                    s.out.writeAll(",\"tokens\":[") catch return;
                    for (seg.tokens, 0..) |t, k| s.out.print("{s}{d}", .{ if (k == 0) "" else ",", t }) catch return;
                    s.out.writeAll("]}\n") catch return;
                } else {
                    s.out.print("[{d:0>2}:{d:0>2}.{d:0>2} --> {d:0>2}:{d:0>2}.{d:0>2}] v{d:<2} {d:3.0}Hz d={d:.2} {s}{s}\n", .{ seg.t0 / 6000, (seg.t0 / 100) % 60, seg.t0 % 100, seg.t1 / 6000, (seg.t1 / 100) % 60, seg.t1 % 100, seg.voice, seg.pitch_hz, seg.voice_dist, @as([]const u8, if (seg.quoted) "Q " else "  "), seg.text }) catch return;
                }
                s.out.flush() catch {};
            }
        };
        const sink = Sink{ .out = out, .json = json };
        const stats = try engine.transcribe(&spec, samples, .{ .condition_on_previous = context, .voices = voices, .learn_names = learn }, &sink, Sink.emit);
        const total_ns = timer.lap();
        if (zw.ops.profile.enabled) {
            const pr = zw.ops.profile;
            std.debug.print("zwhisper: profile linear {d:.0} ms, attention {d:.0} ms, gelu {d:.0} ms, norm {d:.0} ms\n", .{ ms(pr.linear), ms(pr.attention), ms(pr.gelu), ms(pr.norm) });
        }
        const audio_s = @as(f64, @floatFromInt(samples.len)) / 16000.0;
        if (engine.hotwords.items.len > 0) {
            std.debug.print("zwhisper: hotwords:", .{});
            for (engine.hotwords.items) |h| std.debug.print(" {s}", .{h.text});
            std.debug.print("\n", .{});
        }
        if (stats.dropped > 0) std.debug.print("zwhisper: dropped {d} segment(s) that recited the prompt\n", .{stats.dropped});
        std.debug.print("zwhisper: {d:.1} s audio, {d} windows, {d} tokens, {d} segments, {d} voices; mel {d:.0} ms, load {d:.0} ms, encode {d:.0} ms, decode {d:.0} ms, voices {d:.0} ms, total {d:.0} ms ({d:.1}x realtime, {d} threads)\n", .{
            audio_s, stats.windows, stats.tokens, stats.segments, stats.voices, ms(mel_ns), ms(load_ns), ms(stats.encode_ns), ms(stats.decode_ns), ms(stats.voice_ns), ms(total_ns), audio_s / (@as(f64, @floatFromInt(total_ns)) / 1e9), threads,
        });
        return;
    }
    std.debug.print("{s}", .{usage});
    std.process.exit(2);
}

fn embedMain(io: Io, out: *Io.Writer, gpa: std.mem.Allocator, args: []const []const u8) !void {
    var mapped = try MappedFile.open(io, args[2]);
    defer mapped.close();
    var f = try zw.gguf.File.parse(gpa, mapped.data);
    defer f.deinit();
    var threads: usize = @max(1, (std.Thread.getCpuCount() catch 4) - 1);
    var paths: std.ArrayList([]const u8) = .empty;
    defer paths.deinit(gpa);
    var raw_text: ?[]const u8 = null;
    var i: usize = 3;
    while (i < args.len) : (i += 1) {
        if (std.mem.eql(u8, args[i], "-t") and i + 1 < args.len) {
            i += 1;
            threads = try std.fmt.parseInt(usize, args[i], 10);
        } else if (std.mem.eql(u8, args[i], "--text") and i + 1 < args.len) {
            i += 1;
            raw_text = args[i];
        } else try paths.append(gpa, args[i]);
    }
    const m = try zw.bert.Model.load(gpa, &f, threads);
    defer m.deinit();
    const vec = try gpa.alloc(f32, m.n_embd);
    defer gpa.free(vec);
    if (raw_text) |t| {
        const ids = try m.encode(gpa, t);
        defer gpa.free(ids);
        try out.print("tokens:", .{});
        for (ids) |id| try out.print(" {s}", .{m.tokens[id]});
        try out.writeAll("\n");
        var timer = clock.Timer.start();
        try m.embed(gpa, t, vec);
        const ns = timer.lap();
        try out.print("embedding[0..8] =", .{});
        for (vec[0..8]) |v| try out.print(" {d:.4}", .{v});
        try out.print("  ({d:.1} ms)\n", .{ms(ns)});
        return;
    }
    for (paths.items) |path| {
        var tf = MappedFile.open(io, path) catch continue;
        defer tf.close();
        var text: std.ArrayList(u8) = .empty;
        defer text.deinit(gpa);
        var it = std.mem.splitScalar(u8, tf.data, '\n');
        while (it.next()) |line| {
            const ts = std.mem.indexOf(u8, line, "\"text\":\"") orelse continue;
            const te = std.mem.lastIndexOf(u8, line, "\",\"tokens\"") orelse continue;
            try text.appendSlice(gpa, std.mem.trim(u8, line[ts + 8 .. te], " "));
            try text.append(gpa, ' ');
        }
        try m.embed(gpa, text.items, vec);
        try out.print("{s}", .{path});
        for (vec) |v| try out.print(" {d:.5}", .{v});
        try out.writeAll("\n");
        try out.flush();
    }
}

fn llmMain(io: Io, out: *Io.Writer, gpa: std.mem.Allocator, cmd: []const u8, args: []const []const u8) !void {
    var mapped = try MappedFile.open(io, args[2]);
    defer mapped.close();
    var f = zw.gguf.File.parse(gpa, mapped.data) catch |err| {
        std.debug.print("zwhisper: {s}: {t}\n", .{ args[2], err });
        std.process.exit(1);
    };
    defer f.deinit();
    if (std.mem.eql(u8, cmd, "ggufinfo")) {
        try out.print("gguf v{d}: {d} metadata keys, {d} tensors\n", .{ f.version, f.kv.count(), f.tensors.count() });
        var it = f.kv.iterator();
        while (it.next()) |kv| {
            const v = kv.value_ptr.*;
            switch (v) {
                .string => |s| try out.print("  {s} = \"{s}\"\n", .{ kv.key_ptr.*, s[0..@min(s.len, 80)] }),
                .array => |a| try out.print("  {s} = array[{d}] of type {d}\n", .{ kv.key_ptr.*, a.count, a.elem }),
                inline else => |x| try out.print("  {s} = {any}\n", .{ kv.key_ptr.*, x }),
            }
        }
        var tt = f.tensors.iterator();
        var shown: usize = 0;
        while (tt.next()) |kv| : (shown += 1) {
            if (shown >= 12 and !std.mem.eql(u8, args[args.len - 1], "--all")) break;
            const t = kv.value_ptr.*;
            try out.print("  tensor {s} {t} [{d} {d} {d} {d}]\n", .{ kv.key_ptr.*, t.ty, t.ne[0], t.ne[1], t.ne[2], t.ne[3] });
        }
        return;
    }
    var threads: usize = @max(1, (std.Thread.getCpuCount() catch 4) - 1);
    var max_new: usize = 512;
    var n_ctx: usize = 4096;
    var system: ?[]const u8 = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.";
    var raw = false;
    var prompt_text: []const u8 = "";
    var i: usize = 3;
    while (i < args.len) : (i += 1) {
        if (std.mem.eql(u8, args[i], "-t") and i + 1 < args.len) {
            i += 1;
            threads = try std.fmt.parseInt(usize, args[i], 10);
        } else if (std.mem.eql(u8, args[i], "--max") and i + 1 < args.len) {
            i += 1;
            max_new = try std.fmt.parseInt(usize, args[i], 10);
        } else if (std.mem.eql(u8, args[i], "--ctx") and i + 1 < args.len) {
            i += 1;
            n_ctx = try std.fmt.parseInt(usize, args[i], 10);
        } else if (std.mem.eql(u8, args[i], "--system") and i + 1 < args.len) {
            i += 1;
            system = if (args[i].len == 0) null else args[i];
        } else if (std.mem.eql(u8, args[i], "--raw")) {
            raw = true;
        } else if (std.mem.eql(u8, args[i], "--profile")) {
            zw.ops.profile.enabled = true;
        } else if (std.mem.eql(u8, args[i], "--rep") and i + 1 < args.len) {
            i += 1;
            zw.llm.Model.repeat_penalty = try std.fmt.parseFloat(f32, args[i]);
        } else {
            prompt_text = args[i];
        }
    }
    var timer = clock.Timer.start();
    const m = try zw.llm.Model.load(gpa, &f, .{ .n_ctx = n_ctx, .threads = threads });
    defer m.deinit();
    const load_ns = timer.lap();
    if (std.mem.eql(u8, cmd, "llmtokens")) {
        const toks = try m.tok.encode(gpa, prompt_text);
        defer gpa.free(toks);
        for (toks) |t| try out.print("{d}'{s}' ", .{ t, m.tok.text(t) });
        try out.writeAll("\n");
        return;
    }
    if (std.mem.eql(u8, cmd, "features")) {
        // every extra arg is a transcript JSONL; one line of feature floats per file
        for (args[3..]) |path| {
            if (path.len > 0 and path[0] == '-') continue;
            var tf = MappedFile.open(io, path) catch continue;
            defer tf.close();
            var text: std.ArrayList(u8) = .empty;
            defer text.deinit(gpa);
            var it = std.mem.splitScalar(u8, tf.data, '\n');
            while (it.next()) |line| {
                const ts = std.mem.indexOf(u8, line, "\"text\":\"") orelse continue;
                const te = std.mem.lastIndexOf(u8, line, "\",\"tokens\"") orelse continue;
                try text.appendSlice(gpa, std.mem.trim(u8, line[ts + 8 .. te], " "));
                try text.append(gpa, '\n');
            }
            const toks = try m.tok.encode(gpa, text.items);
            defer gpa.free(toks);
            const feat = try gpa.alloc(f32, m.n_embd);
            defer gpa.free(feat);
            try m.features(toks[0..@min(toks.len, m.n_ctx)], feat);
            try out.print("{s}", .{path});
            for (feat) |v| try out.print(" {d:.5}", .{v});
            try out.writeAll("\n");
            try out.flush();
        }
        return;
    }

    if (std.mem.eql(u8, cmd, "scenes")) {
        // prompt_text is a transcript JSONL path; lines are built as the worker does
        var tf = try MappedFile.open(io, prompt_text);
        defer tf.close();
        var lines: std.ArrayList(u8) = .empty;
        defer lines.deinit(gpa);
        var it = std.mem.splitScalar(u8, tf.data, '\n');
        while (it.next()) |line| {
            if (line.len == 0) continue;
            const t0s = std.mem.indexOf(u8, line, "\"t0\":") orelse continue;
            const t0 = std.fmt.parseInt(u32, line[t0s + 5 .. std.mem.indexOfScalarPos(u8, line, t0s, ',').?], 10) catch continue;
            const ts = std.mem.indexOf(u8, line, "\"text\":\"") orelse continue;
            const te = std.mem.lastIndexOf(u8, line, "\",\"tokens\"") orelse std.mem.lastIndexOfScalar(u8, line, '"').?;
            const text = std.mem.trim(u8, line[ts + 8 .. te], " ");
            const sec = t0 / 100;
            var hb: [48]u8 = undefined;
            try lines.appendSlice(gpa, try std.fmt.bufPrint(&hb, "[{d}:{d:0>2} = {d}] ", .{ sec / 60, sec % 60, sec }));
            // unescape the few JSON escapes our own writer produces
            var k: usize = 0;
            while (k < text.len) : (k += 1) {
                if (text[k] == '\\' and k + 1 < text.len) {
                    k += 1;
                    try lines.append(gpa, switch (text[k]) {
                        'n' => '\n',
                        else => text[k],
                    });
                } else try lines.append(gpa, text[k]);
            }
            try lines.append(gpa, '\n');
        }
        var timer2 = clock.Timer.start();
        const scs = try zw.llm.scenes(m, gpa, "", "untitled", 0, lines.items, .{});
        const ns = timer2.lap();
        for (scs) |sc| {
            try out.print("{d:6.0}s  {t:<7} {d}  {s}\n", .{ sc.start_s, sc.music, sc.intensity, sc.summary });
            if (sc.mood.len > 0) try out.print("        mood: {s}\n", .{sc.mood});
            if (sc.song_title.len > 0) try out.print("        song: {s} - {s}\n", .{ sc.song_artist, sc.song_title });
        }
        std.debug.print("zwhisper: {d} scenes in {d:.1} s ({d} prompt tokens in context)\n", .{ scs.len, @as(f64, @floatFromInt(ns)) / 1e9, m.n_past });
        return;
    }
    const toks = if (raw) try m.tok.encode(gpa, prompt_text) else try m.chatPrompt(gpa, system, prompt_text);
    defer gpa.free(toks);
    const Sink = struct {
        out: *Io.Writer,
        fn emit(s: *const @This(), text: []const u8) void {
            s.out.writeAll(text) catch {};
            s.out.flush() catch {};
        }
    };
    const sink = Sink{ .out = out };
    const st = try m.generate(toks, max_new, &sink, Sink.emit);
    try out.writeAll("\n");
    try out.flush();
    if (zw.ops.profile.enabled) {
        const pr = zw.ops.profile;
        std.debug.print("zwhisper: profile linear {d:.0} ms, attention {d:.0} ms, norm {d:.0} ms\n", .{ ms(pr.linear), ms(pr.attention), ms(pr.norm) });
    }
    std.debug.print("zwhisper: {d} layers, {d} embd, {d}/{d} heads, vocab {d}; load {d:.0} ms, prompt {d} tokens in {d:.0} ms ({d:.0} tok/s), generated {d} tokens in {d:.0} ms ({d:.1} tok/s), {d} threads\n", .{
        m.n_layer, m.n_embd, m.n_head, m.n_head_kv, m.n_vocab, ms(load_ns), st.prompt_tokens, ms(st.prompt_ns), @as(f64, @floatFromInt(st.prompt_tokens)) / (@as(f64, @floatFromInt(st.prompt_ns)) / 1e9), st.new_tokens, ms(st.gen_ns), @as(f64, @floatFromInt(st.new_tokens)) / (@as(f64, @floatFromInt(@max(st.gen_ns, 1))) / 1e9), threads,
    });
}

fn ms(ns: u64) f64 {
    return @as(f64, @floatFromInt(ns)) / 1e6;
}

fn jsonString(w: *Io.Writer, s: []const u8) !void {
    try w.writeByte('"');
    for (s) |c| switch (c) {
        '"' => try w.writeAll("\\\""),
        '\\' => try w.writeAll("\\\\"),
        '\n' => try w.writeAll("\\n"),
        else => if (c < 0x20) try w.print("\\u{x:0>4}", .{c}) else try w.writeByte(c),
    };
    try w.writeByte('"');
}

fn writeFile(io: Io, path: []const u8, bytes: []const u8) !void {
    const file = try Io.Dir.cwd().createFile(io, path, .{});
    defer file.close(io);
    var buf: [64 * 1024]u8 = undefined;
    var w: Io.File.Writer = .init(file, io, &buf);
    try w.interface.writeAll(bytes);
    try w.interface.flush();
}

const MappedFile = struct {
    data: []align(std.heap.page_size_min) const u8,

    fn open(io: Io, path: []const u8) !MappedFile {
        const file = try Io.Dir.cwd().openFile(io, path, .{});
        defer file.close(io);
        const len = try file.length(io);
        if (len == 0) return error.EmptyFile;
        const data = try std.posix.mmap(null, @intCast(len), .{ .READ = true }, .{ .TYPE = .PRIVATE }, file.handle, 0);
        return .{ .data = data };
    }

    fn close(f: *MappedFile) void {
        std.posix.munmap(f.data);
        f.* = undefined;
    }
};
