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
