//! C ABI for embedding zwhisper (see include/zwhisper.h). Built as a static library.
//!
//! One handle holds the mapped model file, the parsed model and an engine. Calls on a
//! handle are not thread-safe with each other; use one handle per thread.
const std = @import("std");
const zw = @import("root.zig");
const linux = std.os.linux;

const gpa = std.heap.smp_allocator;

pub const Handle = struct {
    bytes: []align(std.heap.page_size_min) const u8,
    model: zw.Model,
    engine: *zw.Engine,
};

/// Mirrors `zw_segment` in the header.
pub const Segment = extern struct {
    t0_ms: i64,
    t1_ms: i64,
    text: [*]const u8,
    text_len: usize,
    tokens: [*]const u32,
    n_tokens: usize,
    voice: u32,
    quoted: bool,
    pitch_hz: f32,
    voice_dist: f32,
};

pub const Stats = extern struct {
    windows: u64,
    tokens: u64,
    segments: u64,
    voices: u64,
    encode_ns: u64,
    decode_ns: u64,
    voice_ns: u64,
};

pub const Emit = *const fn (?*anyopaque, *const Segment) callconv(.c) void;

fn mapFile(path: [*:0]const u8) ![]align(std.heap.page_size_min) const u8 {
    const fd_rc = linux.open(path, .{ .ACCMODE = .RDONLY, .CLOEXEC = true }, 0);
    if (linux.errno(fd_rc) != .SUCCESS) return error.OpenFailed;
    const fd: linux.fd_t = @intCast(fd_rc);
    defer _ = linux.close(fd);
    const end = linux.lseek(fd, 0, linux.SEEK.END);
    if (linux.errno(end) != .SUCCESS) return error.StatFailed;
    const len: usize = @intCast(end);
    if (len == 0) return error.EmptyFile;
    const rc = linux.mmap(null, len, .{ .READ = true }, .{ .TYPE = .PRIVATE }, fd, 0);
    if (linux.errno(rc) != .SUCCESS) return error.MmapFailed;
    const ptr: [*]align(std.heap.page_size_min) const u8 = @ptrFromInt(rc);
    return ptr[0..len];
}

/// Open a ggml model file. `threads` <= 0 picks the CPU count minus one.
export fn zw_open(path: [*:0]const u8, threads: c_int) ?*Handle {
    const h = gpa.create(Handle) catch return null;
    errdefer gpa.destroy(h);
    h.bytes = mapFile(path) catch {
        gpa.destroy(h);
        return null;
    };
    h.model = zw.Model.parse(gpa, h.bytes) catch {
        _ = linux.munmap(h.bytes.ptr, h.bytes.len);
        gpa.destroy(h);
        return null;
    };
    const n: usize = if (threads > 0) @intCast(threads) else @max(1, (std.Thread.getCpuCount() catch 4) - 1);
    h.engine = zw.Engine.init(gpa, &h.model, n) catch {
        h.model.deinit();
        _ = linux.munmap(h.bytes.ptr, h.bytes.len);
        gpa.destroy(h);
        return null;
    };
    return h;
}

export fn zw_close(h: *Handle) void {
    h.engine.deinit();
    h.model.deinit();
    _ = linux.munmap(h.bytes.ptr, h.bytes.len);
    gpa.destroy(h);
}

/// Forget every voice seen so far.
export fn zw_reset_voices(h: *Handle) void {
    h.engine.resetVoices() catch {};
}

export fn zw_n_voices(h: *Handle) u32 {
    return @intCast(h.engine.voices.count());
}

const Sink = struct {
    emit: Emit,
    user: ?*anyopaque,
    offset_ms: i64,
    fn call(s: *const Sink, seg: zw.Segment) void {
        const out = Segment{
            .t0_ms = s.offset_ms + @as(i64, seg.t0) * 10,
            .t1_ms = s.offset_ms + @as(i64, seg.t1) * 10,
            .text = seg.text.ptr,
            .text_len = seg.text.len,
            .tokens = seg.tokens.ptr,
            .n_tokens = seg.tokens.len,
            .voice = seg.voice,
            .quoted = seg.quoted,
            .pitch_hz = seg.pitch_hz,
            .voice_dist = seg.voice_dist,
        };
        s.emit(s.user, &out);
    }
};

/// Transcribe `n` samples of 16 kHz mono f32. Segment times are offset by `offset_ms`.
/// `stop`, when non-null, is polled between tokens; set it to end early.
/// Returns 0 on success, 1 when stopped, 2 on error.
export fn zw_transcribe(h: *Handle, samples: [*]const f32, n: usize, offset_ms: i64, voices: bool, split_quotes: bool, condition_on_previous: bool, learn_names: bool, stop: ?*const bool, emit: Emit, user: ?*anyopaque, stats_out: ?*Stats) c_int {
    const audio = samples[0..n];
    var spec = zw.mel.logMel(gpa, audio, h.model.filters, h.model.hp.n_mels) catch return 2;
    defer spec.deinit(gpa);
    const sink = Sink{ .emit = emit, .user = user, .offset_ms = offset_ms };
    const stop_atomic: ?*const std.atomic.Value(bool) = if (stop) |p| @ptrCast(p) else null;
    const stats = h.engine.transcribe(&spec, audio, .{
        .voices = voices,
        .split_quotes = split_quotes,
        .condition_on_previous = condition_on_previous,
        .learn_names = learn_names,
        .stop = stop_atomic,
    }, &sink, Sink.call) catch |err| return if (err == error.Stopped) 1 else 2;
    if (stats_out) |so| so.* = .{
        .windows = stats.windows,
        .tokens = stats.tokens,
        .segments = stats.segments,
        .voices = stats.voices,
        .encode_ns = stats.encode_ns,
        .decode_ns = stats.decode_ns,
        .voice_ns = stats.voice_ns,
    };
    return 0;
}

/// Text the decoder sees before every window (names spelt right, in the book's own
/// prose style, quotes included). Empty clears it. Returns 0, or 2 on failure.
export fn zw_set_prompt(h: *Handle, text: [*]const u8, len: usize) c_int {
    h.engine.setPrompt(text[0..len]) catch return 2;
    return 0;
}

/// Comma-separated words and phrases to favour while decoding. Replaces earlier ones
/// and forgets learned names. Returns 0, or 2 on failure.
export fn zw_set_hotwords(h: *Handle, list: [*]const u8, len: usize) c_int {
    h.engine.setHotwords(list[0..len]) catch return 2;
    return 0;
}

// ---- scene axes -------------------------------------------------------------------------

/// A scene model (indecis directory: config.json, indecis.json, model.safetensors,
/// tokenizer.json) with its mapped files.
pub const SceneHandle = struct {
    files: [4][]align(std.heap.page_size_min) const u8,
    model: *zw.scene.Model,
    energy: ?usize,
    valence: ?usize,
    silence: ?usize,
    answers: []zw.scene.Answer,
};

/// Mirrors `zw_scene_axes` in the header. Probabilities per level; the means are on the
/// level scale (energy 1..5, valence -1..1).
pub const SceneAxes = extern struct {
    silence: f32,
    energy: [5]f32,
    energy_mean: f32,
    valence: [3]f32,
    valence_mean: f32,
    n_tokens: u32,
};

const scene_files = [_][]const u8{ "config.json", "indecis.json", "model.safetensors", "tokenizer.json" };

fn openScene(dir: [*:0]const u8, threads: c_int) !*SceneHandle {
    const h = try gpa.create(SceneHandle);
    errdefer gpa.destroy(h);
    h.* = undefined;
    var mapped: usize = 0;
    errdefer for (h.files[0..mapped]) |f| {
        _ = linux.munmap(f.ptr, f.len);
    };
    const d = std.mem.span(dir);
    for (scene_files, 0..) |name, i| {
        var buf: [4096]u8 = undefined;
        const path = try std.fmt.bufPrint(&buf, "{s}/{s}\x00", .{ d, name });
        h.files[i] = try mapFile(@ptrCast(path.ptr));
        mapped += 1;
    }
    const n_threads: usize = if (threads > 0) @intCast(threads) else @max(1, (std.Thread.getCpuCount() catch 2) - 1);
    h.model = try zw.scene.Model.load(gpa, .{ .config = h.files[0], .meta = h.files[1], .weights = h.files[2], .tokenizer = h.files[3] }, n_threads);
    errdefer h.model.deinit();
    h.answers = try gpa.alloc(zw.scene.Answer, h.model.questions.len);
    h.energy = null;
    h.valence = null;
    h.silence = null;
    for (h.model.questions, 0..) |q, i| {
        if (std.mem.eql(u8, q.name, "energy") and q.options.len == 5) h.energy = i;
        if (std.mem.eql(u8, q.name, "valence") and q.options.len == 3) h.valence = i;
        if (std.mem.eql(u8, q.name, "silence") and q.kind == .noul) h.silence = i;
    }
    return h;
}

/// Open a scene model directory. `threads` <= 0 picks the CPU count minus one.
export fn zw_scene_open(dir: [*:0]const u8, threads: c_int) ?*SceneHandle {
    return openScene(dir, threads) catch null;
}

export fn zw_scene_close(h: *SceneHandle) void {
    gpa.free(h.answers);
    h.model.deinit();
    for (h.files) |f| _ = linux.munmap(f.ptr, f.len);
    gpa.destroy(h);
}

/// Rate one passage (UTF-8, about a minute of narration; longer text is truncated to the
/// model's window). Returns 0, 2 on failure, or 3 when the model lacks one of the axes.
export fn zw_scene_rate(h: *SceneHandle, text: [*]const u8, len: usize, out: *SceneAxes) c_int {
    const e = h.energy orelse return 3;
    const v = h.valence orelse return 3;
    const s = h.silence orelse return 3;
    const ids = h.model.tok.encode(gpa, text[0..len], h.model.max_len) catch return 2;
    defer gpa.free(ids);
    var pooled: [1024]f32 = undefined;
    if (h.model.hidden > pooled.len) return 2;
    h.model.encodeIds(ids, pooled[0..h.model.hidden]);
    h.model.answer(pooled[0..h.model.hidden], h.answers);
    out.* = .{
        .silence = h.answers[s].p,
        .energy = h.answers[e].probs[0..5].*,
        .energy_mean = h.answers[e].expected + 1,
        .valence = h.answers[v].probs[0..3].*,
        .valence_mean = h.answers[v].expected - 1,
        .n_tokens = @intCast(ids.len),
    };
    return 0;
}

/// Library version, for the header to check against.
export fn zw_version() c_int {
    return 3;
}
