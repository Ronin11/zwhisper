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
export fn zw_transcribe(h: *Handle, samples: [*]const f32, n: usize, offset_ms: i64, voices: bool, split_quotes: bool, condition_on_previous: bool, stop: ?*const bool, emit: Emit, user: ?*anyopaque, stats_out: ?*Stats) c_int {
    const audio = samples[0..n];
    var spec = zw.mel.logMel(gpa, audio, h.model.filters, h.model.hp.n_mels) catch return 2;
    defer spec.deinit(gpa);
    const sink = Sink{ .emit = emit, .user = user, .offset_ms = offset_ms };
    const stop_atomic: ?*const std.atomic.Value(bool) = if (stop) |p| @ptrCast(p) else null;
    const stats = h.engine.transcribe(&spec, audio, .{
        .voices = voices,
        .split_quotes = split_quotes,
        .condition_on_previous = condition_on_previous,
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

/// Library version, for the header to check against.
export fn zw_version() c_int {
    return 1;
}
