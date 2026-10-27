//! RIFF WAV reader for the one format the engine takes: 16 kHz mono 16-bit PCM.
//! Anything else is an error, not a resample; decoding is the caller's job.
const std = @import("std");

pub const sample_rate: u32 = 16_000;

pub const Error = error{ NotWav, Unsupported, Truncated, OutOfMemory };

/// Samples in [-1, 1), as whisper.cpp's examples convert them (s / 32768).
pub fn decode(gpa: std.mem.Allocator, bytes: []const u8) Error![]f32 {
    if (bytes.len < 12 or !std.mem.eql(u8, bytes[0..4], "RIFF") or !std.mem.eql(u8, bytes[8..12], "WAVE"))
        return error.NotWav;
    var pos: usize = 12;
    var channels: u16 = 0;
    var rate: u32 = 0;
    var bits: u16 = 0;
    var format: u16 = 0;
    while (pos + 8 <= bytes.len) {
        const id = bytes[pos..][0..4];
        const size: usize = std.mem.readInt(u32, bytes[pos + 4 ..][0..4], .little);
        const body_start = pos + 8;
        if (std.mem.eql(u8, id, "fmt ")) {
            if (body_start + 16 > bytes.len) return error.Truncated;
            const f = bytes[body_start..];
            format = std.mem.readInt(u16, f[0..2], .little);
            channels = std.mem.readInt(u16, f[2..4], .little);
            rate = std.mem.readInt(u32, f[4..8], .little);
            bits = std.mem.readInt(u16, f[14..16], .little);
        } else if (std.mem.eql(u8, id, "data")) {
            if (format != 1 or channels != 1 or rate != sample_rate or bits != 16) return error.Unsupported;
            const avail = @min(size, bytes.len - body_start);
            const n = avail / 2;
            const out = try gpa.alloc(f32, n);
            const pcm = bytes[body_start..][0 .. n * 2];
            for (out, 0..) |*o, i| {
                const s = std.mem.readInt(i16, pcm[i * 2 ..][0..2], .little);
                o.* = @as(f32, @floatFromInt(s)) / 32768.0;
            }
            return out;
        }
        pos = body_start + size + (size & 1);
    }
    return error.Truncated;
}

test "rejects non-wav" {
    try std.testing.expectError(error.NotWav, decode(std.testing.allocator, "hello"));
}

test "decodes a tiny mono file" {
    var buf: [44 + 4]u8 = undefined;
    @memcpy(buf[0..4], "RIFF");
    std.mem.writeInt(u32, buf[4..8], 40, .little);
    @memcpy(buf[8..12], "WAVE");
    @memcpy(buf[12..16], "fmt ");
    std.mem.writeInt(u32, buf[16..20], 16, .little);
    std.mem.writeInt(u16, buf[20..22], 1, .little);
    std.mem.writeInt(u16, buf[22..24], 1, .little);
    std.mem.writeInt(u32, buf[24..28], 16000, .little);
    std.mem.writeInt(u32, buf[28..32], 32000, .little);
    std.mem.writeInt(u16, buf[32..34], 2, .little);
    std.mem.writeInt(u16, buf[34..36], 16, .little);
    @memcpy(buf[36..40], "data");
    std.mem.writeInt(u32, buf[40..44], 4, .little);
    std.mem.writeInt(i16, buf[44..46], 16384, .little);
    std.mem.writeInt(i16, buf[46..48], -32768, .little);
    const s = try decode(std.testing.allocator, &buf);
    defer std.testing.allocator.free(s);
    try std.testing.expectEqual(@as(usize, 2), s.len);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), s[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, -1.0), s[1], 1e-6);
}
