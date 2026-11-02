//! safetensors files: an 8-byte little-endian header length, a JSON header mapping tensor
//! names to {dtype, shape, data_offsets}, then the raw tensor bytes. Tensors point into
//! the mapped file; `toF32` converts F32/F16/BF16/I8 (I8 with a caller-supplied scale).
const std = @import("std");

pub const DType = enum { f32, f16, bf16, i8, other };

pub const Tensor = struct {
    dtype: DType,
    shape: [4]usize,
    n_dims: u8,
    data: []const u8,

    pub fn count(t: Tensor) usize {
        var n: usize = 1;
        for (t.shape[0..t.n_dims]) |d| n *= d;
        return n;
    }
};

pub const File = struct {
    gpa: std.mem.Allocator,
    parsed: std.json.Parsed(std.json.Value),
    tensors: std.StringHashMapUnmanaged(Tensor),

    pub fn parse(gpa: std.mem.Allocator, bytes: []const u8) !File {
        if (bytes.len < 8) return error.Truncated;
        const hlen: usize = @intCast(std.mem.readInt(u64, bytes[0..8], .little));
        if (8 + hlen > bytes.len) return error.Truncated;
        const data = bytes[8 + hlen ..];
        var parsed = try std.json.parseFromSlice(std.json.Value, gpa, bytes[8 .. 8 + hlen], .{});
        errdefer parsed.deinit();
        if (parsed.value != .object) return error.BadHeader;
        var tensors: std.StringHashMapUnmanaged(Tensor) = .empty;
        errdefer tensors.deinit(gpa);
        var it = parsed.value.object.iterator();
        while (it.next()) |e| {
            if (std.mem.eql(u8, e.key_ptr.*, "__metadata__")) continue;
            const o = e.value_ptr.*;
            if (o != .object) return error.BadHeader;
            const dt_s = (o.object.get("dtype") orelse return error.BadHeader).string;
            const dtype: DType = if (std.mem.eql(u8, dt_s, "F32")) .f32 else if (std.mem.eql(u8, dt_s, "F16")) .f16 else if (std.mem.eql(u8, dt_s, "BF16")) .bf16 else if (std.mem.eql(u8, dt_s, "I8")) .i8 else .other;
            const shape_v = (o.object.get("shape") orelse return error.BadHeader).array;
            if (shape_v.items.len > 4) return error.Unsupported;
            var t = Tensor{ .dtype = dtype, .shape = .{ 1, 1, 1, 1 }, .n_dims = @intCast(shape_v.items.len), .data = &.{} };
            for (shape_v.items, 0..) |s, i| t.shape[i] = @intCast(s.integer);
            const offs = (o.object.get("data_offsets") orelse return error.BadHeader).array;
            const a: usize = @intCast(offs.items[0].integer);
            const b: usize = @intCast(offs.items[1].integer);
            if (b > data.len or a > b) return error.Truncated;
            t.data = data[a..b];
            try tensors.put(gpa, e.key_ptr.*, t);
        }
        return .{ .gpa = gpa, .parsed = parsed, .tensors = tensors };
    }

    pub fn deinit(f: *File) void {
        f.tensors.deinit(f.gpa);
        f.parsed.deinit();
        f.* = undefined;
    }

    pub fn tensor(f: *const File, name: []const u8) ?Tensor {
        return f.tensors.get(name);
    }
};

pub fn bf16ToF32(h: u16) f32 {
    return @bitCast(@as(u32, h) << 16);
}

/// Convert to f32. I8 tensors need a per-row scale (`scale.len == rows`) and are
/// dequantized as `q * scale[row]`.
pub fn toF32(t: Tensor, gpa: std.mem.Allocator, scale: ?[]const f32) ![]f32 {
    const n = t.count();
    const out = try gpa.alloc(f32, n);
    errdefer gpa.free(out);
    switch (t.dtype) {
        .f32 => @memcpy(std.mem.sliceAsBytes(out), t.data[0 .. n * 4]),
        .f16 => {
            const h: []align(1) const u16 = @ptrCast(t.data[0 .. n * 2]);
            for (out, h) |*o, v| o.* = @floatCast(@as(f16, @bitCast(v)));
        },
        .bf16 => {
            const h: []align(1) const u16 = @ptrCast(t.data[0 .. n * 2]);
            for (out, h) |*o, v| o.* = bf16ToF32(v);
        },
        .i8 => {
            const s = scale orelse return error.NeedScale;
            const rows = t.shape[0];
            const cols = n / rows;
            if (s.len < rows) return error.NeedScale;
            const q: []const i8 = @ptrCast(t.data[0..n]);
            for (0..rows) |r| for (0..cols) |c| {
                out[r * cols + c] = @as(f32, @floatFromInt(q[r * cols + c])) * s[r];
            };
        },
        .other => return error.Unsupported,
    }
    return out;
}

test "header parse" {
    const gpa = std.testing.allocator;
    const header = "{\"w\":{\"dtype\":\"F32\",\"shape\":[2],\"data_offsets\":[0,8]}}";
    var buf: [8 + header.len + 8]u8 = undefined;
    std.mem.writeInt(u64, buf[0..8], header.len, .little);
    @memcpy(buf[8 .. 8 + header.len], header);
    std.mem.writeInt(u32, buf[8 + header.len ..][0..4], @bitCast(@as(f32, 1.5)), .little);
    std.mem.writeInt(u32, buf[8 + header.len + 4 ..][0..4], @bitCast(@as(f32, -2.0)), .little);
    var f = try File.parse(gpa, &buf);
    defer f.deinit();
    const w = try toF32(f.tensor("w").?, gpa, null);
    defer gpa.free(w);
    try std.testing.expectEqual(@as(f32, 1.5), w[0]);
    try std.testing.expectEqual(@as(f32, -2.0), w[1]);
    try std.testing.expectEqual(@as(f32, 1.0), bf16ToF32(0x3f80));
}
