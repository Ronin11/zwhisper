//! GGUF model files (ggml's successor format, what llama.cpp and the Qwen/Llama
//! releases on Hugging Face use): a header, typed metadata, tensor descriptors, then
//! the tensor data at an aligned offset. The file is mapped and tensors point into it.
const std = @import("std");

pub const magic: u32 = 0x46554747; // "GGUF" little-endian

pub const Type = enum(u32) {
    f32 = 0,
    f16 = 1,
    q4_0 = 2,
    q4_1 = 3,
    q8_0 = 8,
    _,
};

pub const Value = union(enum) {
    u8: u8,
    i8: i8,
    u16: u16,
    i16: i16,
    u32: u32,
    i32: i32,
    f32: f32,
    bool: bool,
    string: []const u8,
    /// Arrays keep their raw element type and the slice of the file they live in.
    array: Array,
    u64: u64,
    i64: i64,
    f64: f64,

    pub fn asU32(v: Value) ?u32 {
        return switch (v) {
            .u8 => |x| x,
            .u16 => |x| x,
            .u32 => |x| x,
            .i32 => |x| if (x >= 0) @intCast(x) else null,
            .u64 => |x| @intCast(x),
            .i64 => |x| if (x >= 0) @intCast(x) else null,
            else => null,
        };
    }
    pub fn asF32(v: Value) ?f32 {
        return switch (v) {
            .f32 => |x| x,
            .f64 => |x| @floatCast(x),
            else => null,
        };
    }
};

pub const Array = struct {
    elem: u32, // value type code
    count: usize,
    /// Bytes of the elements, still in file order (strings are length-prefixed).
    bytes: []const u8,

    /// Iterate string elements.
    pub fn strings(a: Array) StringIter {
        return .{ .buf = a.bytes, .left = a.count };
    }
};

pub const StringIter = struct {
    buf: []const u8,
    pos: usize = 0,
    left: usize,
    pub fn next(it: *StringIter) ?[]const u8 {
        if (it.left == 0 or it.pos + 8 > it.buf.len) return null;
        const len: usize = @intCast(std.mem.readInt(u64, it.buf[it.pos..][0..8], .little));
        it.pos += 8;
        if (it.pos + len > it.buf.len) return null;
        defer it.pos += len;
        it.left -= 1;
        return it.buf[it.pos..][0..len];
    }
};

pub const Tensor = struct {
    ne: [4]usize,
    n_dims: u8,
    ty: Type,
    data: []const u8,

    pub fn count(t: Tensor) usize {
        return t.ne[0] * t.ne[1] * t.ne[2] * t.ne[3];
    }
};

pub const Error = error{ BadMagic, Unsupported, Truncated, OutOfMemory };

pub const File = struct {
    gpa: std.mem.Allocator,
    bytes: []const u8,
    version: u32,
    kv: std.StringHashMapUnmanaged(Value),
    tensors: std.StringHashMapUnmanaged(Tensor),

    pub fn parse(gpa: std.mem.Allocator, bytes: []const u8) Error!File {
        var r = Reader{ .buf = bytes };
        if (try r.readU32() != magic) return error.BadMagic;
        const version = try r.readU32();
        if (version < 2) return error.Unsupported;
        const n_tensors = try r.readU64();
        const n_kv = try r.readU64();
        var kv: std.StringHashMapUnmanaged(Value) = .empty;
        errdefer kv.deinit(gpa);
        for (0..n_kv) |_| {
            const key = try r.string();
            const v = try r.value();
            try kv.put(gpa, key, v);
        }
        var alignment: usize = 32;
        if (kv.get("general.alignment")) |a| if (a.asU32()) |x| {
            alignment = x;
        };
        // tensor descriptors
        const Desc = struct { name: []const u8, ne: [4]usize, n_dims: u8, ty: Type, off: usize };
        const descs = try gpa.alloc(Desc, n_tensors);
        defer gpa.free(descs);
        for (descs) |*d| {
            d.name = try r.string();
            const nd = try r.readU32();
            if (nd == 0 or nd > 4) return error.Unsupported;
            d.ne = .{ 1, 1, 1, 1 };
            for (0..nd) |i| d.ne[i] = @intCast(try r.readU64());
            d.n_dims = @intCast(nd);
            d.ty = @enumFromInt(try r.readU32());
            d.off = @intCast(try r.readU64());
        }
        const data_start = (r.pos + alignment - 1) / alignment * alignment;
        var tensors: std.StringHashMapUnmanaged(Tensor) = .empty;
        errdefer tensors.deinit(gpa);
        for (descs) |d| {
            const n = d.ne[0] * d.ne[1] * d.ne[2] * d.ne[3];
            const size = tensorBytes(d.ty, n) orelse return error.Unsupported;
            const start = data_start + d.off;
            if (start + size > bytes.len) return error.Truncated;
            try tensors.put(gpa, d.name, .{ .ne = d.ne, .n_dims = d.n_dims, .ty = d.ty, .data = bytes[start .. start + size] });
        }
        return .{ .gpa = gpa, .bytes = bytes, .version = version, .kv = kv, .tensors = tensors };
    }

    pub fn deinit(f: *File) void {
        f.kv.deinit(f.gpa);
        f.tensors.deinit(f.gpa);
        f.* = undefined;
    }

    pub fn get(f: *const File, key: []const u8) ?Value {
        return f.kv.get(key);
    }
    pub fn getU32(f: *const File, key: []const u8) ?u32 {
        return if (f.kv.get(key)) |v| v.asU32() else null;
    }
    pub fn getF32(f: *const File, key: []const u8) ?f32 {
        return if (f.kv.get(key)) |v| v.asF32() else null;
    }
    pub fn getString(f: *const File, key: []const u8) ?[]const u8 {
        return if (f.kv.get(key)) |v| (if (v == .string) v.string else null) else null;
    }
    pub fn getArray(f: *const File, key: []const u8) ?Array {
        return if (f.kv.get(key)) |v| (if (v == .array) v.array else null) else null;
    }
    pub fn tensor(f: *const File, name: []const u8) ?Tensor {
        return f.tensors.get(name);
    }
};

pub fn tensorBytes(ty: Type, n: usize) ?usize {
    return switch (ty) {
        .f32 => n * 4,
        .f16 => n * 2,
        .q8_0 => if (n % 32 == 0) n / 32 * 34 else null,
        .q4_0 => if (n % 32 == 0) n / 32 * 18 else null,
        else => null,
    };
}

/// Dequantize any supported tensor to f32. Caller owns the result.
pub fn toF32(t: Tensor, gpa: std.mem.Allocator) ![]f32 {
    const n = t.count();
    const out = try gpa.alloc(f32, n);
    errdefer gpa.free(out);
    switch (t.ty) {
        .f32 => @memcpy(std.mem.sliceAsBytes(out), t.data[0 .. n * 4]),
        .f16 => {
            const h: []align(1) const u16 = @ptrCast(t.data[0 .. n * 2]);
            for (out, h) |*o, v| o.* = @floatCast(@as(f16, @bitCast(v)));
        },
        .q8_0 => {
            var i: usize = 0;
            var p: usize = 0;
            while (i < n) : (i += 32) {
                const d: f32 = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, t.data[p..][0..2], .little))));
                const q: []const i8 = @ptrCast(t.data[p + 2 ..][0..32]);
                for (0..32) |j| out[i + j] = @as(f32, @floatFromInt(q[j])) * d;
                p += 34;
            }
        },
        else => return error.Unsupported,
    }
    return out;
}

const Reader = struct {
    buf: []const u8,
    pos: usize = 0,

    fn bytes(r: *Reader, n: usize) Error![]const u8 {
        if (n > r.buf.len - r.pos) return error.Truncated;
        defer r.pos += n;
        return r.buf[r.pos..][0..n];
    }
    fn readU32(r: *Reader) Error!u32 {
        return std.mem.readInt(u32, (try r.bytes(4))[0..4], .little);
    }
    fn readU64(r: *Reader) Error!u64 {
        return std.mem.readInt(u64, (try r.bytes(8))[0..8], .little);
    }
    fn string(r: *Reader) Error![]const u8 {
        const len: usize = @intCast(try r.readU64());
        return r.bytes(len);
    }
    fn scalarSize(ty: u32) ?usize {
        return switch (ty) {
            0, 1, 7 => 1,
            2, 3 => 2,
            4, 5, 6 => 4,
            10, 11, 12 => 8,
            else => null,
        };
    }
    fn value(r: *Reader) Error!Value {
        const ty = try r.readU32();
        return r.valueOf(ty);
    }
    fn valueOf(r: *Reader, ty: u32) Error!Value {
        switch (ty) {
            0 => return .{ .u8 = (try r.bytes(1))[0] },
            1 => return .{ .i8 = @bitCast((try r.bytes(1))[0]) },
            2 => return .{ .u16 = std.mem.readInt(u16, (try r.bytes(2))[0..2], .little) },
            3 => return .{ .i16 = std.mem.readInt(i16, (try r.bytes(2))[0..2], .little) },
            4 => return .{ .u32 = try r.readU32() },
            5 => return .{ .i32 = std.mem.readInt(i32, (try r.bytes(4))[0..4], .little) },
            6 => return .{ .f32 = @bitCast(try r.readU32()) },
            7 => return .{ .bool = (try r.bytes(1))[0] != 0 },
            8 => return .{ .string = try r.string() },
            9 => {
                const elem = try r.readU32();
                const count: usize = @intCast(try r.readU64());
                const start = r.pos;
                if (scalarSize(elem)) |sz| {
                    _ = try r.bytes(sz * count);
                } else if (elem == 8) {
                    for (0..count) |_| _ = try r.string();
                } else return error.Unsupported;
                return .{ .array = .{ .elem = elem, .count = count, .bytes = r.buf[start..r.pos] } };
            },
            10 => return .{ .u64 = try r.readU64() },
            11 => return .{ .i64 = @bitCast(try r.readU64()) },
            12 => return .{ .f64 = @bitCast(try r.readU64()) },
            else => return error.Unsupported,
        }
    }
};

test "rejects non-gguf" {
    try std.testing.expectError(error.BadMagic, File.parse(std.testing.allocator, "GGML...."));
}

test "q8_0 dequantizes" {
    var blk: [34]u8 = undefined;
    std.mem.writeInt(u16, blk[0..2], @bitCast(@as(f16, 0.5)), .little);
    for (0..32) |i| blk[2 + i] = @bitCast(@as(i8, @intCast(@as(i32, @intCast(i)) - 16)));
    const t = Tensor{ .ne = .{ 32, 1, 1, 1 }, .n_dims = 1, .ty = .q8_0, .data = &blk };
    const f = try toF32(t, std.testing.allocator);
    defer std.testing.allocator.free(f);
    try std.testing.expectApproxEqAbs(@as(f32, -8.0), f[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 7.5), f[31], 1e-6);
}
