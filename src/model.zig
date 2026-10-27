//! The ggml Whisper model file (`ggml-*.bin`), as whisper.cpp writes and reads it.
//!
//! Layout, all little-endian:
//!   u32 magic 0x67676d6c ("lmgg")
//!   11 x i32 hparams
//!   i32 n_mel, i32 n_fft, f32[n_mel*n_fft] mel filters
//!   i32 n_vocab, then n_vocab x (u32 len, bytes) tokens
//!   tensors until EOF: i32 n_dims, i32 name_len, i32 ttype, i32 ne[n_dims], name, data
//!
//! The file is mapped read-only and tensors point into the mapping. ggml's ne[0] is
//! the innermost (contiguous) dimension, so a 2-d weight with ne = [n_in, n_out] is
//! a row-major [n_out][n_in] matrix: y = W x.
const std = @import("std");

pub const magic: u32 = 0x67676d6c;

pub const Hparams = struct {
    n_vocab: u32,
    n_audio_ctx: u32,
    n_audio_state: u32,
    n_audio_head: u32,
    n_audio_layer: u32,
    n_text_ctx: u32,
    n_text_state: u32,
    n_text_head: u32,
    n_text_layer: u32,
    n_mels: u32,
    ftype: u32,

    pub fn isMultilingual(h: Hparams) bool {
        return h.n_vocab >= 51865;
    }
};

/// Special token ids. whisper.cpp's defaults are for English-only models; multilingual
/// models shift them by one (plus language tokens), which we don't support yet.
pub const Tokens = struct {
    eot: u32 = 50256,
    sot: u32 = 50257,
    translate: u32 = 50357,
    transcribe: u32 = 50358,
    solm: u32 = 50359,
    prev: u32 = 50360,
    nosp: u32 = 50361,
    not: u32 = 50362,
    beg: u32 = 50363,
};

pub const Type = enum(u32) { f32 = 0, f16 = 1, _ };

pub const Tensor = struct {
    ne: [4]usize,
    n_dims: u8,
    ty: Type,
    /// Raw bytes inside the mapping.
    data: []const u8,

    pub fn count(t: Tensor) usize {
        return t.ne[0] * t.ne[1] * t.ne[2] * t.ne[3];
    }

    /// Element `i` in flat (ne[0]-innermost) order, as f32.
    pub fn at(t: Tensor, i: usize) f32 {
        return switch (t.ty) {
            .f32 => @bitCast(std.mem.readInt(u32, t.data[i * 4 ..][0..4], .little)),
            .f16 => @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, t.data[i * 2 ..][0..2], .little)))),
            _ => unreachable,
        };
    }

    /// Copy out as f16 (lossless for f16 files). Caller owns the slice.
    pub fn toF16(t: Tensor, gpa: std.mem.Allocator) ![]f16 {
        const n = t.count();
        const out = try gpa.alloc(f16, n);
        switch (t.ty) {
            .f16 => @memcpy(std.mem.sliceAsBytes(out), t.data[0 .. n * 2]),
            .f32 => {
                const floats: []align(1) const f32 = @ptrCast(t.data[0 .. n * 4]);
                for (out, floats) |*o, f| o.* = @floatCast(f);
            },
            _ => return error.UnsupportedTensorType,
        }
        return out;
    }

    /// Copy out as f32. Caller owns the slice.
    pub fn toF32(t: Tensor, gpa: std.mem.Allocator) ![]f32 {
        const n = t.count();
        const out = try gpa.alloc(f32, n);
        switch (t.ty) {
            .f32 => @memcpy(std.mem.sliceAsBytes(out), t.data[0 .. n * 4]),
            .f16 => {
                const halves: []align(1) const u16 = @ptrCast(t.data[0 .. n * 2]);
                for (out, halves) |*o, h| o.* = @floatCast(@as(f16, @bitCast(h)));
            },
            _ => return error.UnsupportedTensorType,
        }
        return out;
    }
};

pub const Error = error{ BadMagic, Truncated, UnsupportedTensorType, UnsupportedModel, OutOfMemory };

pub const Model = struct {
    gpa: std.mem.Allocator,
    bytes: []const u8,
    hp: Hparams,
    tok: Tokens,
    /// [n_mel][n_fft] mel filterbank, n_fft = 201.
    filters: []f32,
    filters_n_mel: u32,
    filters_n_fft: u32,
    /// id -> UTF-8 bytes, straight from the file (GPT-2 byte-level BPE already decoded).
    vocab: [][]const u8,
    /// bytes -> id, for prompts and hotwords.
    token_ids: std.StringHashMapUnmanaged(u32),
    tensors: std.StringHashMapUnmanaged(Tensor),

    /// Parse the mapped file. `bytes` must outlive the model.
    pub fn parse(gpa: std.mem.Allocator, bytes: []const u8) Error!Model {
        var r = Reader{ .buf = bytes };
        if (try r.readU32() != magic) return error.BadMagic;
        var hp: Hparams = undefined;
        inline for (@typeInfo(Hparams).@"struct".field_names) |name| @field(hp, name) = try r.readU32();
        if (hp.isMultilingual()) return error.UnsupportedModel;

        const n_mel = try r.readU32();
        const n_fft = try r.readU32();
        const filters = try gpa.alloc(f32, n_mel * n_fft);
        errdefer gpa.free(filters);
        const fbytes = try r.bytes(filters.len * 4);
        @memcpy(std.mem.sliceAsBytes(filters), fbytes);

        const n_vocab_file = try r.readU32();
        const vocab = try gpa.alloc([]const u8, hp.n_vocab);
        errdefer gpa.free(vocab);
        var i: usize = 0;
        while (i < n_vocab_file) : (i += 1) {
            const len = try r.readU32();
            vocab[i] = try r.bytes(len);
        }
        // English-only models list fewer tokens than n_vocab; whisper.cpp names the rest.
        while (i < hp.n_vocab) : (i += 1) vocab[i] = "";
        var token_ids: std.StringHashMapUnmanaged(u32) = .empty;
        errdefer token_ids.deinit(gpa);
        try token_ids.ensureTotalCapacity(gpa, @intCast(n_vocab_file));
        for (vocab[0..n_vocab_file], 0..) |v, id| {
            if (v.len > 0) token_ids.putAssumeCapacity(v, @intCast(id));
        }

        var tensors: std.StringHashMapUnmanaged(Tensor) = .empty;
        errdefer tensors.deinit(gpa);
        while (r.remaining() >= 12) {
            const n_dims = try r.readU32();
            const name_len = try r.readU32();
            const ty: Type = @enumFromInt(try r.readU32());
            if (n_dims == 0 or n_dims > 4) return error.Truncated;
            var ne = [4]usize{ 1, 1, 1, 1 };
            for (0..n_dims) |d| ne[d] = try r.readU32();
            const name = try r.bytes(name_len);
            const count = ne[0] * ne[1] * ne[2] * ne[3];
            const size = switch (ty) {
                .f32 => count * 4,
                .f16 => count * 2,
                _ => return error.UnsupportedTensorType,
            };
            const data = try r.bytes(size);
            try tensors.put(gpa, name, .{ .ne = ne, .n_dims = @intCast(n_dims), .ty = ty, .data = data });
        }
        return .{
            .gpa = gpa,
            .bytes = bytes,
            .hp = hp,
            .tok = .{},
            .filters = filters,
            .filters_n_mel = n_mel,
            .filters_n_fft = n_fft,
            .vocab = vocab,
            .token_ids = token_ids,
            .tensors = tensors,
        };
    }

    pub fn deinit(m: *Model) void {
        m.tensors.deinit(m.gpa);
        m.token_ids.deinit(m.gpa);
        m.gpa.free(m.vocab);
        m.gpa.free(m.filters);
        m.* = undefined;
    }

    pub fn tensor(m: *const Model, name: []const u8) ?Tensor {
        return m.tensors.get(name);
    }

    /// Fetch a tensor by printf-style name with one layer index, e.g. "encoder.blocks.{d}.attn.query.weight".
    pub fn layerTensor(m: *const Model, comptime fmt: []const u8, layer: usize) ?Tensor {
        var buf: [96]u8 = undefined;
        const name = std.fmt.bufPrint(&buf, fmt, .{layer}) catch return null;
        return m.tensors.get(name);
    }

    pub fn tokenText(m: *const Model, id: u32) []const u8 {
        return if (id < m.vocab.len) m.vocab[id] else "";
    }

    /// Id of the token whose text is exactly `s`, if any.
    pub fn tokenId(m: *const Model, s: []const u8) ?u32 {
        return m.token_ids.get(s);
    }
};

const Reader = struct {
    buf: []const u8,
    pos: usize = 0,

    fn remaining(r: Reader) usize {
        return r.buf.len - r.pos;
    }
    fn bytes(r: *Reader, n: usize) Error![]const u8 {
        if (n > r.remaining()) return error.Truncated;
        defer r.pos += n;
        return r.buf[r.pos..][0..n];
    }
    fn readU32(r: *Reader) Error!u32 {
        const b = try r.bytes(4);
        return std.mem.readInt(u32, b[0..4], .little);
    }
};

test "bad magic" {
    try std.testing.expectError(error.BadMagic, Model.parse(std.testing.allocator, "nope"));
}

test "f16 conversion" {
    const one: u16 = 0x3c00;
    var raw: [2]u8 = undefined;
    std.mem.writeInt(u16, &raw, one, .little);
    const t = Tensor{ .ne = .{ 1, 1, 1, 1 }, .n_dims = 1, .ty = .f16, .data = &raw };
    try std.testing.expectEqual(@as(f32, 1.0), t.at(0));
}
