//! A BERT-style sentence encoder (all-MiniLM-L6-v2 and kin) from a GGUF file: WordPiece
//! tokens, token + position + type embeddings, post-norm transformer blocks with
//! bidirectional attention, mean pooling, L2 normalisation. 22M parameters: the frozen
//! feature extractor for the trained tagger, a few hundred milliseconds per window.
const std = @import("std");
const ops = @import("ops.zig");
const gguf = @import("gguf.zig");
const clock = @import("clock.zig");

pub const Layer = struct {
    wq: []f16,
    bq: []f32,
    wk: []f16,
    bk: []f32,
    wv: []f16,
    bv: []f32,
    wo: []f16,
    bo: []f32,
    ln1_w: []f32,
    ln1_b: []f32,
    w_up: []f16,
    b_up: []f32,
    w_down: []f16,
    b_down: []f32,
    ln2_w: []f32,
    ln2_b: []f32,
};

pub const Model = struct {
    gpa: std.mem.Allocator,
    arena: std.heap.ArenaAllocator,
    n_vocab: usize,
    n_embd: usize,
    n_head: usize,
    n_ff: usize,
    n_layer: usize,
    n_ctx: usize,
    eps: f32,
    tok_embd: []f16, // [n_vocab][n_embd]
    pos_embd: []f16, // [n_ctx][n_embd]
    type_embd: []f32, // [2][n_embd]
    embd_ln_w: []f32,
    embd_ln_b: []f32,
    layers: []Layer,
    // vocab
    tokens: [][]const u8,
    ids: std.StringHashMapUnmanaged(u32),
    cls: u32,
    sep: u32,
    unk: u32,
    // scratch for n_ctx tokens
    x: []f32,
    q: []f32,
    k: []f32,
    v: []f32,
    attn: []f32,
    h: []f32,
    up: []f32,
    attn_scratch: []f32,
    pool: *ops.Pool,

    pub fn load(gpa: std.mem.Allocator, f: *const gguf.File, threads: usize) !*Model {
        const arch = f.getString("general.architecture") orelse return error.Unsupported;
        if (!std.mem.eql(u8, arch, "bert")) return error.Unsupported;
        const m = try gpa.create(Model);
        errdefer gpa.destroy(m);
        m.* = undefined;
        m.gpa = gpa;
        m.arena = std.heap.ArenaAllocator.init(gpa);
        errdefer m.arena.deinit();
        const a = m.arena.allocator();
        m.n_embd = f.getU32("bert.embedding_length") orelse return error.Unsupported;
        m.n_head = f.getU32("bert.attention.head_count") orelse return error.Unsupported;
        m.n_ff = f.getU32("bert.feed_forward_length") orelse return error.Unsupported;
        m.n_layer = f.getU32("bert.block_count") orelse return error.Unsupported;
        m.n_ctx = f.getU32("bert.context_length") orelse 512;
        m.eps = f.getF32("bert.attention.layer_norm_epsilon") orelse 1e-12;
        const te = f.tensor("token_embd.weight") orelse return error.MissingTensor;
        m.n_vocab = te.ne[1];
        m.tok_embd = try loadH(a, te);
        m.pos_embd = try loadH(a, f.tensor("position_embd.weight") orelse return error.MissingTensor);
        m.type_embd = try gguf.toF32(f.tensor("token_types.weight") orelse return error.MissingTensor, a);
        m.embd_ln_w = try gguf.toF32(f.tensor("token_embd_norm.weight") orelse return error.MissingTensor, a);
        m.embd_ln_b = try gguf.toF32(f.tensor("token_embd_norm.bias") orelse return error.MissingTensor, a);
        m.layers = try a.alloc(Layer, m.n_layer);
        for (m.layers, 0..) |*l, i| {
            l.wq = try loadH(a, try need(f, "blk.{d}.attn_q.weight", i));
            l.bq = try gguf.toF32(try need(f, "blk.{d}.attn_q.bias", i), a);
            l.wk = try loadH(a, try need(f, "blk.{d}.attn_k.weight", i));
            l.bk = try gguf.toF32(try need(f, "blk.{d}.attn_k.bias", i), a);
            l.wv = try loadH(a, try need(f, "blk.{d}.attn_v.weight", i));
            l.bv = try gguf.toF32(try need(f, "blk.{d}.attn_v.bias", i), a);
            l.wo = try loadH(a, try need(f, "blk.{d}.attn_output.weight", i));
            l.bo = try gguf.toF32(try need(f, "blk.{d}.attn_output.bias", i), a);
            l.ln1_w = try gguf.toF32(try need(f, "blk.{d}.attn_output_norm.weight", i), a);
            l.ln1_b = try gguf.toF32(try need(f, "blk.{d}.attn_output_norm.bias", i), a);
            l.w_up = try loadH(a, try need(f, "blk.{d}.ffn_up.weight", i));
            l.b_up = try gguf.toF32(try need(f, "blk.{d}.ffn_up.bias", i), a);
            l.w_down = try loadH(a, try need(f, "blk.{d}.ffn_down.weight", i));
            l.b_down = try gguf.toF32(try need(f, "blk.{d}.ffn_down.bias", i), a);
            l.ln2_w = try gguf.toF32(try need(f, "blk.{d}.layer_output_norm.weight", i), a);
            l.ln2_b = try gguf.toF32(try need(f, "blk.{d}.layer_output_norm.bias", i), a);
        }
        // vocab
        const toks = f.getArray("tokenizer.ggml.tokens") orelse return error.NoVocab;
        m.tokens = try a.alloc([]const u8, toks.count);
        m.ids = .empty;
        try m.ids.ensureTotalCapacity(a, @intCast(toks.count));
        var it = toks.strings();
        var i: usize = 0;
        while (it.next()) |s| : (i += 1) {
            m.tokens[i] = s;
            m.ids.putAssumeCapacity(s, @intCast(i));
        }
        m.cls = f.getU32("tokenizer.ggml.cls_token_id") orelse m.ids.get("[CLS]") orelse 101;
        m.sep = f.getU32("tokenizer.ggml.seperator_token_id") orelse m.ids.get("[SEP]") orelse 102;
        m.unk = m.ids.get("[UNK]") orelse 100;
        const n = m.n_ctx;
        m.x = try a.alloc(f32, n * m.n_embd);
        m.q = try a.alloc(f32, n * m.n_embd);
        m.k = try a.alloc(f32, n * m.n_embd);
        m.v = try a.alloc(f32, n * m.n_embd);
        m.attn = try a.alloc(f32, n * m.n_embd);
        m.h = try a.alloc(f32, n * m.n_embd);
        m.up = try a.alloc(f32, n * m.n_ff);
        m.attn_scratch = try a.alloc(f32, ops.attentionScratch(n, n, m.n_head, m.n_embd));
        m.pool = try ops.Pool.init(gpa, threads);
        return m;
    }

    pub fn deinit(m: *Model) void {
        m.pool.deinit(m.gpa);
        m.arena.deinit();
        m.gpa.destroy(m);
    }

    // ---- WordPiece (uncased) ------------------------------------------------------------

    fn isPunct(c: u8) bool {
        return (c >= 33 and c <= 47) or (c >= 58 and c <= 64) or (c >= 91 and c <= 96) or (c >= 123 and c <= 126);
    }

    /// Tokenize `text` into WordPiece ids (no [CLS]/[SEP]). Caller owns the result.
    pub fn encode(m: *const Model, gpa: std.mem.Allocator, text: []const u8) ![]u32 {
        var out: std.ArrayList(u32) = .empty;
        errdefer out.deinit(gpa);
        var word: std.ArrayList(u8) = .empty;
        defer word.deinit(gpa);
        var piece: std.ArrayList(u8) = .empty;
        defer piece.deinit(gpa);
        var i: usize = 0;
        while (i <= text.len) : (i += 1) {
            const c: u8 = if (i < text.len) text[i] else ' ';
            const boundary = std.ascii.isWhitespace(c) or isPunct(c) or c >= 0x80 or i == text.len;
            if (!boundary) {
                try word.append(gpa, std.ascii.toLower(c));
                continue;
            }
            if (word.items.len > 0) {
                try m.wordpiece(gpa, word.items, &piece, &out);
                word.clearRetainingCapacity();
            }
            if (isPunct(c)) {
                const one = [_]u8{ 0xe2, 0x96, 0x81, c };
                try out.append(gpa, m.ids.get(&one) orelse m.ids.get(one[3..]) orelse m.unk);
            }
            // non-ASCII bytes are skipped (MiniLM strips accents; the rest is rare in narration)
        }
        return out.toOwnedSlice(gpa);
    }

    fn wordpiece(m: *const Model, gpa: std.mem.Allocator, w: []const u8, piece: *std.ArrayList(u8), out: *std.ArrayList(u32)) !void {
        if (w.len > 100) {
            try out.append(gpa, m.unk);
            return;
        }
        var start: usize = 0;
        var first = true;
        var got: std.ArrayList(u32) = .empty;
        defer got.deinit(gpa);
        while (start < w.len) {
            var end = w.len;
            var found: ?u32 = null;
            while (end > start) : (end -= 1) {
                // llama.cpp's BERT files store word-initial pieces with a "▁" prefix and
                // continuation pieces bare (the "##" is dropped)
                piece.clearRetainingCapacity();
                if (first) try piece.appendSlice(gpa, "\xe2\x96\x81");
                try piece.appendSlice(gpa, w[start..end]);
                if (m.ids.get(piece.items)) |id| {
                    found = id;
                    break;
                }
            }
            const id = found orelse {
                try out.append(gpa, m.unk);
                return;
            };
            try got.append(gpa, id);
            start = end;
            first = false;
        }
        try out.appendSlice(gpa, got.items);
    }

    // ---- forward --------------------------------------------------------------------------

    /// Encode one sequence of up to n_ctx - 2 WordPiece ids; mean-pooled vector into `out`.
    fn encodeChunk(m: *Model, ids: []const u32, out: []f32) void {
        const ne = m.n_embd;
        const n = ids.len + 2;
        std.debug.assert(n <= m.n_ctx);
        // embeddings
        for (0..n) |p| {
            const t: u32 = if (p == 0) m.cls else if (p == n - 1) m.sep else ids[p - 1];
            const row = m.h[p * ne ..][0..ne];
            for (row, m.tok_embd[t * ne ..][0..ne], m.pos_embd[p * ne ..][0..ne], m.type_embd[0..ne]) |*r, te, pe, ty| {
                r.* = @as(f32, @floatCast(te)) + @as(f32, @floatCast(pe)) + ty;
            }
        }
        ops.layerNormEps(m.pool, m.h, n, ne, m.embd_ln_w, m.embd_ln_b, m.eps, m.x);
        for (m.layers) |*l| {
            ops.linear(m.pool, m.x, n, ne, @as([]const f16, l.wq), l.bq, ne, m.q);
            ops.linear(m.pool, m.x, n, ne, @as([]const f16, l.wk), l.bk, ne, m.k);
            ops.linear(m.pool, m.x, n, ne, @as([]const f16, l.wv), l.bv, ne, m.v);
            ops.attention(m.pool, m.q, m.k, m.v, n, n, ne, m.n_head, false, m.attn_scratch, m.attn);
            ops.linear(m.pool, m.attn, n, ne, @as([]const f16, l.wo), l.bo, ne, m.h);
            ops.add(m.h[0 .. n * ne], m.x[0 .. n * ne]);
            ops.layerNormEps(m.pool, m.h, n, ne, l.ln1_w, l.ln1_b, m.eps, m.x);
            ops.linear(m.pool, m.x, n, ne, @as([]const f16, l.w_up), l.b_up, m.n_ff, m.up);
            ops.gelu(m.pool, m.up[0 .. n * m.n_ff]);
            ops.linear(m.pool, m.up, n, m.n_ff, @as([]const f16, l.w_down), l.b_down, ne, m.h);
            ops.add(m.h[0 .. n * ne], m.x[0 .. n * ne]);
            ops.layerNormEps(m.pool, m.h, n, ne, l.ln2_w, l.ln2_b, m.eps, m.x);
        }
        @memset(out, 0);
        for (0..n) |p| ops.add(out, m.x[p * ne ..][0..ne]);
        ops.scale(out, 1.0 / @as(f32, @floatFromInt(n)));
    }

    /// Sentence embedding of `text`: chunks of up to n_ctx - 2 pieces, each mean-pooled,
    /// averaged by length, then L2-normalised. `out` has n_embd floats.
    pub fn embed(m: *Model, gpa: std.mem.Allocator, text: []const u8, out: []f32) !void {
        const ids = try m.encode(gpa, text);
        defer gpa.free(ids);
        @memset(out, 0);
        const tmp = try gpa.alloc(f32, m.n_embd);
        defer gpa.free(tmp);
        const max = m.n_ctx - 2;
        var i: usize = 0;
        var total: usize = 0;
        while (i < ids.len or (i == 0 and ids.len == 0)) {
            const n = @min(max, ids.len - i);
            m.encodeChunk(ids[i .. i + n], tmp);
            ops.axpy(out, @floatFromInt(n + 2), tmp);
            total += n + 2;
            i += n;
            if (ids.len == 0) break;
        }
        if (total > 0) ops.scale(out, 1.0 / @as(f32, @floatFromInt(total)));
        var norm: f32 = 0;
        for (out) |v| norm += v * v;
        if (norm > 0) ops.scale(out, 1.0 / @sqrt(norm));
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

fn loadH(a: std.mem.Allocator, t: gguf.Tensor) ![]f16 {
    const n = t.count();
    const out = try a.alloc(f16, n);
    switch (t.ty) {
        .f16 => @memcpy(std.mem.sliceAsBytes(out), t.data[0 .. n * 2]),
        .f32 => {
            const fl: []align(1) const f32 = @ptrCast(t.data[0 .. n * 4]);
            for (out, fl) |*o, v| o.* = @floatCast(v);
        },
        else => return error.Unsupported,
    }
    return out;
}
