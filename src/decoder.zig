//! The text decoder: token + learned position embeddings, pre-norm blocks with masked
//! self-attention over a KV cache and cross-attention over the encoder output, a final
//! layer norm, and logits tied to the token embedding.
//!
//! `setAudio` projects one window's encoder output into per-layer cross K/V once;
//! `forward` then feeds any number of new tokens, extends the cache, and returns the
//! logits of the last one. `reset` starts a new sequence on the same audio.
const std = @import("std");
const ops = @import("ops.zig");
const model = @import("model.zig");
const gemm = @import("gemm.zig");

pub const Layer = struct {
    attn_ln_w: []f32,
    attn_ln_b: []f32,
    q_w: gemm.Q8,
    q_b: []f32,
    k_w: gemm.Q8,
    v_w: gemm.Q8,
    v_b: []f32,
    o_w: gemm.Q8,
    o_b: []f32,
    cross_ln_w: []f32,
    cross_ln_b: []f32,
    cq_w: gemm.Q8,
    cq_b: []f32,
    ck_w: gemm.Q8,
    cv_w: gemm.Q8,
    cv_b: []f32,
    co_w: gemm.Q8,
    co_b: []f32,
    mlp_ln_w: []f32,
    mlp_ln_b: []f32,
    mlp0_w: gemm.Q8,
    mlp0_b: []f32,
    mlp1_w: gemm.Q8,
    mlp1_b: []f32,
    // state
    cache_k: []f32, // [n_text_ctx][n_state]
    cache_v: []f32,
    cross_k: []f32, // [n_audio_ctx][n_state]
    cross_v: []f32,
};

pub const Decoder = struct {
    arena: std.heap.ArenaAllocator,
    n_vocab: usize,
    n_state: usize,
    n_head: usize,
    n_ctx: usize,
    n_audio_ctx: usize,
    n_layer: usize,
    te: gemm.Q8, // [n_vocab][n_state]
    pe: []f32, // [n_ctx][n_state]
    ln_w: []f32,
    ln_b: []f32,
    layers: []Layer,
    n_past: usize = 0,
    have_audio: bool = false,

    // scratch for up to n_ctx tokens in one call
    x: []f32,
    x_ln: []f32,
    q: []f32,
    attn: []f32,
    hidden: []f32,
    last_ln: []f32,
    attn_scratch: []f32,

    pub fn load(gpa: std.mem.Allocator, m: *const model.Model) !*Decoder {
        const d = try gpa.create(Decoder);
        errdefer gpa.destroy(d);
        d.* = .{
            .arena = std.heap.ArenaAllocator.init(gpa),
            .n_vocab = m.hp.n_vocab,
            .n_state = m.hp.n_text_state,
            .n_head = m.hp.n_text_head,
            .n_ctx = m.hp.n_text_ctx,
            .n_audio_ctx = m.hp.n_audio_ctx,
            .n_layer = m.hp.n_text_layer,
            .te = undefined,
            .pe = undefined,
            .ln_w = undefined,
            .ln_b = undefined,
            .layers = undefined,
            .x = undefined,
            .x_ln = undefined,
            .q = undefined,
            .attn = undefined,
            .hidden = undefined,
            .last_ln = undefined,
            .attn_scratch = undefined,
        };
        errdefer d.arena.deinit();
        const a = d.arena.allocator();
        d.te = try getQ(a, m, "decoder.token_embedding.weight");
        d.pe = try get(a, m, "decoder.positional_embedding");
        d.ln_w = try get(a, m, "decoder.ln.weight");
        d.ln_b = try get(a, m, "decoder.ln.bias");
        d.layers = try a.alloc(Layer, d.n_layer);
        const ns = d.n_state;
        for (d.layers, 0..) |*l, i| {
            l.attn_ln_w = try getL(a, m, "decoder.blocks.{d}.attn_ln.weight", i);
            l.attn_ln_b = try getL(a, m, "decoder.blocks.{d}.attn_ln.bias", i);
            l.q_w = try getQL(a, m, "decoder.blocks.{d}.attn.query.weight", i);
            l.q_b = try getL(a, m, "decoder.blocks.{d}.attn.query.bias", i);
            l.k_w = try getQL(a, m, "decoder.blocks.{d}.attn.key.weight", i);
            l.v_w = try getQL(a, m, "decoder.blocks.{d}.attn.value.weight", i);
            l.v_b = try getL(a, m, "decoder.blocks.{d}.attn.value.bias", i);
            l.o_w = try getQL(a, m, "decoder.blocks.{d}.attn.out.weight", i);
            l.o_b = try getL(a, m, "decoder.blocks.{d}.attn.out.bias", i);
            l.cross_ln_w = try getL(a, m, "decoder.blocks.{d}.cross_attn_ln.weight", i);
            l.cross_ln_b = try getL(a, m, "decoder.blocks.{d}.cross_attn_ln.bias", i);
            l.cq_w = try getQL(a, m, "decoder.blocks.{d}.cross_attn.query.weight", i);
            l.cq_b = try getL(a, m, "decoder.blocks.{d}.cross_attn.query.bias", i);
            l.ck_w = try getQL(a, m, "decoder.blocks.{d}.cross_attn.key.weight", i);
            l.cv_w = try getQL(a, m, "decoder.blocks.{d}.cross_attn.value.weight", i);
            l.cv_b = try getL(a, m, "decoder.blocks.{d}.cross_attn.value.bias", i);
            l.co_w = try getQL(a, m, "decoder.blocks.{d}.cross_attn.out.weight", i);
            l.co_b = try getL(a, m, "decoder.blocks.{d}.cross_attn.out.bias", i);
            l.mlp_ln_w = try getL(a, m, "decoder.blocks.{d}.mlp_ln.weight", i);
            l.mlp_ln_b = try getL(a, m, "decoder.blocks.{d}.mlp_ln.bias", i);
            l.mlp0_w = try getQL(a, m, "decoder.blocks.{d}.mlp.0.weight", i);
            l.mlp0_b = try getL(a, m, "decoder.blocks.{d}.mlp.0.bias", i);
            l.mlp1_w = try getQL(a, m, "decoder.blocks.{d}.mlp.2.weight", i);
            l.mlp1_b = try getL(a, m, "decoder.blocks.{d}.mlp.2.bias", i);
            l.cache_k = try a.alloc(f32, d.n_ctx * ns);
            l.cache_v = try a.alloc(f32, d.n_ctx * ns);
            l.cross_k = try a.alloc(f32, d.n_audio_ctx * ns);
            l.cross_v = try a.alloc(f32, d.n_audio_ctx * ns);
        }
        d.x = try a.alloc(f32, d.n_ctx * ns);
        d.x_ln = try a.alloc(f32, d.n_ctx * ns);
        d.q = try a.alloc(f32, d.n_ctx * ns);
        d.attn = try a.alloc(f32, d.n_ctx * ns);
        d.hidden = try a.alloc(f32, d.n_ctx * 4 * ns);
        d.last_ln = try a.alloc(f32, ns);
        d.attn_scratch = try a.alloc(f32, @max(
            ops.attentionScratch(d.n_ctx, d.n_ctx, d.n_head, ns),
            ops.attentionScratch(d.n_ctx, d.n_audio_ctx, d.n_head, ns),
        ));
        return d;
    }

    pub fn deinit(d: *Decoder, gpa: std.mem.Allocator) void {
        d.arena.deinit();
        gpa.destroy(d);
    }

    /// Project a window's encoder output ([n_audio_ctx][n_state]) into cross K/V.
    pub fn setAudio(d: *Decoder, pool: *ops.Pool, enc: []const f32) void {
        const ns = d.n_state;
        for (d.layers) |*l| {
            ops.linear(pool, enc, d.n_audio_ctx, ns, l.ck_w, null, ns, l.cross_k);
            ops.linear(pool, enc, d.n_audio_ctx, ns, l.cv_w, l.cv_b, ns, l.cross_v);
        }
        d.have_audio = true;
        d.n_past = 0;
    }

    pub fn reset(d: *Decoder) void {
        d.n_past = 0;
    }

    /// Feed `tokens` at positions n_past.., extend the cache, write the last token's logits.
    pub fn forward(d: *Decoder, pool: *ops.Pool, tokens: []const u32, logits: []f32) void {
        std.debug.assert(d.have_audio);
        const n = tokens.len;
        const ns = d.n_state;
        std.debug.assert(n > 0 and d.n_past + n <= d.n_ctx and logits.len >= d.n_vocab);

        for (tokens, 0..) |tok, i| {
            const row = d.x[i * ns ..][0..ns];
            d.te.row(tok, row);
            ops.add(row, d.pe[(d.n_past + i) * ns ..][0..ns]);
        }
        const n_kv = d.n_past + n;
        for (d.layers) |*l| {
            // masked self-attention over the cache
            ops.layerNorm(pool, d.x, n, ns, l.attn_ln_w, l.attn_ln_b, d.x_ln);
            ops.linear(pool, d.x_ln, n, ns, l.q_w, l.q_b, ns, d.q);
            ops.linear(pool, d.x_ln, n, ns, l.k_w, null, ns, l.cache_k[d.n_past * ns ..]);
            ops.linear(pool, d.x_ln, n, ns, l.v_w, l.v_b, ns, l.cache_v[d.n_past * ns ..]);
            ops.attention(pool, d.q, l.cache_k[0 .. n_kv * ns], l.cache_v[0 .. n_kv * ns], n, n_kv, ns, d.n_head, true, d.attn_scratch, d.attn);
            ops.linear(pool, d.attn, n, ns, l.o_w, l.o_b, ns, d.x_ln);
            ops.add(d.x[0 .. n * ns], d.x_ln[0 .. n * ns]);

            // cross-attention over the audio
            ops.layerNorm(pool, d.x, n, ns, l.cross_ln_w, l.cross_ln_b, d.x_ln);
            ops.linear(pool, d.x_ln, n, ns, l.cq_w, l.cq_b, ns, d.q);
            ops.attention(pool, d.q, l.cross_k, l.cross_v, n, d.n_audio_ctx, ns, d.n_head, false, d.attn_scratch, d.attn);
            ops.linear(pool, d.attn, n, ns, l.co_w, l.co_b, ns, d.x_ln);
            ops.add(d.x[0 .. n * ns], d.x_ln[0 .. n * ns]);

            // mlp
            ops.layerNorm(pool, d.x, n, ns, l.mlp_ln_w, l.mlp_ln_b, d.x_ln);
            ops.linear(pool, d.x_ln, n, ns, l.mlp0_w, l.mlp0_b, 4 * ns, d.hidden);
            ops.gelu(pool, d.hidden[0 .. n * 4 * ns]);
            ops.linear(pool, d.hidden, n, 4 * ns, l.mlp1_w, l.mlp1_b, ns, d.x_ln);
            ops.add(d.x[0 .. n * ns], d.x_ln[0 .. n * ns]);
        }
        d.n_past += n;
        // logits for the last token only
        ops.layerNorm(pool, d.x[(n - 1) * ns ..][0..ns], 1, ns, d.ln_w, d.ln_b, d.last_ln);
        ops.linear(pool, d.last_ln, 1, ns, d.te, null, d.n_vocab, logits);
    }
};

/// Load a 2-d weight as int8 blocks (quantized from the file's f16; the f16 copy is temporary).
fn getQ(a: std.mem.Allocator, m: *const model.Model, name: []const u8) !gemm.Q8 {
    const t = m.tensor(name) orelse {
        std.debug.print("zwhisper: model is missing tensor {s}\n", .{name});
        return error.MissingTensor;
    };
    const tmp = try t.toF16(std.heap.page_allocator);
    defer std.heap.page_allocator.free(tmp);
    return gemm.Q8.quantize(a, tmp, t.ne[1], t.ne[0]);
}

fn getQL(a: std.mem.Allocator, m: *const model.Model, comptime fmt: []const u8, layer: usize) !gemm.Q8 {
    var buf: [96]u8 = undefined;
    const name = try std.fmt.bufPrint(&buf, fmt, .{layer});
    return getQ(a, m, name);
}

fn getH(a: std.mem.Allocator, m: *const model.Model, name: []const u8) ![]f16 {
    const t = m.tensor(name) orelse {
        std.debug.print("zwhisper: model is missing tensor {s}\n", .{name});
        return error.MissingTensor;
    };
    return t.toF16(a);
}

fn getHL(a: std.mem.Allocator, m: *const model.Model, comptime fmt: []const u8, layer: usize) ![]f16 {
    var buf: [96]u8 = undefined;
    const name = try std.fmt.bufPrint(&buf, fmt, .{layer});
    return getH(a, m, name);
}

fn get(a: std.mem.Allocator, m: *const model.Model, name: []const u8) ![]f32 {
    const t = m.tensor(name) orelse {
        std.debug.print("zwhisper: model is missing tensor {s}\n", .{name});
        return error.MissingTensor;
    };
    return t.toF32(a);
}

fn getL(a: std.mem.Allocator, m: *const model.Model, comptime fmt: []const u8, layer: usize) ![]f32 {
    var buf: [96]u8 = undefined;
    const name = try std.fmt.bufPrint(&buf, fmt, .{layer});
    return get(a, m, name);
}
