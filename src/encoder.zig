//! The audio encoder: two strided convolutions, sinusoidal positions, a stack of
//! pre-norm transformer blocks, and a final layer norm. One call turns a 30 s window
//! of log-mel (80 x 3000) into 1500 x n_state audio features.
//!
//! Weights are converted to f32 at load and live in one arena. The convolutions run as
//! im2col + linear so the same kernel serves everything; their output is already
//! [time][n_state], the layout the blocks want (ggml transposes after its conv).
const std = @import("std");
const ops = @import("ops.zig");
const model = @import("model.zig");
const mel = @import("mel.zig");

pub const Layer = struct {
    attn_ln_w: []f32,
    attn_ln_b: []f32,
    q_w: []f16,
    q_b: []f32,
    k_w: []f16,
    v_w: []f16,
    v_b: []f32,
    o_w: []f16,
    o_b: []f32,
    mlp_ln_w: []f32,
    mlp_ln_b: []f32,
    mlp0_w: []f16,
    mlp0_b: []f32,
    mlp1_w: []f16,
    mlp1_b: []f32,
};

pub const Encoder = struct {
    arena: std.heap.ArenaAllocator,
    n_mels: usize,
    n_state: usize,
    n_head: usize,
    n_ctx: usize,
    n_layer: usize,
    conv1_w: []f16, // [n_state][n_mels*3]
    conv1_b: []f32,
    conv2_w: []f16, // [n_state][n_state*3]
    conv2_b: []f32,
    pe: []f32, // [n_ctx][n_state]
    ln_post_w: []f32,
    ln_post_b: []f32,
    layers: []Layer,

    // scratch, sized for one window
    x: []f32, // [n_ctx][n_state]
    x_ln: []f32,
    q: []f32,
    k: []f32,
    v: []f32,
    attn: []f32,
    hidden: []f32, // [n_ctx][4*n_state]
    col: []f32, // im2col, max(2*n_ctx * n_mels*3, n_ctx * n_state*3)
    conv: []f32, // [2*n_ctx][n_state]
    attn_scratch: []f32,

    pub fn load(gpa: std.mem.Allocator, m: *const model.Model) !*Encoder {
        const e = try gpa.create(Encoder);
        errdefer gpa.destroy(e);
        e.arena = std.heap.ArenaAllocator.init(gpa);
        errdefer e.arena.deinit();
        const a = e.arena.allocator();
        const hp = m.hp;
        e.n_mels = hp.n_mels;
        e.n_state = hp.n_audio_state;
        e.n_head = hp.n_audio_head;
        e.n_ctx = hp.n_audio_ctx;
        e.n_layer = hp.n_audio_layer;

        e.conv1_w = try getH(a, m, "encoder.conv1.weight");
        e.conv1_b = try get(a, m, "encoder.conv1.bias");
        e.conv2_w = try getH(a, m, "encoder.conv2.weight");
        e.conv2_b = try get(a, m, "encoder.conv2.bias");
        e.pe = try get(a, m, "encoder.positional_embedding");
        e.ln_post_w = try get(a, m, "encoder.ln_post.weight");
        e.ln_post_b = try get(a, m, "encoder.ln_post.bias");
        e.layers = try a.alloc(Layer, e.n_layer);
        for (e.layers, 0..) |*l, i| {
            l.attn_ln_w = try getL(a, m, "encoder.blocks.{d}.attn_ln.weight", i);
            l.attn_ln_b = try getL(a, m, "encoder.blocks.{d}.attn_ln.bias", i);
            l.q_w = try getHL(a, m, "encoder.blocks.{d}.attn.query.weight", i);
            l.q_b = try getL(a, m, "encoder.blocks.{d}.attn.query.bias", i);
            l.k_w = try getHL(a, m, "encoder.blocks.{d}.attn.key.weight", i);
            l.v_w = try getHL(a, m, "encoder.blocks.{d}.attn.value.weight", i);
            l.v_b = try getL(a, m, "encoder.blocks.{d}.attn.value.bias", i);
            l.o_w = try getHL(a, m, "encoder.blocks.{d}.attn.out.weight", i);
            l.o_b = try getL(a, m, "encoder.blocks.{d}.attn.out.bias", i);
            l.mlp_ln_w = try getL(a, m, "encoder.blocks.{d}.mlp_ln.weight", i);
            l.mlp_ln_b = try getL(a, m, "encoder.blocks.{d}.mlp_ln.bias", i);
            l.mlp0_w = try getHL(a, m, "encoder.blocks.{d}.mlp.0.weight", i);
            l.mlp0_b = try getL(a, m, "encoder.blocks.{d}.mlp.0.bias", i);
            l.mlp1_w = try getHL(a, m, "encoder.blocks.{d}.mlp.2.weight", i);
            l.mlp1_b = try getL(a, m, "encoder.blocks.{d}.mlp.2.bias", i);
        }

        const T = 2 * e.n_ctx;
        const ns = e.n_state;
        e.x = try a.alloc(f32, e.n_ctx * ns);
        e.x_ln = try a.alloc(f32, e.n_ctx * ns);
        e.q = try a.alloc(f32, e.n_ctx * ns);
        e.k = try a.alloc(f32, e.n_ctx * ns);
        e.v = try a.alloc(f32, e.n_ctx * ns);
        e.attn = try a.alloc(f32, e.n_ctx * ns);
        e.hidden = try a.alloc(f32, e.n_ctx * 4 * ns);
        e.col = try a.alloc(f32, @max(T * e.n_mels * 3, e.n_ctx * ns * 3));
        e.conv = try a.alloc(f32, T * ns);
        e.attn_scratch = try a.alloc(f32, ops.attentionScratch(e.n_ctx, e.n_ctx, e.n_head, ns));
        return e;
    }

    pub fn deinit(e: *Encoder, gpa: std.mem.Allocator) void {
        e.arena.deinit();
        gpa.destroy(e);
    }

    /// Encode the window of `2 * n_ctx` mel frames starting at frame `offset`.
    /// Returns a view of the internal [n_ctx][n_state] output, valid until the next call.
    pub fn encode(e: *Encoder, pool: *ops.Pool, spec: *const mel.Mel, offset: usize) []const f32 {
        const T = 2 * e.n_ctx;
        const ns = e.n_state;
        std.debug.assert(spec.n_mel == e.n_mels);

        // conv1: k=3, pad 1, stride 1. col[t][c*3+k] = mel[c][offset+t+k-1]
        {
            const kw = e.n_mels * 3;
            for (0..T) |t| {
                const row = e.col[t * kw ..][0..kw];
                for (0..e.n_mels) |c| {
                    const src = spec.data[c * spec.n_len ..];
                    for (0..3) |k| {
                        const ti = @as(isize, @intCast(t + k)) - 1;
                        const fi = @as(isize, @intCast(offset)) + ti;
                        row[c * 3 + k] = if (ti < 0 or ti >= @as(isize, @intCast(T)) or fi >= @as(isize, @intCast(spec.n_len))) 0 else src[@intCast(fi)];
                    }
                }
            }
            ops.linear(pool, e.col, T, kw, e.conv1_w, e.conv1_b, ns, e.conv);
            ops.gelu(pool, e.conv[0 .. T * ns]);
        }
        // conv2: k=3, pad 1, stride 2. col[t'][c*3+k] = conv[2t'+k-1][c]
        {
            const kw = ns * 3;
            for (0..e.n_ctx) |t| {
                const row = e.col[t * kw ..][0..kw];
                for (0..3) |k| {
                    const ti = @as(isize, @intCast(2 * t + k)) - 1;
                    if (ti < 0 or ti >= @as(isize, @intCast(T))) {
                        for (0..ns) |c| row[c * 3 + k] = 0;
                    } else {
                        const src = e.conv[@as(usize, @intCast(ti)) * ns ..][0..ns];
                        for (0..ns) |c| row[c * 3 + k] = src[c];
                    }
                }
            }
            ops.linear(pool, e.col, e.n_ctx, kw, e.conv2_w, e.conv2_b, ns, e.x);
            ops.gelu(pool, e.x);
            ops.add(e.x, e.pe[0 .. e.n_ctx * ns]);
        }

        for (e.layers) |*l| {
            ops.layerNorm(pool, e.x, e.n_ctx, ns, l.attn_ln_w, l.attn_ln_b, e.x_ln);
            ops.linear(pool, e.x_ln, e.n_ctx, ns, l.q_w, l.q_b, ns, e.q);
            ops.linear(pool, e.x_ln, e.n_ctx, ns, l.k_w, null, ns, e.k);
            ops.linear(pool, e.x_ln, e.n_ctx, ns, l.v_w, l.v_b, ns, e.v);
            ops.attention(pool, e.q, e.k, e.v, e.n_ctx, e.n_ctx, ns, e.n_head, false, e.attn_scratch, e.attn);
            ops.linear(pool, e.attn, e.n_ctx, ns, l.o_w, l.o_b, ns, e.x_ln);
            ops.add(e.x, e.x_ln);

            ops.layerNorm(pool, e.x, e.n_ctx, ns, l.mlp_ln_w, l.mlp_ln_b, e.x_ln);
            ops.linear(pool, e.x_ln, e.n_ctx, ns, l.mlp0_w, l.mlp0_b, 4 * ns, e.hidden);
            ops.gelu(pool, e.hidden);
            ops.linear(pool, e.hidden, e.n_ctx, 4 * ns, l.mlp1_w, l.mlp1_b, ns, e.x_ln);
            ops.add(e.x, e.x_ln);
        }
        ops.layerNorm(pool, e.x, e.n_ctx, ns, e.ln_post_w, e.ln_post_b, e.x_ln);
        return e.x_ln;
    }
};

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
