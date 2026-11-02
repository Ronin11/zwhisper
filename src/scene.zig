//! Scene axes: a ModernBERT-style encoder fine-tuned with indecis, answering three typed
//! questions about a minute of narration: silence (yes/no), energy (1..5) and valence
//! (dark/neutral/light). Loads the indecis model directory as saved by `indecis train`
//! or `indecis compact`: config.json, indecis.json, model.safetensors, tokenizer.json.
//!
//! The forward pass follows indecis/internal/modernbert: pre-norm layers without biases,
//! rotary positions, sliding-window attention on every layer but each third, GeGLU,
//! mean pooling over all tokens, linear heads, Frank-Hall decoding for ordinal scores.
const std = @import("std");
const ops = @import("ops.zig");
const safetensors = @import("safetensors.zig");
const spm = @import("spm.zig");

pub const Kind = enum { noul, choice, score };

pub const Question = struct {
    name: []const u8,
    kind: Kind,
    options: []const []const u8,
    n_logits: usize,
    w: []f32, // [n_logits][hidden]
    b: []f32,
    temperature: f32,
};

/// One answered question: probabilities over options (or `p` for noul), the expected
/// level for scores (0-based) and the most likely option.
pub const Answer = struct {
    p: f32 = 0,
    probs: [8]f32 = @splat(0),
    n: usize = 0,
    expected: f32 = 0,
    choice: usize = 0,
};

const Layer = struct {
    attn_norm: ?[]f32,
    wqkv: []f16, // [3H][H]
    wo: []f16, // [H][H]
    mlp_norm: []f32,
    wi: []f16, // [2I][H]
    wo_mlp: []f16, // [H][I]
    global: bool,
};

pub const Model = struct {
    gpa: std.mem.Allocator,
    arena: std.heap.ArenaAllocator,
    tok: *spm.Tokenizer,
    hidden: usize,
    n_head: usize,
    inter: usize,
    n_layer: usize,
    eps: f32,
    window: usize, // keys each side on local layers
    theta_global: f32,
    theta_local: f32,
    max_len: usize,
    // embeddings: int8 rows with a scale, or f16 rows
    emb_q: ?[]i8,
    emb_scale: ?[]f32,
    emb_h: ?[]f16,
    emb_norm: []f32,
    final_norm: []f32,
    zeros: []f32,
    layers: []Layer,
    questions: []Question,
    // scratch
    x: []f32,
    xn: []f32,
    qkv: []f32,
    ctx: []f32,
    z: []f32,
    g: []f32,
    scores: []f32,
    cos_g: []f32,
    sin_g: []f32,
    cos_l: []f32,
    sin_l: []f32,
    pool: *ops.Pool,

    pub const Files = struct {
        config: []const u8,
        meta: []const u8,
        weights: []const u8,
        tokenizer: []const u8,
    };

    fn num(v: std.json.Value) f64 {
        return switch (v) {
            .integer => |i| @floatFromInt(i),
            .float => |f| f,
            else => 0,
        };
    }

    pub fn load(gpa: std.mem.Allocator, files: Files, threads: usize) !*Model {
        const m = try gpa.create(Model);
        errdefer gpa.destroy(m);
        m.* = undefined;
        m.gpa = gpa;
        m.arena = std.heap.ArenaAllocator.init(gpa);
        errdefer m.arena.deinit();
        const a = m.arena.allocator();

        var cfg = try std.json.parseFromSlice(std.json.Value, gpa, files.config, .{});
        defer cfg.deinit();
        const c = cfg.value.object;
        m.hidden = @intFromFloat(num(c.get("hidden_size") orelse return error.BadConfig));
        m.n_head = @intFromFloat(num(c.get("num_attention_heads") orelse return error.BadConfig));
        m.inter = @intFromFloat(num(c.get("intermediate_size") orelse return error.BadConfig));
        m.n_layer = @intFromFloat(num(c.get("num_hidden_layers") orelse return error.BadConfig));
        const every: usize = @intFromFloat(num(c.get("global_attn_every_n_layers") orelse .{ .integer = 3 }));
        m.window = @as(usize, @intFromFloat(num(c.get("local_attention") orelse .{ .integer = 128 }))) / 2;
        m.theta_global = @floatCast(num(c.get("global_rope_theta") orelse .{ .float = 160000 }));
        m.theta_local = @floatCast(num(c.get("local_rope_theta") orelse .{ .float = 160000 }));
        m.eps = @floatCast(num(c.get("norm_eps") orelse .{ .float = 1e-5 }));

        var meta = try std.json.parseFromSlice(std.json.Value, gpa, files.meta, .{});
        defer meta.deinit();
        const mo = meta.value.object;
        m.max_len = @intFromFloat(num(mo.get("max_len") orelse .{ .integer = 256 }));
        const temps = if (mo.get("temperatures")) |t| t.object else null;

        var st = try safetensors.File.parse(gpa, files.weights);
        defer st.deinit();
        const H = m.hidden;
        m.emb_norm = try f32Of(&st, a, "embeddings.norm.weight");
        m.final_norm = try f32Of(&st, a, "final_norm.weight");
        m.zeros = try a.alloc(f32, H);
        @memset(m.zeros, 0);
        // embeddings
        const emb = st.tensor("embeddings.tok_embeddings.weight") orelse return error.MissingTensor;
        m.emb_q = null;
        m.emb_scale = null;
        m.emb_h = null;
        if (emb.dtype == .i8) {
            m.emb_scale = try f32Of(&st, a, "indecis.embeddings.scale");
            const q = try a.alloc(i8, emb.count());
            @memcpy(q, @as([]const i8, @ptrCast(emb.data[0..emb.count()])));
            m.emb_q = q;
        } else {
            const h = try a.alloc(f16, emb.count());
            switch (emb.dtype) {
                .bf16 => {
                    const src: []align(1) const u16 = @ptrCast(emb.data[0 .. emb.count() * 2]);
                    for (h, src) |*o, v| o.* = @floatCast(safetensors.bf16ToF32(v));
                },
                .f16 => @memcpy(std.mem.sliceAsBytes(h), emb.data[0 .. emb.count() * 2]),
                .f32 => {
                    const src: []align(1) const f32 = @ptrCast(emb.data[0 .. emb.count() * 4]);
                    for (h, src) |*o, v| o.* = @floatCast(v);
                },
                else => return error.Unsupported,
            }
            // rows retrained exactly by indecis override the table
            if (st.tensor("indecis.embeddings.exact_rows")) |rows_t| {
                const rows = try safetensors.toF32(rows_t, gpa, null);
                defer gpa.free(rows);
                const vals = try safetensors.toF32(st.tensor("indecis.embeddings.exact_values") orelse return error.MissingTensor, gpa, null);
                defer gpa.free(vals);
                for (rows, 0..) |r, i| {
                    const row: usize = @intFromFloat(r);
                    for (0..H) |j| h[row * H + j] = @floatCast(vals[i * H + j]);
                }
            }
            m.emb_h = h;
        }
        m.layers = try a.alloc(Layer, m.n_layer);
        for (m.layers, 0..) |*l, i| {
            var buf: [64]u8 = undefined;
            l.global = (i % every) == 0;
            l.attn_norm = if (i == 0) null else try f32Of(&st, a, try std.fmt.bufPrint(&buf, "layers.{d}.attn_norm.weight", .{i}));
            l.wqkv = try f16Of(&st, a, try std.fmt.bufPrint(&buf, "layers.{d}.attn.Wqkv.weight", .{i}));
            l.wo = try f16Of(&st, a, try std.fmt.bufPrint(&buf, "layers.{d}.attn.Wo.weight", .{i}));
            l.mlp_norm = try f32Of(&st, a, try std.fmt.bufPrint(&buf, "layers.{d}.mlp_norm.weight", .{i}));
            l.wi = try f16Of(&st, a, try std.fmt.bufPrint(&buf, "layers.{d}.mlp.Wi.weight", .{i}));
            l.wo_mlp = try f16Of(&st, a, try std.fmt.bufPrint(&buf, "layers.{d}.mlp.Wo.weight", .{i}));
        }
        // questions
        const schema = (mo.get("schema") orelse return error.BadMeta).array;
        m.questions = try a.alloc(Question, schema.items.len);
        for (schema.items, m.questions) |qv, *q| {
            const qo = qv.object;
            q.name = try a.dupe(u8, qo.get("name").?.string);
            const kind_s = qo.get("kind").?.string;
            q.kind = if (std.mem.eql(u8, kind_s, "noul")) .noul else if (std.mem.eql(u8, kind_s, "choice")) .choice else if (std.mem.eql(u8, kind_s, "score")) .score else return error.Unsupported;
            var opts: std.ArrayList([]const u8) = .empty;
            if (qo.get("options")) |ol| for (ol.array.items) |o| {
                try opts.append(a, try a.dupe(u8, switch (o) {
                    .string => |s| s,
                    .object => |oo| oo.get("name").?.string,
                    else => return error.BadMeta,
                }));
            };
            q.options = try opts.toOwnedSlice(a);
            q.n_logits = switch (q.kind) {
                .noul => 1,
                .choice => q.options.len,
                .score => q.options.len - 1,
            };
            var buf: [96]u8 = undefined;
            q.w = try f32Of(&st, a, try std.fmt.bufPrint(&buf, "heads.{s}.weight", .{q.name}));
            q.b = try f32Of(&st, a, try std.fmt.bufPrint(&buf, "heads.{s}.bias", .{q.name}));
            q.temperature = 1;
            if (temps) |t| if (t.get(q.name)) |tv| {
                q.temperature = @floatCast(num(tv));
            };
            if (q.temperature <= 0) q.temperature = 1;
        }
        m.tok = try spm.Tokenizer.init(gpa, files.tokenizer);
        errdefer m.tok.deinit();
        // scratch for max_len tokens
        const n = m.max_len;
        m.x = try a.alloc(f32, n * H);
        m.xn = try a.alloc(f32, n * H);
        m.qkv = try a.alloc(f32, n * 3 * H);
        m.ctx = try a.alloc(f32, n * H);
        m.z = try a.alloc(f32, n * 2 * m.inter);
        m.g = try a.alloc(f32, n * m.inter);
        m.scores = try a.alloc(f32, threads * n);
        const d = H / m.n_head;
        m.cos_g = try a.alloc(f32, n * d / 2);
        m.sin_g = try a.alloc(f32, n * d / 2);
        m.cos_l = try a.alloc(f32, n * d / 2);
        m.sin_l = try a.alloc(f32, n * d / 2);
        ropeTable(m.cos_g, m.sin_g, m.theta_global, n, d);
        ropeTable(m.cos_l, m.sin_l, m.theta_local, n, d);
        m.pool = try ops.Pool.init(gpa, threads);
        return m;
    }

    pub fn deinit(m: *Model) void {
        m.pool.deinit(m.gpa);
        m.tok.deinit();
        m.arena.deinit();
        m.gpa.destroy(m);
    }

    fn f32Of(st: *const safetensors.File, a: std.mem.Allocator, name: []const u8) ![]f32 {
        const t = st.tensor(name) orelse {
            std.debug.print("scene: model is missing tensor {s}\n", .{name});
            return error.MissingTensor;
        };
        return safetensors.toF32(t, a, null);
    }

    fn f16Of(st: *const safetensors.File, a: std.mem.Allocator, name: []const u8) ![]f16 {
        const f = try f32Of(st, a, name);
        const h = try a.alloc(f16, f.len);
        for (h, f) |*o, v| o.* = @floatCast(v);
        return h;
    }

    /// Same rounding as the reference: frequencies and positions in f32.
    fn ropeTable(cos: []f32, sin: []f32, theta: f32, n: usize, d: usize) void {
        const half = d / 2;
        for (0..half) |i| {
            const inv: f32 = @floatCast(1.0 / std.math.pow(f64, theta, @as(f64, @floatFromInt(2 * i)) / @as(f64, @floatFromInt(d))));
            for (0..n) |p| {
                const f: f64 = @as(f32, @floatFromInt(p)) * inv;
                cos[p * half + i] = @floatCast(@cos(f));
                sin[p * half + i] = @floatCast(@sin(f));
            }
        }
    }

    fn embRow(m: *const Model, id: u32, dst: []f32) void {
        const H = m.hidden;
        if (m.emb_q) |q| {
            const s = m.emb_scale.?[id];
            for (dst, q[id * H ..][0..H]) |*o, v| o.* = @as(f32, @floatFromInt(v)) * s;
        } else {
            for (dst, m.emb_h.?[id * H ..][0..H]) |*o, v| o.* = @floatCast(v);
        }
    }

    const AttnJob = struct {
        m: *const Model,
        n: usize,
        global: bool,
        fn run(j: *const AttnJob, task: usize) void {
            // task = head * n + query row
            j.m.attendRow(task / j.n, task % j.n, j.n, j.global, task % j.m.pool.size());
        }
    };

    fn attendRow(m: *const Model, h: usize, i: usize, n: usize, global: bool, slot: usize) void {
        const H = m.hidden;
        const d = H / m.n_head;
        const q = m.qkv[i * 3 * H + h * d ..][0..d];
        const lo: usize = if (global) 0 else (if (i > m.window) i - m.window else 0);
        const hi: usize = if (global) n else @min(n, i + m.window + 1);
        const sc = m.scores[slot * n ..][0..n];
        const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(d)));
        var mx: f32 = -std.math.inf(f32);
        for (lo..hi) |j| {
            const k = m.qkv[j * 3 * H + H + h * d ..][0..d];
            var s: f32 = 0;
            for (q, k) |a, b| s += a * b;
            s *= scale;
            sc[j] = s;
            mx = @max(mx, s);
        }
        var sum: f32 = 0;
        for (lo..hi) |j| {
            sc[j] = @exp(sc[j] - mx);
            sum += sc[j];
        }
        const out = m.ctx[i * H + h * d ..][0..d];
        @memset(out, 0);
        for (lo..hi) |j| {
            const p = sc[j] / sum;
            const v = m.qkv[j * 3 * H + 2 * H + h * d ..][0..d];
            for (out, v) |*o, vv| o.* += p * vv;
        }
    }

    fn rope(m: *const Model, n: usize, global: bool) void {
        const H = m.hidden;
        const d = H / m.n_head;
        const half = d / 2;
        const cos = if (global) m.cos_g else m.cos_l;
        const sin = if (global) m.sin_g else m.sin_l;
        for (0..n) |p| {
            const c = cos[p * half ..][0..half];
            const s = sin[p * half ..][0..half];
            for (0..m.n_head) |h| {
                for (0..2) |which| { // q then k
                    const x = m.qkv[p * 3 * H + which * H + h * d ..][0..d];
                    for (0..half) |i| {
                        const x1 = x[i];
                        const x2 = x[i + half];
                        x[i] = x1 * c[i] - x2 * s[i];
                        x[i + half] = x2 * c[i] + x1 * s[i];
                    }
                }
            }
        }
    }

    /// erf-GELU of the first half times the second half, per row.
    fn geglu(m: *const Model, n: usize) void {
        const I = m.inter;
        for (0..n) |r| {
            const z = m.z[r * 2 * I ..][0 .. 2 * I];
            const g = m.g[r * I ..][0..I];
            for (0..I) |i| g[i] = gelu(z[i]) * z[I + i];
        }
    }

    /// Mean-pooled final hidden state of the ids into `pooled` (hidden floats).
    pub fn encodeIds(m: *Model, ids: []const u32, pooled: []f32) void {
        const H = m.hidden;
        const n = ids.len;
        std.debug.assert(n > 0 and n <= m.max_len);
        for (ids, 0..) |id, p| m.embRow(id, m.xn[p * H ..][0..H]);
        ops.layerNormEps(m.pool, m.xn, n, H, m.emb_norm, m.zeros, m.eps, m.x);
        for (m.layers) |*l| {
            const inp = if (l.attn_norm) |w| blk: {
                ops.layerNormEps(m.pool, m.x, n, H, w, m.zeros, m.eps, m.xn);
                break :blk m.xn;
            } else m.x;
            ops.linear(m.pool, inp, n, H, @as([]const f16, l.wqkv), null, 3 * H, m.qkv);
            m.rope(n, l.global);
            const job = AttnJob{ .m = m, .n = n, .global = l.global };
            m.pool.parallelFor(m.n_head * n, &job, AttnJob.run);
            ops.linear(m.pool, m.ctx, n, H, @as([]const f16, l.wo), null, H, m.xn);
            ops.add(m.x[0 .. n * H], m.xn[0 .. n * H]);
            ops.layerNormEps(m.pool, m.x, n, H, l.mlp_norm, m.zeros, m.eps, m.xn);
            ops.linear(m.pool, m.xn, n, H, @as([]const f16, l.wi), null, 2 * m.inter, m.z);
            m.geglu(n);
            ops.linear(m.pool, m.g, n, m.inter, @as([]const f16, l.wo_mlp), null, H, m.xn);
            ops.add(m.x[0 .. n * H], m.xn[0 .. n * H]);
        }
        ops.layerNormEps(m.pool, m.x, n, H, m.final_norm, m.zeros, m.eps, m.xn);
        @memset(pooled, 0);
        for (0..n) |p| ops.add(pooled, m.xn[p * H ..][0..H]);
        ops.scale(pooled, 1.0 / @as(f32, @floatFromInt(n)));
    }

    /// Answer every question for a pooled vector.
    pub fn answer(m: *const Model, pooled: []const f32, out: []Answer) void {
        const H = m.hidden;
        for (m.questions, out) |q, *ans| {
            var z: [8]f32 = undefined;
            for (0..q.n_logits) |r| {
                var s: f32 = q.b[r];
                for (q.w[r * H ..][0..H], pooled) |w, x| s += w * x;
                z[r] = s / q.temperature;
            }
            ans.* = .{};
            switch (q.kind) {
                .noul => {
                    ans.p = sigmoid(z[0]);
                    ans.n = 0;
                },
                .choice => {
                    var mx: f32 = -std.math.inf(f32);
                    for (z[0..q.n_logits]) |v| mx = @max(mx, v);
                    var sum: f32 = 0;
                    for (0..q.n_logits) |i| {
                        ans.probs[i] = @exp(z[i] - mx);
                        sum += ans.probs[i];
                    }
                    for (0..q.n_logits) |i| ans.probs[i] /= sum;
                    ans.n = q.n_logits;
                },
                .score => {
                    // P(level > k), made monotone, then differences
                    var gt: [8]f32 = undefined;
                    for (0..q.n_logits) |k| {
                        gt[k] = sigmoid(z[k]);
                        if (k > 0) gt[k] = @min(gt[k], gt[k - 1]);
                    }
                    const levels = q.options.len;
                    for (0..levels) |k| {
                        const above: f32 = if (k > 0) gt[k - 1] else 1.0;
                        const below: f32 = if (k < q.n_logits) gt[k] else 0.0;
                        ans.probs[k] = above - below;
                    }
                    ans.n = levels;
                },
            }
            if (ans.n > 0) {
                var best: usize = 0;
                for (0..ans.n) |k| {
                    ans.expected += @as(f32, @floatFromInt(k)) * ans.probs[k];
                    if (ans.probs[k] > ans.probs[best]) best = k;
                }
                ans.choice = best;
            }
        }
    }

    /// Tokenize, encode and answer. `out` has one slot per question.
    pub fn rate(m: *Model, gpa: std.mem.Allocator, text: []const u8, out: []Answer) !void {
        const ids = try m.tok.encode(gpa, text, m.max_len);
        defer gpa.free(ids);
        const pooled = try gpa.alloc(f32, m.hidden);
        defer gpa.free(pooled);
        m.encodeIds(ids, pooled);
        m.answer(pooled, out);
    }
};

fn sigmoid(x: f32) f32 {
    return 1.0 / (1.0 + @exp(-x));
}

/// erf via Abramowitz & Stegun 7.1.26 (|error| < 1.5e-7), enough for the exact GELU.
fn erf(x: f32) f32 {
    const sign: f32 = if (x < 0) -1 else 1;
    const ax = @abs(x);
    const t = 1.0 / (1.0 + 0.3275911 * ax);
    const y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * @exp(-ax * ax);
    return sign * y;
}

fn gelu(x: f32) f32 {
    return 0.5 * x * (1.0 + erf(x * 0.7071067811865476));
}

test "ordinal decoding sums to one" {
    const gpa = std.testing.allocator;
    _ = gpa;
    var q = Question{ .name = "e", .kind = .score, .options = &.{ "1", "2", "3" }, .n_logits = 2, .w = undefined, .b = undefined, .temperature = 1 };
    var w = [_]f32{ 0, 0, 0, 0 };
    var b = [_]f32{ 1.0, -1.0 };
    q.w = &w;
    q.b = &b;
    var m: Model = undefined;
    m.hidden = 2;
    m.questions = @constCast(&[_]Question{q});
    var ans: [1]Answer = undefined;
    m.answer(&[_]f32{ 0, 0 }, &ans);
    var sum: f32 = 0;
    for (ans[0].probs[0..3]) |p| sum += p;
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), sum, 1e-5);
    try std.testing.expect(ans[0].probs[1] > ans[0].probs[0] and ans[0].probs[1] > ans[0].probs[2]);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), gelu(0.5) * 2 - 0.5 + 0.5 - (gelu(0.5) * 2 - 0.5), 1e-6);
}
