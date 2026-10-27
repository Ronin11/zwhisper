//! Register-blocked matrix kernels.
//!
//! `nt`: C[m][n] = A[m][k] . B[n][k]^T, both operands contiguous along k, which is how a
//! linear layer and the attention scores Q.K^T look. A 4x4 tile of rows is kept in
//! 16 vector accumulators across the whole k loop; one group of 4 B rows stays in L1
//! while every A row in the block streams past it.
//!
//! `pv`: C[m][d] = P[m][t] . V[t][d] (attention probabilities times values), the
//! other orientation: V rows are the reduction axis, so this is a sum of scaled rows.
//!
//! Vector width follows the target: 16 lanes on AVX-512, 8 on AVX2, 4 on NEON.
const std = @import("std");
const builtin = @import("builtin");

pub const V: usize = blk: {
    const cpu = builtin.cpu;
    if (cpu.arch == .x86_64) {
        if (std.Target.x86.featureSetHas(cpu.features, .avx512f)) break :blk 16;
        if (std.Target.x86.featureSetHas(cpu.features, .avx2)) break :blk 8;
        break :blk 4;
    }
    break :blk 4;
};
pub const Vec = @Vector(V, f32);

/// Rows of A per task block and rows of B per group; tuned for L1/L2 sizes of ~48 KB/1 MB.
pub const mc: usize = 64;
pub const nc: usize = 128;

inline fn load(p: [*]const f32) Vec {
    return p[0..V].*;
}

/// Load V elements of a weight row as f32; f16 rows are widened on the fly.
inline fn loadW(comptime T: type, p: [*]const T) Vec {
    if (T == f32) return p[0..V].*;
    const h: @Vector(V, f16) = p[0..V].*;
    return @floatCast(h);
}

/// 4x4 micro-kernel: c[i][j] (+)= dot(a_i, b_j) over k; a_i = a + i*lda, b_j = b + j*ldb.
inline fn micro4x4(comptime TB: type, a: [*]const f32, lda: usize, b: [*]const TB, ldb: usize, k: usize, c: [*]f32, ldc: usize, accumulate: bool) void {
    var acc: [4][4]Vec = undefined;
    inline for (0..4) |i| inline for (0..4) |j| {
        acc[i][j] = @splat(0);
    };
    const a0 = a;
    const a1 = a + lda;
    const a2 = a + 2 * lda;
    const a3 = a + 3 * lda;
    const b0 = b;
    const b1 = b + ldb;
    const b2 = b + 2 * ldb;
    const b3 = b + 3 * ldb;
    var p: usize = 0;
    while (p + V <= k) : (p += V) {
        const va = [4]Vec{ load(a0 + p), load(a1 + p), load(a2 + p), load(a3 + p) };
        const vb = [4]Vec{ loadW(TB, b0 + p), loadW(TB, b1 + p), loadW(TB, b2 + p), loadW(TB, b3 + p) };
        inline for (0..4) |i| inline for (0..4) |j| {
            acc[i][j] = @mulAdd(Vec, va[i], vb[j], acc[i][j]);
        };
    }
    var s: [4][4]f32 = undefined;
    inline for (0..4) |i| inline for (0..4) |j| {
        s[i][j] = @reduce(.Add, acc[i][j]);
    };
    while (p < k) : (p += 1) {
        const av = [4]f32{ a0[p], a1[p], a2[p], a3[p] };
        const bv = [4]f32{ @floatCast(b0[p]), @floatCast(b1[p]), @floatCast(b2[p]), @floatCast(b3[p]) };
        inline for (0..4) |i| inline for (0..4) |j| {
            s[i][j] += av[i] * bv[j];
        };
    }
    inline for (0..4) |i| inline for (0..4) |j| {
        const dst = &c[i * ldc + j];
        dst.* = if (accumulate) dst.* + s[i][j] else s[i][j];
    };
}

/// 1xN rows: a single A row against 4 B rows (used for remainders and m < 4).
inline fn micro1x4(comptime TB: type, a: [*]const f32, b: [*]const TB, ldb: usize, k: usize, c: [*]f32, accumulate: bool) void {
    var acc: [4]Vec = .{ @splat(0), @splat(0), @splat(0), @splat(0) };
    const b0 = b;
    const b1 = b + ldb;
    const b2 = b + 2 * ldb;
    const b3 = b + 3 * ldb;
    var p: usize = 0;
    while (p + V <= k) : (p += V) {
        const va = load(a + p);
        acc[0] = @mulAdd(Vec, va, loadW(TB, b0 + p), acc[0]);
        acc[1] = @mulAdd(Vec, va, loadW(TB, b1 + p), acc[1]);
        acc[2] = @mulAdd(Vec, va, loadW(TB, b2 + p), acc[2]);
        acc[3] = @mulAdd(Vec, va, loadW(TB, b3 + p), acc[3]);
    }
    var s = [4]f32{ @reduce(.Add, acc[0]), @reduce(.Add, acc[1]), @reduce(.Add, acc[2]), @reduce(.Add, acc[3]) };
    while (p < k) : (p += 1) {
        s[0] += a[p] * @as(f32, @floatCast(b0[p]));
        s[1] += a[p] * @as(f32, @floatCast(b1[p]));
        s[2] += a[p] * @as(f32, @floatCast(b2[p]));
        s[3] += a[p] * @as(f32, @floatCast(b3[p]));
    }
    inline for (0..4) |j| {
        c[j] = if (accumulate) c[j] + s[j] else s[j];
    }
}

inline fn dot1(comptime TB: type, a: [*]const f32, b: [*]const TB, k: usize) f32 {
    var acc: Vec = @splat(0);
    var p: usize = 0;
    while (p + V <= k) : (p += V) acc = @mulAdd(Vec, load(a + p), loadW(TB, b + p), acc);
    var s = @reduce(.Add, acc);
    while (p < k) : (p += 1) s += a[p] * @as(f32, @floatCast(b[p]));
    return s;
}

/// One block: rows [m0,m1) of A against rows [n0,n1) of B.
pub fn ntBlock(comptime TB: type, a: [*]const f32, lda: usize, b: [*]const TB, ldb: usize, c: [*]f32, ldc: usize, m0: usize, m1: usize, n0: usize, n1: usize, k: usize, accumulate: bool) void {
    var j = n0;
    while (j + 4 <= n1) : (j += 4) {
        const bj = b + j * ldb;
        var i = m0;
        while (i + 4 <= m1) : (i += 4) {
            micro4x4(TB, a + i * lda, lda, bj, ldb, k, c + i * ldc + j, ldc, accumulate);
        }
        while (i < m1) : (i += 1) {
            micro1x4(TB, a + i * lda, bj, ldb, k, c + i * ldc + j, accumulate);
        }
    }
    while (j < n1) : (j += 1) {
        const bj = b + j * ldb;
        for (m0..m1) |i| {
            const dst = &c[i * ldc + j];
            const s = dot1(TB, a + i * lda, bj, k);
            dst.* = if (accumulate) dst.* + s else s;
        }
    }
}

/// C[m][d] (=|+=) P[m][t] . V[t][d]; d must be a multiple of V (head dims are 64).
pub fn pvBlock(p: [*]const f32, ldp: usize, v: [*]const f32, ldv: usize, c: [*]f32, ldc: usize, m: usize, t: usize, d: usize) void {
    std.debug.assert(d % V == 0);
    const nd = d / V;
    var i: usize = 0;
    // four query rows at a time, all of d in registers (d=64 -> 4 or 8 or 16 vectors per row)
    while (i + 4 <= m) : (i += 4) {
        var dv: usize = 0;
        while (dv < nd) : (dv += 4) {
            const w = @min(4, nd - dv);
            var acc: [4][4]Vec = undefined;
            inline for (0..4) |r| inline for (0..4) |q| {
                acc[r][q] = @splat(0);
            };
            for (0..t) |tt| {
                const vrow = v + tt * ldv + dv * V;
                const p0: Vec = @splat(p[(i + 0) * ldp + tt]);
                const p1: Vec = @splat(p[(i + 1) * ldp + tt]);
                const p2: Vec = @splat(p[(i + 2) * ldp + tt]);
                const p3: Vec = @splat(p[(i + 3) * ldp + tt]);
                inline for (0..4) |q| {
                    if (q < w) {
                        const vv = load(vrow + q * V);
                        acc[0][q] = @mulAdd(Vec, p0, vv, acc[0][q]);
                        acc[1][q] = @mulAdd(Vec, p1, vv, acc[1][q]);
                        acc[2][q] = @mulAdd(Vec, p2, vv, acc[2][q]);
                        acc[3][q] = @mulAdd(Vec, p3, vv, acc[3][q]);
                    }
                }
            }
            inline for (0..4) |r| inline for (0..4) |q| {
                if (q < w) c[(i + r) * ldc + (dv + q) * V ..][0..V].* = acc[r][q];
            };
        }
    }
    while (i < m) : (i += 1) {
        const out = c[i * ldc ..][0..d];
        @memset(out, 0);
        for (0..t) |tt| {
            const s: Vec = @splat(p[i * ldp + tt]);
            const vrow = v + tt * ldv;
            var q: usize = 0;
            while (q < d) : (q += V) {
                out[q..][0..V].* = @mulAdd(Vec, s, load(vrow + q), @as(Vec, out[q..][0..V].*));
            }
        }
    }
}

// ---------------------------------------------------------------------------------------
// Int8 weights: each row is split into blocks of `q8_block`, every block scaled by one
// f32 (absmax / 127). The dot product widens int8 to f32 per vector and applies the
// scale once per block, so the decoder reads a quarter of the bytes f32 would need.
// ---------------------------------------------------------------------------------------

pub const q8_block: usize = 32;

pub const Q8 = struct {
    /// [n][k] int8
    q: []i8,
    /// [n][k / q8_block] f32
    scales: []f32,
    n: usize,
    k: usize,

    pub fn blocks(w: Q8) usize {
        return w.k / q8_block;
    }

    /// Quantize an [n][k] matrix given as f16 (or f32). k must be a multiple of q8_block.
    pub fn quantize(gpa: std.mem.Allocator, w: anytype, n: usize, k: usize) !Q8 {
        std.debug.assert(k % q8_block == 0 and w.len >= n * k);
        const nb = k / q8_block;
        const q = try gpa.alloc(i8, n * k);
        errdefer gpa.free(q);
        const scales = try gpa.alloc(f32, n * nb);
        errdefer gpa.free(scales);
        for (0..n) |i| for (0..nb) |b| {
            const src = w[i * k + b * q8_block ..][0..q8_block];
            var amax: f32 = 0;
            for (src) |v| amax = @max(amax, @abs(@as(f32, @floatCast(v))));
            const scale: f32 = amax / 127.0;
            const inv: f32 = if (amax > 0) 127.0 / amax else 0;
            scales[i * nb + b] = scale;
            const dst = q[i * k + b * q8_block ..][0..q8_block];
            for (dst, src) |*d, v| {
                const r = @round(@as(f32, @floatCast(v)) * inv);
                d.* = @intFromFloat(@min(@max(r, -127), 127));
            }
        };
        return .{ .q = q, .scales = scales, .n = n, .k = k };
    }

    pub fn deinit(w: *Q8, gpa: std.mem.Allocator) void {
        gpa.free(w.q);
        gpa.free(w.scales);
        w.* = undefined;
    }

    /// Dequantize row `i` into `out` (k floats).
    pub fn row(w: Q8, i: usize, out: []f32) void {
        const nb = w.blocks();
        for (0..nb) |b| {
            const sc = w.scales[i * nb + b];
            const src = w.q[i * w.k + b * q8_block ..][0..q8_block];
            const dst = out[b * q8_block ..][0..q8_block];
            for (dst, src) |*d, v| d.* = @as(f32, @floatFromInt(v)) * sc;
        }
    }
};

inline fn loadQ(p: [*]const i8) Vec {
    const v: @Vector(V, i8) = p[0..V].*;
    return @floatFromInt(v);
}

/// Dot of an f32 row with one int8 row across all blocks.
inline fn dotQ(a: [*]const f32, q: [*]const i8, sc: [*]const f32, nb: usize) f32 {
    const chunks = q8_block / V;
    var acc: Vec = @splat(0);
    for (0..nb) |b| {
        var t: Vec = @splat(0);
        inline for (0..chunks) |c| {
            const off = b * q8_block + c * V;
            t = @mulAdd(Vec, load(a + off), loadQ(q + off), t);
        }
        acc = @mulAdd(Vec, t, @splat(sc[b]), acc);
    }
    return @reduce(.Add, acc);
}

/// 4 A rows x 4 int8 rows.
inline fn micro4x4Q(a: [*]const f32, lda: usize, w: Q8, j: usize, c: [*]f32, ldc: usize, accumulate: bool) void {
    const chunks = q8_block / V;
    const nb = w.blocks();
    var acc: [4][4]Vec = undefined;
    inline for (0..4) |i| inline for (0..4) |jj| {
        acc[i][jj] = @splat(0);
    };
    var qp: [4][*]const i8 = undefined;
    var sp: [4][*]const f32 = undefined;
    inline for (0..4) |jj| {
        qp[jj] = w.q.ptr + (j + jj) * w.k;
        sp[jj] = w.scales.ptr + (j + jj) * nb;
    }
    for (0..nb) |b| {
        var t: [4][4]Vec = undefined;
        inline for (0..4) |i| inline for (0..4) |jj| {
            t[i][jj] = @splat(0);
        };
        inline for (0..chunks) |ch| {
            const off = b * q8_block + ch * V;
            const va = [4]Vec{ load(a + off), load(a + lda + off), load(a + 2 * lda + off), load(a + 3 * lda + off) };
            const vq = [4]Vec{ loadQ(qp[0] + off), loadQ(qp[1] + off), loadQ(qp[2] + off), loadQ(qp[3] + off) };
            inline for (0..4) |i| inline for (0..4) |jj| {
                t[i][jj] = @mulAdd(Vec, va[i], vq[jj], t[i][jj]);
            };
        }
        inline for (0..4) |jj| {
            const s: Vec = @splat(sp[jj][b]);
            inline for (0..4) |i| {
                acc[i][jj] = @mulAdd(Vec, t[i][jj], s, acc[i][jj]);
            }
        }
    }
    inline for (0..4) |i| inline for (0..4) |jj| {
        const dst = &c[i * ldc + jj];
        const v = @reduce(.Add, acc[i][jj]);
        dst.* = if (accumulate) dst.* + v else v;
    };
}

/// C[m0..m1][n0..n1] = A . W^T for int8 W.
pub fn ntBlockQ(a: [*]const f32, lda: usize, w: Q8, c: [*]f32, ldc: usize, m0: usize, m1: usize, n0: usize, n1: usize, accumulate: bool) void {
    const nb = w.blocks();
    var j = n0;
    while (j + 4 <= n1) : (j += 4) {
        var i = m0;
        while (i + 4 <= m1) : (i += 4) micro4x4Q(a + i * lda, lda, w, j, c + i * ldc + j, ldc, accumulate);
        while (i < m1) : (i += 1) {
            inline for (0..4) |jj| {
                const dst = &c[i * ldc + j + jj];
                const v = dotQ(a + i * lda, w.q.ptr + (j + jj) * w.k, w.scales.ptr + (j + jj) * nb, nb);
                dst.* = if (accumulate) dst.* + v else v;
            }
        }
    }
    while (j < n1) : (j += 1) {
        for (m0..m1) |i| {
            const dst = &c[i * ldc + j];
            const v = dotQ(a + i * lda, w.q.ptr + j * w.k, w.scales.ptr + j * nb, nb);
            dst.* = if (accumulate) dst.* + v else v;
        }
    }
}

/// One entry point for every weight type: `[]const f32`, `[]const f16`, or `Q8`.
pub fn ntBlockW(a: [*]const f32, lda: usize, w: anytype, k: usize, c: [*]f32, ldc: usize, m0: usize, m1: usize, n0: usize, n1: usize, accumulate: bool) void {
    const W = @TypeOf(w);
    if (W == Q8) {
        std.debug.assert(w.k == k);
        ntBlockQ(a, lda, w, c, ldc, m0, m1, n0, n1, accumulate);
    } else {
        const TB = @typeInfo(W).pointer.child;
        ntBlock(TB, a, lda, w.ptr, k, c, ldc, m0, m1, n0, n1, k, accumulate);
    }
}

test "int8 block matches f32 within quantization error" {
    const gpa = std.testing.allocator;
    const m = 6;
    const n = 7;
    const k = 2 * q8_block;
    const a = try gpa.alloc(f32, m * k);
    defer gpa.free(a);
    const w = try gpa.alloc(f16, n * k);
    defer gpa.free(w);
    var prng = std.Random.DefaultPrng.init(11);
    const r = prng.random();
    for (a) |*x| x.* = r.float(f32) - 0.5;
    for (w) |*x| x.* = @floatCast(r.float(f32) - 0.5);
    var q = try Q8.quantize(gpa, w, n, k);
    defer q.deinit(gpa);
    const c = try gpa.alloc(f32, m * n);
    defer gpa.free(c);
    ntBlockQ(a.ptr, k, q, c.ptr, n, 0, m, 0, n, false);
    for (0..m) |i| for (0..n) |j| {
        var s: f32 = 0;
        for (0..k) |p| s += a[i * k + p] * @as(f32, @floatCast(w[j * k + p]));
        try std.testing.expectApproxEqAbs(s, c[i * n + j], 0.02);
    };
    var rowbuf: [k]f32 = undefined;
    q.row(2, &rowbuf);
    for (0..k) |p| try std.testing.expectApproxEqAbs(@as(f32, @floatCast(w[2 * k + p])), rowbuf[p], 0.005);
}

test "nt block matches naive" {
    const m = 7;
    const n = 9;
    const k = 37;
    var a: [m * k]f32 = undefined;
    var b: [n * k]f32 = undefined;
    var prng = std.Random.DefaultPrng.init(3);
    const r = prng.random();
    for (&a) |*x| x.* = r.float(f32) - 0.5;
    for (&b) |*x| x.* = r.float(f32) - 0.5;
    var c: [m * n]f32 = undefined;
    ntBlock(f32, &a, k, &b, k, &c, n, 0, m, 0, n, k, false);
    for (0..m) |i| for (0..n) |j| {
        var s: f32 = 0;
        for (0..k) |p| s += a[i * k + p] * b[j * k + p];
        try std.testing.expectApproxEqAbs(s, c[i * n + j], 1e-4);
    };
}

test "nt block with f16 weights" {
    const m = 5;
    const n = 6;
    const k = 2 * V + 3;
    var a: [m * k]f32 = undefined;
    var b: [n * k]f16 = undefined;
    var prng = std.Random.DefaultPrng.init(4);
    const r = prng.random();
    for (&a) |*x| x.* = r.float(f32) - 0.5;
    for (&b) |*x| x.* = @floatCast(r.float(f32) - 0.5);
    var c: [m * n]f32 = undefined;
    ntBlock(f16, &a, k, &b, k, &c, n, 0, m, 0, n, k, false);
    for (0..m) |i| for (0..n) |j| {
        var s: f32 = 0;
        for (0..k) |p| s += a[i * k + p] * @as(f32, @floatCast(b[j * k + p]));
        try std.testing.expectApproxEqAbs(s, c[i * n + j], 1e-4);
    };
}

test "pv block matches naive" {
    const m = 6;
    const t = 11;
    const d = 2 * V;
    var p: [m * t]f32 = undefined;
    var v: [t * d]f32 = undefined;
    var prng = std.Random.DefaultPrng.init(5);
    const r = prng.random();
    for (&p) |*x| x.* = r.float(f32);
    for (&v) |*x| x.* = r.float(f32) - 0.5;
    var c: [m * d]f32 = undefined;
    pvBlock(&p, t, &v, d, &c, d, m, t, d);
    for (0..m) |i| for (0..d) |q| {
        var s: f32 = 0;
        for (0..t) |tt| s += p[i * t + tt] * v[tt * d + q];
        try std.testing.expectApproxEqAbs(s, c[i * d + q], 1e-4);
    };
}
