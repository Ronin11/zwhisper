//! Tensor kernels: f32, row-major, SIMD through @Vector, work spread over a persistent
//! thread pool. Shapes are passed explicitly; nothing here allocates.
//!
//! A "matrix" is `[]f32` with a row count and a row stride equal to its column count.
//! Weights come from the model as [n_out][n_in] (ggml ne[0] innermost), so a linear layer
//! is y[i][o] = sum_k x[i][k] * w[o][k] + b[o]: every output is a dot product of two
//! contiguous rows, which is what the kernels are built around.
const std = @import("std");
const gemm = @import("gemm.zig");
const clock = @import("clock.zig");

pub const V = gemm.V;
pub const Vec = gemm.Vec;

/// Wall time per kind of op, for `ZW_PROFILE`. Only the calling thread updates these.
pub const Profile = struct {
    linear: u64 = 0,
    attention: u64 = 0,
    gelu: u64 = 0,
    norm: u64 = 0,
    enabled: bool = false,
};
pub var profile = Profile{};

// ---------------------------------------------------------------------------------------
// Thread pool: one job at a time. Workers spin on a generation counter, pull task indices
// from the job, and ack; the caller joins in and waits for every ack, so the job can live
// on the caller's stack.
// ---------------------------------------------------------------------------------------

pub const Pool = struct {
    threads: []std.Thread,
    gen: std.atomic.Value(u32) = .init(0),
    acks: std.atomic.Value(usize) = .init(0),
    job: std.atomic.Value(?*Job) = .init(null),
    quit: std.atomic.Value(bool) = .init(false),
    sleepers: std.atomic.Value(usize) = .init(0),

    const Job = struct {
        ctx: *anyopaque,
        run: *const fn (*anyopaque, usize) void,
        n: usize,
        next: std.atomic.Value(usize) = .init(0),
    };

    /// `n_threads` includes the calling thread; 1 means no workers.
    pub fn init(gpa: std.mem.Allocator, n_threads: usize) !*Pool {
        const p = try gpa.create(Pool);
        errdefer gpa.destroy(p);
        p.* = .{ .threads = &.{} };
        const n_workers = if (n_threads > 1) n_threads - 1 else 0;
        p.threads = try gpa.alloc(std.Thread, n_workers);
        var spawned: usize = 0;
        errdefer {
            p.quit.store(true, .release);
            p.bumpGen();
            for (p.threads[0..spawned]) |t| t.join();
            gpa.free(p.threads);
        }
        for (p.threads) |*t| {
            t.* = try std.Thread.spawn(.{}, worker, .{p});
            spawned += 1;
        }
        return p;
    }

    pub fn deinit(p: *Pool, gpa: std.mem.Allocator) void {
        p.quit.store(true, .release);
        p.bumpGen();
        for (p.threads) |t| t.join();
        gpa.free(p.threads);
        gpa.destroy(p);
    }

    pub fn size(p: *const Pool) usize {
        return p.threads.len + 1;
    }

    /// Spin briefly, then sleep on the generation word until the caller bumps it.
    fn waitForGen(p: *Pool, seen: u32) void {
        var spins: u32 = 0;
        while (p.gen.load(.acquire) == seen) {
            if (spins < 50_000) {
                spins += 1;
                std.atomic.spinLoopHint();
                continue;
            }
            _ = p.sleepers.fetchAdd(1, .acq_rel);
            if (p.gen.load(.acquire) == seen) {
                _ = std.os.linux.futex_4arg(&p.gen.raw, .{ .cmd = .WAIT, .private = true }, seen, null);
            }
            _ = p.sleepers.fetchSub(1, .acq_rel);
        }
    }

    fn bumpGen(p: *Pool) void {
        _ = p.gen.fetchAdd(1, .release);
        if (p.sleepers.load(.acquire) > 0) {
            _ = std.os.linux.futex_4arg(&p.gen.raw, .{ .cmd = .WAKE, .private = true }, std.math.maxInt(i32), null);
        }
    }

    fn worker(p: *Pool) void {
        // Generations start at 0 and only advance once every worker has acked the previous
        // one, so a worker that starts late can never miss a job by reading a newer value.
        var seen: u32 = 0;
        while (true) {
            p.waitForGen(seen);
            seen = p.gen.load(.acquire);
            if (p.quit.load(.acquire)) return;
            if (p.job.load(.acquire)) |job| drain(job);
            _ = p.acks.fetchAdd(1, .acq_rel);
        }
    }

    fn drain(job: *Job) void {
        while (true) {
            const i = job.next.fetchAdd(1, .acq_rel);
            if (i >= job.n) break;
            job.run(job.ctx, i);
        }
    }

    /// Run `func(ctx, i)` for i in 0..n, on all threads, returning when every task is done.
    pub fn parallelFor(p: *Pool, n: usize, ctx: anytype, comptime func: fn (@TypeOf(ctx), usize) void) void {
        const Ctx = @TypeOf(ctx);
        const Wrap = struct {
            fn call(raw: *anyopaque, i: usize) void {
                const c: Ctx = @ptrCast(@alignCast(raw));
                func(c, i);
            }
        };
        if (n == 0) return;
        if (p.threads.len == 0 or n == 1) {
            for (0..n) |i| func(ctx, i);
            return;
        }
        var job = Job{ .ctx = @ptrCast(@constCast(ctx)), .run = Wrap.call, .n = n };
        p.job.store(&job, .release);
        p.acks.store(0, .release);
        p.bumpGen();
        drain(&job);
        while (p.acks.load(.acquire) < p.threads.len) std.atomic.spinLoopHint();
        p.job.store(null, .release);
    }
};

// ---------------------------------------------------------------------------------------
// Vector helpers
// ---------------------------------------------------------------------------------------

pub inline fn dot(a: []const f32, b: []const f32) f32 {
    std.debug.assert(a.len == b.len);
    var acc: Vec = @splat(0);
    var acc2: Vec = @splat(0);
    var i: usize = 0;
    while (i + 2 * V <= a.len) : (i += 2 * V) {
        const a0: Vec = a[i..][0..V].*;
        const b0: Vec = b[i..][0..V].*;
        const a1: Vec = a[i + V ..][0..V].*;
        const b1: Vec = b[i + V ..][0..V].*;
        acc = @mulAdd(Vec, a0, b0, acc);
        acc2 = @mulAdd(Vec, a1, b1, acc2);
    }
    while (i + V <= a.len) : (i += V) {
        const a0: Vec = a[i..][0..V].*;
        const b0: Vec = b[i..][0..V].*;
        acc = @mulAdd(Vec, a0, b0, acc);
    }
    var s = @reduce(.Add, acc + acc2);
    while (i < a.len) : (i += 1) s += a[i] * b[i];
    return s;
}

/// e^x for a vector, Cephes-style range reduction and a degree-5 polynomial; relative
/// error ~2e-7 over [-87, 88]. Inputs below -87 flush to 0.
pub inline fn expVec(x_in: Vec) Vec {
    const lo: Vec = @splat(-87.3);
    const hi: Vec = @splat(88.3);
    const x = @min(@max(x_in, lo), hi);
    const log2e: Vec = @splat(1.44269504088896341);
    const half: Vec = @splat(0.5);
    const fx = @floor(@mulAdd(Vec, x, log2e, half));
    const c1: Vec = @splat(0.693359375);
    const c2: Vec = @splat(-2.12194440e-4);
    var r = x - fx * c1;
    r = r - fx * c2;
    const p0: Vec = @splat(1.9875691500e-4);
    const p1: Vec = @splat(1.3981999507e-3);
    const p2: Vec = @splat(8.3334519073e-3);
    const p3: Vec = @splat(4.1665795894e-2);
    const p4: Vec = @splat(1.6666665459e-1);
    const p5: Vec = @splat(5.0000001201e-1);
    var y = @mulAdd(Vec, p0, r, p1);
    y = @mulAdd(Vec, y, r, p2);
    y = @mulAdd(Vec, y, r, p3);
    y = @mulAdd(Vec, y, r, p4);
    y = @mulAdd(Vec, y, r, p5);
    const r2 = r * r;
    y = @mulAdd(Vec, y, r2, r);
    const one: Vec = @splat(1.0);
    y = y + one;
    // scale by 2^fx through the exponent bits
    const bias: @Vector(V, i32) = @splat(127);
    const e: @Vector(V, i32) = @intFromFloat(fx);
    const bits: @Vector(V, i32) = (e + bias) << @splat(23);
    const scale_v: Vec = @bitCast(bits);
    return y * scale_v;
}

/// y += a * x
pub inline fn axpy(y: []f32, a: f32, x: []const f32) void {
    const av: Vec = @splat(a);
    var i: usize = 0;
    while (i + V <= y.len) : (i += V) {
        const yv: Vec = y[i..][0..V].*;
        const xv: Vec = x[i..][0..V].*;
        y[i..][0..V].* = @mulAdd(Vec, av, xv, yv);
    }
    while (i < y.len) : (i += 1) y[i] += a * x[i];
}

pub inline fn scale(x: []f32, a: f32) void {
    const av: Vec = @splat(a);
    var i: usize = 0;
    while (i + V <= x.len) : (i += V) {
        const xv: Vec = x[i..][0..V].*;
        x[i..][0..V].* = xv * av;
    }
    while (i < x.len) : (i += 1) x[i] *= a;
}

pub fn add(x: []f32, y: []const f32) void {
    var i: usize = 0;
    while (i + V <= x.len) : (i += V) {
        const a: Vec = x[i..][0..V].*;
        const bv: Vec = y[i..][0..V].*;
        x[i..][0..V].* = a + bv;
    }
    while (i < x.len) : (i += 1) x[i] += y[i];
}

// ---------------------------------------------------------------------------------------
// Linear: out[m][n] = x[m][k] . w[n][k]^T + b[n]
// ---------------------------------------------------------------------------------------

fn LinearJob(comptime W: type) type {
    return struct {
        x: []const f32,
        w: W,
        b: ?[]const f32,
        out: []f32,
        m: usize,
        n: usize,
        k: usize,
        mb: usize, // rows of x per task
        nb: usize, // rows of w per task
        n_mblocks: usize,

        fn run(j: *const @This(), task: usize) void {
            const mi = task % j.n_mblocks;
            const ni = task / j.n_mblocks;
            const m0 = mi * j.mb;
            const m1 = @min(m0 + j.mb, j.m);
            const n0 = ni * j.nb;
            const n1 = @min(n0 + j.nb, j.n);
            gemm.ntBlockW(j.x.ptr, j.k, j.w, j.k, j.out.ptr, j.n, m0, m1, n0, n1, false);
            if (j.b) |b| {
                for (m0..m1) |i| {
                    const row = j.out[i * j.n ..];
                    for (n0..n1) |o| row[o] += b[o];
                }
            }
        }
    };
}

/// Below this many multiply-adds an op runs on the calling thread: the pool's barrier
/// costs more than the work, and the decoder is nothing but such ops.
pub const small_op_macs: usize = 1 << 16;

/// `w` is `[]const f32`, `[]const f16` or `gemm.Q8`, logically [n][k].
pub fn linear(pool: *Pool, x: []const f32, m: usize, k: usize, w: anytype, b: ?[]const f32, n: usize, out: []f32) void {
    const W = @TypeOf(w);
    std.debug.assert(x.len >= m * k and out.len >= m * n);
    if (b) |bb| std.debug.assert(bb.len >= n);
    var t = clock.Timer.start();
    if (m * n * k < small_op_macs) {
        gemm.ntBlockW(x.ptr, k, w, k, out.ptr, n, 0, m, 0, n, false);
        if (b) |bb| for (0..m) |i| {
            const row = out[i * n ..];
            for (0..n) |o| row[o] += bb[o];
        };
        if (profile.enabled) profile.linear += t.lap();
        return;
    }
    const mb: usize = if (m >= gemm.mc) gemm.mc else m;
    // weight rows per task: enough tasks to keep every thread busy a few times over
    var nb: usize = gemm.nc;
    const n_mblocks = (m + mb - 1) / mb;
    const want = pool.size() * 3;
    while (n_mblocks * ((n + nb - 1) / nb) < want and nb > 4) nb /= 2;
    const job = LinearJob(W){ .x = x, .w = w, .b = b, .out = out, .m = m, .n = n, .k = k, .mb = mb, .nb = nb, .n_mblocks = n_mblocks };
    pool.parallelFor(n_mblocks * ((n + nb - 1) / nb), &job, LinearJob(W).run);
    if (profile.enabled) profile.linear += t.lap();
}

// ---------------------------------------------------------------------------------------
// Elementwise and row-wise
// ---------------------------------------------------------------------------------------

/// ggml's GELU: the tanh approximation, tanh(u) = 1 - 2 / (1 + e^(2u)).
pub inline fn geluScalar(x: f32) f32 {
    const c: f32 = 0.7978845608028654; // sqrt(2/pi)
    return 0.5 * x * (1.0 + std.math.tanh(c * (x + 0.044715 * x * x * x)));
}

inline fn geluVec(x: Vec) Vec {
    const c: Vec = @splat(0.7978845608028654);
    const k: Vec = @splat(0.044715);
    const one: Vec = @splat(1.0);
    const two: Vec = @splat(2.0);
    const half: Vec = @splat(0.5);
    const u = c * (x + k * x * x * x);
    const t = one - two / (one + expVec(two * u));
    return half * x * (one + t);
}

const GeluJob = struct {
    x: []f32,
    chunk: usize,
    fn run(j: *const GeluJob, task: usize) void {
        const s = j.x[task * j.chunk .. @min((task + 1) * j.chunk, j.x.len)];
        var i: usize = 0;
        while (i + V <= s.len) : (i += V) s[i..][0..V].* = geluVec(s[i..][0..V].*);
        while (i < s.len) : (i += 1) s[i] = geluScalar(s[i]);
    }
};

pub fn gelu(pool: *Pool, x: []f32) void {
    var t = clock.Timer.start();
    const chunk: usize = 32 * 1024;
    const job = GeluJob{ .x = x, .chunk = chunk };
    pool.parallelFor((x.len + chunk - 1) / chunk, &job, GeluJob.run);
    if (profile.enabled) profile.gelu += t.lap();
}

const NormJob = struct {
    x: []const f32,
    w: []const f32,
    b: []const f32,
    out: []f32,
    n: usize,
    rows_per_task: usize,
    m: usize,
    eps: f32,
    fn run(j: *const NormJob, task: usize) void {
        const r0 = task * j.rows_per_task;
        const r1 = @min(r0 + j.rows_per_task, j.m);
        for (r0..r1) |i| normRow(j.x[i * j.n ..][0..j.n], j.w, j.b, j.eps, j.out[i * j.n ..][0..j.n]);
    }
};

inline fn normRow(row: []const f32, w: []const f32, b: []const f32, eps: f32, o: []f32) void {
    const n = row.len;
    var sv: Vec = @splat(0);
    var i: usize = 0;
    while (i + V <= n) : (i += V) sv += @as(Vec, row[i..][0..V].*);
    var sum = @reduce(.Add, sv);
    while (i < n) : (i += 1) sum += row[i];
    const mean = sum / @as(f32, @floatFromInt(n));
    const mv: Vec = @splat(mean);
    var qv: Vec = @splat(0);
    i = 0;
    while (i + V <= n) : (i += V) {
        const d = @as(Vec, row[i..][0..V].*) - mv;
        qv = @mulAdd(Vec, d, d, qv);
    }
    var sq = @reduce(.Add, qv);
    while (i < n) : (i += 1) {
        const d = row[i] - mean;
        sq += d * d;
    }
    const inv = 1.0 / @sqrt(sq / @as(f32, @floatFromInt(n)) + eps);
    const iv: Vec = @splat(inv);
    i = 0;
    while (i + V <= n) : (i += V) {
        const d = (@as(Vec, row[i..][0..V].*) - mv) * iv;
        o[i..][0..V].* = @mulAdd(Vec, d, @as(Vec, w[i..][0..V].*), @as(Vec, b[i..][0..V].*));
    }
    while (i < n) : (i += 1) o[i] = (row[i] - mean) * inv * w[i] + b[i];
}

/// LayerNorm over each row of x[m][n] into out, eps as ggml_norm (1e-5).
pub fn layerNorm(pool: *Pool, x: []const f32, m: usize, n: usize, w: []const f32, b: []const f32, out: []f32) void {
    layerNormEps(pool, x, m, n, w, b, 1e-5, out);
}

/// LayerNorm with an explicit epsilon (BERT uses 1e-12).
pub fn layerNormEps(pool: *Pool, x: []const f32, m: usize, n: usize, w: []const f32, b: []const f32, eps: f32, out: []f32) void {
    var t = clock.Timer.start();
    const rows_per_task: usize = 64;
    const job = NormJob{ .x = x, .w = w, .b = b, .out = out, .n = n, .rows_per_task = rows_per_task, .m = m, .eps = eps };
    pool.parallelFor((m + rows_per_task - 1) / rows_per_task, &job, NormJob.run);
    if (profile.enabled) profile.norm += t.lap();
}

/// RMSNorm over each row of x[m][n] into out: x / sqrt(mean(x^2) + eps) * w.
pub fn rmsNorm(x: []const f32, m: usize, n: usize, w: []const f32, eps: f32, out: []f32) void {
    for (0..m) |i| {
        const row = x[i * n ..][0..n];
        var sv: Vec = @splat(0);
        var j: usize = 0;
        while (j + V <= n) : (j += V) {
            const v: Vec = row[j..][0..V].*;
            sv = @mulAdd(Vec, v, v, sv);
        }
        var ss = @reduce(.Add, sv);
        while (j < n) : (j += 1) ss += row[j] * row[j];
        const inv = 1.0 / @sqrt(ss / @as(f32, @floatFromInt(n)) + eps);
        const iv: Vec = @splat(inv);
        const o = out[i * n ..][0..n];
        j = 0;
        while (j + V <= n) : (j += V) o[j..][0..V].* = @as(Vec, row[j..][0..V].*) * iv * @as(Vec, w[j..][0..V].*);
        while (j < n) : (j += 1) o[j] = row[j] * inv * w[j];
    }
}

inline fn siluVec(x: Vec) Vec {
    const one: Vec = @splat(1.0);
    return x / (one + expVec(-x));
}

/// gate = silu(gate) * up, elementwise.
pub fn siluMul(gate: []f32, up: []const f32) void {
    var i: usize = 0;
    while (i + V <= gate.len) : (i += V) gate[i..][0..V].* = siluVec(gate[i..][0..V].*) * @as(Vec, up[i..][0..V].*);
    while (i < gate.len) : (i += 1) gate[i] = gate[i] / (1.0 + @exp(-gate[i])) * up[i];
}

/// In-place softmax of a row, vectorized exp.
pub fn softmaxRow(x: []f32) void {
    var mxv: Vec = @splat(-std.math.inf(f32));
    var i: usize = 0;
    while (i + V <= x.len) : (i += V) mxv = @max(mxv, @as(Vec, x[i..][0..V].*));
    var mx = @reduce(.Max, mxv);
    while (i < x.len) : (i += 1) mx = @max(mx, x[i]);
    const mv: Vec = @splat(mx);
    var sv: Vec = @splat(0);
    i = 0;
    while (i + V <= x.len) : (i += V) {
        const e = expVec(@as(Vec, x[i..][0..V].*) - mv);
        x[i..][0..V].* = e;
        sv += e;
    }
    var sum = @reduce(.Add, sv);
    while (i < x.len) : (i += 1) {
        x[i] = @exp(x[i] - mx);
        sum += x[i];
    }
    scale(x, 1.0 / sum);
}

// ---------------------------------------------------------------------------------------
// Attention: q[n_q][n_state], k/v[n_kv][n_state], n_head heads of d = n_state / n_head.
// out[n_q][n_state]. With `causal`, query i may only see keys j <= n_kv - n_q + i.
// Scores are scaled by 1/sqrt(d) (whisper.cpp scales q and k by d^-0.25 each; same thing).
//
// Per head, K is transposed once into KT[d][n_kv_pad] so scores come out of the same
// broadcast-FMA kernel as P.V: S = Q_h . KT, softmax rows, O_h = S . V_h.
// ---------------------------------------------------------------------------------------

pub const q_block: usize = 64;

/// Shapes for one attention call. Query heads may share key/value heads (grouped-query
/// attention): head h reads kv head h / (n_head / n_head_kv).
pub const AttnShape = struct {
    n_q: usize,
    n_kv: usize,
    n_head: usize,
    n_head_kv: usize,
    d: usize,
    /// Row strides, in floats.
    q_stride: usize,
    kv_stride: usize,
    out_stride: usize,
    causal: bool,
    /// When set, `k` is already [n_head_kv][d][kt_stride] (keys along the last axis),
    /// as a cache kept in that layout; no transpose is done. kt_stride must be a
    /// multiple of V and columns beyond n_kv must hold finite values.
    kt_stride: ?usize = null,
};

const AttnJob = struct {
    q: []const f32,
    k: []const f32,
    v: []const f32,
    out: []f32,
    sh: AttnShape,
    n_kv_pad: usize,
    kt: []f32, // [n_head_kv][d][n_kv_pad]
    scores: []f32, // per task: q_block * n_kv_pad
    n_qblocks: usize,

    fn transpose(j: *const AttnJob, h: usize) void {
        const d = j.sh.d;
        const kt = j.kt[h * d * j.n_kv_pad ..][0 .. d * j.n_kv_pad];
        for (0..j.sh.n_kv) |t| {
            const src = j.k[t * j.sh.kv_stride + h * d ..][0..d];
            for (0..d) |p| kt[p * j.n_kv_pad + t] = src[p];
        }
        for (0..d) |p| for (j.sh.n_kv..j.n_kv_pad) |t| {
            kt[p * j.n_kv_pad + t] = 0;
        };
    }

    fn run(j: *const AttnJob, task: usize) void {
        const sh = j.sh;
        const d = sh.d;
        const h = task / j.n_qblocks;
        const hk = h / (sh.n_head / sh.n_head_kv);
        const qb = task % j.n_qblocks;
        const q0 = qb * q_block;
        const q1 = @min(q0 + q_block, sh.n_q);
        const m = q1 - q0;
        const sc = j.scores[task * q_block * j.n_kv_pad ..][0 .. m * j.n_kv_pad];
        const kt = j.kt[hk * d * j.n_kv_pad ..];
        // S[m][cols] = Q[q0..q1, head h] . KT; only the columns that hold keys
        const cols = (sh.n_kv + V - 1) / V * V;
        gemm.pvBlock(j.q.ptr + q0 * sh.q_stride + h * d, sh.q_stride, kt.ptr, j.n_kv_pad, sc.ptr, j.n_kv_pad, m, d, cols);
        const scl: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(d)));
        for (0..m) |i| {
            const row = sc[i * j.n_kv_pad ..][0..cols];
            const limit = if (sh.causal) sh.n_kv - sh.n_q + q0 + i + 1 else sh.n_kv;
            scale(row[0..limit], scl);
            softmaxRow(row[0..limit]);
            @memset(row[limit..], 0);
        }
        // O[m][d] = S . V_h
        gemm.pvBlock(sc.ptr, j.n_kv_pad, j.v.ptr + hk * d, sh.kv_stride, j.out.ptr + q0 * sh.out_stride + h * d, sh.out_stride, m, sh.n_kv, d);
    }
};

/// Scratch floats needed by `attentionEx` for these shapes.
pub fn attentionScratchEx(n_q: usize, n_kv: usize, n_head: usize, n_head_kv: usize, d: usize) usize {
    const n_kv_pad = (n_kv + V - 1) / V * V;
    const n_qblocks = (n_q + q_block - 1) / q_block;
    return n_head_kv * d * n_kv_pad + n_head * n_qblocks * q_block * n_kv_pad;
}

/// Scratch floats when keys are kept transposed with stride `kt_stride`.
pub fn attentionScratchKT(n_q: usize, kt_stride: usize, n_head: usize) usize {
    const n_qblocks = (n_q + q_block - 1) / q_block;
    return n_head * n_qblocks * q_block * kt_stride;
}

/// Scratch floats needed by `attention` for these shapes.
pub fn attentionScratch(n_q: usize, n_kv: usize, n_head: usize, n_state: usize) usize {
    return attentionScratchEx(n_q, n_kv, n_head, n_head, n_state / n_head);
}

pub fn attentionEx(pool: *Pool, q: []const f32, k: []const f32, v: []const f32, sh: AttnShape, scratch: []f32, out: []f32) void {
    var t = clock.Timer.start();
    std.debug.assert(sh.d % V == 0 and sh.n_head % sh.n_head_kv == 0);
    const n_kv_pad = if (sh.kt_stride) |st| st else (sh.n_kv + V - 1) / V * V;
    const n_qblocks = (sh.n_q + q_block - 1) / q_block;
    const kt_len = if (sh.kt_stride != null) 0 else sh.n_head_kv * sh.d * n_kv_pad;
    std.debug.assert(scratch.len >= kt_len + sh.n_head * n_qblocks * q_block * n_kv_pad);
    const job = AttnJob{
        .q = q,
        .k = k,
        .v = v,
        .out = out,
        .sh = sh,
        .n_kv_pad = n_kv_pad,
        .kt = if (sh.kt_stride != null) @constCast(k) else scratch[0..kt_len],
        .scores = scratch[kt_len..],
        .n_qblocks = n_qblocks,
    };
    const small = sh.n_q * sh.n_kv * sh.n_head * sh.d < small_op_macs;
    if (sh.kt_stride == null) {
        if (small) {
            for (0..sh.n_head_kv) |h| AttnJob.transpose(&job, h);
        } else pool.parallelFor(sh.n_head_kv, &job, AttnJob.transpose);
    }
    if (small) {
        for (0..sh.n_head * n_qblocks) |i| AttnJob.run(&job, i);
    } else pool.parallelFor(sh.n_head * n_qblocks, &job, AttnJob.run);
    if (profile.enabled) profile.attention += t.lap();
}

/// Multi-head attention with q, k, v all [rows][n_state] and n_head heads of n_state / n_head.
pub fn attention(pool: *Pool, q: []const f32, k: []const f32, v: []const f32, n_q: usize, n_kv: usize, n_state: usize, n_head: usize, causal: bool, scratch: []f32, out: []f32) void {
    attentionEx(pool, q, k, v, .{
        .n_q = n_q,
        .n_kv = n_kv,
        .n_head = n_head,
        .n_head_kv = n_head,
        .d = n_state / n_head,
        .q_stride = n_state,
        .kv_stride = n_state,
        .out_stride = n_state,
        .causal = causal,
    }, scratch, out);
}

// ---------------------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------------------

test "linear agrees with a plain loop" {
    const gpa = std.testing.allocator;
    const pool = try Pool.init(gpa, 3);
    defer pool.deinit(gpa);
    const m = 70;
    const k = 37;
    const n = 133;
    const x = try gpa.alloc(f32, m * k);
    defer gpa.free(x);
    const w = try gpa.alloc(f32, n * k);
    defer gpa.free(w);
    var b: [n]f32 = undefined;
    var prng = std.Random.DefaultPrng.init(1);
    const r = prng.random();
    for (x) |*v| v.* = r.float(f32) - 0.5;
    for (w) |*v| v.* = r.float(f32) - 0.5;
    for (&b) |*v| v.* = r.float(f32) - 0.5;
    const out = try gpa.alloc(f32, m * n);
    defer gpa.free(out);
    linear(pool, x, m, k, w, &b, n, out);
    for (0..m) |i| for (0..n) |o| {
        var s: f32 = b[o];
        for (0..k) |p| s += x[i * k + p] * w[o * k + p];
        try std.testing.expectApproxEqAbs(s, out[i * n + o], 1e-4);
    };
}

test "layer norm has zero mean and unit variance" {
    const gpa = std.testing.allocator;
    const pool = try Pool.init(gpa, 1);
    defer pool.deinit(gpa);
    const x = [_]f32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17 };
    const w: [17]f32 = @splat(1);
    const b: [17]f32 = @splat(0);
    var out: [17]f32 = undefined;
    layerNorm(pool, &x, 1, 17, &w, &b, &out);
    var mean: f32 = 0;
    for (out) |v| mean += v;
    try std.testing.expectApproxEqAbs(@as(f32, 0), mean / 17, 1e-5);
    var vr: f32 = 0;
    for (out) |v| vr += v * v;
    try std.testing.expectApproxEqAbs(@as(f32, 1), vr / 17, 1e-3);
}

test "attention matches a naive implementation, causal and not" {
    const gpa = std.testing.allocator;
    const pool = try Pool.init(gpa, 2);
    defer pool.deinit(gpa);
    const n_q = 70;
    const n_kv = 90;
    const n_head = 2;
    const n_state = 2 * V * n_head;
    const d = n_state / n_head;
    const q = try gpa.alloc(f32, n_q * n_state);
    defer gpa.free(q);
    const k = try gpa.alloc(f32, n_kv * n_state);
    defer gpa.free(k);
    const v = try gpa.alloc(f32, n_kv * n_state);
    defer gpa.free(v);
    var prng = std.Random.DefaultPrng.init(9);
    const r = prng.random();
    for (q) |*x| x.* = r.float(f32) - 0.5;
    for (k) |*x| x.* = r.float(f32) - 0.5;
    for (v) |*x| x.* = r.float(f32) - 0.5;
    const scratch = try gpa.alloc(f32, attentionScratch(n_q, n_kv, n_head, n_state));
    defer gpa.free(scratch);
    const out = try gpa.alloc(f32, n_q * n_state);
    defer gpa.free(out);
    const sc = try gpa.alloc(f32, n_kv);
    defer gpa.free(sc);
    for ([_]bool{ false, true }) |causal| {
        attention(pool, q, k, v, n_q, n_kv, n_state, n_head, causal, scratch, out);
        for (0..n_head) |h| for (0..n_q) |i| {
            const limit = if (causal) n_kv - n_q + i + 1 else n_kv;
            for (0..limit) |t| {
                var s: f32 = 0;
                for (0..d) |p| s += q[i * n_state + h * d + p] * k[t * n_state + h * d + p];
                sc[t] = s / @sqrt(@as(f32, @floatFromInt(d)));
            }
            var mx: f32 = -std.math.inf(f32);
            for (sc[0..limit]) |s| mx = @max(mx, s);
            var sum: f32 = 0;
            for (sc[0..limit]) |*s| {
                s.* = @exp(s.* - mx);
                sum += s.*;
            }
            for (0..d) |p| {
                var o: f32 = 0;
                for (0..limit) |t| o += sc[t] / sum * v[t * n_state + h * d + p];
                try std.testing.expectApproxEqAbs(o, out[i * n_state + h * d + p], 1e-4);
            }
        };
    }
}

test "vector exp and gelu match scalar" {
    var arr: [V]f32 = undefined;
    for (&arr, 0..) |*v, i| v.* = -10.0 + 20.0 * @as(f32, @floatFromInt(i)) / @as(f32, @floatFromInt(V));
    const xs: Vec = arr;
    const e: [V]f32 = expVec(xs);
    const g: [V]f32 = geluVec(xs);
    for (0..V) |i| {
        try std.testing.expectApproxEqRel(@exp(arr[i]), e[i], 1e-6);
        try std.testing.expectApproxEqAbs(geluScalar(arr[i]), g[i], 1e-5);
    }
}
