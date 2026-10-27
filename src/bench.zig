//! Micro-benchmarks for the kernels the model spends its time in.
const std = @import("std");
const ops = @import("ops.zig");
const clock = @import("clock.zig");

fn fill(prng: *std.Random.DefaultPrng, x: []f32) void {
    const r = prng.random();
    for (x) |*v| v.* = r.float(f32) - 0.5;
}

pub fn run(gpa: std.mem.Allocator, out: *std.Io.Writer, n_threads: usize) !void {
    const pool = try ops.Pool.init(gpa, n_threads);
    defer pool.deinit(gpa);
    var prng = std.Random.DefaultPrng.init(7);

    const cases = [_]struct { name: []const u8, m: usize, k: usize, n: usize }{
        .{ .name = "enc qkv   1500x512 . 512x512", .m = 1500, .k = 512, .n = 512 },
        .{ .name = "enc mlp0  1500x512 . 2048x512", .m = 1500, .k = 512, .n = 2048 },
        .{ .name = "enc mlp1  1500x2048 . 512x2048", .m = 1500, .k = 2048, .n = 512 },
        .{ .name = "conv2 1500x1536 . 512x1536", .m = 1500, .k = 1536, .n = 512 },
        .{ .name = "dec matvec 1x512 . 512x512", .m = 1, .k = 512, .n = 512 },
        .{ .name = "dec mlp0   1x512 . 2048x512", .m = 1, .k = 512, .n = 2048 },
        .{ .name = "dec logits 1x512 . 51864x512", .m = 1, .k = 512, .n = 51864 },
    };
    for (cases) |c| {
        const x = try gpa.alloc(f32, c.m * c.k);
        defer gpa.free(x);
        const w = try gpa.alloc(f32, c.n * c.k);
        defer gpa.free(w);
        const y = try gpa.alloc(f32, c.m * c.n);
        defer gpa.free(y);
        fill(&prng, x);
        fill(&prng, w);
        const flops = 2.0 * @as(f64, @floatFromInt(c.m * c.k * c.n));
        // warm up, then time the best of several
        ops.linear(pool, x, c.m, c.k, w, null, c.n, y);
        var best: u64 = std.math.maxInt(u64);
        var reps: usize = 0;
        var total: u64 = 0;
        while (reps < 50 and total < 500_000_000) : (reps += 1) {
            var t = clock.Timer.start();
            ops.linear(pool, x, c.m, c.k, w, null, c.n, y);
            const ns = t.lap();
            best = @min(best, ns);
            total += ns;
        }
        const bytes = @as(f64, @floatFromInt((c.n * c.k + c.m * c.k + c.m * c.n) * 4));
        try out.print("{s:<34} {d:8.3} ms  {d:7.1} GFLOP/s  {d:6.1} GB/s\n", .{ c.name, @as(f64, @floatFromInt(best)) / 1e6, flops / @as(f64, @floatFromInt(best)), bytes / @as(f64, @floatFromInt(best)) });
    }

    // attention, encoder shape
    {
        const n = 1500;
        const ns = 512;
        const h = 8;
        const q = try gpa.alloc(f32, n * ns);
        defer gpa.free(q);
        const k = try gpa.alloc(f32, n * ns);
        defer gpa.free(k);
        const v = try gpa.alloc(f32, n * ns);
        defer gpa.free(v);
        const o = try gpa.alloc(f32, n * ns);
        defer gpa.free(o);
        const scratch = try gpa.alloc(f32, ops.attentionScratch(n, n, h, ns));
        defer gpa.free(scratch);
        fill(&prng, q);
        fill(&prng, k);
        fill(&prng, v);
        ops.attention(pool, q, k, v, n, n, ns, h, false, scratch, o);
        var best: u64 = std.math.maxInt(u64);
        for (0..5) |_| {
            var t = clock.Timer.start();
            ops.attention(pool, q, k, v, n, n, ns, h, false, scratch, o);
            best = @min(best, t.lap());
        }
        const flops = 4.0 * @as(f64, @floatFromInt(n * n * ns));
        try out.print("{s:<34} {d:8.3} ms  {d:7.1} GFLOP/s\n", .{ "enc attention 1500x1500 x8 heads", @as(f64, @floatFromInt(best)) / 1e6, flops / @as(f64, @floatFromInt(best)) });
    }

    // pool overhead: an empty parallelFor
    {
        const Nop = struct {
            fn run(_: *const u8, _: usize) void {}
        };
        const dummy: u8 = 0;
        var t = clock.Timer.start();
        for (0..1000) |_| pool.parallelFor(pool.size() * 2, &dummy, Nop.run);
        const ns = t.lap();
        try out.print("{s:<34} {d:8.3} us per call\n", .{ "pool parallelFor overhead", @as(f64, @floatFromInt(ns)) / 1e3 / 1000.0 });
    }
}
