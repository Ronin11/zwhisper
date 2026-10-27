//! Who is speaking: a voice tag per span for narrators who do character voices.
//!
//! Two cheap cues, both local to the span of audio:
//!   - pitch: median fundamental frequency over voiced frames, by normalized
//!     autocorrelation on the 16 kHz samples. Narrators drop or raise their pitch for
//!     characters, which is the clearest single signal.
//!   - timbre: the log-mel spectral envelope (each frame's shape with its loudness
//!     removed), averaged over the span. Formant structure survives the averaging;
//!     phonetic content mostly averages out over a sentence.
//!
//! `Voices` clusters spans online: every feature is standardized by running statistics
//! over the book so far, and a span joins the nearest voice when it is within
//! `dist_thold` standard units, otherwise it starts a new one. Ids are small integers in
//! order of first appearance. Nothing here is a speaker model; it is a consistent label
//! the map tool can use to tell dialogue from narration and to notice a new character.
const std = @import("std");
const ops = @import("ops.zig");
const mel = @import("mel.zig");

pub const sample_rate: f32 = 16_000;

pub const Pitch = struct {
    /// Median F0 in Hz over voiced frames; 0 when nothing voiced was found.
    hz: f32 = 0,
    /// Share of frames judged voiced.
    voiced: f32 = 0,
};

const frame: usize = 1024;
const hop: usize = 512;
const lag_min: usize = 40; // 400 Hz
const lag_max: usize = 320; // 50 Hz
const voicing_thold: f32 = 0.55;

/// F0 of `s` (16 kHz mono). `scratch` must hold at least s.len / hop + 1 floats.
pub fn pitch(s: []const f32, scratch: []f32) Pitch {
    if (s.len < frame) return .{};
    var n_voiced: usize = 0;
    var n_frames: usize = 0;
    var r: [lag_max + 1]f32 = undefined;
    var start: usize = 0;
    while (start + frame <= s.len) : (start += hop) {
        n_frames += 1;
        const x = s[start..][0..frame];
        var mean: f32 = 0;
        for (x) |v| mean += v;
        mean /= @floatFromInt(frame);
        var e0: f32 = 0;
        for (x) |v| e0 += (v - mean) * (v - mean);
        if (e0 < 1e-4 * @as(f32, @floatFromInt(frame))) continue; // silence
        var r_max: f32 = 0;
        var lag = lag_min;
        while (lag <= lag_max) : (lag += 1) {
            r[lag] = corr(x, mean, lag);
            r_max = @max(r_max, r[lag]);
        }
        if (r_max < voicing_thold) continue;
        // The shortest lag that is a local peak near the global maximum is the period;
        // longer near-equal peaks are its multiples.
        var best: usize = 0;
        lag = lag_min + 1;
        while (lag < lag_max) : (lag += 1) {
            if (r[lag] >= 0.9 * r_max and r[lag] >= r[lag - 1] and r[lag] >= r[lag + 1]) {
                best = lag;
                break;
            }
        }
        if (best == 0) continue;
        // parabolic interpolation around the peak
        const y0 = r[best - 1];
        const y1 = r[best];
        const y2 = r[best + 1];
        const denom = y0 - 2 * y1 + y2;
        const delta: f32 = if (@abs(denom) > 1e-9) 0.5 * (y0 - y2) / denom else 0;
        scratch[n_voiced] = sample_rate / (@as(f32, @floatFromInt(best)) + delta);
        n_voiced += 1;
    }
    if (n_voiced == 0) return .{ .hz = 0, .voiced = 0 };
    const v = scratch[0..n_voiced];
    std.mem.sort(f32, v, {}, std.sort.asc(f32));
    return .{ .hz = v[n_voiced / 2], .voiced = @as(f32, @floatFromInt(n_voiced)) / @as(f32, @floatFromInt(n_frames)) };
}

inline fn corr(x: []const f32, mean: f32, lag: usize) f32 {
    const n = x.len - lag;
    const mv: ops.Vec = @splat(mean);
    var nv: ops.Vec = @splat(0);
    var av: ops.Vec = @splat(0);
    var bv: ops.Vec = @splat(0);
    var i: usize = 0;
    while (i + ops.V <= n) : (i += ops.V) {
        const a = @as(ops.Vec, x[i..][0..ops.V].*) - mv;
        const b = @as(ops.Vec, x[i + lag ..][0..ops.V].*) - mv;
        nv = @mulAdd(ops.Vec, a, b, nv);
        av = @mulAdd(ops.Vec, a, a, av);
        bv = @mulAdd(ops.Vec, b, b, bv);
    }
    var num = @reduce(.Add, nv);
    var ea = @reduce(.Add, av);
    var eb = @reduce(.Add, bv);
    while (i < n) : (i += 1) {
        const a = x[i] - mean;
        const b = x[i + lag] - mean;
        num += a * b;
        ea += a * a;
        eb += b * b;
    }
    return num / (@sqrt(ea * eb) + 1e-9);
}

/// Bands the mel bins are pooled into for the envelope; fewer dimensions, less noise.
pub const n_bands: usize = 16;

/// Spectral envelope over mel frames [f0, f1): per frame, the log-mel bins minus the
/// frame's mean (loudness), pooled into `n_bands` bands and averaged over frames.
/// Frames below the span's loudness median are skipped so pauses don't dilute it.
pub fn envelope(spec: *const mel.Mel, f0: usize, f1: usize, out: []f32) void {
    @memset(out, 0);
    const a = @min(f0, spec.n_len);
    const b = @min(f1, spec.n_len);
    if (b <= a) return;
    // loudness threshold: mean of frame means
    var thr: f32 = 0;
    for (a..b) |t| {
        var m: f32 = 0;
        for (0..spec.n_mel) |j| m += spec.data[j * spec.n_len + t];
        thr += m / @as(f32, @floatFromInt(spec.n_mel));
    }
    thr /= @floatFromInt(b - a);
    var used: usize = 0;
    const per = spec.n_mel / n_bands;
    for (a..b) |t| {
        var m: f32 = 0;
        for (0..spec.n_mel) |j| m += spec.data[j * spec.n_len + t];
        m /= @floatFromInt(spec.n_mel);
        if (m < thr) continue;
        used += 1;
        for (0..n_bands) |k| {
            var acc: f32 = 0;
            for (k * per..(k + 1) * per) |j| acc += spec.data[j * spec.n_len + t] - m;
            out[k] += acc / @as(f32, @floatFromInt(per));
        }
    }
    if (used > 0) ops.scale(out, 1.0 / @as(f32, @floatFromInt(used)));
}

pub const Assignment = struct {
    id: u32,
    /// Standardized distance to the voice joined (0 for the very first span).
    distance: f32,
    /// True when this span started a new voice.
    new: bool,
};

/// Online clustering of spans into voices on standardized features.
pub const Voices = struct {
    gpa: std.mem.Allocator,
    /// Feature layout: [n_env envelope dims][1 log2 pitch]. Pitch 0 means unknown.
    dim: usize,
    n_env: usize,
    /// Centroids [n][dim], running means of raw features (pitch mean over known values).
    centroids: std.ArrayList(f32) = .empty,
    counts: std.ArrayList(u32) = .empty,
    pitch_counts: std.ArrayList(u32) = .empty,
    /// Welford statistics over every span seen, per dimension.
    n_seen: usize = 0,
    mean: []f32,
    m2: []f32,
    n_pitch_seen: usize = 0,
    /// Distance in standard units beyond which a span starts a new voice.
    dist_thold: f32 = 1.1,
    /// Weight of the pitch dimension relative to one envelope dimension.
    pitch_weight: f32 = 24,
    /// Spans to see before distances mean anything; they all join voice 1.
    warmup: usize = 4,

    pub fn init(gpa: std.mem.Allocator, n_env: usize) !Voices {
        const dim = n_env + 1;
        const mean = try gpa.alloc(f32, dim);
        errdefer gpa.free(mean);
        @memset(mean, 0);
        const m2 = try gpa.alloc(f32, dim);
        @memset(m2, 0);
        return .{ .gpa = gpa, .dim = dim, .n_env = n_env, .mean = mean, .m2 = m2 };
    }

    pub fn deinit(v: *Voices) void {
        v.centroids.deinit(v.gpa);
        v.counts.deinit(v.gpa);
        v.pitch_counts.deinit(v.gpa);
        v.gpa.free(v.mean);
        v.gpa.free(v.m2);
    }

    pub fn count(v: *const Voices) usize {
        return v.counts.items.len;
    }

    fn sigma(v: *const Voices, d: usize) f32 {
        const n: f32 = @floatFromInt(if (d == v.n_env) v.n_pitch_seen else v.n_seen);
        if (n < 2) return 1;
        return @max(@sqrt(v.m2[d] / (n - 1)), 1e-3);
    }

    /// Standardized distance between a feature vector and centroid i.
    fn distance(v: *const Voices, f: []const f32, i: usize) f32 {
        const c = v.centroids.items[i * v.dim ..][0..v.dim];
        var acc: f32 = 0;
        var wsum: f32 = 0;
        for (0..v.n_env) |d| {
            const z = (f[d] - c[d]) / v.sigma(d);
            acc += z * z;
            wsum += 1;
        }
        const p = v.n_env;
        if (f[p] != 0 and v.pitch_counts.items[i] > 0) {
            const z = (f[p] - c[p]) / v.sigma(p);
            acc += v.pitch_weight * z * z;
            wsum += v.pitch_weight;
        }
        return @sqrt(acc / wsum);
    }

    fn observe(v: *Voices, f: []const f32) void {
        v.n_seen += 1;
        const n: f32 = @floatFromInt(v.n_seen);
        for (0..v.n_env) |d| {
            const delta = f[d] - v.mean[d];
            v.mean[d] += delta / n;
            v.m2[d] += delta * (f[d] - v.mean[d]);
        }
        const p = v.n_env;
        if (f[p] != 0) {
            v.n_pitch_seen += 1;
            const np: f32 = @floatFromInt(v.n_pitch_seen);
            const delta = f[p] - v.mean[p];
            v.mean[p] += delta / np;
            v.m2[p] += delta * (f[p] - v.mean[p]);
        }
    }

    fn join(v: *Voices, i: usize, f: []const f32) void {
        const c = v.centroids.items[i * v.dim ..][0..v.dim];
        const n: f32 = @floatFromInt(v.counts.items[i]);
        for (0..v.n_env) |d| c[d] = (c[d] * n + f[d]) / (n + 1);
        v.counts.items[i] += 1;
        const p = v.n_env;
        if (f[p] != 0) {
            const np: f32 = @floatFromInt(v.pitch_counts.items[i]);
            c[p] = (c[p] * np + f[p]) / (np + 1);
            v.pitch_counts.items[i] += 1;
        }
    }

    /// Assign a span with features `f` (len dim; f[n_env] = log2 pitch or 0).
    pub fn assign(v: *Voices, f: []const f32) !Assignment {
        std.debug.assert(f.len == v.dim);
        v.observe(f);
        if (v.count() == 0) {
            try v.centroids.appendSlice(v.gpa, f);
            try v.counts.append(v.gpa, 1);
            try v.pitch_counts.append(v.gpa, if (f[v.n_env] != 0) 1 else 0);
            return .{ .id = 1, .distance = 0, .new = true };
        }
        var best: usize = 0;
        var best_d: f32 = std.math.inf(f32);
        for (0..v.count()) |i| {
            const d = v.distance(f, i);
            if (d < best_d) {
                best_d = d;
                best = i;
            }
        }
        if (v.n_seen <= v.warmup or best_d <= v.dist_thold) {
            v.join(best, f);
            return .{ .id = @intCast(best + 1), .distance = best_d, .new = false };
        }
        try v.centroids.appendSlice(v.gpa, f);
        try v.counts.append(v.gpa, 1);
        try v.pitch_counts.append(v.gpa, if (f[v.n_env] != 0) 1 else 0);
        return .{ .id = @intCast(v.count()), .distance = best_d, .new = true };
    }
};

test "pitch of a sine" {
    const gpa = std.testing.allocator;
    const n = 16000;
    const s = try gpa.alloc(f32, n);
    defer gpa.free(s);
    for (s, 0..) |*v, i| v.* = 0.3 * @sin(2.0 * std.math.pi * 150.0 * @as(f32, @floatFromInt(i)) / 16000.0);
    const scratch = try gpa.alloc(f32, n / hop + 1);
    defer gpa.free(scratch);
    const p = pitch(s, scratch);
    try std.testing.expect(p.voiced > 0.9);
    try std.testing.expectApproxEqRel(@as(f32, 150.0), p.hz, 0.02);
}

test "silence is unvoiced" {
    const gpa = std.testing.allocator;
    const s = try gpa.alloc(f32, 8000);
    defer gpa.free(s);
    @memset(s, 0);
    const scratch = try gpa.alloc(f32, 8000 / hop + 1);
    defer gpa.free(scratch);
    try std.testing.expectEqual(@as(f32, 0), pitch(s, scratch).hz);
}

test "voices split and merge" {
    var v = try Voices.init(std.testing.allocator, 2);
    defer v.deinit();
    v.warmup = 0;
    v.dist_thold = 1.0;
    // two tight groups far apart in envelope space; pitch unknown
    const a = [_]f32{ 0, 0, 0 };
    const a2 = [_]f32{ 0.1, -0.1, 0 };
    const b = [_]f32{ 5, 5, 0 };
    const b2 = [_]f32{ 5.1, 4.9, 0 };
    try std.testing.expectEqual(@as(u32, 1), (try v.assign(&a)).id);
    try std.testing.expectEqual(@as(u32, 2), (try v.assign(&b)).id);
    try std.testing.expectEqual(@as(u32, 1), (try v.assign(&a2)).id);
    try std.testing.expectEqual(@as(u32, 2), (try v.assign(&b2)).id);
    try std.testing.expectEqual(@as(usize, 2), v.count());
}
