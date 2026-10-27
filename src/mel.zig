//! Log-mel spectrogram, step for step the way whisper.cpp does it, so the encoder sees
//! the same input and the reference logits can be compared directly.
//!
//! 16 kHz audio, 400-sample periodic Hann window, hop 160, 201 bins, 80 mel filters from
//! the model file, log10 with a 1e-10 floor, then clamp to (max - 8) and (x + 4) / 4.
//! whisper.cpp reflect-pads 200 samples at the front and appends 30 s of zeros so the last
//! window is always full; `n_len` counts every frame including that padding.
const std = @import("std");

pub const sample_rate: usize = 16_000;
pub const n_fft: usize = 400;
pub const hop: usize = 160;
pub const n_bins: usize = n_fft / 2 + 1; // 201
pub const chunk_frames: usize = 3000; // 30 s of frames, one encoder window

pub const Mel = struct {
    n_mel: usize,
    n_len: usize,
    /// Frames of audio actually present (before the 30 s zero pad), as whisper.cpp's n_len_org.
    n_len_org: usize,
    /// [n_mel][n_len]
    data: []f32,

    pub fn deinit(m: *Mel, gpa: std.mem.Allocator) void {
        gpa.free(m.data);
        m.* = undefined;
    }
};

const Tables = struct {
    hann: [n_fft]f32,
    sin: [n_fft]f32,
    cos: [n_fft]f32,

    fn init() Tables {
        @setEvalBranchQuota(20000);
        var t: Tables = undefined;
        for (0..n_fft) |i| {
            const x = 2.0 * std.math.pi * @as(f64, @floatFromInt(i)) / @as(f64, @floatFromInt(n_fft));
            t.hann[i] = @floatCast(0.5 - 0.5 * @cos(x));
            t.sin[i] = @floatCast(@sin(x));
            t.cos[i] = @floatCast(@cos(x));
        }
        return t;
    }
};

const tables: Tables = Tables.init();

/// Naive DFT of a real input of odd length N, whisper.cpp style (table step = 400 / N).
fn dft(t: *const Tables, in: []const f32, out: []f32) void {
    const n = in.len;
    const step = n_fft / n;
    for (0..n) |k| {
        var re: f32 = 0;
        var im: f32 = 0;
        for (0..n) |j| {
            const idx = (k * j * step) % n_fft;
            re += in[j] * t.cos[idx];
            im -= in[j] * t.sin[idx];
        }
        out[2 * k] = re;
        out[2 * k + 1] = im;
    }
}

/// Cooley-Tukey on a real input; `out` holds N complex pairs; `scratch` needs 4*N floats.
fn fft(t: *const Tables, in: []const f32, out: []f32, scratch: []f32) void {
    const n = in.len;
    if (n == 1) {
        out[0] = in[0];
        out[1] = 0;
        return;
    }
    if (n % 2 == 1) return dft(t, in, out);
    const half = n / 2;
    // even/odd split into the front of scratch, their FFTs into the back
    const even = scratch[0..half];
    const odd = scratch[half..n];
    for (0..half) |i| {
        even[i] = in[2 * i];
        odd[i] = in[2 * i + 1];
    }
    const even_fft = scratch[n .. n + n];
    const odd_fft = scratch[n + n .. n + 3 * half * 2 - n + n]; // n + n .. 3n
    fft(t, even, even_fft, scratch[3 * n ..]);
    fft(t, odd, odd_fft[0 .. 2 * half], scratch[3 * n ..]);
    const step = n_fft / n;
    for (0..half) |k| {
        const idx = k * step;
        const re = t.cos[idx];
        const im = -t.sin[idx];
        const re_odd = odd_fft[2 * k];
        const im_odd = odd_fft[2 * k + 1];
        out[2 * k] = even_fft[2 * k] + re * re_odd - im * im_odd;
        out[2 * k + 1] = even_fft[2 * k + 1] + re * im_odd + im * re_odd;
        out[2 * (k + half)] = even_fft[2 * k] - re * re_odd + im * im_odd;
        out[2 * (k + half) + 1] = even_fft[2 * k + 1] - re * im_odd - im * re_odd;
    }
}

/// Compute the spectrogram of `samples` with `filters` ([n_mel][n_bins]) from the model.
pub fn logMel(gpa: std.mem.Allocator, samples: []const f32, filters: []const f32, n_mel: usize) !Mel {
    std.debug.assert(filters.len == n_mel * n_bins);
    const t = &tables;

    const pad_front = n_fft / 2; // 200, reflected
    const pad_tail = sample_rate * 30;
    const padded_len = samples.len + pad_tail + 2 * pad_front;
    const padded = try gpa.alloc(f32, padded_len);
    defer gpa.free(padded);
    @memset(padded, 0);
    @memcpy(padded[pad_front..][0..samples.len], samples);
    // reflective pad: samples[1..201] reversed at the front
    for (0..pad_front) |i| padded[i] = if (i + 1 < samples.len) samples[pad_front - i] else 0;

    const n_len = (padded_len - n_fft) / hop;
    const n_len_org = 1 + (samples.len + pad_front - n_fft) / hop;
    const n_real = samples.len + pad_front; // frames past this are all zeros
    const data = try gpa.alloc(f32, n_mel * n_len);
    errdefer gpa.free(data);

    var fft_in: [n_fft]f32 = undefined;
    var fft_out: [2 * n_fft]f32 = undefined;
    var scratch: [16 * n_fft]f32 = undefined;
    var power: [n_bins]f32 = undefined;
    const floor: f32 = @floatCast(@log10(1e-10));

    const n_frames_real = @min(n_real / hop + 1, n_len);
    for (0..n_len) |i| {
        if (i >= n_frames_real) {
            for (0..n_mel) |j| data[j * n_len + i] = floor;
            continue;
        }
        const off = i * hop;
        const avail = @min(n_fft, n_real - off);
        for (0..avail) |j| fft_in[j] = t.hann[j] * padded[off + j];
        for (avail..n_fft) |j| fft_in[j] = 0;
        fft(t, &fft_in, &fft_out, &scratch);
        for (0..n_bins) |j| power[j] = fft_out[2 * j] * fft_out[2 * j] + fft_out[2 * j + 1] * fft_out[2 * j + 1];
        for (0..n_mel) |j| {
            const row = filters[j * n_bins ..][0..n_bins];
            var sum: f64 = 0;
            for (0..n_bins) |k| sum += @as(f64, power[k]) * @as(f64, row[k]);
            data[j * n_len + i] = @floatCast(@log10(@max(sum, 1e-10)));
        }
    }

    var mmax: f64 = -1e20;
    for (data) |v| mmax = @max(mmax, @as(f64, v));
    mmax -= 8.0;
    for (data) |*v| {
        if (@as(f64, v.*) < mmax) v.* = @floatCast(mmax);
        v.* = (v.* + 4.0) / 4.0;
    }
    return .{ .n_mel = n_mel, .n_len = n_len, .n_len_org = n_len_org, .data = data };
}

test "silence is a flat spectrogram of the clamp floor" {
    const n_mel = 80;
    const filters = try std.testing.allocator.alloc(f32, n_mel * n_bins);
    defer std.testing.allocator.free(filters);
    @memset(filters, 0.01);
    const samples = try std.testing.allocator.alloc(f32, sample_rate);
    defer std.testing.allocator.free(samples);
    @memset(samples, 0);
    var m = try logMel(std.testing.allocator, samples, filters, n_mel);
    defer m.deinit(std.testing.allocator);
    try std.testing.expectEqual(@as(usize, (sample_rate + 30 * sample_rate + 400 - 400) / 160), m.n_len);
    // every value is (log10(1e-10) + 4) / 4 = -1.5
    for (m.data) |v| try std.testing.expectApproxEqAbs(@as(f32, -1.5), v, 1e-5);
}

test "fft of an impulse is flat" {
    const t = &tables;
    var in: [n_fft]f32 = @splat(0);
    in[0] = 1;
    var out: [2 * n_fft]f32 = undefined;
    var scratch: [16 * n_fft]f32 = undefined;
    fft(t, &in, &out, &scratch);
    for (0..n_fft) |k| {
        try std.testing.expectApproxEqAbs(@as(f32, 1), out[2 * k], 1e-5);
        try std.testing.expectApproxEqAbs(@as(f32, 0), out[2 * k + 1], 1e-5);
    }
}
