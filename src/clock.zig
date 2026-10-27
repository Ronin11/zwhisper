//! Monotonic nanoseconds without an Io handle, for timing inside the engine.
const std = @import("std");

pub fn nowNs() u64 {
    var ts: std.os.linux.timespec = undefined;
    _ = std.os.linux.clock_gettime(.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * 1_000_000_000 + @as(u64, @intCast(ts.nsec));
}

pub const Timer = struct {
    last: u64,

    pub fn start() Timer {
        return .{ .last = nowNs() };
    }

    /// Nanoseconds since start or the previous lap.
    pub fn lap(t: *Timer) u64 {
        const now = nowNs();
        const d = now - t.last;
        t.last = now;
        return d;
    }
};
