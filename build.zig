const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // The engine: pure Zig, no system dependencies.
    const mod = b.addModule("zwhisper", .{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
        .optimize = optimize,
    });

    const exe = b.addExecutable(.{
        .name = "zwhisper",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/main.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{.{ .name = "zwhisper", .module = mod }},
        }),
    });
    b.installArtifact(exe);

    // Static library with the C ABI from include/zwhisper.h (for the Rust crate).
    const lib = b.addLibrary(.{
        .name = "zwhisper",
        .linkage = .static,
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/capi.zig"),
            .target = target,
            .optimize = optimize,
            // linked into position-independent executables and shared objects
            .pic = true,
            // the host process (Rust, glibc or bionic) owns threads and TLS: use its
            // pthreads rather than Zig's raw clone threads
            .link_libc = true,
        }),
    });
    b.installArtifact(lib);
    b.installFile("include/zwhisper.h", "include/zwhisper.h");

    const run_step = b.step("run", "Run the CLI");
    const run_cmd = b.addRunArtifact(exe);
    run_step.dependOn(&run_cmd.step);
    run_cmd.step.dependOn(b.getInstallStep());
    run_cmd.addPassthruArgs();

    const test_step = b.step("test", "Run unit tests");
    test_step.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = mod })).step);
}
