//! zwhisper: Whisper speech-to-text for long-form narration, in Zig.
//!
//! Reads whisper.cpp's ggml model files, runs the encoder and decoder on the CPU, and
//! streams timed segments. Built for Underscore's job: hours of one narrator, where the
//! timing of speech and pauses matters more than every word.
pub const model = @import("model.zig");
pub const mel = @import("mel.zig");
pub const wav = @import("wav.zig");
pub const ops = @import("ops.zig");
pub const encoder = @import("encoder.zig");
pub const decoder = @import("decoder.zig");
pub const transcribe = @import("transcribe.zig");
pub const clock = @import("clock.zig");
pub const bench = @import("bench.zig");
pub const gemm = @import("gemm.zig");
pub const voice = @import("voice.zig");
pub const tokenizer = @import("tokenizer.zig");
pub const gguf = @import("gguf.zig");
pub const bpe = @import("bpe.zig");
pub const llm = @import("llm.zig");
pub const bert = @import("bert.zig");

pub const Model = model.Model;
pub const Engine = transcribe.Engine;
pub const Segment = transcribe.Segment;

test {
    _ = model;
    _ = mel;
    _ = wav;
    _ = ops;
    _ = encoder;
    _ = decoder;
    _ = transcribe;
    _ = gemm;
    _ = voice;
    _ = tokenizer;
    _ = gguf;
    _ = bpe;
    _ = llm;
    _ = bert;
}
