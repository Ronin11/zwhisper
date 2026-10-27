# zwhisper

Whisper speech-to-text in Zig, built for long-form narration: hours of one narrator,
where the timing of speech and pauses matters as much as the words, and where the
narrator plays several characters.

It reads whisper.cpp's `ggml-*.bin` model files directly, runs the encoder and decoder
on the CPU in pure Zig (no ggml, no C), and streams timed segments as they are decoded.
Each segment carries a voice label, so a map tool can tell narration from dialogue and
notice when a new character starts talking. The target is Underscore's transcriber
(a phone, running for a whole book at low priority), with a C ABI for its Rust crate.

```
zig build -Doptimize=ReleaseFast
zig-out/bin/zwhisper transcribe models/ggml-base.en.bin book.wav            # text, one span per line
zig-out/bin/zwhisper transcribe models/ggml-base.en.bin book.wav --json     # JSON lines, streamed
zig-out/bin/zwhisper bench 8                                                # kernel micro-benchmarks
zig build test
```

Input is 16 kHz mono 16-bit WAV (decode with ffmpeg: `-ac 1 -ar 16000 -c:a pcm_s16le`).
Models: English-only `tiny.en`, `base.en`, `small.en` from
`huggingface.co/ggerganov/whisper.cpp`, into `models/`.

## Output

One line per span, as soon as its closing timestamp is decoded:

```
{"t0":1916,"t1":2200,"voice":2,"quoted":true,"pitch":188,"dist":0.41,"text":" \"I still don't know what's happening,\"","tokens":[...]}
```

- `t0`, `t1`: centiseconds from the start of the audio.
- `quoted`: the span is inside quotation marks (dialogue). Segments are split at quotes
  so each line of dialogue is its own span; times inside a segment are interpolated.
- `voice`: a label in order of first appearance; 1 is almost always the narrator.
- `pitch`: median F0 over the span, Hz (0 if nothing voiced).
- `dist`: how far the span was from the voice it joined, in standard units.
- `tokens`: the segment's token ids, timestamps included.

## Names and odd words

Three levers, all on by default or one flag away:

- `--hotwords Carl,Donut,"Borant Corporation"`: tokens that continue a listed word get a
  logit boost while decoding (bias 5, `--hotword-bias`). On five minutes of *Dungeon
  Crawler Carl* with `base.en` this turns "Borenk" into "Borant" and nudges overall
  agreement with `small.en` up a little. Cheap and safe; the first lever to reach for.
- Learned names: a capitalised word seen twice mid-sentence becomes a hotword, so a name
  the model gets right once stays spelt that way. `--no-learn` turns it off.
- `--prompt '"Carl," said Princess Donut. Mordecai nodded.'`: the decoder reads this
  before every window and copies its spelling and style. Write it as a line of the
  book's own prose, quotes and all: a bare list of names makes the model drop quotation
  marks. With `small.en` it fixes spellings the hotwords alone miss, at the cost of the
  model occasionally reciting the prompt over an ambiguous start; segments that are a
  verbatim run of the prompt are dropped for that reason.

Bigger models help most: `small.en` already gets Borant, Van Damme, Hobbit and Wizard of
Oz right where `base.en` guesses.

## The language model, on the phone

The same engine runs a decoder-only model from a GGUF file (Qwen2 family: grouped-query
attention, rotary positions, SwiGLU, int8 weights), so the scene step that used to need
a server can run next to the transcriber:

```
zig-out/bin/zwhisper llm models/qwen2.5-1.5b-instruct-q8_0.gguf "What is the capital of France?"
zig-out/bin/zwhisper scenes models/qwen2.5-1.5b-instruct-q8_0.gguf transcript.jsonl
```

`scenes` is template-driven: the JSON scaffolding is fed as forced tokens and the model
only fills the values, with enums and digits chosen by comparing logits. A small model
can't hand back placeholders, comments or an unterminated list that way. On this
machine Qwen2.5 1.5B reads a five-minute window (2000 tokens) in 12 s and generates at
34 tokens per second; 0.5B is about three times faster. The prompt side is what costs on
a phone, which is the argument for distilling the task into a small tagger next.

## How it works

| Stage | File | Notes |
|---|---|---|
| Model file | `model.zig` | mmap, hparams, mel filters, vocab, tensors in place |
| Spectrogram | `mel.zig` | whisper.cpp's exact pipeline (400-pt FFT, 80 mels, pad, clamp) |
| Kernels | `gemm.zig`, `ops.zig` | 4x4 register-blocked NT GEMM, broadcast-FMA for attention, vector exp/GELU/softmax, f16 weights widened in the kernel, thread pool with futex-sleeping workers |
| Encoder | `encoder.zig` | conv as im2col+GEMM, 6/6/12 pre-norm blocks, per-head transposed-key attention |
| Decoder | `decoder.zig` | KV cache, cross K/V projected once per window, logits for the last token only |
| Decoding | `transcribe.zig` | greedy, whisper.cpp's timestamp rules, previous-window conditioning, prompt and hotwords, streaming emit |
| Prompts | `tokenizer.zig` | whisper.cpp's greedy longest-match tokenizer for prompts and hotwords |
| LLM | `gguf.zig`, `bpe.zig`, `llm.zig` | GGUF loader, byte-level BPE with merges, Qwen2 forward pass with a transposed key cache, greedy generation with repetition penalty, template-driven scenes |
| Voices | `voice.zig` | pitch by autocorrelation, 16-band spectral envelope, online clustering |

Encoder weights are stored as f16 (what the files hold) and widened to f32 inside the
dot product, so the encoder's results are the same as f32. Decoder weights are int8 in
blocks of 32 with one f32 scale each (whisper.cpp's Q8_0 layout in spirit), which is
what makes decoding fast: it is bound by memory bandwidth, and int8 reads a quarter of
the bytes. On the 30 s clip the only change from f32 is one segment split in two.

## Accuracy

Checked against whisper.cpp (vendored in the whisper-rs crate, built in `ref/`):

- First-step logits after `[sot]`: argmax identical, correlation 0.99998.
- zwhisper's mel fed through whisper.cpp's encoder gives whisper.cpp's own logits to
  within 0.003.
- 30 s clip, `base.en`: token stream and timestamps identical to whisper.cpp.
- 10 min clip, `base.en`: 1928 tokens each, 98.4% word agreement; the differences are
  capitalisation and commas, which is what f32 vs f16 activations buys.

## Speed

Ryzen 9800X3D, `base.en`, release build:

| | whisper.cpp (4 threads) | zwhisper (4 threads) | zwhisper (7 threads) |
|---|---|---|---|
| 30 s clip | 3.3 s | 0.50 s | 0.42 s |

Encode is ~150 ms per 30 s window; decode is ~1.8 ms per token with int8 weights.
Voice tagging adds ~2% of the audio length.

## Voices

Heuristic, not a speaker model. On *Dungeon Crawler Carl* (Jeff Hays) five minutes
come out as three voices: narration, Carl's raised dialogue register (170-200 Hz) and a
third for shouted lines. On *We Are Legion* (Ray Porter), where the transcript has no
quote marks, the narrator holds 41 of 84 spans and Dr. Landis's lines group into a
couple of neighbouring voices, with a few one-off exclamations on their own. Good enough
to separate narration from dialogue and to flag a change of speaker; not good enough to
name every character consistently across a book. The next step there is a small
speaker-embedding network (ECAPA/WeSpeaker class, ~6 M parameters) run on each span,
which this engine can host the same way it hosts Whisper.

## Reference harness

`ref/ref.c` links the vendored whisper.cpp and prints segments, first-step logits, or
runs zwhisper's mel through whisper.cpp (`melcheck`). `bench/compare_logits.py` and
`bench/compare_segments.py` diff the outputs. Ground truth for the clips used so far is
in `ref/*.jsonl`.

## Embedding

`zig build` also produces `zig-out/lib/libzwhisper.a` with the C ABI in
`include/zwhisper.h`: open a model, feed 16 kHz f32 samples, get spans through a
callback as they are decoded, with a stop flag polled between tokens. Voice clusters
live in the handle so labels stay stable across calls on one book. Underscore's
`crates/transcribe` builds and links it from `build.rs` (`ZWHISPER_DIR`, or a checkout
beside the Underscore repo), for the host and for Android arm64
(`-Dtarget=aarch64-linux-android`).

## Not yet

- Multilingual models (token ids shift; English-only models only for now).
- Temperature fallback on a failed window (whisper.cpp retries; zwhisper keeps what it
  has and moves on by a window).
- Int8 for the encoder too (it is compute-bound, so the win is memory, not time).
- A real speaker embedding for voices; token-level timestamps for sharper span edges.
- Measured numbers on a phone.
