// Reference harness over the vendored whisper.cpp: ground truth for zwhisper.
//
//   ref <model> <wav16k> segments [threads]   JSON lines: {"t0","t1","text","tokens":[...]}
//   ref <model> <wav16k> mel <out.f32>        log-mel as whisper.cpp computes it, [n_mel][n_len] f32
//   ref <model> <wav16k> logits <out.f32>     logits after decoding the prompt [sot] on window 0
//   ref <model> <wav16k> encoder <out.f32>    (not exposed by the API; see mel + logits)
//
// WAV must be 16 kHz mono 16-bit PCM (what testdata/ holds).
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include "whisper.h"

static float *read_wav(const char *path, int *n_out) {
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); exit(1); }
    fseek(f, 0, SEEK_END); long len = ftell(f); fseek(f, 0, SEEK_SET);
    uint8_t *buf = (uint8_t *)malloc(len);
    if (fread(buf, 1, len, f) != (size_t)len) { perror("read"); exit(1); }
    fclose(f);
    // find "data" chunk
    long p = 12;
    while (p + 8 <= len) {
        uint32_t sz; memcpy(&sz, buf + p + 4, 4);
        if (!memcmp(buf + p, "data", 4)) {
            int n = sz / 2;
            float *out = (float *)malloc(sizeof(float) * n);
            const int16_t *s = (const int16_t *)(buf + p + 8);
            for (int i = 0; i < n; i++) out[i] = s[i] / 32768.0f;
            free(buf); *n_out = n; return out;
        }
        p += 8 + sz + (sz & 1);
    }
    fprintf(stderr, "no data chunk\n"); exit(1);
}

static void quiet_log(enum ggml_log_level level, const char *text, void *user) { (void)level; (void)text; (void)user; }

static void json_str(const char *s) {
    putchar('"');
    for (; *s; s++) {
        unsigned char c = *s;
        if (c == '"' || c == '\\') { putchar('\\'); putchar(c); }
        else if (c < 0x20) printf("\\u%04x", c);
        else putchar(c);
    }
    putchar('"');
}

int main(int argc, char **argv) {
    if (argc < 4) { fprintf(stderr, "usage: ref <model> <wav> segments|mel|logits [arg]\n"); return 2; }
    if (!getenv("REF_VERBOSE")) whisper_log_set(quiet_log, NULL);
    struct whisper_context_params cp = whisper_context_default_params();
    cp.use_gpu = false;
    struct whisper_context *ctx = whisper_init_from_file_with_params(argv[1], cp);
    if (!ctx) { fprintf(stderr, "load failed\n"); return 1; }
    int n; float *pcm = read_wav(argv[2], &n);
    const char *mode = argv[3];
    int threads = 8;

    if (!strcmp(mode, "segments")) {
        if (argc > 4) threads = atoi(argv[4]);
        struct whisper_full_params p = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
        p.n_threads = threads; p.language = "en"; p.print_progress = false; p.print_special = false;
        p.temperature_inc = 0.0f; // no fallback: a single greedy pass, same as zwhisper's first milestone
        p.suppress_blank = true; p.suppress_nst = false;
        if (whisper_full(ctx, p, pcm, n) != 0) { fprintf(stderr, "full failed\n"); return 1; }
        int ns = whisper_full_n_segments(ctx);
        for (int i = 0; i < ns; i++) {
            printf("{\"t0\":%lld,\"t1\":%lld,\"text\":", (long long)whisper_full_get_segment_t0(ctx, i), (long long)whisper_full_get_segment_t1(ctx, i));
            json_str(whisper_full_get_segment_text(ctx, i));
            printf(",\"tokens\":[");
            int nt = whisper_full_n_tokens(ctx, i);
            for (int j = 0; j < nt; j++) printf("%s%d", j ? "," : "", whisper_full_get_token_id(ctx, i, j));
            printf("]}\n");
        }
        struct whisper_timings *t = whisper_get_timings(ctx);
        fprintf(stderr, "timings: mel %.1f ms, encode %.1f ms, decode %.1f ms, prompt %.1f ms, batchd %.1f ms\n",
                t->sample_ms, t->encode_ms, t->decode_ms, t->prompt_ms, t->batchd_ms);
        free(t);
    } else if (!strcmp(mode, "melcheck")) {
        // feed zwhisper's mel ([n_mel][n_len] f32 at argv[4]) through whisper.cpp, dump logits to argv[5]
        if (argc < 6) { fprintf(stderr, "need mel in and logits out\n"); return 2; }
        FILE *mf = fopen(argv[4], "rb"); if (!mf) { perror(argv[4]); return 1; }
        fseek(mf, 0, SEEK_END); long ml = ftell(mf); fseek(mf, 0, SEEK_SET);
        float *mel = (float *)malloc(ml); if (fread(mel, 1, ml, mf) != (size_t)ml) { perror("mel"); return 1; } fclose(mf);
        int n_mel = whisper_model_n_mels(ctx); int n_len = (int)(ml / sizeof(float) / n_mel);
        if (whisper_set_mel(ctx, mel, n_len, n_mel) != 0) { fprintf(stderr, "set_mel failed\n"); return 1; }
        if (whisper_encode(ctx, 0, threads) != 0) { fprintf(stderr, "encode failed\n"); return 1; }
        whisper_token prompt[1] = { whisper_token_sot(ctx) };
        if (whisper_decode(ctx, prompt, 1, 0, threads) != 0) { fprintf(stderr, "decode failed\n"); return 1; }
        int nv = whisper_n_vocab(ctx); float *lg = whisper_get_logits(ctx);
        FILE *o = fopen(argv[5], "wb"); fwrite(lg, sizeof(float), nv, o); fclose(o);
        fprintf(stderr, "melcheck: n_len=%d wrote %d logits\n", n_len, nv);
    } else if (!strcmp(mode, "mel") || !strcmp(mode, "logits")) {
        if (argc < 5) { fprintf(stderr, "need output path\n"); return 2; }
        if (whisper_pcm_to_mel(ctx, pcm, n, threads) != 0) { fprintf(stderr, "mel failed\n"); return 1; }
        if (!strcmp(mode, "mel")) {
            // whisper.h has no mel getter; re-run the public path via whisper_set_mel is pointless.
            // Instead expose through the state: not public. So dump via the encoder-independent
            // route: the harness recomputes nothing; it asks the state through whisper_n_len.
            int n_len = whisper_n_len(ctx);
            int n_mel = whisper_model_n_mels(ctx);
            fprintf(stderr, "mel n_len=%d n_mel=%d (data not exported by whisper.h; use zwhisper's own test)\n", n_len, n_mel);
            return 3;
        }
        if (whisper_encode(ctx, 0, threads) != 0) { fprintf(stderr, "encode failed\n"); return 1; }
        whisper_token prompt[1] = { whisper_token_sot(ctx) };
        if (whisper_decode(ctx, prompt, 1, 0, threads) != 0) { fprintf(stderr, "decode failed\n"); return 1; }
        int nv = whisper_n_vocab(ctx);
        float *lg = whisper_get_logits(ctx);
        FILE *o = fopen(argv[4], "wb");
        fwrite(lg, sizeof(float), nv, o);
        fclose(o);
        int best = 0; for (int i = 1; i < nv; i++) if (lg[i] > lg[best]) best = i;
        fprintf(stderr, "wrote %d logits; argmax %d '%s' = %f\n", nv, best, whisper_token_to_str(ctx, best), lg[best]);
    }
    whisper_free(ctx);
    free(pcm);
    return 0;
}
