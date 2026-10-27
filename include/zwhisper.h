/* zwhisper C ABI. Times in milliseconds; audio is 16 kHz mono f32. */
#ifndef ZWHISPER_H
#define ZWHISPER_H
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct zw_handle zw_handle;

typedef struct zw_segment {
    int64_t t0_ms;
    int64_t t1_ms;
    const uint8_t *text;   /* UTF-8, not NUL-terminated, valid during the callback only */
    size_t text_len;
    const uint32_t *tokens; /* token ids of the enclosing segment, valid during the callback */
    size_t n_tokens;
    uint32_t voice;        /* 1-based label in order of first appearance; 0 when voices are off */
    bool quoted;           /* inside quotation marks */
    float pitch_hz;        /* median F0, 0 if unvoiced */
    float voice_dist;      /* distance to the voice joined, standard units */
} zw_segment;

typedef struct zw_stats {
    uint64_t windows, tokens, segments, voices;
    uint64_t encode_ns, decode_ns, voice_ns;
} zw_stats;

typedef void (*zw_emit)(void *user, const zw_segment *seg);

/* threads <= 0: CPU count minus one. NULL on failure (missing or bad model file). */
zw_handle *zw_open(const char *model_path, int threads);
void zw_close(zw_handle *h);

/* 0 ok, 1 stopped through *stop, 2 error. Segments arrive on the calling thread. */
int zw_transcribe(zw_handle *h, const float *samples, size_t n, int64_t offset_ms,
                  bool voices, bool split_quotes, bool condition_on_previous,
                  const bool *stop, zw_emit emit, void *user, zw_stats *stats_or_null);

void zw_reset_voices(zw_handle *h);
uint32_t zw_n_voices(zw_handle *h);
int zw_version(void);

#ifdef __cplusplus
}
#endif
#endif
