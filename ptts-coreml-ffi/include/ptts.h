// C ABI over the CoreML Pocket TTS driver. See ../src/lib.rs for the full contract.
#ifndef PTTS_H
#define PTTS_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#define PTTS_UNIT_CPU 0
#define PTTS_UNIT_ANE 2

typedef struct PttsHandle PttsHandle;

/// One utterance's audio and timings. `pcm` is 24 kHz mono, owned by the handle until its next
/// call.
typedef struct {
    const float *pcm;
    size_t pcm_len;
    uint32_t sample_rate;
    uint32_t frames;
    double total_ms;
    double ttfa_ms;
    double per_frame_ms;
    double mean_frame_ms;
    double max_frame_ms;
    double rtf;
} PttsResult;

/// Called with each frame's audio as it is decoded, on the generating thread. Return false to stop.
typedef bool (*PttsFrameFn)(const float *pcm, size_t n, void *user);

/// Compile the graphs if needed (about 10 s on an iPhone 16 Pro, once per install): 1 if it did, 0 if not, -1 on error.
/// `ptts_new` compiles by itself when needed; this lets an app say that it is happening.
int32_t ptts_prepare(const char *dir);
/// `unit` is PTTS_UNIT_ANE or PTTS_UNIT_CPU; `lang` is en, fr, de, es, pt or none.
PttsHandle *ptts_new(const char *dir, uint32_t unit, const char *lang);
bool ptts_speak(PttsHandle *h, const char *text, PttsFrameFn cb, void *user, PttsResult *out);
/// NUL-separated voice names, ending in a second NUL. Borrowed.
const char *ptts_voices(const PttsHandle *h);
bool ptts_set_voice(PttsHandle *h, const char *name);
/// The last error on `h`, or the last construction error when `h` is NULL.
const char *ptts_last_error(const PttsHandle *h);
void ptts_free(PttsHandle *h);

#endif
