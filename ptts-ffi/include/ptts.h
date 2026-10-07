// C interface to the Core ML Phonon driver, which the PhononTTS Swift package wraps.
// Declared by hand: keep it in step with ../src/lib.rs, which documents each call in full.
#ifndef PTTS_H
#define PTTS_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#define PTTS_UNIT_CPU 0
#define PTTS_UNIT_ANE 2

typedef struct PttsHandle PttsHandle;

/// One utterance's timings. The audio itself went to the callback.
typedef struct {
    uint32_t frames;
    size_t samples;   // 24 kHz mono samples delivered
    double ttfa_ms;   // from the call to the first audio
    double total_ms;
} PttsResult;

/// Called with each frame's audio as it is decoded, on the generating thread. Return false to stop.
typedef bool (*PttsFrameFn)(const float *pcm, size_t n, void *user);

/// Load a model bundle, compiling it for this device first if needed (about 10 s on an
/// iPhone 16 Pro, once per install). `unit` is PTTS_UNIT_ANE or PTTS_UNIT_CPU; `lang` is en, fr,
/// de, es, pt or none. NULL on failure; then ptts_last_error(NULL), on the same thread, says why.
PttsHandle *ptts_new(const char *dir, uint32_t unit, const char *lang);
bool ptts_speak(PttsHandle *h, const char *text, PttsFrameFn cb, void *user, PttsResult *out);
/// NUL-separated voice names, ending in a second NUL. Borrowed.
const char *ptts_voices(const PttsHandle *h);
bool ptts_set_voice(PttsHandle *h, const char *name);
/// The last error on `h`, or the last ptts_new failure on this thread when `h` is NULL.
const char *ptts_last_error(const PttsHandle *h);
void ptts_free(PttsHandle *h);

#endif
