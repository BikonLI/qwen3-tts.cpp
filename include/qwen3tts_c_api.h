/* qwen3tts_c_api.h - C API wrapper for qwen3-tts.cpp (Nim FFI) */
#ifndef QWEN3TTS_C_API_H
#define QWEN3TTS_C_API_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque handle */
typedef struct Qwen3Tts Qwen3Tts;

/* Generation parameters */
typedef struct Qwen3TtsParams {
    int32_t max_audio_tokens;    /* default: 4096 */
    float   temperature;         /* default: 0.9, 0=greedy */
    float   top_p;               /* default: 1.0 */
    int32_t top_k;               /* default: 50, 0=disabled */
    int32_t n_threads;           /* default: 4 */
    float   repetition_penalty;  /* default: 1.05 */
    int32_t language_id;         /* -1=auto, 2050=en, 2058=ja, 2055=zh, etc. */

    int32_t task_type;           /* 0=auto,1=voice_clone,2=custom_voice,3=voice_design */
    int32_t model_variant;       /* 0=auto,1=base,2=custom_voice,3=voice_design */

    const char* instruct;        /* optional */
    const char* speaker;         /* optional */
    const char* reference_text;  /* required for base voice_clone unless x_vector_only_mode */
    int32_t x_vector_only_mode;  /* 0=false, 1=true */
} Qwen3TtsParams;

/* Generated audio result */
typedef struct Qwen3TtsAudio {
    const float* samples;  /* PCM float32 mono */
    int32_t n_samples;
    int32_t sample_rate;   /* always 24000 */
} Qwen3TtsAudio;

/* Fill params with defaults */
void qwen3_tts_default_params(Qwen3TtsParams* params);

/* Create TTS engine and load models from directory.
 * model_dir must contain one TTS gguf (0.6B/1.7B, Base/CustomVoice/VoiceDesign)
 * and one tokenizer gguf.
 * n_threads sets the default thread count used when synthesis params are NULL
 * or when params->n_threads <= 0.
 * Returns NULL on failure. */
Qwen3Tts* qwen3_tts_create(const char* model_dir, int32_t n_threads);

/* Check if models are loaded */
int qwen3_tts_is_loaded(const Qwen3Tts* tts);

/* Synthesize text to audio. Returns NULL on failure.
 * Caller must free with qwen3_tts_free_audio(). */
Qwen3TtsAudio* qwen3_tts_synthesize(
    Qwen3Tts* tts,
    const char* text,
    const Qwen3TtsParams* params);

/* Get sample rate (always 24000) */
int32_t qwen3_tts_sample_rate(const Qwen3Tts* tts);

/* Free generated audio */
void qwen3_tts_free_audio(Qwen3TtsAudio* audio);

/* Destroy TTS engine */
void qwen3_tts_destroy(Qwen3Tts* tts);

/* Synthesize with voice cloning from WAV file.
 * reference_audio_path: path to reference WAV (24kHz mono recommended).
 * Returns NULL on failure. Caller must free with qwen3_tts_free_audio(). */
Qwen3TtsAudio* qwen3_tts_synthesize_with_voice_file(
    Qwen3Tts* tts,
    const char* text,
    const char* reference_audio_path,
    const Qwen3TtsParams* params);

/* Synthesize with voice cloning from raw samples.
 * ref_samples: 24kHz mono float32 normalized to [-1, 1].
 * Returns NULL on failure. Caller must free with qwen3_tts_free_audio(). */
Qwen3TtsAudio* qwen3_tts_synthesize_with_voice_samples(
    Qwen3Tts* tts,
    const char* text,
    const float* ref_samples,
    int32_t n_ref_samples,
    const Qwen3TtsParams* params);

/* Extract speaker embedding from WAV file (for caching).
 * embedding_out: caller-allocated buffer for the embedding.
 * max_size: size of embedding_out in floats.
 * Returns the actual embedding size (typically 1024), or -1 on failure. */
int32_t qwen3_tts_extract_embedding_file(
    Qwen3Tts* tts,
    const char* reference_audio_path,
    float* embedding_out,
    int32_t max_size);

/* Synthesize with pre-computed speaker embedding (skips encoder).
 * embedding: speaker embedding from qwen3_tts_extract_embedding_file().
 * embedding_size: must match the size returned by extract.
 * Returns NULL on failure. Caller must free with qwen3_tts_free_audio(). */
Qwen3TtsAudio* qwen3_tts_synthesize_with_embedding(
    Qwen3Tts* tts,
    const char* text,
    const float* embedding,
    int32_t embedding_size,
    const Qwen3TtsParams* params);

/* Get last error message (or empty string) */
const char* qwen3_tts_get_error(const Qwen3Tts* tts);

#ifdef __cplusplus
}
#endif

#endif /* QWEN3TTS_C_API_H */
