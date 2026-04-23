/* qwen3tts_c_api.cpp - C API wrapper for Nim FFI.
 *
 * Wraps qwen3_tts::Qwen3TTS C++ class in a C-linkage API.
 * Synthesis calls use @autoreleasepool on macOS to drain Metal
 * Objective-C objects when called from background threads. */

#include "qwen3tts_c_api.h"

#include "qwen3_tts.h"

#ifdef __APPLE__
#include <objc/objc.h>
#include <objc/message.h>
// Minimal autorelease pool without importing Foundation
static void * new_autorelease_pool() {
    id pool = ((id(*)(id, SEL))objc_msgSend)(
        (id)objc_getClass("NSAutoreleasePool"),
        sel_registerName("new"));
    return (void *)pool;
}
static void drain_autorelease_pool(void * pool) {
    ((void(*)(id, SEL))objc_msgSend)((id)pool, sel_registerName("drain"));
}
#define AUTORELEASE_BEGIN void * _pool = new_autorelease_pool();
#define AUTORELEASE_END   drain_autorelease_pool(_pool);
#else
#define AUTORELEASE_BEGIN
#define AUTORELEASE_END
#endif

#include <cstring>
#include <cstdlib>

// Opaque handle - backs the C typedef
struct Qwen3Tts {
    qwen3_tts::Qwen3TTS engine;
    std::string last_error;
    int32_t default_n_threads = 4;
};

// Helper: convert C params to C++ params
static qwen3_tts::tts_params to_cpp_params(const Qwen3TtsParams * p, int32_t default_n_threads) {
    qwen3_tts::tts_params params;
    const int32_t sane_default_threads = default_n_threads > 0 ? default_n_threads : 4;
    params.n_threads = sane_default_threads;
    if (p) {
        params.max_audio_tokens  = p->max_audio_tokens;
        params.temperature       = p->temperature;
        params.top_p             = p->top_p;
        params.top_k             = p->top_k;
        if (p->n_threads > 0) {
            params.n_threads = p->n_threads;
        }
        params.repetition_penalty = p->repetition_penalty;
        params.seed              = p->seed;
        params.language_id       = p->language_id;
        params.task_type = static_cast<qwen3_tts::tts_task_type>(p->task_type);
        params.model_variant = static_cast<qwen3_tts::tts_model_variant>(p->model_variant);
        params.instruct = p->instruct ? p->instruct : "";
        params.speaker = p->speaker ? p->speaker : "";
        params.reference_text = p->reference_text ? p->reference_text : "";
        params.x_vector_only_mode = p->x_vector_only_mode != 0;
        params.subtalker_dosample = p->subtalker_dosample != 0;
        params.subtalker_top_k = p->subtalker_top_k;
        params.subtalker_top_p = p->subtalker_top_p;
        params.subtalker_temperature = p->subtalker_temperature;
        params.min_new_tokens = p->min_new_tokens;
        params.non_streaming_mode = p->non_streaming_mode != 0;
        params.talker_attention_window = p->talker_attention_window;
        // ICL (voice clone with ref_codes)
        if (p->ref_codes && p->ref_code_frames > 0) {
            params.ref_code_frames = p->ref_code_frames;
            params.ref_codes.assign(p->ref_codes,
                p->ref_codes + (size_t)p->ref_code_frames * 16 /* n_codebooks */);
        }
        if (p->ref_text_tokens && p->ref_text_token_count > 0) {
            params.ref_text_tokens.assign(p->ref_text_tokens,
                p->ref_text_tokens + p->ref_text_token_count);
        }
    }
    return params;
}

// Helper: convert C++ result to heap-allocated C audio struct
static Qwen3TtsAudio * to_c_audio(const qwen3_tts::tts_result & result) {
    if (!result.success || result.audio.empty()) {
        return nullptr;
    }
    auto * out = new Qwen3TtsAudio;
    auto * buf = new float[result.audio.size()];
    std::memcpy(buf, result.audio.data(), result.audio.size() * sizeof(float));
    out->samples     = buf;
    out->n_samples   = (int32_t)result.audio.size();
    out->sample_rate = result.sample_rate;
    return out;
}

// ============================================================
// C API implementation
// ============================================================

extern "C" {

void qwen3_tts_default_params(Qwen3TtsParams * params) {
    if (!params) return;
    params->max_audio_tokens  = 4096;
    params->temperature       = 0.9f;
    params->top_p             = 1.0f;
    params->top_k             = 50;
    params->n_threads         = 4;
    params->repetition_penalty = 1.05f;
    params->seed              = -1;
    params->language_id       = -1;
    params->task_type         = 0;
    params->model_variant     = 0;
    params->instruct          = nullptr;
    params->speaker           = nullptr;
    params->reference_text    = nullptr;
    params->x_vector_only_mode = 0;
    params->subtalker_dosample = 1;
    params->subtalker_top_k   = 50;
    params->subtalker_top_p   = 1.0f;
    params->subtalker_temperature = 0.9f;
    params->min_new_tokens    = 2;
    params->non_streaming_mode = 0;
    params->talker_attention_window = 0;
    params->ref_codes          = nullptr;
    params->ref_code_frames    = 0;
    params->ref_text_tokens    = nullptr;
    params->ref_text_token_count = 0;
}

Qwen3Tts * qwen3_tts_create(const char * model_dir, int32_t n_threads) {
    if (!model_dir) return nullptr;
    auto * tts = new Qwen3Tts;
    tts->default_n_threads = n_threads > 0 ? n_threads : 4;
    if (!tts->engine.load_models(model_dir)) {
        tts->last_error = tts->engine.get_error();
        delete tts;
        return nullptr;
    }
    return tts;
}

int qwen3_tts_is_loaded(const Qwen3Tts * tts) {
    return (tts && tts->engine.is_loaded()) ? 1 : 0;
}

Qwen3TtsAudio * qwen3_tts_synthesize(
        Qwen3Tts * tts, const char * text,
        const Qwen3TtsParams * params) {
    if (!tts || !text) return nullptr;
    AUTORELEASE_BEGIN
    auto cpp_params = to_cpp_params(params, tts->default_n_threads);
    auto result = tts->engine.synthesize(text, cpp_params);
    if (!result.success) {
        tts->last_error = result.error_msg;
    }
    auto * out = to_c_audio(result);
    AUTORELEASE_END
    return out;
}

int32_t qwen3_tts_sample_rate(const Qwen3Tts * tts) {
    (void)tts;
    return 24000;
}

void qwen3_tts_free_audio(Qwen3TtsAudio * audio) {
    if (!audio) return;
    delete[] audio->samples;
    delete audio;
}

void qwen3_tts_destroy(Qwen3Tts * tts) {
    delete tts;
}

Qwen3TtsAudio * qwen3_tts_synthesize_with_voice_file(
        Qwen3Tts * tts, const char * text,
        const char * reference_audio_path,
        const Qwen3TtsParams * params) {
    if (!tts || !text || !reference_audio_path) return nullptr;
    AUTORELEASE_BEGIN
    auto cpp_params = to_cpp_params(params, tts->default_n_threads);
    auto result = tts->engine.synthesize_with_voice(text, reference_audio_path, cpp_params);
    if (!result.success) {
        tts->last_error = result.error_msg;
    }
    auto * out = to_c_audio(result);
    AUTORELEASE_END
    return out;
}

Qwen3TtsAudio * qwen3_tts_synthesize_with_voice_samples(
        Qwen3Tts * tts, const char * text,
        const float * ref_samples, int32_t n_ref_samples,
        const Qwen3TtsParams * params) {
    if (!tts || !text || !ref_samples || n_ref_samples <= 0) return nullptr;
    AUTORELEASE_BEGIN
    auto cpp_params = to_cpp_params(params, tts->default_n_threads);
    auto result = tts->engine.synthesize_with_voice(text, ref_samples, n_ref_samples, cpp_params);
    if (!result.success) {
        tts->last_error = result.error_msg;
    }
    auto * out = to_c_audio(result);
    AUTORELEASE_END
    return out;
}

int32_t qwen3_tts_extract_embedding_file(
        Qwen3Tts * tts, const char * reference_audio_path,
        float * embedding_out, int32_t max_size) {
    if (!tts || !reference_audio_path || !embedding_out || max_size <= 0) return -1;

    // Load WAV and resample to 24kHz
    std::vector<float> ref_samples;
    int ref_sample_rate;
    if (!qwen3_tts::load_audio_file(reference_audio_path, ref_samples, ref_sample_rate)) {
        tts->last_error = "Failed to load reference audio: " + std::string(reference_audio_path);
        return -1;
    }

    // Resample if needed (same logic as synthesize_with_voice)
    if (ref_sample_rate != 24000) {
        // Simple linear resampling
        double ratio = (double)ref_sample_rate / 24000;
        int output_len = (int)((double)ref_samples.size() / ratio);
        std::vector<float> resampled(output_len);
        for (int i = 0; i < output_len; ++i) {
            double src_idx = i * ratio;
            int idx0 = (int)src_idx;
            int idx1 = idx0 + 1;
            double frac = src_idx - idx0;
            if (idx1 >= (int)ref_samples.size()) {
                resampled[i] = ref_samples.back();
            } else {
                resampled[i] = (float)((1.0 - frac) * ref_samples[idx0] + frac * ref_samples[idx1]);
            }
        }
        ref_samples = std::move(resampled);
    }

    AUTORELEASE_BEGIN
    std::vector<float> embedding;
    if (!tts->engine.extract_speaker_embedding(ref_samples.data(), (int32_t)ref_samples.size(), embedding)) {
        tts->last_error = tts->engine.get_error();
        AUTORELEASE_END
        return -1;
    }
    AUTORELEASE_END

    int32_t emb_size = (int32_t)embedding.size();
    if (emb_size > max_size) emb_size = max_size;
    std::memcpy(embedding_out, embedding.data(), emb_size * sizeof(float));
    return emb_size;
}

Qwen3TtsAudio * qwen3_tts_synthesize_with_embedding(
        Qwen3Tts * tts, const char * text,
        const float * embedding, int32_t embedding_size,
        const Qwen3TtsParams * params) {
    if (!tts || !text || !embedding || embedding_size <= 0) return nullptr;
    AUTORELEASE_BEGIN
    auto cpp_params = to_cpp_params(params, tts->default_n_threads);
    auto result = tts->engine.synthesize_with_embedding(text, embedding, embedding_size, cpp_params);
    if (!result.success) {
        tts->last_error = result.error_msg;
    }
    auto * out = to_c_audio(result);
    AUTORELEASE_END
    return out;
}

Qwen3TtsCodes * qwen3_tts_encode_audio_file(
        Qwen3Tts * tts, const char * reference_audio_path) {
    if (!tts || !reference_audio_path) return nullptr;
    AUTORELEASE_BEGIN
    std::vector<int32_t> codes;
    int32_t n_frames = 0;
    int32_t n_codebooks = 0;
    if (!tts->engine.encode_audio(reference_audio_path, codes, n_frames, n_codebooks)) {
        tts->last_error = tts->engine.get_error();
        AUTORELEASE_END
        return nullptr;
    }
    auto * out = new Qwen3TtsCodes;
    auto * buf = new int32_t[codes.size()];
    std::memcpy(buf, codes.data(), codes.size() * sizeof(int32_t));
    out->codes = buf;
    out->n_frames = n_frames;
    out->n_codebooks = n_codebooks;
    AUTORELEASE_END
    return out;
}

Qwen3TtsCodes * qwen3_tts_encode_audio_samples(
        Qwen3Tts * tts, const float * ref_samples, int32_t n_ref_samples) {
    if (!tts || !ref_samples || n_ref_samples <= 0) return nullptr;
    AUTORELEASE_BEGIN
    std::vector<int32_t> codes;
    int32_t n_frames = 0;
    int32_t n_codebooks = 0;
    if (!tts->engine.encode_audio(ref_samples, n_ref_samples, codes, n_frames, n_codebooks)) {
        tts->last_error = tts->engine.get_error();
        AUTORELEASE_END
        return nullptr;
    }
    auto * out = new Qwen3TtsCodes;
    auto * buf = new int32_t[codes.size()];
    std::memcpy(buf, codes.data(), codes.size() * sizeof(int32_t));
    out->codes = buf;
    out->n_frames = n_frames;
    out->n_codebooks = n_codebooks;
    AUTORELEASE_END
    return out;
}

void qwen3_tts_free_codes(Qwen3TtsCodes * codes) {
    if (!codes) return;
    delete[] codes->codes;
    delete codes;
}

const char * qwen3_tts_get_error(const Qwen3Tts * tts) {
    if (!tts) return "";
    return tts->last_error.c_str();
}

} // extern "C"
