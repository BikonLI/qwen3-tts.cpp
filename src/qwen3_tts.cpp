#include "audio/audio_tokenizer_decoder.h"
#include "audio/audio_tokenizer_encoder.h"
#include "gguf_loader.h"
#include "qwen3_tts.h"
#include "core/text_tokenizer.h"
#include "core/tts_transformer.h"

#include <SDL3/SDL.h>

#include <chrono>
#include <cstdarg>
#include <cmath>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <deque>
#include <limits>
#include <thread>

#ifdef __APPLE__
#include <mach/mach.h>
#elif defined(__linux__)
#include <sys/resource.h>
#else
#endif

namespace qwen3_tts {

    namespace fs = std::filesystem;

    static bool stream_debug_enabled() {
        const char * env = std::getenv("QWEN3_TTS_STREAM_DEBUG");
        if (!env || env[0] == '\0') {
            return false;
        }
        std::string v = env;
        std::transform(v.begin(), v.end(), v.begin(), [](unsigned char c) { return (char)std::tolower(c); });
        return !(v == "0" || v == "false" || v == "off" || v == "no");
    }

    struct tts_stream_session::impl {
        ~impl() {
            if (worker.joinable()) {
                if (worker.get_id() == std::this_thread::get_id()) {
                    worker.detach();
                } else {
                    worker.join();
                }
            }
        }

        mutable std::mutex mutex;
        std::condition_variable cv;
        std::deque<tts_audio_chunk> queue;
        int32_t queue_capacity = 8;
        bool drop_oldest_on_overflow = false;
        int32_t default_poll_timeout_ms = 100;
        int32_t dropped_chunks = 0;
        bool finished = false;
        bool has_error = false;
        std::string error_msg;
        int64_t next_chunk_id = 0;
        std::thread worker;
    };

    tts_stream_session::tts_stream_session(const std::shared_ptr<impl> &impl_ptr)
        : impl_(impl_ptr) {
    }

    tts_stream_session::~tts_stream_session() = default;

    tts_stream_poll_status tts_stream_session::poll(tts_audio_chunk &chunk, int32_t timeout_ms) {
        if (!impl_) {
            return tts_stream_poll_status::error;
        }

        std::unique_lock<std::mutex> lock(impl_->mutex);
        const int32_t wait_ms = timeout_ms >= 0 ? timeout_ms : impl_->default_poll_timeout_ms;

        if (impl_->queue.empty() && !impl_->finished && !impl_->has_error) {
            if (wait_ms > 0) {
                impl_->cv.wait_for(lock, std::chrono::milliseconds(wait_ms), [&]() {
                    return !impl_->queue.empty() || impl_->finished || impl_->has_error;
                });
            }
        }

        if (!impl_->queue.empty()) {
            chunk = std::move(impl_->queue.front());
            impl_->queue.pop_front();
            impl_->cv.notify_all();
            return tts_stream_poll_status::chunk;
        }

        if (impl_->has_error) {
            return tts_stream_poll_status::error;
        }

        if (impl_->finished) {
            return tts_stream_poll_status::finished;
        }

        return tts_stream_poll_status::timeout;
    }

    bool tts_stream_session::is_finished() const {
        if (!impl_) {
            return true;
        }
        std::lock_guard<std::mutex> lock(impl_->mutex);
        return impl_->finished && impl_->queue.empty();
    }

    bool tts_stream_session::has_error() const {
        if (!impl_) {
            return true;
        }
        std::lock_guard<std::mutex> lock(impl_->mutex);
        return impl_->has_error;
    }

    std::string tts_stream_session::get_error() const {
        if (!impl_) {
            return {};
        }
        std::lock_guard<std::mutex> lock(impl_->mutex);
        return impl_->error_msg;
    }

    int32_t tts_stream_session::dropped_chunks() const {
        if (!impl_) {
            return 0;
        }
        std::lock_guard<std::mutex> lock(impl_->mutex);
        return impl_->dropped_chunks;
    }

    static void stream_set_error(const std::shared_ptr<tts_stream_session::impl> &impl,
                                 const std::string &error_msg) {
        if (!impl) {
            return;
        }
        {
            std::lock_guard<std::mutex> lock(impl->mutex);
            if (!impl->has_error) {
                impl->error_msg = error_msg;
            }
            impl->has_error = true;
            impl->finished = true;
        }
        impl->cv.notify_all();
    }

    static void stream_mark_finished(const std::shared_ptr<tts_stream_session::impl> &impl) {
        if (!impl) {
            return;
        }
        {
            std::lock_guard<std::mutex> lock(impl->mutex);
            impl->finished = true;
        }
        impl->cv.notify_all();
    }

    static bool stream_emit_chunk(const std::shared_ptr<tts_stream_session::impl> &impl,
                                  tts_audio_chunk chunk,
                                  const tts_stream_callbacks &callbacks) {
        if (!impl) {
            return false;
        }

        {
            std::unique_lock<std::mutex> lock(impl->mutex);
            const int32_t capacity = std::max(1, impl->queue_capacity);
            while ((int32_t)impl->queue.size() >= capacity) {
                if (impl->drop_oldest_on_overflow) {
                    impl->queue.pop_front();
                    impl->dropped_chunks++;
                    continue;
                }
                impl->cv.wait(lock, [&]() {
                    return (int32_t)impl->queue.size() < capacity || impl->finished || impl->has_error;
                });
                if (impl->finished || impl->has_error) {
                    return false;
                }
            }
            chunk.chunk_id = impl->next_chunk_id++;
            chunk.dropped_before = impl->dropped_chunks;
            impl->queue.push_back(chunk);
        }
        impl->cv.notify_all();

        if (callbacks.on_audio_chunk) {
            try {
                callbacks.on_audio_chunk(chunk);
            } catch (...) {
                stream_set_error(impl, "Streaming audio callback threw an exception");
                if (callbacks.on_error) {
                    try {
                        callbacks.on_error("Streaming audio callback threw an exception");
                    } catch (...) {
                    }
                }
                return false;
            }
        }

        return true;
    }

    static std::string to_lower_copy(const std::string &s) {
        std::string out = s;
        std::transform(out.begin(), out.end(), out.begin(), [](unsigned char c) { return (char)std::tolower(c); });
        return out;
    }

    static tts_model_variant infer_model_variant_from_path(const std::string &path) {
        const std::string lower = to_lower_copy(path);
        if (lower.find("voicedesign") != std::string::npos) {
            return tts_model_variant::voice_design;
        }
        if (lower.find("customvoice") != std::string::npos) {
            return tts_model_variant::custom_voice;
        }
        if (lower.find("base") != std::string::npos) {
            return tts_model_variant::base;
        }
        return tts_model_variant::auto_variant;
    }

    static tts_model_variant infer_model_variant_from_gguf(struct gguf_context *ctx) {
        if (!ctx) {
            return tts_model_variant::auto_variant;
        }
        const int64_t name_key = gguf_find_key(ctx, "general.name");
        if (name_key >= 0) {
            const char *name = gguf_get_val_str(ctx, name_key);
            if (name && name[0] != '\0') {
                return infer_model_variant_from_path(name);
            }
        }
        return tts_model_variant::auto_variant;
    }

    static const char *variant_name(tts_model_variant v) {
        switch (v) {
            case tts_model_variant::base: return "base";
            case tts_model_variant::custom_voice: return "custom_voice";
            case tts_model_variant::voice_design: return "voice_design";
            default: return "auto";
        }
    }

    static const char *task_name(tts_task_type t) {
        switch (t) {
            case tts_task_type::voice_clone: return "voice_clone";
            case tts_task_type::custom_voice: return "custom_voice";
            case tts_task_type::voice_design: return "voice_design";
            default: return "auto";
        }
    }

    static const char *model_id_name(tts_model_id id) {
        switch (id) {
            case tts_model_id::model_06b_base: return "06b_base";
            case tts_model_id::model_06b_custom_voice: return "06b_custom_voice";
            case tts_model_id::model_1_7b_base: return "1.7b_base";
            case tts_model_id::model_1_7b_custom_voice: return "1.7b_custom_voice";
            case tts_model_id::model_1_7b_voice_design: return "1.7b_voice_design";
            default: return "unknown";
        }
    }

    static bool is_supported_custom_voice_speaker(const std::string &speaker_lower);
    static bool map_custom_voice_speaker_codec_id(const std::string &speaker_lower, int32_t &codec_id_out);
    static int32_t map_dialect_language_if_needed(const std::string &speaker_lower, int32_t language_id);

    static std::string pick_first_sorted(std::vector<std::string> &v) {
        if (v.empty()) {
            return {};
        }
        std::sort(v.begin(), v.end());
        return v.front();
    }

    static bool discover_tts_model_path(const std::string &model_dir,
                                        std::string &tts_model_path,
                                        std::string &error_msg) {
        tts_model_path.clear();

        std::vector<std::string> q8_candidates;
        std::vector<std::string> f16_candidates;
        std::vector<std::string> fallback_tts;

        std::error_code ec;
        if (!fs::exists(model_dir, ec) || !fs::is_directory(model_dir, ec)) {
            error_msg = "Model directory does not exist: " + model_dir;
            return false;
        }

        for (const auto &entry : fs::directory_iterator(model_dir, ec)) {
            if (!entry.is_regular_file()) {
                continue;
            }
            if (entry.path().extension().string() != ".gguf") {
                continue;
            }
            const std::string path = entry.path().string();
            const std::string name = to_lower_copy(entry.path().filename().string());
            if (name.find("tokenizer") != std::string::npos) {
                continue;
            }

            if (name.find("q8_0") != std::string::npos) {
                q8_candidates.push_back(path);
            } else if (name.find("f16") != std::string::npos) {
                f16_candidates.push_back(path);
            } else {
                fallback_tts.push_back(path);
            }
        }

        if (!q8_candidates.empty()) {
            tts_model_path = pick_first_sorted(q8_candidates);
        } else if (!f16_candidates.empty()) {
            tts_model_path = pick_first_sorted(f16_candidates);
        } else {
            tts_model_path = pick_first_sorted(fallback_tts);
        }

        if (tts_model_path.empty()) {
            error_msg = "Failed to discover TTS GGUF file in model dir: " + model_dir;
            return false;
        }

        return true;
    }

    static bool discover_tokenizer_model_path(const std::string &tokenizer_path_or_dir,
                                              std::string &tokenizer_model_path,
                                              std::string &error_msg) {
        tokenizer_model_path.clear();

        std::error_code ec;
        if (fs::exists(tokenizer_path_or_dir, ec) && fs::is_regular_file(tokenizer_path_or_dir, ec)) {
            if (fs::path(tokenizer_path_or_dir).extension().string() != ".gguf") {
                error_msg = "Tokenizer path is not a GGUF file: " + tokenizer_path_or_dir;
                return false;
            }
            tokenizer_model_path = tokenizer_path_or_dir;
            return true;
        }

        if (!fs::exists(tokenizer_path_or_dir, ec) || !fs::is_directory(tokenizer_path_or_dir, ec)) {
            error_msg = "Tokenizer path does not exist: " + tokenizer_path_or_dir;
            return false;
        }

        std::vector<std::string> tokenizer_candidates;
        std::vector<std::string> fallback_gguf;
        for (const auto &entry : fs::directory_iterator(tokenizer_path_or_dir, ec)) {
            if (!entry.is_regular_file()) {
                continue;
            }
            if (entry.path().extension().string() != ".gguf") {
                continue;
            }

            const std::string path = entry.path().string();
            const std::string name = to_lower_copy(entry.path().filename().string());
            if (name.find("tokenizer") != std::string::npos) {
                tokenizer_candidates.push_back(path);
            } else {
                fallback_gguf.push_back(path);
            }
        }

        if (!tokenizer_candidates.empty()) {
            tokenizer_model_path = pick_first_sorted(tokenizer_candidates);
        } else {
            tokenizer_model_path = pick_first_sorted(fallback_gguf);
        }

        if (tokenizer_model_path.empty()) {
            error_msg = "Failed to discover tokenizer GGUF in: " + tokenizer_path_or_dir;
            return false;
        }

        return true;
    }

    static bool discover_model_paths(const std::string &model_dir,
                                     std::string &tts_model_path,
                                     std::string &tokenizer_model_path,
                                     std::string &error_msg) {
        if (!discover_tts_model_path(model_dir, tts_model_path, error_msg)) {
            return false;
        }
        if (!discover_tokenizer_model_path(model_dir, tokenizer_model_path, error_msg)) {
            return false;
        }
        return true;
    }

    static int64_t get_time_ms() {
        return std::chrono::duration_cast<std::chrono::milliseconds>(
                   std::chrono::steady_clock::now().time_since_epoch()
        )
            .count();
    }

    struct process_memory_snapshot {
        uint64_t rss_bytes = 0;
        uint64_t phys_footprint_bytes = 0;
    };

    static bool get_process_memory_snapshot(process_memory_snapshot &out) {
#ifdef __APPLE__
        mach_task_basic_info_data_t basic_info = {};
        mach_msg_type_number_t basic_count = MACH_TASK_BASIC_INFO_COUNT;
        if (task_info(
                mach_task_self(), MACH_TASK_BASIC_INFO, reinterpret_cast<task_info_t>(&basic_info), &basic_count
            ) != KERN_SUCCESS) {
            return false;
        }
        out.rss_bytes = (uint64_t)basic_info.resident_size;

        task_vm_info_data_t vm_info = {};
        mach_msg_type_number_t vm_count = TASK_VM_INFO_COUNT;
        if (task_info(mach_task_self(), TASK_VM_INFO, reinterpret_cast<task_info_t>(&vm_info), &vm_count) ==
            KERN_SUCCESS) {
            out.phys_footprint_bytes = (uint64_t)vm_info.phys_footprint;
        } else {
            out.phys_footprint_bytes = out.rss_bytes;
        }
        return true;
#elif defined(__linux__)
        struct rusage usage = {};
        if (getrusage(RUSAGE_SELF, &usage) != 0) {
            return false;
        }
        out.rss_bytes = (uint64_t)usage.ru_maxrss * 1024ULL;
        out.phys_footprint_bytes = out.rss_bytes;
        return true;
#else
        return false;
#endif
    }

    static std::string format_bytes(uint64_t bytes) {
        static const char *units[] = {"B", "KB", "MB", "GB", "TB"};
        double val = (double)bytes;
        int unit = 0;
        while (val >= 1024.0 && unit < 4) {
            val /= 1024.0;
            ++unit;
        }
        char buf[64];
        snprintf(buf, sizeof(buf), "%.2f %s", val, units[unit]);
        return std::string(buf);
    }

    static void log_memory_usage(const char *label) {
        process_memory_snapshot mem;
        if (!get_process_memory_snapshot(mem)) {
            fprintf(stderr, "  [mem] %-24s unavailable\n", label);
            return;
        }
        fprintf(
            stderr, "  [mem] %-24s rss=%s  phys=%s\n", label, format_bytes(mem.rss_bytes).c_str(),
            format_bytes(mem.phys_footprint_bytes).c_str()
        );
    }

    static void resample_linear(
        const float *input, int input_len, int input_rate, std::vector<float> &output, int output_rate
    ) {
        double ratio = (double)input_rate / output_rate;
        int output_len = (int)((double)input_len / ratio);
        output.resize(output_len);

        for (int i = 0; i < output_len; ++i) {
            double src_idx = i * ratio;
            int idx0 = (int)src_idx;
            int idx1 = idx0 + 1;
            double frac = src_idx - idx0;

            if (idx1 >= input_len) {
                output[i] = input[input_len - 1];
            } else {
                output[i] = (float)((1.0 - frac) * input[idx0] + frac * input[idx1]);
            }
        }
    }

    Qwen3TTS::Qwen3TTS()
        : tokenizer_(std::make_unique<TextTokenizer>()),
          transformer_(std::make_unique<TTSTransformer>()),
          audio_encoder_(std::make_unique<AudioTokenizerEncoder>()),
          audio_decoder_(std::make_unique<AudioTokenizerDecoder>()) {
    }

    Qwen3TTS::~Qwen3TTS() {
        std::shared_ptr<tts_stream_session::impl> stream_impl;
        {
            std::lock_guard<std::mutex> lock(stream_mutex_);
            stream_impl = stream_impl_;
            stream_impl_.reset();
        }
        if (stream_impl && stream_impl->worker.joinable()) {
            if (stream_impl->worker.get_id() == std::this_thread::get_id()) {
                stream_impl->worker.detach();
            } else {
                stream_impl->worker.join();
            }
        }
    }

    bool Qwen3TTS::load_models(const std::string &model_dir) {
        std::string tts_model_path;
        std::string tokenizer_model_path;
        if (!discover_model_paths(model_dir, tts_model_path, tokenizer_model_path, error_msg_)) {
            return false;
        }
        return load_models_from_files(tts_model_path, tokenizer_model_path);
    }

    bool Qwen3TTS::load_models_from_files(const std::string &tts_model_path,
                                          const std::string &tokenizer_model_path) {
        int64_t t_start = get_time_ms();
        log_memory_usage("load/start");

        models_loaded_ = false;
        transformer_->unload_model();
        audio_decoder_->unload_model();
        transformer_loaded_ = false;
        decoder_loaded_ = false;
        loaded_hidden_size_ = 0;

        if (tts_model_path.empty() || tokenizer_model_path.empty()) {
            error_msg_ = "Model path is empty";
            return false;
        }
        if (!fs::exists(tts_model_path) || !fs::is_regular_file(tts_model_path)) {
            error_msg_ = "TTS model file does not exist: " + tts_model_path;
            return false;
        }
        if (!fs::exists(tokenizer_model_path) || !fs::is_regular_file(tokenizer_model_path)) {
            error_msg_ = "Tokenizer model file does not exist: " + tokenizer_model_path;
            return false;
        }

        tts_model_path_ = tts_model_path;
        decoder_model_path_ = tokenizer_model_path;
        loaded_model_variant_ = infer_model_variant_from_path(tts_model_path);
        encoder_loaded_ = false;
        transformer_loaded_ = false;
        decoder_loaded_ = false;

        const char *low_mem_env = std::getenv("QWEN3_TTS_LOW_MEM");
        low_mem_mode_ = low_mem_env && low_mem_env[0] != '\0' && low_mem_env[0] != '0';
        if (low_mem_mode_) {
            fprintf(stderr, "  Low-memory mode enabled (lazy decoder + component unloads)\n");
        }

        // Load TTS model (contains text tokenizer + transformer for generation)
        fprintf(stderr, "Loading TTS model from %s...\n", tts_model_path.c_str());

        // Load text tokenizer from TTS model
        int64_t t_tokenizer_start = get_time_ms();
        {
            GGUFLoader loader;
            if (!loader.open(tts_model_path)) {
                error_msg_ = "Failed to open TTS model: " + loader.get_error();
                return false;
            }

            // Prefer GGUF metadata for variant detection; path-based inference is fallback only.
            tts_model_variant meta_variant = infer_model_variant_from_gguf(loader.get_ctx());
            if (meta_variant != tts_model_variant::auto_variant) {
                loaded_model_variant_ = meta_variant;
            }

            if (!tokenizer_->load_from_gguf(loader.get_ctx())) {
                error_msg_ = "Failed to load text tokenizer: " + tokenizer_->get_error();
                return false;
            }
            fprintf(
                stderr, "  Text tokenizer loaded: vocab_size=%d (%lld ms)\n", tokenizer_->get_config().vocab_size,
                (long long)(get_time_ms() - t_tokenizer_start)
            );
        }
        log_memory_usage("load/after-tokenizer");

        // Speaker encoder is loaded lazily on first voice cloning request.
        fprintf(stderr, "  Speaker encoder: deferred (lazy load)\n");

        // Load TTS transformer from TTS model
        int64_t t_transformer_start = get_time_ms();
        if (!transformer_->load_model(tts_model_path)) {
            error_msg_ = "Failed to load TTS transformer: " + transformer_->get_error();
            return false;
        }
        transformer_loaded_ = true;
        loaded_hidden_size_ = transformer_->get_config().hidden_size;
        fprintf(
            stderr, "  TTS transformer loaded: hidden_size=%d, n_layers=%d (%lld ms)\n",
            transformer_->get_config().hidden_size, transformer_->get_config().n_layers,
            (long long)(get_time_ms() - t_transformer_start)
        );
        fprintf(stderr, "  Inferred model variant: %s\n", variant_name(loaded_model_variant_));
        log_memory_usage("load/after-transformer");

        if (!low_mem_mode_) {
            // Load vocoder (audio decoder) from tokenizer model
            fprintf(stderr, "Loading vocoder from %s...\n", tokenizer_model_path.c_str());
            int64_t t_decoder_start = get_time_ms();
            if (!audio_decoder_->load_model(tokenizer_model_path)) {
                error_msg_ = "Failed to load vocoder: " + audio_decoder_->get_error();
                return false;
            }
            decoder_loaded_ = true;
            fprintf(
                stderr, "  Vocoder loaded: sample_rate=%d, n_codebooks=%d (%lld ms)\n",
                audio_decoder_->get_config().sample_rate, audio_decoder_->get_config().n_codebooks,
                (long long)(get_time_ms() - t_decoder_start)
            );
            log_memory_usage("load/after-vocoder");
        } else {
            fprintf(stderr, "  Vocoder: deferred (lazy load)\n");
        }

        models_loaded_ = true;

        int64_t t_end = get_time_ms();
        fprintf(stderr, "All models loaded in %lld ms\n", (long long)(t_end - t_start));
        log_memory_usage("load/end");

        return true;
    }

    tts_result Qwen3TTS::synthesize(const std::string &text, const tts_params &params) {
        tts_result result;

        if (!models_loaded_) {
            result.error_msg = "Models not loaded";
            return result;
        }

        if (params.task_type == tts_task_type::voice_clone && !params.x_vector_only_mode) {
            result.error_msg = "voice_clone task requires reference audio unless x_vector_only_mode=true";
            return result;
        }

        // For basic synthesis without voice cloning, we use a zero speaker embedding
        // This will use the model's default voice characteristics
        std::vector<float> zero_embedding(transformer_->get_config().hidden_size, 0.0f);

        return synthesize_internal(text, zero_embedding.data(), params, result);
    }

    tts_result Qwen3TTS::synthesize_with_voice(
        const std::string &text, const std::string &reference_audio, const tts_params &params
    ) {
        tts_result result;

        if (params.task_type == tts_task_type::voice_clone &&
            !params.x_vector_only_mode &&
            params.reference_text.empty()) {
            result.error_msg = "voice_clone task requires reference_text when x_vector_only_mode is false";
            return result;
        }

        std::vector<float> ref_samples;
        int ref_sample_rate;
        if (!load_audio_file(reference_audio, ref_samples, ref_sample_rate)) {
            result.error_msg = "Failed to load reference audio: " + reference_audio;
            return result;
        }

        const int target_rate = 24000;
        if (ref_sample_rate != target_rate) {
            fprintf(stderr, "Resampling audio from %d Hz to %d Hz...\n", ref_sample_rate, target_rate);
            std::vector<float> resampled;
            resample_linear(ref_samples.data(), (int)ref_samples.size(), ref_sample_rate, resampled, target_rate);
            ref_samples = std::move(resampled);
        }

        return synthesize_with_voice(text, ref_samples.data(), (int32_t)ref_samples.size(), params);
    }

    tts_result Qwen3TTS::synthesize_with_voice(
        const std::string &text, const float *ref_samples, int32_t n_ref_samples, const tts_params &params
    ) {
        tts_result result;

        if (!models_loaded_) {
            result.error_msg = "Models not loaded";
            return result;
        }

        if (!encoder_loaded_) {
            if (tts_model_path_.empty()) {
                result.error_msg = "Internal error: missing TTS model path for lazy encoder load";
                return result;
            }
            int64_t t_encoder_load_start = get_time_ms();
            if (!audio_encoder_->load_model(tts_model_path_)) {
                result.error_msg = "Failed to load speaker encoder: " + audio_encoder_->get_error();
                return result;
            }
            encoder_loaded_ = true;
            if (params.print_timing) {
                fprintf(
                    stderr, "  Speaker encoder lazy-loaded in %lld ms\n",
                    (long long)(get_time_ms() - t_encoder_load_start)
                );
                log_memory_usage("voice/after-encoder-load");
            }
        }

        if (params.n_threads > 0 && !audio_encoder_->set_n_threads(params.n_threads)) {
            result.error_msg = "Failed to set speaker encoder thread count: " + audio_encoder_->get_error();
            return result;
        }

        int64_t t_encode_start = get_time_ms();
        std::vector<float> speaker_embedding;

        if (!audio_encoder_->encode(ref_samples, n_ref_samples, speaker_embedding)) {
            result.error_msg = "Failed to extract speaker embedding: " + audio_encoder_->get_error();
            return result;
        }
        result.t_encode_ms = get_time_ms() - t_encode_start;

        if (params.print_progress) {
            fprintf(stderr, "Speaker embedding extracted: %zu floats\n", speaker_embedding.size());
        }

        return synthesize_internal(text, speaker_embedding.data(), params, result);
    }

    bool Qwen3TTS::extract_speaker_embedding(
        const float *ref_samples, int32_t n_ref_samples, std::vector<float> &embedding, const tts_params &params
    ) {
        if (!models_loaded_) {
            error_msg_ = "Models not loaded";
            return false;
        }

        if (!encoder_loaded_) {
            if (tts_model_path_.empty()) {
                error_msg_ = "Internal error: missing TTS model path for lazy encoder load";
                return false;
            }
            int64_t t_encoder_load_start = get_time_ms();
            if (!audio_encoder_->load_model(tts_model_path_)) {
                error_msg_ = "Failed to load speaker encoder: " + audio_encoder_->get_error();
                return false;
            }
            encoder_loaded_ = true;
            if (params.print_timing) {
                fprintf(
                    stderr, "  Speaker encoder lazy-loaded in %lld ms\n",
                    (long long)(get_time_ms() - t_encoder_load_start)
                );
            }
        }

        if (params.n_threads > 0 && !audio_encoder_->set_n_threads(params.n_threads)) {
            error_msg_ = "Failed to set speaker encoder thread count: " + audio_encoder_->get_error();
            return false;
        }

        if (!audio_encoder_->encode(ref_samples, n_ref_samples, embedding)) {
            error_msg_ = "Failed to extract speaker embedding: " + audio_encoder_->get_error();
            return false;
        }

        return true;
    }

    tts_result Qwen3TTS::synthesize_with_embedding(
        const std::string &text, const float *embedding, int32_t embedding_size, const tts_params &params
    ) {
        tts_result result;

        if (!models_loaded_) {
            result.error_msg = "Models not loaded";
            return result;
        }

        if (embedding == nullptr || embedding_size <= 0) {
            result.error_msg = "Invalid speaker embedding";
            return result;
        }

        return synthesize_internal(text, embedding, params, result);
    }

    tts_result Qwen3TTS::synthesize_internal(
        const std::string &text, const float *speaker_embedding, const tts_params &params, tts_result &result
    ) {
        {
            std::lock_guard<std::mutex> lock(stream_mutex_);
            if (stream_active_) {
                result.error_msg = "Streaming session is active; concurrent non-stream synthesis is not supported";
                return result;
            }
        }

        if (loaded_model_variant_ != tts_model_variant::auto_variant &&
            params.model_variant != tts_model_variant::auto_variant &&
            params.model_variant != loaded_model_variant_) {
            result.error_msg = std::string("Requested --model-variant '") +
                               variant_name(params.model_variant) +
                               "' does not match loaded model variant '" +
                               variant_name(loaded_model_variant_) +
                               "'.";
            return result;
        }

        tts_model_variant effective_variant =
            (params.model_variant != tts_model_variant::auto_variant) ? params.model_variant : loaded_model_variant_;
        if (effective_variant == tts_model_variant::auto_variant) {
            effective_variant = tts_model_variant::base;
        }

        if (params.task_type != tts_task_type::auto_task &&
            loaded_model_variant_ != tts_model_variant::auto_variant) {
            bool compatible = true;
            switch (params.task_type) {
                case tts_task_type::voice_clone:
                    compatible = (loaded_model_variant_ == tts_model_variant::base);
                    break;
                case tts_task_type::custom_voice:
                    compatible = (loaded_model_variant_ == tts_model_variant::custom_voice);
                    break;
                case tts_task_type::voice_design:
                    compatible = (loaded_model_variant_ == tts_model_variant::voice_design);
                    break;
                default:
                    compatible = true;
                    break;
            }
            if (!compatible) {
                result.error_msg = std::string("Task '") + task_name(params.task_type) +
                                   "' is incompatible with loaded model variant '" +
                                   variant_name(loaded_model_variant_) +
                                   "'. Please use a matching model family.";
                return result;
            }
        }

        if (params.task_type == tts_task_type::voice_design &&
            effective_variant != tts_model_variant::voice_design) {
            result.error_msg = std::string("voice_design task requires voice_design model variant, got ") +
                               variant_name(effective_variant);
            return result;
        }
        if (params.task_type == tts_task_type::custom_voice &&
            effective_variant == tts_model_variant::base) {
            result.error_msg = "custom_voice task requires custom_voice or voice_design variant";
            return result;
        }

        int32_t effective_language_id = params.language_id;
        int32_t speaker_codec_id = -1;
        const bool use_custom_voice_speaker =
            (params.task_type == tts_task_type::custom_voice ||
             effective_variant == tts_model_variant::custom_voice);
        if (use_custom_voice_speaker) {
            const std::string speaker_lower = to_lower_copy(params.speaker);
            if (speaker_lower.empty()) {
                result.error_msg = "speaker is required for custom_voice";
                return result;
            }
            if (!is_supported_custom_voice_speaker(speaker_lower)) {
                result.error_msg = "Unsupported speaker: " + params.speaker;
                return result;
            }
            if (!map_custom_voice_speaker_codec_id(speaker_lower, speaker_codec_id)) {
                result.error_msg = "Unsupported speaker: " + params.speaker;
                return result;
            }
            effective_language_id = map_dialect_language_if_needed(speaker_lower, effective_language_id);
        }

        int64_t t_total_start = get_time_ms();
        auto sample_memory = [&](const char *stage) {
            process_memory_snapshot mem;
            if (!get_process_memory_snapshot(mem)) {
                return;
            }
            if (result.mem_rss_start_bytes == 0) {
                result.mem_rss_start_bytes = mem.rss_bytes;
                result.mem_phys_start_bytes = mem.phys_footprint_bytes;
            }
            result.mem_rss_end_bytes = mem.rss_bytes;
            result.mem_phys_end_bytes = mem.phys_footprint_bytes;
            if (mem.rss_bytes > result.mem_rss_peak_bytes) {
                result.mem_rss_peak_bytes = mem.rss_bytes;
            }
            if (mem.phys_footprint_bytes > result.mem_phys_peak_bytes) {
                result.mem_phys_peak_bytes = mem.phys_footprint_bytes;
            }
            if (params.print_timing) {
                fprintf(
                    stderr, "  [mem] %-24s rss=%s  phys=%s\n", stage, format_bytes(mem.rss_bytes).c_str(),
                    format_bytes(mem.phys_footprint_bytes).c_str()
                );
            }
        };
        sample_memory("synth/start");

        // Step 2: Tokenize input text
        int64_t t_tokenize_start = get_time_ms();
        std::vector<int32_t> text_tokens = tokenizer_->encode_for_tts(
            text,
            "",
            params.speaker,
            params.reference_text
        );
        std::vector<int32_t> instruct_tokens = tokenizer_->encode_instruct_for_tts(params.instruct);
        result.t_tokenize_ms = get_time_ms() - t_tokenize_start;
        sample_memory("synth/after-tokenize");

        if (text_tokens.empty()) {
            result.error_msg = "Failed to tokenize text";
            return result;
        }

        if (params.print_progress) {
            fprintf(stderr, "Text tokenized: %zu tokens\n", text_tokens.size());
            fprintf(stderr, "  Tokens: ");
            for (size_t i = 0; i < std::min(text_tokens.size(), (size_t)10); ++i) {
                fprintf(stderr, "%d ", text_tokens[i]);
            }
            if (text_tokens.size() > 10)
                fprintf(stderr, "...");
            fprintf(stderr, "\n");
        }

        // Step 3: Generate speech codes using TTS transformer
        int64_t t_generate_start = get_time_ms();
        if (!transformer_loaded_) {
            int64_t t_reload_start = get_time_ms();
            if (!transformer_->load_model(tts_model_path_)) {
                result.error_msg = "Failed to reload TTS transformer: " + transformer_->get_error();
                return result;
            }
            transformer_loaded_ = true;
            if (params.n_threads > 0 && !transformer_->set_n_threads(params.n_threads)) {
                result.error_msg = "Failed to set transformer thread count: " + transformer_->get_error();
                return result;
            }
            if (params.print_timing) {
                fprintf(stderr, "  Transformer reloaded in %lld ms\n", (long long)(get_time_ms() - t_reload_start));
                sample_memory("synth/after-transformer-reload");
            }
        } else if (params.n_threads > 0 && !transformer_->set_n_threads(params.n_threads)) {
            result.error_msg = "Failed to set transformer thread count: " + transformer_->get_error();
            return result;
        }
        transformer_->set_talker_attention_window(params.talker_attention_window);
        transformer_->clear_kv_cache();

        const float * generation_speaker_embedding = speaker_embedding;
        if (speaker_codec_id >= 0) {
            generation_speaker_embedding = nullptr;
        }

        std::vector<int32_t> speech_codes;
        if (!transformer_->generate(
            text_tokens.data(), (int32_t)text_tokens.size(), generation_speaker_embedding,
            params.max_audio_tokens,
            speech_codes, effective_language_id, speaker_codec_id,
            params.repetition_penalty, params.temperature, params.top_k,
                params.top_p,
                instruct_tokens.empty() ? nullptr : instruct_tokens.data(),
                (int32_t)instruct_tokens.size(),
                nullptr,
                params.seed
            )) {
            result.error_msg = "Failed to generate speech codes: " + transformer_->get_error();
            return result;
        }

        result.t_generate_ms = get_time_ms() - t_generate_start;
        sample_memory("synth/after-generate");

        int n_codebooks = transformer_->get_config().n_codebooks;
        int n_frames = (int)speech_codes.size() / n_codebooks;

        if (params.print_progress) {
            fprintf(stderr, "Speech codes generated: %d frames x %d codebooks\n", n_frames, n_codebooks);
        }

        if (n_frames == 0) {
            result.error_msg = "No speech codes generated";
            return result;
        }

        if (low_mem_mode_) {
            transformer_->unload_model();
            transformer_loaded_ = false;
            sample_memory("synth/after-transformer-unload");
        }

        // Step 4: Decode speech codes to waveform using vocoder
        int64_t t_decode_start = get_time_ms();
        if (!decoder_loaded_) {
            int64_t t_decoder_load_start = get_time_ms();
            if (decoder_model_path_.empty()) {
                result.error_msg = "Internal error: missing vocoder model path";
                return result;
            }
            if (!audio_decoder_->load_model(decoder_model_path_)) {
                result.error_msg = "Failed to load vocoder: " + audio_decoder_->get_error();
                return result;
            }
            decoder_loaded_ = true;
            if (params.n_threads > 0 && !audio_decoder_->set_n_threads(params.n_threads)) {
                result.error_msg = "Failed to set vocoder thread count: " + audio_decoder_->get_error();
                return result;
            }
            if (params.print_timing) {
                fprintf(
                    stderr, "  Vocoder lazy-loaded in %lld ms\n", (long long)(get_time_ms() - t_decoder_load_start)
                );
                sample_memory("synth/after-vocoder-load");
            }
        } else if (params.n_threads > 0 && !audio_decoder_->set_n_threads(params.n_threads)) {
            result.error_msg = "Failed to set vocoder thread count: " + audio_decoder_->get_error();
            return result;
        }

        if (!audio_decoder_->decode(speech_codes.data(), n_frames, result.audio)) {
            result.error_msg = "Failed to decode speech codes: " + audio_decoder_->get_error();
            return result;
        }
        result.t_decode_ms = get_time_ms() - t_decode_start;
        sample_memory("synth/after-decode");

        if (low_mem_mode_) {
            audio_decoder_->unload_model();
            decoder_loaded_ = false;
            sample_memory("synth/after-vocoder-unload");
        }

        result.sample_rate = audio_decoder_->get_config().sample_rate;
        result.success = true;
        result.t_total_ms = get_time_ms() - t_total_start;
        sample_memory("synth/end");

        if (params.print_timing) {
            const double audio_sec =
                result.sample_rate > 0 ? (double)result.audio.size() / (double)result.sample_rate : 0.0;
            const double wall_sec = (double)result.t_total_ms / 1000.0;
            const double realtime_factor = audio_sec > 0.0 ? wall_sec / audio_sec : 0.0;
            const double x_realtime = wall_sec > 0.0 ? audio_sec / wall_sec : 0.0;
            fprintf(stderr, "\nTiming:\n");
            fprintf(stderr, "  Tokenization:    %lld ms\n", (long long)result.t_tokenize_ms);
            fprintf(stderr, "  Speaker encode:  %lld ms\n", (long long)result.t_encode_ms);
            fprintf(stderr, "  Code generation: %lld ms\n", (long long)result.t_generate_ms);
            fprintf(stderr, "  Vocoder decode:  %lld ms\n", (long long)result.t_decode_ms);
            fprintf(stderr, "  Total:           %lld ms\n", (long long)result.t_total_ms);
            fprintf(stderr, "  Audio duration:  %.2f s\n", audio_sec);
            fprintf(stderr, "  Throughput:      %.2fx realtime (RTF=%.3f)\n", x_realtime, realtime_factor);
            fprintf(stderr, "\nMemory:\n");
            fprintf(
                stderr, "  RSS start/end:   %s -> %s\n", format_bytes(result.mem_rss_start_bytes).c_str(),
                format_bytes(result.mem_rss_end_bytes).c_str()
            );
            fprintf(stderr, "  RSS peak:        %s\n", format_bytes(result.mem_rss_peak_bytes).c_str());
            fprintf(
                stderr, "  Phys start/end:  %s -> %s\n", format_bytes(result.mem_phys_start_bytes).c_str(),
                format_bytes(result.mem_phys_end_bytes).c_str()
            );
            fprintf(stderr, "  Phys peak:       %s\n", format_bytes(result.mem_phys_peak_bytes).c_str());
        }

        return result;
    }

    void Qwen3TTS::set_progress_callback(tts_progress_callback_t callback) {
        progress_callback_ = callback;
    }

    std::shared_ptr<tts_stream_session> Qwen3TTS::synthesize_stream(
        const std::string &text,
        const tts_stream_params &params,
        const tts_stream_callbacks &callbacks
    ) {
        if (!models_loaded_) {
            error_msg_ = "Models not loaded";
            if (callbacks.on_error) {
                callbacks.on_error(error_msg_);
            }
            return nullptr;
        }

        const int32_t hidden_size = loaded_hidden_size_ > 0 ? loaded_hidden_size_ : transformer_->get_config().hidden_size;
        if (hidden_size <= 0) {
            error_msg_ = "Invalid hidden size for streaming";
            if (callbacks.on_error) {
                callbacks.on_error(error_msg_);
            }
            return nullptr;
        }

        std::vector<float> zero_embedding((size_t)hidden_size, 0.0f);
        return synthesize_stream_internal(text, zero_embedding, true, params, callbacks);
    }

    std::shared_ptr<tts_stream_session> Qwen3TTS::synthesize_with_voice_stream(
        const std::string &text,
        const std::string &reference_audio,
        const tts_stream_params &params,
        const tts_stream_callbacks &callbacks
    ) {
        if (params.tts.task_type == tts_task_type::voice_clone &&
            !params.tts.x_vector_only_mode &&
            params.tts.reference_text.empty()) {
            error_msg_ = "voice_clone task requires reference_text when x_vector_only_mode is false";
            if (callbacks.on_error) {
                callbacks.on_error(error_msg_);
            }
            return nullptr;
        }

        std::vector<float> ref_samples;
        int ref_sample_rate = 0;
        if (!load_audio_file(reference_audio, ref_samples, ref_sample_rate)) {
            error_msg_ = "Failed to load reference audio: " + reference_audio;
            if (callbacks.on_error) {
                callbacks.on_error(error_msg_);
            }
            return nullptr;
        }

        const int target_rate = 24000;
        if (ref_sample_rate != target_rate) {
            std::vector<float> resampled;
            resample_linear(ref_samples.data(), (int)ref_samples.size(), ref_sample_rate, resampled, target_rate);
            ref_samples = std::move(resampled);
        }

        return synthesize_with_voice_stream(text, ref_samples.data(), (int32_t)ref_samples.size(), params, callbacks);
    }

    std::shared_ptr<tts_stream_session> Qwen3TTS::synthesize_with_voice_stream(
        const std::string &text,
        const float *ref_samples,
        int32_t n_ref_samples,
        const tts_stream_params &params,
        const tts_stream_callbacks &callbacks
    ) {
        if (!models_loaded_) {
            error_msg_ = "Models not loaded";
            if (callbacks.on_error) {
                callbacks.on_error(error_msg_);
            }
            return nullptr;
        }

        if (!ref_samples || n_ref_samples <= 0) {
            error_msg_ = "Invalid reference samples";
            if (callbacks.on_error) {
                callbacks.on_error(error_msg_);
            }
            return nullptr;
        }

        if (!encoder_loaded_) {
            if (tts_model_path_.empty()) {
                error_msg_ = "Internal error: missing TTS model path for lazy encoder load";
                if (callbacks.on_error) {
                    callbacks.on_error(error_msg_);
                }
                return nullptr;
            }
            if (!audio_encoder_->load_model(tts_model_path_)) {
                error_msg_ = "Failed to load speaker encoder: " + audio_encoder_->get_error();
                if (callbacks.on_error) {
                    callbacks.on_error(error_msg_);
                }
                return nullptr;
            }
            encoder_loaded_ = true;
        }

        if (params.tts.n_threads > 0 && !audio_encoder_->set_n_threads(params.tts.n_threads)) {
            error_msg_ = "Failed to set speaker encoder thread count: " + audio_encoder_->get_error();
            if (callbacks.on_error) {
                callbacks.on_error(error_msg_);
            }
            return nullptr;
        }

        std::vector<float> speaker_embedding;
        if (!audio_encoder_->encode(ref_samples, n_ref_samples, speaker_embedding)) {
            error_msg_ = "Failed to extract speaker embedding: " + audio_encoder_->get_error();
            if (callbacks.on_error) {
                callbacks.on_error(error_msg_);
            }
            return nullptr;
        }

        return synthesize_stream_internal(text, speaker_embedding, true, params, callbacks);
    }

    std::shared_ptr<tts_stream_session> Qwen3TTS::synthesize_with_embedding_stream(
        const std::string &text,
        const float *embedding,
        int32_t embedding_size,
        const tts_stream_params &params,
        const tts_stream_callbacks &callbacks
    ) {
        if (!models_loaded_) {
            error_msg_ = "Models not loaded";
            if (callbacks.on_error) {
                callbacks.on_error(error_msg_);
            }
            return nullptr;
        }

        if (!embedding || embedding_size <= 0) {
            error_msg_ = "Invalid speaker embedding";
            if (callbacks.on_error) {
                callbacks.on_error(error_msg_);
            }
            return nullptr;
        }

        std::vector<float> speaker_embedding((size_t)embedding_size);
        std::copy(embedding, embedding + embedding_size, speaker_embedding.begin());
        return synthesize_stream_internal(text, speaker_embedding, true, params, callbacks);
    }

    std::shared_ptr<tts_stream_session> Qwen3TTS::synthesize_stream_internal(
        const std::string &text,
        const std::vector<float> &speaker_embedding,
        bool has_speaker_embedding,
        const tts_stream_params &params,
        const tts_stream_callbacks &callbacks
    ) {
        if (!models_loaded_) {
            error_msg_ = "Models not loaded";
            if (callbacks.on_error) {
                callbacks.on_error(error_msg_);
            }
            return nullptr;
        }
        if (text.empty()) {
            error_msg_ = "Input text is empty";
            if (callbacks.on_error) {
                callbacks.on_error(error_msg_);
            }
            return nullptr;
        }

        {
            std::lock_guard<std::mutex> lock(stream_mutex_);
            if (stream_active_) {
                error_msg_ = "Another streaming session is already active";
                if (callbacks.on_error) {
                    callbacks.on_error(error_msg_);
                }
                return nullptr;
            }
            stream_active_ = true;
        }

        auto impl = std::make_shared<tts_stream_session::impl>();
        impl->queue_capacity = std::max(1, params.stream_queue_capacity);
        impl->drop_oldest_on_overflow = params.stream_drop_oldest_on_overflow;
        impl->default_poll_timeout_ms = std::max(0, params.stream_poll_timeout_ms);

        std::shared_ptr<tts_stream_session> session(new tts_stream_session(impl));

        {
            std::lock_guard<std::mutex> lock(stream_mutex_);
            stream_impl_ = impl;
        }

        try {
            impl->worker = std::thread([this, impl, callbacks, text, speaker_embedding, has_speaker_embedding, params]() {
                struct stream_active_reset {
                    Qwen3TTS * self;
                    ~stream_active_reset() {
                        std::lock_guard<std::mutex> lock(self->stream_mutex_);
                        self->stream_active_ = false;
                        self->stream_impl_.reset();
                    }
                } active_reset{this};

                auto report_error = [&](const std::string &msg) {
                    {
                        std::lock_guard<std::mutex> lock(stream_mutex_);
                        error_msg_ = msg;
                    }
                    stream_set_error(impl, msg);
                    if (callbacks.on_error) {
                        try {
                            callbacks.on_error(msg);
                        } catch (...) {
                        }
                    }
                };

                tts_params p = params.tts;

                if (loaded_model_variant_ != tts_model_variant::auto_variant &&
                    p.model_variant != tts_model_variant::auto_variant &&
                    p.model_variant != loaded_model_variant_) {
                    report_error(std::string("Requested --model-variant '") +
                                 variant_name(p.model_variant) +
                                 "' does not match loaded model variant '" +
                                 variant_name(loaded_model_variant_) +
                                 "'.");
                    return;
                }

                tts_model_variant effective_variant =
                    (p.model_variant != tts_model_variant::auto_variant) ? p.model_variant : loaded_model_variant_;
                if (effective_variant == tts_model_variant::auto_variant) {
                    effective_variant = tts_model_variant::base;
                }

                if (p.task_type != tts_task_type::auto_task &&
                    loaded_model_variant_ != tts_model_variant::auto_variant) {
                    bool compatible = true;
                    switch (p.task_type) {
                        case tts_task_type::voice_clone:
                            compatible = (loaded_model_variant_ == tts_model_variant::base);
                            break;
                        case tts_task_type::custom_voice:
                            compatible = (loaded_model_variant_ == tts_model_variant::custom_voice);
                            break;
                        case tts_task_type::voice_design:
                            compatible = (loaded_model_variant_ == tts_model_variant::voice_design);
                            break;
                        default:
                            compatible = true;
                            break;
                    }
                    if (!compatible) {
                        report_error(std::string("Task '") + task_name(p.task_type) +
                                     "' is incompatible with loaded model variant '" +
                                     variant_name(loaded_model_variant_) + "'.");
                        return;
                    }
                }

                if (p.task_type == tts_task_type::voice_design &&
                    effective_variant != tts_model_variant::voice_design) {
                    report_error(std::string("voice_design task requires voice_design model variant, got ") +
                                 variant_name(effective_variant));
                    return;
                }

                if (p.task_type == tts_task_type::custom_voice &&
                    effective_variant == tts_model_variant::base) {
                    report_error("custom_voice task requires custom_voice or voice_design variant");
                    return;
                }

                int32_t effective_language_id = p.language_id;
                int32_t speaker_codec_id = -1;
                const bool use_custom_voice_speaker =
                    (p.task_type == tts_task_type::custom_voice ||
                     effective_variant == tts_model_variant::custom_voice);
                if (use_custom_voice_speaker) {
                    const std::string speaker_lower = to_lower_copy(p.speaker);
                    if (speaker_lower.empty()) {
                        report_error("speaker is required for custom_voice");
                        return;
                    }
                    if (!is_supported_custom_voice_speaker(speaker_lower)) {
                        report_error("Unsupported speaker: " + p.speaker);
                        return;
                    }
                    if (!map_custom_voice_speaker_codec_id(speaker_lower, speaker_codec_id)) {
                        report_error("Unsupported speaker: " + p.speaker);
                        return;
                    }
                    effective_language_id = map_dialect_language_if_needed(speaker_lower, effective_language_id);
                }

                std::vector<int32_t> text_tokens = tokenizer_->encode_for_tts(
                    text,
                    "",
                    p.speaker,
                    p.reference_text
                );
                std::vector<int32_t> instruct_tokens = tokenizer_->encode_instruct_for_tts(p.instruct);
                if (text_tokens.empty()) {
                    report_error("Failed to tokenize text");
                    return;
                }

                if (!transformer_loaded_) {
                    if (!transformer_->load_model(tts_model_path_)) {
                        report_error("Failed to reload TTS transformer: " + transformer_->get_error());
                        return;
                    }
                    transformer_loaded_ = true;
                }
                if (p.n_threads > 0 && !transformer_->set_n_threads(p.n_threads)) {
                    report_error("Failed to set transformer thread count: " + transformer_->get_error());
                    return;
                }
                const int32_t stream_talker_window =
                    params.stream_talker_attention_window >= 0
                        ? params.stream_talker_attention_window
                        : p.talker_attention_window;
                transformer_->set_talker_attention_window(stream_talker_window);

                if (!decoder_loaded_) {
                    if (decoder_model_path_.empty()) {
                        report_error("Internal error: missing vocoder model path");
                        return;
                    }
                    if (!audio_decoder_->load_model(decoder_model_path_)) {
                        report_error("Failed to load vocoder: " + audio_decoder_->get_error());
                        return;
                    }
                    decoder_loaded_ = true;
                }
                if (p.n_threads > 0 && !audio_decoder_->set_n_threads(p.n_threads)) {
                    report_error("Failed to set vocoder thread count: " + audio_decoder_->get_error());
                    return;
                }

                if (!audio_decoder_->begin_stream(
                        params.stream_decoder_left_context_frames,
                        params.stream_decoder_lookahead_frames)) {
                    report_error("Failed to initialize decoder stream: " + audio_decoder_->get_error());
                    return;
                }

                const int32_t n_codebooks = transformer_->get_config().n_codebooks;
                const int32_t chunk_target_frames = std::max(1, params.stream_chunk_frames);
                const int32_t left_context_frames = std::max(0, params.stream_decoder_left_context_frames);
                const int32_t lookahead_frames = std::max(0, params.stream_decoder_lookahead_frames);
                std::vector<int32_t> pending_codes;
                pending_codes.reserve((size_t)chunk_target_frames * (size_t)n_codebooks);
                int32_t pending_frames = 0;
                bool any_chunk_emitted = false;
                const bool stream_dbg = stream_debug_enabled();
                if (params.stream_adaptive_tuning) {
                    fprintf(stderr, "[stream] adaptive tuning is disabled; using fixed chunk/context\n");
                }
                int64_t stream_t0_ms = get_time_ms();
                int64_t t_tok_ms = 0;
                int64_t t_gen_cb_ms = 0;
                int64_t t_decode_ms = 0;
                int64_t t_emit_ms = 0;
                int32_t n_decode_calls = 0;
                int32_t n_emit_calls = 0;
                int32_t n_frames_seen = 0;
                int64_t t_enqueue_wait_ms = 0;
                int32_t n_enqueue_waits = 0;
                std::atomic<int32_t> decode_queue_peak{0};
                std::atomic<int32_t> output_queue_peak{0};

                auto update_peak = [](std::atomic<int32_t> &peak, int32_t value) {
                    int32_t cur = peak.load();
                    while (value > cur && !peak.compare_exchange_weak(cur, value)) {
                    }
                };

                struct decode_work_item {
                    std::vector<int32_t> codes;
                    int32_t n_frames = 0;
                };
                std::mutex decode_mutex;
                std::condition_variable decode_cv;
                std::deque<decode_work_item> decode_queue;
                const size_t decode_work_capacity = (size_t)std::max(
                    1,
                    params.stream_parallel_decode
                        ? std::max(2, params.stream_queue_capacity / 2)
                        : 1
                );
                bool decode_input_done = false;
                std::atomic<bool> decode_failed{false};
                std::string decode_error;

                auto dbg_log = [&](const char * fmt, ...) {
                    if (!stream_dbg || !fmt) {
                        return;
                    }
                    va_list args;
                    va_start(args, fmt);
                    fprintf(stderr, "[stream-dbg] ");
                    vfprintf(stderr, fmt, args);
                    fprintf(stderr, "\n");
                    va_end(args);
                };

                dbg_log(
                    "start chunk_frames=%d left_ctx=%d lookahead=%d max_tokens=%d",
                    chunk_target_frames,
                    left_context_frames,
                    lookahead_frames,
                    p.max_audio_tokens
                );

                auto set_decode_error = [&](const std::string &msg) {
                    std::lock_guard<std::mutex> lock(decode_mutex);
                    if (!decode_failed.load()) {
                        decode_error = msg;
                    }
                    decode_failed.store(true);
                    decode_cv.notify_all();
                };

                auto get_decode_error = [&]() -> std::string {
                    std::lock_guard<std::mutex> lock(decode_mutex);
                    return decode_error;
                };

                std::thread decode_worker([&]() {
                    while (true) {
                        decode_work_item work;
                        {
                            std::unique_lock<std::mutex> lock(decode_mutex);
                            decode_cv.wait(lock, [&]() {
                                return !decode_queue.empty() || decode_input_done || decode_failed.load();
                            });
                            if (decode_failed.load()) {
                                return;
                            }
                            if (decode_queue.empty()) {
                                if (decode_input_done) {
                                    break;
                                }
                                continue;
                            }
                            work = std::move(decode_queue.front());
                            decode_queue.pop_front();
                        }
                        decode_cv.notify_all();

                        if (work.n_frames <= 0 || work.codes.empty()) {
                            continue;
                        }

                        std::vector<float> chunk_audio;
                        const int64_t t0_ms = get_time_ms();
                        if (!audio_decoder_->decode_append(work.codes.data(), work.n_frames, chunk_audio)) {
                            set_decode_error("Streaming decode failed: " + audio_decoder_->get_error());
                            return;
                        }
                        const int64_t t1_ms = get_time_ms();
                        t_decode_ms += (t1_ms - t0_ms);
                        n_decode_calls++;

                        const int32_t out_samples_count = (int32_t)chunk_audio.size();

                        if (!chunk_audio.empty()) {
                            tts_audio_chunk chunk;
                            chunk.audio = std::move(chunk_audio);
                            chunk.sample_rate = audio_decoder_->get_config().sample_rate;
                            chunk.is_last = false;

                            const int64_t t_emit0_ms = get_time_ms();
                            if (!stream_emit_chunk(impl, std::move(chunk), callbacks)) {
                                std::string emit_error;
                                {
                                    std::lock_guard<std::mutex> lock(impl->mutex);
                                    emit_error = impl->error_msg;
                                }
                                if (emit_error.empty()) {
                                    emit_error = "Failed to emit stream audio chunk";
                                }
                                set_decode_error(emit_error);
                                return;
                            }
                            const int64_t t_emit1_ms = get_time_ms();
                            t_emit_ms += (t_emit1_ms - t_emit0_ms);
                            n_emit_calls++;
                            any_chunk_emitted = true;
                        }

                        int32_t queue_size = 0;
                        {
                            std::lock_guard<std::mutex> lock(impl->mutex);
                            queue_size = (int32_t)impl->queue.size();
                        }
                        update_peak(output_queue_peak, queue_size);

                        dbg_log(
                            "emit decode_call=%d decode_ms=%lld emit_calls=%d out_samples=%d queue=%d",
                            n_decode_calls,
                            (long long)(t1_ms - t0_ms),
                            n_emit_calls,
                            out_samples_count,
                            queue_size
                        );
                    }

                    std::vector<float> tail_samples;
                    const int64_t t_tail0_ms = get_time_ms();
                    if (!audio_decoder_->end_stream(tail_samples, params.stream_full_flush)) {
                        set_decode_error("Failed to finalize decoder stream: " + audio_decoder_->get_error());
                        return;
                    }
                    const int64_t t_tail1_ms = get_time_ms();
                    t_decode_ms += (t_tail1_ms - t_tail0_ms);

                    if (!tail_samples.empty()) {
                        tts_audio_chunk chunk;
                        chunk.audio = std::move(tail_samples);
                        chunk.sample_rate = audio_decoder_->get_config().sample_rate;
                        chunk.is_last = true;
                        const int64_t t_emit0_ms = get_time_ms();
                        if (!stream_emit_chunk(impl, std::move(chunk), callbacks)) {
                            std::string emit_error;
                            {
                                std::lock_guard<std::mutex> lock(impl->mutex);
                                emit_error = impl->error_msg;
                            }
                            if (emit_error.empty()) {
                                emit_error = "Failed to emit final stream audio chunk";
                            }
                            set_decode_error(emit_error);
                            return;
                        }
                        const int64_t t_emit1_ms = get_time_ms();
                        t_emit_ms += (t_emit1_ms - t_emit0_ms);
                        n_emit_calls++;
                        any_chunk_emitted = true;
                    } else if (!any_chunk_emitted) {
                        tts_audio_chunk terminal;
                        terminal.sample_rate = audio_decoder_->get_config().sample_rate;
                        terminal.is_last = true;
                        const int64_t t_emit0_ms = get_time_ms();
                        if (!stream_emit_chunk(impl, std::move(terminal), callbacks)) {
                            std::string emit_error;
                            {
                                std::lock_guard<std::mutex> lock(impl->mutex);
                                emit_error = impl->error_msg;
                            }
                            if (emit_error.empty()) {
                                emit_error = "Failed to emit stream terminal chunk";
                            }
                            set_decode_error(emit_error);
                            return;
                        }
                        const int64_t t_emit1_ms = get_time_ms();
                        t_emit_ms += (t_emit1_ms - t_emit0_ms);
                        n_emit_calls++;
                    }
                });

                struct decode_worker_guard {
                    std::mutex &mutex;
                    std::condition_variable &cv;
                    bool &input_done;
                    std::thread &worker;
                    ~decode_worker_guard() {
                        {
                            std::lock_guard<std::mutex> lock(mutex);
                            input_done = true;
                        }
                        cv.notify_all();
                        if (worker.joinable()) {
                            worker.join();
                        }
                    }
                } decode_guard{decode_mutex, decode_cv, decode_input_done, decode_worker};

                auto enqueue_decode = [&](std::vector<int32_t> &&codes, int32_t n_frames) -> bool {
                    if (n_frames <= 0 || codes.empty()) {
                        return true;
                    }

                    std::unique_lock<std::mutex> lock(decode_mutex);
                    const int64_t t_wait0_ms = get_time_ms();
                    decode_cv.wait(lock, [&]() {
                        return decode_queue.size() < decode_work_capacity || decode_failed.load();
                    });
                    const int64_t t_wait1_ms = get_time_ms();
                    if (t_wait1_ms > t_wait0_ms) {
                        t_enqueue_wait_ms += (t_wait1_ms - t_wait0_ms);
                        n_enqueue_waits++;
                    }
                    if (decode_failed.load()) {
                        return false;
                    }

                    decode_work_item item;
                    item.codes = std::move(codes);
                    item.n_frames = n_frames;
                    decode_queue.push_back(std::move(item));
                    update_peak(decode_queue_peak, (int32_t)decode_queue.size());
                    lock.unlock();
                    decode_cv.notify_all();
                    return true;
                };

                auto fail_from_decode_worker = [&]() -> bool {
                    std::string msg = get_decode_error();
                    if (msg.empty()) {
                        msg = "Streaming decode worker failed";
                    }
                    report_error(msg);
                    return false;
                };

                auto emit_pending = [&]() -> bool {
                    if (pending_frames <= 0) {
                        return true;
                    }

                    std::vector<int32_t> chunk_codes = std::move(pending_codes);
                    const int32_t chunk_frames = pending_frames;
                    pending_codes.clear();
                    pending_frames = 0;
                    const int32_t reserve_frames = std::max(1, chunk_target_frames);
                    pending_codes.reserve((size_t)reserve_frames * (size_t)n_codebooks);

                    if (!enqueue_decode(std::move(chunk_codes), chunk_frames)) {
                        return fail_from_decode_worker();
                    }
                    return true;
                };

                const float * generation_speaker_embedding = has_speaker_embedding ? speaker_embedding.data() : nullptr;
                if (speaker_codec_id >= 0) {
                    generation_speaker_embedding = nullptr;
                }

                std::vector<int32_t> generated_codes;
                t_tok_ms = get_time_ms() - stream_t0_ms;
                const bool gen_ok = transformer_->generate(
                    text_tokens.data(),
                    (int32_t)text_tokens.size(),
                    generation_speaker_embedding,
                    p.max_audio_tokens,
                    generated_codes,
                    effective_language_id,
                    speaker_codec_id,
                    p.repetition_penalty,
                    p.temperature,
                    p.top_k,
                    p.top_p,
                    instruct_tokens.empty() ? nullptr : instruct_tokens.data(),
                    (int32_t)instruct_tokens.size(),
                    [&](const int32_t *frame_codes, int32_t cb_count, int32_t frame_index) -> bool {
                        if (decode_failed.load()) {
                            return fail_from_decode_worker();
                        }

                        int64_t cb_t0_ms = get_time_ms();
                        if (cb_count != n_codebooks) {
                            report_error("Unexpected codebook count in streaming callback");
                            return false;
                        }

                        if (callbacks.on_codes_frame) {
                            try {
                                callbacks.on_codes_frame(frame_codes, cb_count, frame_index);
                            } catch (...) {
                                report_error("Streaming codes callback threw an exception");
                                return false;
                            }
                        }

                        pending_codes.insert(pending_codes.end(), frame_codes, frame_codes + cb_count);
                        pending_frames++;
                        n_frames_seen = frame_index + 1;

                        if (progress_callback_) {
                            progress_callback_(frame_index + 1, p.max_audio_tokens);
                        }

                        const int32_t chunk_target_now = std::max(1, chunk_target_frames);
                        if (pending_frames >= chunk_target_now) {
                            bool ok = emit_pending();
                            t_gen_cb_ms += (get_time_ms() - cb_t0_ms);
                            return ok;
                        }

                        t_gen_cb_ms += (get_time_ms() - cb_t0_ms);

                        return true;
                    },
                    p.seed
                );

                if (!gen_ok) {
                    if (!impl->has_error) {
                        report_error("Failed to generate speech codes: " + transformer_->get_error());
                    }
                    return;
                }

                if (!emit_pending()) {
                    return;
                }

                {
                    std::lock_guard<std::mutex> lock(decode_mutex);
                    decode_input_done = true;
                }
                decode_cv.notify_all();
                if (decode_worker.joinable()) {
                    decode_worker.join();
                }

                if (decode_failed.load()) {
                    fail_from_decode_worker();
                    return;
                }

                if (callbacks.on_finish) {
                    try {
                        callbacks.on_finish();
                    } catch (...) {
                        report_error("Streaming finish callback threw an exception");
                        return;
                    }
                }

                if (low_mem_mode_) {
                    transformer_->unload_model();
                    transformer_loaded_ = false;
                    audio_decoder_->unload_model();
                    decoder_loaded_ = false;
                }

                (void)any_chunk_emitted;
                if (stream_dbg) {
                    int64_t t_total_ms = get_time_ms() - stream_t0_ms;
                    int32_t queue_final = 0;
                    {
                        std::lock_guard<std::mutex> lock(impl->mutex);
                        queue_final = (int32_t)impl->queue.size();
                    }
                    dbg_log(
                        "done total_ms=%lld tokenize_ms=%lld gen_cb_ms=%lld decode_ms=%lld emit_ms=%lld enqueue_wait_ms=%lld enqueue_waits=%d frames=%d decode_calls=%d emit_calls=%d decode_q_peak=%d out_q_peak=%d queue_final=%d",
                        (long long)t_total_ms,
                        (long long)t_tok_ms,
                        (long long)t_gen_cb_ms,
                        (long long)t_decode_ms,
                        (long long)t_emit_ms,
                        (long long)t_enqueue_wait_ms,
                        n_enqueue_waits,
                        n_frames_seen,
                        n_decode_calls,
                        n_emit_calls,
                        decode_queue_peak.load(),
                        output_queue_peak.load(),
                        queue_final
                    );
                }
                stream_mark_finished(impl);
            });
        } catch (const std::exception &ex) {
            {
                std::lock_guard<std::mutex> lock(stream_mutex_);
                stream_active_ = false;
                stream_impl_.reset();
            }
            error_msg_ = std::string("Failed to start streaming worker: ") + ex.what();
            if (callbacks.on_error) {
                callbacks.on_error(error_msg_);
            }
            return nullptr;
        }

        return session;
    }

    static tts_result make_error_result(const std::string &error_msg) {
        tts_result result;
        result.success = false;
        result.error_msg = error_msg;
        return result;
    }

    static bool map_custom_voice_speaker_codec_id(const std::string &speaker_lower, int32_t &codec_id_out) {
        struct speaker_codec_item {
            const char *name;
            int32_t codec_id;
        };
        static const speaker_codec_item kSpeakerCodecMap[] = {
            { "vivian", 3065 },
            { "serena", 3066 },
            { "uncle_fu", 3010 },
            { "dylan", 2878 },
            { "eric", 2875 },
            { "ryan", 3061 },
            { "aiden", 2861 },
            { "ono_anna", 2873 },
            { "sohee", 2864 },
        };
        for (const speaker_codec_item &item : kSpeakerCodecMap) {
            if (speaker_lower == item.name) {
                codec_id_out = item.codec_id;
                return true;
            }
        }
        return false;
    }

    static bool is_supported_custom_voice_speaker(const std::string &speaker_lower) {
        int32_t codec_id = -1;
        return map_custom_voice_speaker_codec_id(speaker_lower, codec_id);
    }

    static int32_t map_dialect_language_if_needed(const std::string &speaker_lower, int32_t language_id) {
        const bool chinese_or_auto = (language_id < 0 || language_id == 2055);
        if (!chinese_or_auto) {
            return language_id;
        }
        if (speaker_lower == "dylan") {
            return 2074; // beijing_dialect
        }
        if (speaker_lower == "eric") {
            return 2062; // sichuan_dialect
        }
        return language_id;
    }

    struct Qwen3TTSModelHub::model_slot {
        tts_model_id model_id = tts_model_id::model_06b_base;
        tts_model_variant expected_variant = tts_model_variant::auto_variant;
        int32_t expected_hidden_size = 0;
        std::string model_dir;
        std::string tts_model_path;
        std::string tokenizer_model_path;
        std::unique_ptr<Qwen3TTS> engine;
    };

    Qwen3TTSModelHub::Qwen3TTSModelHub() = default;

    Qwen3TTSModelHub::~Qwen3TTSModelHub() = default;

    bool Qwen3TTSModelHub::set_shared_tokenizer_12hz(const std::string &tokenizer_model_path_or_dir) {
        std::string resolved;
        if (!discover_tokenizer_model_path(tokenizer_model_path_or_dir, resolved, error_msg_)) {
            return false;
        }
        shared_tokenizer_path_ = resolved;
        return true;
    }

    bool Qwen3TTSModelHub::load_models_06b_base(const std::string &model_dir) {
        return load_model_internal(
            tts_model_id::model_06b_base,
            model_dir,
            tts_model_variant::base,
            1024
        );
    }

    bool Qwen3TTSModelHub::load_models_06b_custom_voice(const std::string &model_dir) {
        return load_model_internal(
            tts_model_id::model_06b_custom_voice,
            model_dir,
            tts_model_variant::custom_voice,
            1024
        );
    }

    bool Qwen3TTSModelHub::load_models_1_7b_base(const std::string &model_dir) {
        return load_model_internal(
            tts_model_id::model_1_7b_base,
            model_dir,
            tts_model_variant::base,
            2048
        );
    }

    bool Qwen3TTSModelHub::load_models_1_7b_custom_voice(const std::string &model_dir) {
        return load_model_internal(
            tts_model_id::model_1_7b_custom_voice,
            model_dir,
            tts_model_variant::custom_voice,
            2048
        );
    }

    bool Qwen3TTSModelHub::load_models_1_7b_voice_design(const std::string &model_dir) {
        return load_model_internal(
            tts_model_id::model_1_7b_voice_design,
            model_dir,
            tts_model_variant::voice_design,
            2048
        );
    }

    bool Qwen3TTSModelHub::unload_model(tts_model_id model_id) {
        auto it = models_.find(model_id);
        if (it == models_.end()) {
            error_msg_ = std::string("Model is not loaded: ") + model_id_name(model_id);
            return false;
        }
        models_.erase(it);
        return true;
    }

    bool Qwen3TTSModelHub::is_model_loaded(tts_model_id model_id) const {
        const model_slot *slot = find_slot(model_id);
        return slot && slot->engine && slot->engine->is_loaded();
    }

    std::vector<tts_model_id> Qwen3TTSModelHub::loaded_models() const {
        std::vector<tts_model_id> ids;
        for (const auto &kv : models_) {
            if (kv.second && kv.second->engine && kv.second->engine->is_loaded()) {
                ids.push_back(kv.first);
            }
        }
        return ids;
    }

    tts_result Qwen3TTSModelHub::synthesize_06b_base_with_voice(
        const std::string &text,
        const std::string &reference_audio,
        const tts_voice_clone_params &params
    ) {
        model_slot *slot = find_slot(tts_model_id::model_06b_base);
        if (!slot || !slot->engine || !slot->engine->is_loaded()) {
            error_msg_ = "Model not loaded: 0.6B Base";
            return make_error_result(error_msg_);
        }

        tts_params p = build_common_tts_params(params.common);
        p.task_type = tts_task_type::voice_clone;
        p.model_variant = tts_model_variant::base;
        p.reference_text = params.reference_text;
        p.x_vector_only_mode = params.x_vector_only_mode;

        tts_result result = slot->engine->synthesize_with_voice(text, reference_audio, p);
        if (!result.success) {
            error_msg_ = result.error_msg;
        }
        return result;
    }

    tts_result Qwen3TTSModelHub::synthesize_06b_base_with_embedding(
        const std::string &text,
        const float *embedding,
        int32_t embedding_size,
        const tts_common_params &params
    ) {
        model_slot *slot = find_slot(tts_model_id::model_06b_base);
        if (!slot || !slot->engine || !slot->engine->is_loaded()) {
            error_msg_ = "Model not loaded: 0.6B Base";
            return make_error_result(error_msg_);
        }

        tts_params p = build_common_tts_params(params);
        p.task_type = tts_task_type::voice_clone;
        p.model_variant = tts_model_variant::base;
        p.x_vector_only_mode = true;

        tts_result result = slot->engine->synthesize_with_embedding(text, embedding, embedding_size, p);
        if (!result.success) {
            error_msg_ = result.error_msg;
        }
        return result;
    }

    tts_result Qwen3TTSModelHub::synthesize_1_7b_base_with_voice(
        const std::string &text,
        const std::string &reference_audio,
        const tts_voice_clone_params &params
    ) {
        model_slot *slot = find_slot(tts_model_id::model_1_7b_base);
        if (!slot || !slot->engine || !slot->engine->is_loaded()) {
            error_msg_ = "Model not loaded: 1.7B Base";
            return make_error_result(error_msg_);
        }

        tts_params p = build_common_tts_params(params.common);
        p.task_type = tts_task_type::voice_clone;
        p.model_variant = tts_model_variant::base;
        p.reference_text = params.reference_text;
        p.x_vector_only_mode = params.x_vector_only_mode;

        tts_result result = slot->engine->synthesize_with_voice(text, reference_audio, p);
        if (!result.success) {
            error_msg_ = result.error_msg;
        }
        return result;
    }

    tts_result Qwen3TTSModelHub::synthesize_1_7b_base_with_embedding(
        const std::string &text,
        const float *embedding,
        int32_t embedding_size,
        const tts_common_params &params
    ) {
        model_slot *slot = find_slot(tts_model_id::model_1_7b_base);
        if (!slot || !slot->engine || !slot->engine->is_loaded()) {
            error_msg_ = "Model not loaded: 1.7B Base";
            return make_error_result(error_msg_);
        }

        tts_params p = build_common_tts_params(params);
        p.task_type = tts_task_type::voice_clone;
        p.model_variant = tts_model_variant::base;
        p.x_vector_only_mode = true;

        tts_result result = slot->engine->synthesize_with_embedding(text, embedding, embedding_size, p);
        if (!result.success) {
            error_msg_ = result.error_msg;
        }
        return result;
    }

    tts_result Qwen3TTSModelHub::synthesize_06b_custom_voice(
        const std::string &text,
        const tts_custom_voice_params &params
    ) {
        model_slot *slot = find_slot(tts_model_id::model_06b_custom_voice);
        if (!slot || !slot->engine || !slot->engine->is_loaded()) {
            error_msg_ = "Model not loaded: 0.6B CustomVoice";
            return make_error_result(error_msg_);
        }

        std::string speaker = to_lower_copy(params.speaker);
        if (speaker.empty()) {
            error_msg_ = "speaker is required for 0.6B CustomVoice";
            return make_error_result(error_msg_);
        }
        if (!is_supported_custom_voice_speaker(speaker)) {
            error_msg_ = "Unsupported speaker: " + params.speaker;
            return make_error_result(error_msg_);
        }

        tts_params p = build_common_tts_params(params.common);
        p.task_type = tts_task_type::custom_voice;
        p.model_variant = tts_model_variant::custom_voice;
        p.speaker = speaker;
        p.language_id = map_dialect_language_if_needed(speaker, p.language_id);

        tts_result result = slot->engine->synthesize(text, p);
        if (!result.success) {
            error_msg_ = result.error_msg;
        }
        return result;
    }

    tts_result Qwen3TTSModelHub::synthesize_1_7b_custom_voice(
        const std::string &text,
        const tts_custom_voice_params &params
    ) {
        model_slot *slot = find_slot(tts_model_id::model_1_7b_custom_voice);
        if (!slot || !slot->engine || !slot->engine->is_loaded()) {
            error_msg_ = "Model not loaded: 1.7B CustomVoice";
            return make_error_result(error_msg_);
        }

        std::string speaker = to_lower_copy(params.speaker);
        if (speaker.empty()) {
            error_msg_ = "speaker is required for 1.7B CustomVoice";
            return make_error_result(error_msg_);
        }
        if (!is_supported_custom_voice_speaker(speaker)) {
            error_msg_ = "Unsupported speaker: " + params.speaker;
            return make_error_result(error_msg_);
        }

        tts_params p = build_common_tts_params(params.common);
        p.task_type = tts_task_type::custom_voice;
        p.model_variant = tts_model_variant::custom_voice;
        p.speaker = speaker;
        p.language_id = map_dialect_language_if_needed(speaker, p.language_id);

        tts_result result = slot->engine->synthesize(text, p);
        if (!result.success) {
            error_msg_ = result.error_msg;
        }
        return result;
    }

    tts_result Qwen3TTSModelHub::synthesize_1_7b_custom_voice_with_instruct(
        const std::string &text,
        const tts_custom_voice_instruct_params &params
    ) {
        model_slot *slot = find_slot(tts_model_id::model_1_7b_custom_voice);
        if (!slot || !slot->engine || !slot->engine->is_loaded()) {
            error_msg_ = "Model not loaded: 1.7B CustomVoice";
            return make_error_result(error_msg_);
        }

        std::string speaker = to_lower_copy(params.speaker);
        if (speaker.empty()) {
            error_msg_ = "speaker is required for 1.7B CustomVoice";
            return make_error_result(error_msg_);
        }
        if (!is_supported_custom_voice_speaker(speaker)) {
            error_msg_ = "Unsupported speaker: " + params.speaker;
            return make_error_result(error_msg_);
        }

        tts_params p = build_common_tts_params(params.common);
        p.task_type = tts_task_type::custom_voice;
        p.model_variant = tts_model_variant::custom_voice;
        p.speaker = speaker;
        p.instruct = params.instruct;
        p.language_id = map_dialect_language_if_needed(speaker, p.language_id);

        tts_result result = slot->engine->synthesize(text, p);
        if (!result.success) {
            error_msg_ = result.error_msg;
        }
        return result;
    }

    tts_result Qwen3TTSModelHub::synthesize_1_7b_voice_design(
        const std::string &text,
        const tts_voice_design_params &params
    ) {
        model_slot *slot = find_slot(tts_model_id::model_1_7b_voice_design);
        if (!slot || !slot->engine || !slot->engine->is_loaded()) {
            error_msg_ = "Model not loaded: 1.7B VoiceDesign";
            return make_error_result(error_msg_);
        }

        if (params.instruct.empty()) {
            error_msg_ = "instruct is required for 1.7B VoiceDesign";
            return make_error_result(error_msg_);
        }

        tts_params p = build_common_tts_params(params.common);
        p.task_type = tts_task_type::voice_design;
        p.model_variant = tts_model_variant::voice_design;
        p.instruct = params.instruct;

        tts_result result = slot->engine->synthesize(text, p);
        if (!result.success) {
            error_msg_ = result.error_msg;
        }
        return result;
    }

    tts_result Qwen3TTSModelHub::synthesize_with_voice(
        tts_model_id model_id,
        const std::string &text,
        const std::string &reference_audio,
        const tts_voice_clone_params &params
    ) {
        switch (model_id) {
            case tts_model_id::model_06b_base:
                return synthesize_06b_base_with_voice(text, reference_audio, params);
            case tts_model_id::model_1_7b_base:
                return synthesize_1_7b_base_with_voice(text, reference_audio, params);
            default:
                error_msg_ = std::string("synthesize_with_voice is not supported for model: ") +
                             model_id_name(model_id);
                return make_error_result(error_msg_);
        }
    }

    tts_result Qwen3TTSModelHub::synthesize_with_instruct(
        tts_model_id model_id,
        const std::string &text,
        const std::string &instruct,
        const tts_custom_voice_params &params
    ) {
        switch (model_id) {
            case tts_model_id::model_1_7b_custom_voice: {
                tts_custom_voice_instruct_params p;
                p.common = params.common;
                p.speaker = params.speaker;
                p.instruct = instruct;
                return synthesize_1_7b_custom_voice_with_instruct(text, p);
            }
            case tts_model_id::model_1_7b_voice_design: {
                tts_voice_design_params p;
                p.common = params.common;
                p.instruct = instruct;
                return synthesize_1_7b_voice_design(text, p);
            }
            case tts_model_id::model_06b_custom_voice:
                error_msg_ = "0.6B CustomVoice does not support instruct";
                return make_error_result(error_msg_);
            default:
                error_msg_ = std::string("synthesize_with_instruct is not supported for model: ") +
                             model_id_name(model_id);
                return make_error_result(error_msg_);
        }
    }

    bool Qwen3TTSModelHub::load_model_internal(
        tts_model_id model_id,
        const std::string &model_dir,
        tts_model_variant expected_variant,
        int32_t expected_hidden_size
    ) {
        std::string tts_model_path;
        if (!discover_tts_model_path(model_dir, tts_model_path, error_msg_)) {
            return false;
        }

        std::string tokenizer_source = shared_tokenizer_path_.empty() ? model_dir : shared_tokenizer_path_;
        std::string tokenizer_model_path;
        if (!discover_tokenizer_model_path(tokenizer_source, tokenizer_model_path, error_msg_)) {
            return false;
        }

        std::unique_ptr<Qwen3TTS> engine = std::make_unique<Qwen3TTS>();
        if (!engine->load_models_from_files(tts_model_path, tokenizer_model_path)) {
            error_msg_ = engine->get_error();
            return false;
        }

        tts_model_variant loaded_variant = engine->loaded_model_variant();
        if (loaded_variant == tts_model_variant::auto_variant) {
            loaded_variant = infer_model_variant_from_path(tts_model_path + " " + model_dir);
        }
        if (expected_variant != tts_model_variant::auto_variant &&
            loaded_variant != tts_model_variant::auto_variant &&
            loaded_variant != expected_variant) {
            error_msg_ = std::string("Loaded model variant mismatch for ") +
                         model_id_name(model_id) +
                         ": expected " + variant_name(expected_variant) +
                         ", got " + variant_name(loaded_variant);
            return false;
        }

        int32_t hidden_size = engine->loaded_hidden_size();
        if (expected_hidden_size > 0 && hidden_size != expected_hidden_size) {
            error_msg_ = std::string("Loaded hidden_size mismatch for ") +
                         model_id_name(model_id) +
                         ": expected " + std::to_string(expected_hidden_size) +
                         ", got " + std::to_string(hidden_size);
            return false;
        }

        auto slot = std::make_unique<model_slot>();
        slot->model_id = model_id;
        slot->expected_variant = expected_variant;
        slot->expected_hidden_size = expected_hidden_size;
        slot->model_dir = model_dir;
        slot->tts_model_path = tts_model_path;
        slot->tokenizer_model_path = tokenizer_model_path;
        slot->engine = std::move(engine);

        models_[model_id] = std::move(slot);
        return true;
    }

    tts_params Qwen3TTSModelHub::build_common_tts_params(const tts_common_params &params) const {
        tts_params out;
        out.max_audio_tokens = params.max_audio_tokens;
        out.temperature = params.temperature;
        out.top_p = params.top_p;
        out.top_k = params.top_k;
        out.n_threads = params.n_threads;
        out.print_progress = params.print_progress;
        out.print_timing = params.print_timing;
        out.repetition_penalty = params.repetition_penalty;
        out.seed = params.seed;
        out.language_id = params.language_id;
        out.talker_attention_window = params.talker_attention_window;
        return out;
    }

    Qwen3TTSModelHub::model_slot *Qwen3TTSModelHub::find_slot(tts_model_id model_id) {
        auto it = models_.find(model_id);
        if (it == models_.end()) {
            return nullptr;
        }
        return it->second.get();
    }

    const Qwen3TTSModelHub::model_slot *Qwen3TTSModelHub::find_slot(tts_model_id model_id) const {
        auto it = models_.find(model_id);
        if (it == models_.end()) {
            return nullptr;
        }
        return it->second.get();
    }

    // WAV file loading (16-bit PCM or 32-bit float)
    bool load_audio_file(const std::string &path, std::vector<float> &samples, int &sample_rate) {
        FILE *f = fopen(path.c_str(), "rb");
        if (!f) {
            fprintf(stderr, "ERROR: Cannot open WAV file: %s\n", path.c_str());
            return false;
        }

        // Read RIFF header
        char riff[4];
        if (fread(riff, 1, 4, f) != 4 || strncmp(riff, "RIFF", 4) != 0) {
            fprintf(stderr, "ERROR: Not a RIFF file\n");
            fclose(f);
            return false;
        }

        uint32_t file_size;
        if (fread(&file_size, 4, 1, f) != 1) {
            fclose(f);
            return false;
        }

        char wave[4];
        if (fread(wave, 1, 4, f) != 4 || strncmp(wave, "WAVE", 4) != 0) {
            fprintf(stderr, "ERROR: Not a WAVE file\n");
            fclose(f);
            return false;
        }

        // Find fmt and data chunks
        uint16_t audio_format = 0;
        uint16_t num_channels = 0;
        uint32_t sr = 0;
        uint16_t bits_per_sample = 0;

        while (!feof(f)) {
            char chunk_id[4];
            uint32_t chunk_size;

            if (fread(chunk_id, 1, 4, f) != 4)
                break;
            if (fread(&chunk_size, 4, 1, f) != 1)
                break;

            if (strncmp(chunk_id, "fmt ", 4) == 0) {
                if (fread(&audio_format, 2, 1, f) != 1)
                    break;
                if (fread(&num_channels, 2, 1, f) != 1)
                    break;
                if (fread(&sr, 4, 1, f) != 1)
                    break;
                fseek(f, 6, SEEK_CUR); // Skip byte rate and block align
                if (fread(&bits_per_sample, 2, 1, f) != 1)
                    break;

                // Skip any extra format bytes
                if (chunk_size > 16) {
                    fseek(f, chunk_size - 16, SEEK_CUR);
                }
            } else if (strncmp(chunk_id, "data", 4) == 0) {
                sample_rate = sr;

                if (audio_format == 1) { // PCM
                    if (bits_per_sample == 16) {
                        int n_samples = chunk_size / (2 * num_channels);
                        samples.resize(n_samples);

                        std::vector<int16_t> raw(n_samples * num_channels);
                        if (fread(raw.data(), 2, n_samples * num_channels, f) != (size_t)(n_samples * num_channels)) {
                            fclose(f);
                            return false;
                        }

                        // Convert to mono float
                        for (int i = 0; i < n_samples; ++i) {
                            float sum = 0.0f;
                            for (int c = 0; c < num_channels; ++c) {
                                sum += raw[i * num_channels + c] / 32768.0f;
                            }
                            samples[i] = sum / num_channels;
                        }
                    } else if (bits_per_sample == 32) {
                        int n_samples = chunk_size / (4 * num_channels);
                        samples.resize(n_samples);

                        std::vector<int32_t> raw(n_samples * num_channels);
                        if (fread(raw.data(), 4, n_samples * num_channels, f) != (size_t)(n_samples * num_channels)) {
                            fclose(f);
                            return false;
                        }

                        // Convert to mono float
                        for (int i = 0; i < n_samples; ++i) {
                            float sum = 0.0f;
                            for (int c = 0; c < num_channels; ++c) {
                                sum += raw[i * num_channels + c] / 2147483648.0f;
                            }
                            samples[i] = sum / num_channels;
                        }
                    } else {
                        fprintf(stderr, "ERROR: Unsupported bits per sample: %d\n", bits_per_sample);
                        fclose(f);
                        return false;
                    }
                } else if (audio_format == 3) { // IEEE float
                    int n_samples = chunk_size / (4 * num_channels);
                    samples.resize(n_samples);

                    std::vector<float> raw(n_samples * num_channels);
                    if (fread(raw.data(), 4, n_samples * num_channels, f) != (size_t)(n_samples * num_channels)) {
                        fclose(f);
                        return false;
                    }

                    // Convert to mono
                    for (int i = 0; i < n_samples; ++i) {
                        float sum = 0.0f;
                        for (int c = 0; c < num_channels; ++c) {
                            sum += raw[i * num_channels + c];
                        }
                        samples[i] = sum / num_channels;
                    }
                } else {
                    fprintf(stderr, "ERROR: Unsupported audio format: %d\n", audio_format);
                    fclose(f);
                    return false;
                }

                fclose(f);
                return true;
            } else {
                // Skip unknown chunk
                fseek(f, chunk_size, SEEK_CUR);
            }
        }

        fprintf(stderr, "ERROR: No data chunk found\n");
        fclose(f);
        return false;
    }

    // WAV file saving (16-bit PCM at specified sample rate)
    bool save_audio_file(const std::string &path, const std::vector<float> &samples, int sample_rate) {
        FILE *f = fopen(path.c_str(), "wb");
        if (!f) {
            fprintf(stderr, "ERROR: Cannot create WAV file: %s\n", path.c_str());
            return false;
        }

        // WAV header parameters
        uint16_t num_channels = 1;
        uint16_t bits_per_sample = 16;
        uint32_t byte_rate = sample_rate * num_channels * bits_per_sample / 8;
        uint16_t block_align = num_channels * bits_per_sample / 8;
        uint32_t data_size = samples.size() * block_align;
        uint32_t file_size = 36 + data_size;

        // Write RIFF header
        fwrite("RIFF", 1, 4, f);
        fwrite(&file_size, 4, 1, f);
        fwrite("WAVE", 1, 4, f);

        // Write fmt chunk
        fwrite("fmt ", 1, 4, f);
        uint32_t fmt_size = 16;
        fwrite(&fmt_size, 4, 1, f);
        uint16_t audio_format = 1; // PCM
        fwrite(&audio_format, 2, 1, f);
        fwrite(&num_channels, 2, 1, f);
        uint32_t sr = sample_rate;
        fwrite(&sr, 4, 1, f);
        fwrite(&byte_rate, 4, 1, f);
        fwrite(&block_align, 2, 1, f);
        fwrite(&bits_per_sample, 2, 1, f);

        // Write data chunk
        fwrite("data", 1, 4, f);
        fwrite(&data_size, 4, 1, f);

        // Convert float samples to 16-bit PCM and write
        for (size_t i = 0; i < samples.size(); ++i) {
            // Clamp to [-1, 1] and convert to int16
            float sample = samples[i];
            if (sample > 1.0f)
                sample = 1.0f;
            if (sample < -1.0f)
                sample = -1.0f;
            int16_t pcm_sample = (int16_t)(sample * 32767.0f);
            fwrite(&pcm_sample, 2, 1, f);
        }

        fclose(f);
        return true;
    }

    // Blocking audio playback via SDL3. Returns only after playback has finished.
    bool qwen3_tts::play_audio(const std::vector<float> &samples, int sample_rate) {
        if (samples.empty()) {
            return true;
        }
        if (sample_rate <= 0) {
            fprintf(stderr, "ERROR: Invalid sample rate: %d\n", sample_rate);
            return false;
        }
        if (samples.size() > (size_t)INT32_MAX / sizeof(float)) {
            fprintf(stderr, "ERROR: Audio buffer too large for SDL stream\n");
            return false;
        }

        const bool audio_was_initialized = (SDL_WasInit(SDL_INIT_AUDIO) & SDL_INIT_AUDIO) != 0;
        if (!audio_was_initialized) {
            if (!SDL_InitSubSystem(SDL_INIT_AUDIO)) {
                fprintf(stderr, "ERROR: SDL_InitSubSystem(SDL_INIT_AUDIO) failed: %s\n", SDL_GetError());
                return false;
            }
        }

        SDL_AudioSpec spec = {};
        spec.format = SDL_AUDIO_F32;
        spec.channels = 1;
        spec.freq = sample_rate;

        SDL_AudioStream *stream = SDL_OpenAudioDeviceStream(SDL_AUDIO_DEVICE_DEFAULT_PLAYBACK, &spec, nullptr, nullptr);
        if (!stream) {
            fprintf(stderr, "ERROR: SDL_OpenAudioDeviceStream failed: %s\n", SDL_GetError());
            if (!audio_was_initialized) {
                SDL_QuitSubSystem(SDL_INIT_AUDIO);
            }
            return false;
        }

        const int byte_len = (int)(samples.size() * sizeof(float));
        if (!SDL_PutAudioStreamData(stream, samples.data(), byte_len)) {
            fprintf(stderr, "ERROR: SDL_PutAudioStreamData failed: %s\n", SDL_GetError());
            SDL_DestroyAudioStream(stream);
            if (!audio_was_initialized) {
                SDL_QuitSubSystem(SDL_INIT_AUDIO);
            }
            return false;
        }

        // Mark end-of-input so any buffered conversion/resampling can be drained.
        if (!SDL_FlushAudioStream(stream)) {
            fprintf(stderr, "ERROR: SDL_FlushAudioStream failed: %s\n", SDL_GetError());
            SDL_DestroyAudioStream(stream);
            if (!audio_was_initialized) {
                SDL_QuitSubSystem(SDL_INIT_AUDIO);
            }
            return false;
        }

        // SDL_OpenAudioDeviceStream opens device paused; resume to start playback.
        if (!SDL_ResumeAudioStreamDevice(stream)) {
            fprintf(stderr, "ERROR: SDL_ResumeAudioStreamDevice failed: %s\n", SDL_GetError());
            SDL_DestroyAudioStream(stream);
            if (!audio_was_initialized) {
                SDL_QuitSubSystem(SDL_INIT_AUDIO);
            }
            return false;
        }

        // Block until all queued input data has been consumed by the stream.
        while (true) {
            const int queued = SDL_GetAudioStreamQueued(stream);
            if (queued < 0) {
                fprintf(stderr, "ERROR: SDL_GetAudioStreamQueued failed: %s\n", SDL_GetError());
                SDL_DestroyAudioStream(stream);
                if (!audio_was_initialized) {
                    SDL_QuitSubSystem(SDL_INIT_AUDIO);
                }
                return false;
            }
            if (queued == 0) {
                break;
            }
            SDL_Delay(10);
        }

        // Give backend device buffer a moment to drain after stream queue reaches zero.
        SDL_Delay(20);

        SDL_DestroyAudioStream(stream);
        if (!audio_was_initialized) {
            SDL_QuitSubSystem(SDL_INIT_AUDIO);
        }
        return true;
    }

} // namespace qwen3_tts

