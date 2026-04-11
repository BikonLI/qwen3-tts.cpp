#include "gguf_loader.h"
#include "qwen3_tts.h"

#include <SDL3/SDL.h>

#include <chrono>
#include <cmath>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <algorithm>

#ifdef __APPLE__
#include <mach/mach.h>
#elif defined(__linux__)
#include <sys/resource.h>
#else
#endif

namespace qwen3_tts {

    namespace fs = std::filesystem;

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

    Qwen3TTS::Qwen3TTS() = default;

    Qwen3TTS::~Qwen3TTS() = default;

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
        transformer_.unload_model();
        audio_decoder_.unload_model();
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

            if (!tokenizer_.load_from_gguf(loader.get_ctx())) {
                error_msg_ = "Failed to load text tokenizer: " + tokenizer_.get_error();
                return false;
            }
            fprintf(
                stderr, "  Text tokenizer loaded: vocab_size=%d (%lld ms)\n", tokenizer_.get_config().vocab_size,
                (long long)(get_time_ms() - t_tokenizer_start)
            );
        }
        log_memory_usage("load/after-tokenizer");

        // Speaker encoder is loaded lazily on first voice cloning request.
        fprintf(stderr, "  Speaker encoder: deferred (lazy load)\n");

        // Load TTS transformer from TTS model
        int64_t t_transformer_start = get_time_ms();
        if (!transformer_.load_model(tts_model_path)) {
            error_msg_ = "Failed to load TTS transformer: " + transformer_.get_error();
            return false;
        }
        transformer_loaded_ = true;
        loaded_hidden_size_ = transformer_.get_config().hidden_size;
        fprintf(
            stderr, "  TTS transformer loaded: hidden_size=%d, n_layers=%d (%lld ms)\n",
            transformer_.get_config().hidden_size, transformer_.get_config().n_layers,
            (long long)(get_time_ms() - t_transformer_start)
        );
        fprintf(stderr, "  Inferred model variant: %s\n", variant_name(loaded_model_variant_));
        log_memory_usage("load/after-transformer");

        if (!low_mem_mode_) {
            // Load vocoder (audio decoder) from tokenizer model
            fprintf(stderr, "Loading vocoder from %s...\n", tokenizer_model_path.c_str());
            int64_t t_decoder_start = get_time_ms();
            if (!audio_decoder_.load_model(tokenizer_model_path)) {
                error_msg_ = "Failed to load vocoder: " + audio_decoder_.get_error();
                return false;
            }
            decoder_loaded_ = true;
            fprintf(
                stderr, "  Vocoder loaded: sample_rate=%d, n_codebooks=%d (%lld ms)\n",
                audio_decoder_.get_config().sample_rate, audio_decoder_.get_config().n_codebooks,
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
        std::vector<float> zero_embedding(transformer_.get_config().hidden_size, 0.0f);

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
            if (!audio_encoder_.load_model(tts_model_path_)) {
                result.error_msg = "Failed to load speaker encoder: " + audio_encoder_.get_error();
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

        int64_t t_encode_start = get_time_ms();
        std::vector<float> speaker_embedding;

        if (!audio_encoder_.encode(ref_samples, n_ref_samples, speaker_embedding)) {
            result.error_msg = "Failed to extract speaker embedding: " + audio_encoder_.get_error();
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
            if (!audio_encoder_.load_model(tts_model_path_)) {
                error_msg_ = "Failed to load speaker encoder: " + audio_encoder_.get_error();
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

        if (!audio_encoder_.encode(ref_samples, n_ref_samples, embedding)) {
            error_msg_ = "Failed to extract speaker embedding: " + audio_encoder_.get_error();
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
        std::vector<int32_t> text_tokens = tokenizer_.encode_for_tts(
            text,
            "",
            params.speaker,
            params.reference_text
        );
        std::vector<int32_t> instruct_tokens = tokenizer_.encode_instruct_for_tts(params.instruct);
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
            if (!transformer_.load_model(tts_model_path_)) {
                result.error_msg = "Failed to reload TTS transformer: " + transformer_.get_error();
                return result;
            }
            transformer_loaded_ = true;
            if (params.print_timing) {
                fprintf(stderr, "  Transformer reloaded in %lld ms\n", (long long)(get_time_ms() - t_reload_start));
                sample_memory("synth/after-transformer-reload");
            }
        }
        transformer_.clear_kv_cache();

        const float * generation_speaker_embedding = speaker_embedding;
        if (speaker_codec_id >= 0) {
            generation_speaker_embedding = nullptr;
        }

        std::vector<int32_t> speech_codes;
        if (!transformer_.generate(
            text_tokens.data(), (int32_t)text_tokens.size(), generation_speaker_embedding,
            params.max_audio_tokens,
            speech_codes, effective_language_id, speaker_codec_id,
            params.repetition_penalty, params.temperature, params.top_k,
                params.top_p,
                instruct_tokens.empty() ? nullptr : instruct_tokens.data(),
                (int32_t)instruct_tokens.size()
            )) {
            result.error_msg = "Failed to generate speech codes: " + transformer_.get_error();
            return result;
        }
        result.t_generate_ms = get_time_ms() - t_generate_start;
        sample_memory("synth/after-generate");

        int n_codebooks = transformer_.get_config().n_codebooks;
        int n_frames = (int)speech_codes.size() / n_codebooks;

        if (params.print_progress) {
            fprintf(stderr, "Speech codes generated: %d frames x %d codebooks\n", n_frames, n_codebooks);
        }

        if (n_frames == 0) {
            result.error_msg = "No speech codes generated";
            return result;
        }

        if (low_mem_mode_) {
            transformer_.unload_model();
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
            if (!audio_decoder_.load_model(decoder_model_path_)) {
                result.error_msg = "Failed to load vocoder: " + audio_decoder_.get_error();
                return result;
            }
            decoder_loaded_ = true;
            if (params.print_timing) {
                fprintf(
                    stderr, "  Vocoder lazy-loaded in %lld ms\n", (long long)(get_time_ms() - t_decoder_load_start)
                );
                sample_memory("synth/after-vocoder-load");
            }
        }

        if (!audio_decoder_.decode(speech_codes.data(), n_frames, result.audio)) {
            result.error_msg = "Failed to decode speech codes: " + audio_decoder_.get_error();
            return result;
        }
        result.t_decode_ms = get_time_ms() - t_decode_start;
        sample_memory("synth/after-decode");

        if (low_mem_mode_) {
            audio_decoder_.unload_model();
            decoder_loaded_ = false;
            sample_memory("synth/after-vocoder-unload");
        }

        result.sample_rate = audio_decoder_.get_config().sample_rate;
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
        out.language_id = params.language_id;
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

