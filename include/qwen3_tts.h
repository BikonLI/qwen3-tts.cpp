#pragma once

#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace qwen3_tts
{

    class TextTokenizer;
    class TTSTransformer;
    class AudioTokenizerEncoder;
    class AudioTokenizerDecoder;

    enum class tts_task_type
    {
        auto_task = 0,
        voice_clone = 1,
        custom_voice = 2,
        voice_design = 3,
    };

    enum class tts_model_variant
    {
        auto_variant = 0,
        base = 1,
        custom_voice = 2,
        voice_design = 3,
    };

    // Concrete model families supported by this library
    enum class tts_model_id
    {
        model_06b_base = 0,
        model_06b_custom_voice = 1,
        model_1_7b_base = 2,
        model_1_7b_custom_voice = 3,
        model_1_7b_voice_design = 4,
    };

    // TTS generation parameters
    struct tts_params
    {
        // Maximum number of audio tokens to generate
        int32_t max_audio_tokens = 4096;

        // Temperature for sampling (0 = greedy)
        float temperature = 0.9f;

        // Top-p sampling
        float top_p = 1.0f;

        // Top-k sampling (0 = disabled)
        int32_t top_k = 50;

        // Number of threads
        int32_t n_threads = 4;

        // Print progress during generation
        bool print_progress = false;

        // Print timing information
        bool print_timing = true;

        // Repetition penalty for CB0 token generation (HuggingFace style)
        float repetition_penalty = 1.05f;

        // Language ID for codec (2050=en, 2069=ru, 2055=zh, 2058=ja, 2064=ko, 2053=de, 2061=fr, 2054=es)
        // Set to -1 for auto/no-language-control path.
        int32_t language_id = -1;

        // Task and model variant routing
        tts_task_type task_type = tts_task_type::auto_task;
        tts_model_variant model_variant = tts_model_variant::auto_variant;

        // Instruction controls
        std::string instruct;
        std::string speaker;
        std::string reference_text;

        // Base model option: when true, run with x-vector only and do not require reference_text.
        bool x_vector_only_mode = false;
    };

    // Common generation controls shared by all typed APIs below.
    struct tts_common_params
    {
        int32_t max_audio_tokens = 4096;
        float temperature = 0.9f;
        float top_p = 1.0f;
        int32_t top_k = 50;
        int32_t n_threads = 4;
        bool print_progress = false;
        bool print_timing = true;
        float repetition_penalty = 1.05f;
        int32_t language_id = -1;
    };

    // Voice clone params for Base model families (0.6B/1.7B).
    struct tts_voice_clone_params
    {
        tts_common_params common;
        std::string reference_text;
        bool x_vector_only_mode = false;
    };

    // CustomVoice params without instruct (0.6B and 1.7B no-instruct path).
    struct tts_custom_voice_params
    {
        tts_common_params common;
        std::string speaker;
    };

    // 1.7B CustomVoice params with optional instruct.
    struct tts_custom_voice_instruct_params
    {
        tts_common_params common;
        std::string speaker;
        std::string instruct;
    };

    // 1.7B VoiceDesign params (instruct required).
    struct tts_voice_design_params
    {
        tts_common_params common;
        std::string instruct;
    };

    // TTS generation result
    struct tts_result
    {
        // Generated audio samples (24kHz, mono)
        std::vector<float> audio;

        // Sample rate
        int32_t sample_rate = 24000;

        // Success flag
        bool success = false;

        // Error message if failed
        std::string error_msg;

        // Timing info (in milliseconds)
        int64_t t_load_ms = 0;
        int64_t t_tokenize_ms = 0;
        int64_t t_encode_ms = 0;
        int64_t t_generate_ms = 0;
        int64_t t_decode_ms = 0;
        int64_t t_total_ms = 0;

        // Process memory snapshots (bytes)
        uint64_t mem_rss_start_bytes = 0;
        uint64_t mem_rss_end_bytes = 0;
        uint64_t mem_rss_peak_bytes = 0;
        uint64_t mem_phys_start_bytes = 0;
        uint64_t mem_phys_end_bytes = 0;
        uint64_t mem_phys_peak_bytes = 0;
    };

    // Progress callback type
    using tts_progress_callback_t = std::function<void(int tokens_generated, int max_tokens)>;

    struct tts_stream_params
    {
        tts_params tts;

        // Number of code frames to aggregate before emitting one audio chunk.
        int32_t stream_chunk_frames = 4;

        // Bounded output queue capacity for poll mode.
        int32_t stream_queue_capacity = 8;

        // Default poll timeout in milliseconds when caller passes timeout < 0.
        int32_t stream_poll_timeout_ms = 100;

        // Run end-of-stream full flush before finish.
        bool stream_full_flush = true;

        // Decoder left-context frames for chunked incremental decode.
        // Official tokenizer chunk decode uses 25 by default.
        int32_t stream_decoder_left_context_frames = 25;
    };

    struct tts_audio_chunk
    {
        std::vector<float> audio;
        int32_t sample_rate = 24000;
        int64_t chunk_id = 0;
        bool is_last = false;
        int32_t dropped_before = 0;
    };

    using tts_stream_audio_callback_t = std::function<void(const tts_audio_chunk &chunk)>;
    using tts_stream_codes_callback_t =
        std::function<void(const int32_t *frame_codes, int32_t n_codebooks, int32_t frame_index)>;
    using tts_stream_finish_callback_t = std::function<void()>;
    using tts_stream_error_callback_t = std::function<void(const std::string &error_msg)>;

    struct tts_stream_callbacks
    {
        tts_stream_audio_callback_t on_audio_chunk;
        tts_stream_codes_callback_t on_codes_frame;
        tts_stream_finish_callback_t on_finish;
        tts_stream_error_callback_t on_error;
    };

    enum class tts_stream_poll_status
    {
        chunk = 0,
        timeout = 1,
        finished = 2,
        error = 3,
    };

    class tts_stream_session
    {
    public:
        struct impl;

        ~tts_stream_session();

        tts_stream_poll_status poll(tts_audio_chunk &chunk, int32_t timeout_ms = -1);

        bool is_finished() const;
        bool has_error() const;
        std::string get_error() const;
        int32_t dropped_chunks() const;

    private:
        explicit tts_stream_session(const std::shared_ptr<impl> &impl_ptr);

        std::shared_ptr<impl> impl_;
        friend class Qwen3TTS;
    };

    // Main TTS class that orchestrates the full pipeline
    class Qwen3TTS
    {
    public:
        Qwen3TTS();
        ~Qwen3TTS();

        // Load all models from directory
        // model_dir should contain: transformer.gguf, tokenizer.gguf, vocoder.gguf
        bool load_models(const std::string &model_dir);

        // Load model by explicit file paths (for shared tokenizer reuse).
        bool load_models_from_files(const std::string &tts_model_path,
                        const std::string &tokenizer_model_path);

        // Generate speech from text
        // text: input text to synthesize
        // params: generation parameters
        tts_result synthesize(const std::string &text,
                              const tts_params &params = tts_params());

        // Generate speech with voice cloning
        // text: input text to synthesize
        // reference_audio: path to reference audio file (WAV, 24kHz)
        // params: generation parameters
        tts_result synthesize_with_voice(const std::string &text,
                                         const std::string &reference_audio,
                                         const tts_params &params = tts_params());

        // Generate speech with voice cloning from samples
        // text: input text to synthesize
        // ref_samples: reference audio samples (24kHz, mono, normalized to [-1, 1])
        // n_ref_samples: number of reference samples
        // params: generation parameters
        tts_result synthesize_with_voice(const std::string &text,
                                         const float *ref_samples, int32_t n_ref_samples,
                                         const tts_params &params = tts_params());

        // Extract speaker embedding from raw audio samples (for caching)
        // ref_samples: 24kHz mono float32 normalized to [-1, 1]
        // embedding: output vector (resized to hidden_size, typically 1024)
        // Returns true on success
        bool extract_speaker_embedding(const float *ref_samples, int32_t n_ref_samples,
                                       std::vector<float> &embedding,
                                       const tts_params &params = tts_params());

        // Synthesize with pre-computed speaker embedding (skips encoder)
        // embedding: speaker embedding from extract_speaker_embedding()
        // embedding_size: must match hidden_size (typically 1024)
        tts_result synthesize_with_embedding(const std::string &text,
                                             const float *embedding, int32_t embedding_size,
                                             const tts_params &params = tts_params());

        // Set progress callback
        void set_progress_callback(tts_progress_callback_t callback);

        // Stream synthesis (whole text input + chunked audio output).
        std::shared_ptr<tts_stream_session> synthesize_stream(
            const std::string &text,
            const tts_stream_params &params = tts_stream_params(),
            const tts_stream_callbacks &callbacks = tts_stream_callbacks()
        );

        std::shared_ptr<tts_stream_session> synthesize_with_voice_stream(
            const std::string &text,
            const std::string &reference_audio,
            const tts_stream_params &params = tts_stream_params(),
            const tts_stream_callbacks &callbacks = tts_stream_callbacks()
        );

        std::shared_ptr<tts_stream_session> synthesize_with_voice_stream(
            const std::string &text,
            const float *ref_samples,
            int32_t n_ref_samples,
            const tts_stream_params &params = tts_stream_params(),
            const tts_stream_callbacks &callbacks = tts_stream_callbacks()
        );

        std::shared_ptr<tts_stream_session> synthesize_with_embedding_stream(
            const std::string &text,
            const float *embedding,
            int32_t embedding_size,
            const tts_stream_params &params = tts_stream_params(),
            const tts_stream_callbacks &callbacks = tts_stream_callbacks()
        );

        // Get error message
        const std::string &get_error() const { return error_msg_; }

        // Check if models are loaded
        bool is_loaded() const { return models_loaded_; }

        tts_model_variant loaded_model_variant() const { return loaded_model_variant_; }

        int32_t loaded_hidden_size() const { return loaded_hidden_size_; }

    private:
        tts_result synthesize_internal(const std::string &text,
                                       const float *speaker_embedding,
                                       const tts_params &params,
                                       tts_result &result);

        std::shared_ptr<tts_stream_session> synthesize_stream_internal(
            const std::string &text,
            const std::vector<float> &speaker_embedding,
            bool has_speaker_embedding,
            const tts_stream_params &params,
            const tts_stream_callbacks &callbacks
        );

        std::unique_ptr<TextTokenizer> tokenizer_;
        std::unique_ptr<TTSTransformer> transformer_;
        std::unique_ptr<AudioTokenizerEncoder> audio_encoder_;
        std::unique_ptr<AudioTokenizerDecoder> audio_decoder_;

        bool models_loaded_ = false;
        bool encoder_loaded_ = false;
        bool transformer_loaded_ = false;
        bool decoder_loaded_ = false;
        bool low_mem_mode_ = false;
        tts_model_variant loaded_model_variant_ = tts_model_variant::auto_variant;
        int32_t loaded_hidden_size_ = 0;
        std::string error_msg_;
        std::string tts_model_path_;
        std::string decoder_model_path_;
        tts_progress_callback_t progress_callback_;
        mutable std::mutex stream_mutex_;
        bool stream_active_ = false;
        std::shared_ptr<tts_stream_session::impl> stream_impl_;
    };

    // Multi-model manager with model-specific load/synthesis APIs.
    class Qwen3TTSModelHub
    {
    public:
        Qwen3TTSModelHub();
        ~Qwen3TTSModelHub();

        // Register shared Qwen3-TTS-Tokenizer-12Hz GGUF (file path or directory).
        // When set, all load_models_* calls reuse this tokenizer path.
        bool set_shared_tokenizer_12hz(const std::string &tokenizer_model_path_or_dir);

        // Model-specific loaders
        bool load_models_06b_base(const std::string &model_dir);
        bool load_models_06b_custom_voice(const std::string &model_dir);
        bool load_models_1_7b_base(const std::string &model_dir);
        bool load_models_1_7b_custom_voice(const std::string &model_dir);
        bool load_models_1_7b_voice_design(const std::string &model_dir);

        // Model state helpers
        bool unload_model(tts_model_id model_id);
        bool is_model_loaded(tts_model_id model_id) const;
        std::vector<tts_model_id> loaded_models() const;

        // 0.6B Base (voice clone)
        tts_result synthesize_06b_base_with_voice(const std::string &text,
                                                  const std::string &reference_audio,
                                                  const tts_voice_clone_params &params = tts_voice_clone_params());
        tts_result synthesize_06b_base_with_embedding(const std::string &text,
                                                      const float *embedding, int32_t embedding_size,
                                                      const tts_common_params &params = tts_common_params());

        // 1.7B Base (voice clone)
        tts_result synthesize_1_7b_base_with_voice(const std::string &text,
                                                   const std::string &reference_audio,
                                                   const tts_voice_clone_params &params = tts_voice_clone_params());
        tts_result synthesize_1_7b_base_with_embedding(const std::string &text,
                                                       const float *embedding, int32_t embedding_size,
                                                       const tts_common_params &params = tts_common_params());

        // 0.6B CustomVoice (no instruct API)
        tts_result synthesize_06b_custom_voice(const std::string &text,
                                               const tts_custom_voice_params &params = tts_custom_voice_params());

        // 1.7B CustomVoice
        tts_result synthesize_1_7b_custom_voice(const std::string &text,
                                                const tts_custom_voice_params &params = tts_custom_voice_params());
        tts_result synthesize_1_7b_custom_voice_with_instruct(
            const std::string &text,
            const tts_custom_voice_instruct_params &params = tts_custom_voice_instruct_params()
        );

        // 1.7B VoiceDesign
        tts_result synthesize_1_7b_voice_design(const std::string &text,
                                                const tts_voice_design_params &params = tts_voice_design_params());

        // Convenience routers with unified naming.
        tts_result synthesize_with_voice(tts_model_id model_id,
                                         const std::string &text,
                                         const std::string &reference_audio,
                                         const tts_voice_clone_params &params = tts_voice_clone_params());
        tts_result synthesize_with_instruct(tts_model_id model_id,
                                            const std::string &text,
                                            const std::string &instruct,
                                            const tts_custom_voice_params &params = tts_custom_voice_params());

        const std::string &get_error() const { return error_msg_; }

    private:
        struct model_slot;

        bool load_model_internal(tts_model_id model_id,
                                 const std::string &model_dir,
                                 tts_model_variant expected_variant,
                                 int32_t expected_hidden_size);

        tts_params build_common_tts_params(const tts_common_params &params) const;
        model_slot *find_slot(tts_model_id model_id);
        const model_slot *find_slot(tts_model_id model_id) const;

        std::map<tts_model_id, std::unique_ptr<model_slot>> models_;
        std::string shared_tokenizer_path_;
        std::string error_msg_;
    };

    // Utility: Load audio file (WAV format)
    bool load_audio_file(const std::string &path, std::vector<float> &samples,
                         int &sample_rate);

    // Utility: Save audio file (WAV format)
    bool save_audio_file(const std::string &path, const std::vector<float> &samples,
                         int sample_rate);

    // Utility: Play audio (platform-dependent implementation)
    bool play_audio(const std::vector<float> &samples, int sample_rate);

} // namespace qwen3_tts
