#include "qwen3_tts.h"

#include <SDL3/SDL.h>

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>
#include <chrono>

namespace {

namespace fs = std::filesystem;

static std::string to_lower_copy(const std::string &s) {
    std::string out = s;
    std::transform(out.begin(), out.end(), out.begin(), [](unsigned char c) { return (char)std::tolower(c); });
    return out;
}

static bool resolve_gguf_file_path(
    const std::string &path,
    const char *arg_name,
    std::string &resolved,
    std::string &error_msg
) {
    resolved.clear();
    error_msg.clear();

    std::error_code ec;
    if (!fs::exists(path, ec)) {
        error_msg = std::string(arg_name) + " path does not exist: " + path;
        return false;
    }

    if (!fs::is_regular_file(path, ec)) {
        error_msg = std::string(arg_name) + " must be a GGUF file path (directories are not supported): " + path;
        return false;
    }

    if (to_lower_copy(fs::path(path).extension().string()) != ".gguf") {
        error_msg = std::string(arg_name) + " path is not a GGUF file: " + path;
        return false;
    }

    resolved = path;
    return true;
}

static qwen3_tts::tts_model_variant infer_model_variant_from_path(const std::string &path) {
    const std::string lower = to_lower_copy(path);
    if (lower.find("voicedesign") != std::string::npos) {
        return qwen3_tts::tts_model_variant::voice_design;
    }
    if (lower.find("customvoice") != std::string::npos) {
        return qwen3_tts::tts_model_variant::custom_voice;
    }
    if (lower.find("base") != std::string::npos) {
        return qwen3_tts::tts_model_variant::base;
    }
    return qwen3_tts::tts_model_variant::auto_variant;
}

static bool parse_language_id(const std::string &lang, int32_t &out) {
    if (lang == "auto") { out = -1; return true; }
    if (lang == "en" || lang == "english") { out = 2050; return true; }
    if (lang == "ru" || lang == "russian") { out = 2069; return true; }
    if (lang == "zh" || lang == "chinese") { out = 2055; return true; }
    if (lang == "ja" || lang == "japanese") { out = 2058; return true; }
    if (lang == "ko" || lang == "korean") { out = 2064; return true; }
    if (lang == "de" || lang == "german") { out = 2053; return true; }
    if (lang == "fr" || lang == "french") { out = 2061; return true; }
    if (lang == "es" || lang == "spanish") { out = 2054; return true; }
    if (lang == "it" || lang == "italian") { out = 2070; return true; }
    if (lang == "pt" || lang == "portuguese") { out = 2071; return true; }
    if (lang == "beijing_dialect") { out = 2074; return true; }
    if (lang == "sichuan_dialect") { out = 2062; return true; }
    return false;
}

static bool load_text_or_file(const std::string &input, std::string &out_text) {
    out_text = input;
#ifdef _WIN32
    FILE *f = nullptr;
    if (fopen_s(&f, input.c_str(), "rb") != 0) {
        f = nullptr;
    }
#else
    FILE *f = fopen(input.c_str(), "rb");
#endif
    if (!f) {
        return true;
    }

    if (fseek(f, 0, SEEK_END) != 0) {
        fclose(f);
        return false;
    }
    long size = ftell(f);
    if (size < 0) {
        fclose(f);
        return false;
    }
    if (fseek(f, 0, SEEK_SET) != 0) {
        fclose(f);
        return false;
    }

    std::string content((size_t)size, '\0');
    if (size > 0) {
        size_t n = fread(&content[0], 1, (size_t)size, f);
        if (n != (size_t)size) {
            fclose(f);
            return false;
        }
    }
    fclose(f);
    out_text = std::move(content);
    return true;
}

static const char *variant_name(qwen3_tts::tts_model_variant v) {
    switch (v) {
        case qwen3_tts::tts_model_variant::base: return "base";
        case qwen3_tts::tts_model_variant::custom_voice: return "custom_voice";
        case qwen3_tts::tts_model_variant::voice_design: return "voice_design";
        default: return "auto";
    }
}

class WavStreamWriter {
public:
    ~WavStreamWriter() {
        close();
    }

    bool open(const std::string &path, int32_t sample_rate, std::string &error_msg) {
        close();
        if (sample_rate <= 0) {
            error_msg = "Invalid sample rate for WAV writer";
            return false;
        }

#ifdef _WIN32
        if (fopen_s(&file_, path.c_str(), "wb") != 0) {
            file_ = nullptr;
        }
#else
        file_ = fopen(path.c_str(), "wb");
#endif
        if (!file_) {
            error_msg = "Failed to open WAV output: " + path;
            return false;
        }

        path_ = path;
        sample_rate_ = sample_rate;
        data_bytes_ = 0;
        if (!write_header(0, error_msg)) {
            close();
            return false;
        }
        return true;
    }

    bool append(const std::vector<float> &samples, std::string &error_msg) {
        if (!file_ || samples.empty()) {
            return true;
        }

        for (float s : samples) {
            if (s > 1.0f) s = 1.0f;
            if (s < -1.0f) s = -1.0f;
            const int16_t pcm = (int16_t)(s * 32767.0f);
            if (fwrite(&pcm, sizeof(int16_t), 1, file_) != 1) {
                error_msg = "Failed writing WAV payload: " + path_;
                return false;
            }
            data_bytes_ += sizeof(int16_t);
        }

        return true;
    }

    bool close() {
        if (!file_) {
            return true;
        }

        std::string error_msg;
        if (!write_header(data_bytes_, error_msg)) {
            fclose(file_);
            file_ = nullptr;
            return false;
        }

        fclose(file_);
        file_ = nullptr;
        return true;
    }

private:
    bool write_header(uint32_t data_size, std::string &error_msg) {
        if (!file_) {
            error_msg = "WAV file is not open";
            return false;
        }

        if (fseek(file_, 0, SEEK_SET) != 0) {
            error_msg = "Failed to seek WAV file";
            return false;
        }

        const uint16_t num_channels = 1;
        const uint16_t bits_per_sample = 16;
        const uint16_t block_align = (uint16_t)(num_channels * bits_per_sample / 8);
        const uint32_t byte_rate = (uint32_t)(sample_rate_ * block_align);
        const uint32_t file_size = 36u + data_size;
        const uint16_t audio_format = 1;
        const uint32_t fmt_size = 16;

        if (fwrite("RIFF", 1, 4, file_) != 4 ||
            fwrite(&file_size, 4, 1, file_) != 1 ||
            fwrite("WAVE", 1, 4, file_) != 4 ||
            fwrite("fmt ", 1, 4, file_) != 4 ||
            fwrite(&fmt_size, 4, 1, file_) != 1 ||
            fwrite(&audio_format, 2, 1, file_) != 1 ||
            fwrite(&num_channels, 2, 1, file_) != 1 ||
            fwrite(&sample_rate_, 4, 1, file_) != 1 ||
            fwrite(&byte_rate, 4, 1, file_) != 1 ||
            fwrite(&block_align, 2, 1, file_) != 1 ||
            fwrite(&bits_per_sample, 2, 1, file_) != 1 ||
            fwrite("data", 1, 4, file_) != 4 ||
            fwrite(&data_size, 4, 1, file_) != 1) {
            error_msg = "Failed writing WAV header: " + path_;
            return false;
        }

        if (fseek(file_, 0, SEEK_END) != 0) {
            error_msg = "Failed to seek WAV end";
            return false;
        }
        return true;
    }

    FILE *file_ = nullptr;
    std::string path_;
    int32_t sample_rate_ = 24000;
    uint32_t data_bytes_ = 0;
};

class StreamPlayer {
public:
    ~StreamPlayer() {
        close();
    }

    bool open(int32_t sample_rate, std::string &error_msg) {
        if (stream_) {
            return true;
        }
        if (sample_rate <= 0) {
            error_msg = "Invalid sample rate for stream playback";
            return false;
        }

        audio_was_initialized_ = (SDL_WasInit(SDL_INIT_AUDIO) & SDL_INIT_AUDIO) != 0;
        if (!audio_was_initialized_) {
            if (!SDL_InitSubSystem(SDL_INIT_AUDIO)) {
                error_msg = std::string("SDL_InitSubSystem(SDL_INIT_AUDIO) failed: ") + SDL_GetError();
                return false;
            }
            initialized_audio_ = true;
        }

        SDL_AudioSpec spec = {};
        spec.format = SDL_AUDIO_F32;
        spec.channels = 1;
        spec.freq = sample_rate;

        stream_ = SDL_OpenAudioDeviceStream(SDL_AUDIO_DEVICE_DEFAULT_PLAYBACK, &spec, nullptr, nullptr);
        if (!stream_) {
            error_msg = std::string("SDL_OpenAudioDeviceStream failed: ") + SDL_GetError();
            close();
            return false;
        }

        if (!SDL_ResumeAudioStreamDevice(stream_)) {
            error_msg = std::string("SDL_ResumeAudioStreamDevice failed: ") + SDL_GetError();
            close();
            return false;
        }

        return true;
    }

    bool push(const std::vector<float> &samples, std::string &error_msg) {
        if (!stream_ || samples.empty()) {
            return true;
        }

        const int byte_len = (int)(samples.size() * sizeof(float));
        if (!SDL_PutAudioStreamData(stream_, samples.data(), byte_len)) {
            error_msg = std::string("SDL_PutAudioStreamData failed: ") + SDL_GetError();
            return false;
        }
        return true;
    }

    int queued_bytes() const {
        if (!stream_) {
            return 0;
        }
        return SDL_GetAudioStreamQueued(stream_);
    }

    bool finish(std::string &error_msg) {
        if (!stream_) {
            return true;
        }

        if (!SDL_FlushAudioStream(stream_)) {
            error_msg = std::string("SDL_FlushAudioStream failed: ") + SDL_GetError();
            close();
            return false;
        }

        while (true) {
            const int queued = SDL_GetAudioStreamQueued(stream_);
            if (queued < 0) {
                error_msg = std::string("SDL_GetAudioStreamQueued failed: ") + SDL_GetError();
                close();
                return false;
            }
            if (queued == 0) {
                break;
            }
            SDL_Delay(10);
        }
        SDL_Delay(20);
        close();
        return true;
    }

private:
    void close() {
        if (stream_) {
            SDL_DestroyAudioStream(stream_);
            stream_ = nullptr;
        }
        if (initialized_audio_) {
            SDL_QuitSubSystem(SDL_INIT_AUDIO);
            initialized_audio_ = false;
        }
    }

    SDL_AudioStream *stream_ = nullptr;
    bool audio_was_initialized_ = false;
    bool initialized_audio_ = false;
};

struct StreamRunStats {
    size_t n_samples = 0;
    int32_t sample_rate = 24000;
    int32_t dropped_chunks = 0;
    int64_t wall_ms = 0;
};

static bool run_stream_session(
    const std::shared_ptr<qwen3_tts::tts_stream_session> &session,
    const std::string &output_file,
    int32_t poll_timeout_ms,
    bool playback_enabled,
    StreamRunStats &stats,
    std::string &error_msg
) {
    error_msg.clear();
    stats = {};
    if (!session) {
        error_msg = "Stream session is null";
        return false;
    }

    const char *stream_dbg_env = std::getenv("QWEN3_TTS_STREAM_DEBUG");
    bool stream_dbg = false;
    if (stream_dbg_env && stream_dbg_env[0] != '\0') {
        std::string v = stream_dbg_env;
        std::transform(v.begin(), v.end(), v.begin(), [](unsigned char c) { return (char)std::tolower(c); });
        stream_dbg = !(v == "0" || v == "false" || v == "off" || v == "no");
    }

    int64_t poll_ms = 0;
    int64_t push_ms = 0;
    int64_t wav_ms = 0;
    int64_t n_polls = 0;
    int64_t n_poll_timeouts = 0;
    int64_t n_chunks = 0;
    int64_t total_chunk_samples = 0;
    int64_t player_q_peak_bytes = 0;
    int64_t player_q_last_bytes = 0;
    int64_t player_q_probe_errors = 0;
    int64_t player_underrun_events = 0;
    bool player_prev_empty = false;
    auto t_stream_start = std::chrono::steady_clock::now();

    StreamPlayer player;
    WavStreamWriter writer;
    bool writer_opened = false;
    bool player_opened = false;

    while (true) {
        qwen3_tts::tts_audio_chunk chunk;
        auto t_poll0 = std::chrono::steady_clock::now();
        qwen3_tts::tts_stream_poll_status status = session->poll(chunk, poll_timeout_ms);
        auto t_poll1 = std::chrono::steady_clock::now();
        poll_ms += std::chrono::duration_cast<std::chrono::milliseconds>(t_poll1 - t_poll0).count();
        n_polls++;

        if (status == qwen3_tts::tts_stream_poll_status::timeout) {
            n_poll_timeouts++;
            if (playback_enabled && player_opened) {
                const int queued_bytes = player.queued_bytes();
                if (queued_bytes >= 0) {
                    player_q_last_bytes = queued_bytes;
                    player_q_peak_bytes = std::max<int64_t>(player_q_peak_bytes, queued_bytes);
                    const bool now_empty = queued_bytes == 0;
                    if (now_empty && !player_prev_empty) {
                        player_underrun_events++;
                    }
                    player_prev_empty = now_empty;
                } else {
                    player_q_probe_errors++;
                }
            }
            continue;
        }

        if (status == qwen3_tts::tts_stream_poll_status::error) {
            error_msg = session->get_error();
            if (error_msg.empty()) {
                error_msg = "Unknown streaming error";
            }
            return false;
        }

        if (status == qwen3_tts::tts_stream_poll_status::finished) {
            break;
        }

        if (status == qwen3_tts::tts_stream_poll_status::chunk) {
            n_chunks++;
            if (chunk.sample_rate > 0) {
                stats.sample_rate = chunk.sample_rate;
            }

            if (!output_file.empty() && !writer_opened) {
                if (!writer.open(output_file, stats.sample_rate, error_msg)) {
                    return false;
                }
                writer_opened = true;
            }

            if (!chunk.audio.empty()) {
                if (playback_enabled && !player_opened) {
                    if (!player.open(stats.sample_rate, error_msg)) {
                        return false;
                    }
                    player_opened = true;
                }

                if (playback_enabled) {
                    auto t_push0 = std::chrono::steady_clock::now();
                    if (!player.push(chunk.audio, error_msg)) {
                        return false;
                    }
                    auto t_push1 = std::chrono::steady_clock::now();
                    push_ms += std::chrono::duration_cast<std::chrono::milliseconds>(t_push1 - t_push0).count();

                    const int queued_bytes = player.queued_bytes();
                    if (queued_bytes >= 0) {
                        player_q_last_bytes = queued_bytes;
                        player_q_peak_bytes = std::max<int64_t>(player_q_peak_bytes, queued_bytes);
                        player_prev_empty = queued_bytes == 0;
                    } else {
                        player_q_probe_errors++;
                    }
                }

                auto t_wav0 = std::chrono::steady_clock::now();
                if (writer_opened && !writer.append(chunk.audio, error_msg)) {
                    return false;
                }
                auto t_wav1 = std::chrono::steady_clock::now();
                wav_ms += std::chrono::duration_cast<std::chrono::milliseconds>(t_wav1 - t_wav0).count();

                stats.n_samples += chunk.audio.size();
                total_chunk_samples += (int64_t)chunk.audio.size();
            }

            stats.dropped_chunks = std::max(stats.dropped_chunks, chunk.dropped_before);

            if (stream_dbg) {
                const double chunk_ms = stats.sample_rate > 0
                    ? ((double)chunk.audio.size() * 1000.0 / (double)stats.sample_rate)
                    : 0.0;
                const double player_q_ms = (playback_enabled && stats.sample_rate > 0)
                    ? ((double)(player_q_last_bytes / (int64_t)sizeof(float)) * 1000.0 / (double)stats.sample_rate)
                    : 0.0;
                fprintf(
                    stderr,
                    "[stream-cli] chunk=%lld samples=%zu dur_ms=%.1f dropped_before=%d player_q_ms=%.1f\n",
                    (long long)chunk.chunk_id,
                    chunk.audio.size(),
                    chunk_ms,
                    chunk.dropped_before,
                    player_q_ms
                );
            }
            continue;
        }
    }

    if (player_opened && !player.finish(error_msg)) {
        return false;
    }

    if (writer_opened && !writer.close()) {
        error_msg = "Failed to finalize stream WAV file";
        return false;
    }

    stats.dropped_chunks = std::max(stats.dropped_chunks, session->dropped_chunks());
    auto t_stream_end = std::chrono::steady_clock::now();
    stats.wall_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_stream_end - t_stream_start).count();

    if (stream_dbg) {
        int64_t total_ms = stats.wall_ms;
        double avg_chunk_ms = 0.0;
        if (n_chunks > 0 && stats.sample_rate > 0) {
            avg_chunk_ms = ((double)total_chunk_samples * 1000.0 / (double)stats.sample_rate) / (double)n_chunks;
        }
        const double player_q_peak_ms = (playback_enabled && stats.sample_rate > 0)
            ? ((double)(player_q_peak_bytes / (int64_t)sizeof(float)) * 1000.0 / (double)stats.sample_rate)
            : 0.0;
        fprintf(
            stderr,
            "[stream-cli] done total_ms=%lld polls=%lld poll_ms=%lld timeouts=%lld chunks=%lld avg_chunk_audio_ms=%.1f push_ms=%lld wav_ms=%lld player_q_peak_ms=%.1f player_q_probe_errors=%lld dropped=%d\n",
            (long long)total_ms,
            (long long)n_polls,
            (long long)poll_ms,
            (long long)n_poll_timeouts,
            (long long)n_chunks,
            avg_chunk_ms,
            (long long)push_ms,
            (long long)wav_ms,
            player_q_peak_ms,
            (long long)player_q_probe_errors,
            stats.dropped_chunks
        );
        if (playback_enabled) {
            fprintf(
                stderr,
                "[stream-cli] playback underrun_events=%lld\n",
                (long long)player_underrun_events
            );
        }
    }

    return true;
}

void print_usage(const char *program) {
    fprintf(stderr, "Usage: %s -m <tts_model_path.gguf> -mt <tokenizer_path.gguf> -t <text> [options]\n", program);
    fprintf(stderr, "\n");
    fprintf(stderr, "Required:\n");
    fprintf(stderr, "  -m,  --model <path>               Functional TTS model GGUF file path\n");
    fprintf(stderr, "  -mt, --model-tokenizer <path>     Tokenizer model GGUF file path\n");
    fprintf(stderr, "  -t,  --text <text|file>           Input text (or a file path)\n");
    fprintf(stderr, "\n");
    fprintf(stderr, "Optional:\n");
    fprintf(stderr, "  -o, --output <wav>                Output WAV file; omit to play audio (stream/non-stream)\n");
    fprintf(stderr, "  -r, --reference <wav>             Reference audio for voice clone\n");
    fprintf(stderr, "  --ref-text <text>                 Reference transcript for voice clone\n");
    fprintf(stderr, "  --x-vector-only                   Voice clone without reference_text\n");
    fprintf(stderr, "  --speaker <name>                  CustomVoice speaker\n");
    fprintf(stderr, "  --instruct <text>                 Instruct text (CustomVoice/VoiceDesign)\n");
    fprintf(stderr, "  -l, --language <lang>             auto,en,ru,zh,ja,ko,de,fr,es,it,pt,beijing_dialect,sichuan_dialect\n");
    fprintf(stderr, "  --temperature <v>                 Sampling temperature (default: 0.9)\n");
    fprintf(stderr, "  --top-k <n>                       Top-k (default: 50)\n");
    fprintf(stderr, "  --top-p <v>                       Top-p (default: 1.0)\n");
    fprintf(stderr, "  --max-tokens <n>                  Max audio tokens (default: 4096)\n");
    fprintf(stderr, "  --repetition-penalty <v>          Repetition penalty (default: 1.05)\n");
    fprintf(stderr, "  --seed <n>                        Sampling seed (-1 = random, default: -1)\n");
    fprintf(stderr, "  --stream                          Enable streaming generation/playback\n");
    fprintf(stderr, "  --stream-realtime                 Enable realtime preset (RTF-first, safe sampling)\n");
    fprintf(stderr, "  --stream-no-playback              Disable playback (benchmark mode)\n");
    fprintf(stderr, "  --stream-playback                 Enable playback (default)\n");
    fprintf(stderr, "  --stream-chunk-frames <n>         Frames per streamed chunk (default: 12)\n");
    fprintf(stderr, "  --stream-left-context <n>         Decoder left context frames (default: 2)\n");
    fprintf(stderr, "  --stream-lookahead <n>            Decoder right lookahead frames (default: 4)\n");
    fprintf(stderr, "  --stream-talker-window <n>        Talker attention window for stream (default: 0=full)\n");
    fprintf(stderr, "  --talker-window <n>               Talker attention window for non-stream (default: 0=full)\n");
    fprintf(stderr, "  --stream-parallel-decode          Decode in parallel worker (default: on)\n");
    fprintf(stderr, "  --stream-no-parallel-decode       Disable parallel decode worker\n");
    fprintf(stderr, "  --stream-drop-oldest              Drop oldest chunk on queue overflow\n");
    fprintf(stderr, "  --stream-block-on-full            Block producer on queue overflow (default)\n");
    fprintf(stderr, "  --stream-adaptive                Enable adaptive chunk/context tuning\n");
    fprintf(stderr, "  --stream-no-adaptive             Disable adaptive chunk/context tuning\n");
    fprintf(stderr, "  --stream-chunk-min <n>           Adaptive chunk lower bound (default: 2)\n");
    fprintf(stderr, "  --stream-chunk-max <n>           Adaptive chunk upper bound (default: 16)\n");
    fprintf(stderr, "  --stream-left-context-min <n>    Adaptive context lower bound (default: 2)\n");
    fprintf(stderr, "  --stream-adapt-high <ratio>      Expand chunk/reduce ctx threshold (default: 1.15)\n");
    fprintf(stderr, "  --stream-adapt-low <ratio>       Recover quality threshold (default: 0.65)\n");
    fprintf(stderr, "  --stream-queue-cap <n>            Stream queue capacity (default: 8)\n");
    fprintf(stderr, "  --stream-poll-timeout <ms>        Poll timeout for stream client (default: 100)\n");
    fprintf(stderr, "  -j, --threads <n>                 Number of threads\n");
    fprintf(stderr, "  -h, --help                        Show this help\n");
    fprintf(stderr, "\n");
    fprintf(stderr, "Auto task routing:\n");
    fprintf(stderr, "  - if --reference is set: voice_clone\n");
    fprintf(stderr, "  - else if --speaker is set: custom_voice\n");
    fprintf(stderr, "  - else if --instruct is set: voice_design (only when model is voice_design)\n");
    fprintf(stderr, "  - else: plain synthesize\n");
    fprintf(stderr, "\n");
    fprintf(stderr, "Examples:\n");
    fprintf(stderr, "  %s -m models/gguf/0.6b-base/qwen3-tts-12hz-0.6b-base-f16.gguf -mt models/gguf/tokenizer/qwen3-tts-tokenizer-12hz-f16.gguf -t \"Hello\" -r ref.wav --ref-text \"Hello\" -o clone.wav\n", program);
    fprintf(stderr, "  %s -m models/gguf/1.7b-custom-voice/qwen3-tts-12hz-1.7b-custom-voice-f16.gguf -mt models/gguf/tokenizer/qwen3-tts-tokenizer-12hz-f16.gguf -t \"Hello\" --speaker ryan --instruct \"happy\" -o custom.wav\n", program);
    fprintf(stderr, "  %s -m models/gguf/1.7b-voice-design/qwen3-tts-12hz-1.7b-voice-design-f16.gguf -mt models/gguf/tokenizer/qwen3-tts-tokenizer-12hz-f16.gguf -t \"Hello\" --instruct \"female voice, warm tone\" -o design.wav\n", program);
    fprintf(stderr, "  %s -m models/gguf/0.6b-custom-voice/qwen3-tts-12hz-0.6b-custom-voice-f16.gguf -mt models/gguf/tokenizer/qwen3-tts-tokenizer-12hz-f16.gguf -t \".\\text\\test1.txt\" --speaker ryan --stream\n", program);
}

} // namespace

int main(int argc, char **argv) {
    std::string model_path_arg;
    std::string tokenizer_path_arg;
    std::string text_input;
    std::string text;
    std::string output_file;
    std::string reference_audio;
    std::string reference_text;
    std::string speaker;
    std::string instruct;

    const bool stream_dbg = []() {
        const char *stream_dbg_env = std::getenv("QWEN3_TTS_STREAM_DEBUG");
        if (!stream_dbg_env || stream_dbg_env[0] == '\0') {
            return false;
        }
        std::string v = stream_dbg_env;
        std::transform(v.begin(), v.end(), v.begin(), [](unsigned char c) { return (char)std::tolower(c); });
        return !(v == "0" || v == "false" || v == "off" || v == "no");
    }();

    bool stream_mode = false;
    bool stream_realtime = false;
    bool stream_playback = true;
    bool stream_no_playback_requested = false;
    bool temperature_explicit = false;
    bool top_k_explicit = false;
    bool top_p_explicit = false;
    bool repetition_penalty_explicit = false;
    bool stream_left_context_explicit = false;
    bool stream_lookahead_explicit = false;
    int32_t stream_chunk_frames = 12;
    int32_t stream_queue_cap = 8;
    int32_t stream_poll_timeout = 100;
    int32_t stream_decoder_left_context = 2;
    int32_t stream_decoder_lookahead = 4;
    int32_t stream_talker_attention_window = 0;
    bool stream_parallel_decode = true;
    bool stream_drop_oldest_on_overflow = false;
    bool stream_adaptive_tuning = false;
    int32_t stream_chunk_min = 2;
    int32_t stream_chunk_max = 16;
    int32_t stream_left_context_min = 2;
    float stream_adapt_high = 1.15f;
    float stream_adapt_low = 0.65f;
    int32_t talker_attention_window = 0;

    qwen3_tts::tts_params params;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            return 0;
        } else if (arg == "-m" || arg == "--model") {
            if (++i >= argc) {
                fprintf(stderr, "Error: missing --model value\n");
                return 1;
            }
            model_path_arg = argv[i];
        } else if (arg == "-mt" || arg == "--model-tokenizer") {
            if (++i >= argc) {
                fprintf(stderr, "Error: missing --model-tokenizer value\n");
                return 1;
            }
            tokenizer_path_arg = argv[i];
        } else if (arg == "-t" || arg == "--text") {
            if (++i >= argc) {
                fprintf(stderr, "Error: missing --text value\n");
                return 1;
            }
            text_input = argv[i];
        } else if (arg == "-o" || arg == "--output") {
            if (++i >= argc) {
                fprintf(stderr, "Error: missing --output value\n");
                return 1;
            }
            output_file = argv[i];
        } else if (arg == "-r" || arg == "--reference") {
            if (++i >= argc) {
                fprintf(stderr, "Error: missing --reference value\n");
                return 1;
            }
            reference_audio = argv[i];
        } else if (arg == "--ref-text") {
            if (++i >= argc) {
                fprintf(stderr, "Error: missing --ref-text value\n");
                return 1;
            }
            reference_text = argv[i];
        } else if (arg == "--x-vector-only") {
            params.x_vector_only_mode = true;
        } else if (arg == "--speaker") {
            if (++i >= argc) {
                fprintf(stderr, "Error: missing --speaker value\n");
                return 1;
            }
            speaker = argv[i];
        } else if (arg == "--instruct") {
            if (++i >= argc) {
                fprintf(stderr, "Error: missing --instruct value\n");
                return 1;
            }
            instruct = argv[i];
        } else if (arg == "-l" || arg == "--language") {
            if (++i >= argc) {
                fprintf(stderr, "Error: missing --language value\n");
                return 1;
            }
            int32_t language_id = -1;
            if (!parse_language_id(argv[i], language_id)) {
                fprintf(stderr, "Error: unsupported language '%s'\n", argv[i]);
                return 1;
            }
            params.language_id = language_id;
        } else if (arg == "--temperature") {
            if (++i >= argc) {
                fprintf(stderr, "Error: missing --temperature value\n");
                return 1;
            }
            params.temperature = std::stof(argv[i]);
            temperature_explicit = true;
        } else if (arg == "--top-k") {
            if (++i >= argc) {
                fprintf(stderr, "Error: missing --top-k value\n");
                return 1;
            }
            params.top_k = std::stoi(argv[i]);
            top_k_explicit = true;
        } else if (arg == "--top-p") {
            if (++i >= argc) {
                fprintf(stderr, "Error: missing --top-p value\n");
                return 1;
            }
            params.top_p = std::stof(argv[i]);
            top_p_explicit = true;
        } else if (arg == "--max-tokens") {
            if (++i >= argc) {
                fprintf(stderr, "Error: missing --max-tokens value\n");
                return 1;
            }
            params.max_audio_tokens = std::stoi(argv[i]);
        } else if (arg == "--repetition-penalty") {
            if (++i >= argc) {
                fprintf(stderr, "Error: missing --repetition-penalty value\n");
                return 1;
            }
            params.repetition_penalty = std::stof(argv[i]);
            repetition_penalty_explicit = true;
        } else if (arg == "--seed") {
            if (++i >= argc) {
                fprintf(stderr, "Error: missing --seed value\n");
                return 1;
            }
            params.seed = std::stoi(argv[i]);
        } else if (arg == "--stream") {
            stream_mode = true;
        } else if (arg == "--stream-realtime") {
            stream_mode = true;
            stream_realtime = true;
        } else if (arg == "--stream-no-playback") {
            stream_mode = true;
            stream_playback = false;
            stream_no_playback_requested = true;
        } else if (arg == "--stream-playback") {
            stream_playback = true;
            stream_no_playback_requested = false;
        } else if (arg == "--talker-window") {
            if (++i >= argc) {
                fprintf(stderr, "Error: missing --talker-window value\n");
                return 1;
            }
            talker_attention_window = std::max(0, std::stoi(argv[i]));
        } else if (arg == "--stream-talker-window") {
            if (++i >= argc) {
                fprintf(stderr, "Error: missing --stream-talker-window value\n");
                return 1;
            }
            stream_talker_attention_window = std::max(0, std::stoi(argv[i]));
        } else if (arg == "--stream-parallel-decode") {
            stream_parallel_decode = true;
        } else if (arg == "--stream-no-parallel-decode") {
            stream_parallel_decode = false;
        } else if (arg == "--stream-drop-oldest") {
            stream_drop_oldest_on_overflow = true;
        } else if (arg == "--stream-block-on-full") {
            stream_drop_oldest_on_overflow = false;
        } else if (arg == "--stream-chunk-frames") {
            if (++i >= argc) {
                fprintf(stderr, "Error: missing --stream-chunk-frames value\n");
                return 1;
            }
            stream_chunk_frames = std::max(1, std::stoi(argv[i]));
        } else if (arg == "--stream-queue-cap") {
            if (++i >= argc) {
                fprintf(stderr, "Error: missing --stream-queue-cap value\n");
                return 1;
            }
            stream_queue_cap = std::max(1, std::stoi(argv[i]));
        } else if (arg == "--stream-poll-timeout") {
            if (++i >= argc) {
                fprintf(stderr, "Error: missing --stream-poll-timeout value\n");
                return 1;
            }
            stream_poll_timeout = std::max(0, std::stoi(argv[i]));
        } else if (arg == "--stream-left-context") {
            if (++i >= argc) {
                fprintf(stderr, "Error: missing --stream-left-context value\n");
                return 1;
            }
            stream_decoder_left_context = std::max(0, std::stoi(argv[i]));
            stream_left_context_explicit = true;
        } else if (arg == "--stream-lookahead") {
            if (++i >= argc) {
                fprintf(stderr, "Error: missing --stream-lookahead value\n");
                return 1;
            }
            stream_decoder_lookahead = std::max(0, std::stoi(argv[i]));
            stream_lookahead_explicit = true;
        } else if (arg == "--stream-adaptive") {
            stream_adaptive_tuning = true;
        } else if (arg == "--stream-no-adaptive") {
            stream_adaptive_tuning = false;
        } else if (arg == "--stream-chunk-min") {
            if (++i >= argc) {
                fprintf(stderr, "Error: missing --stream-chunk-min value\n");
                return 1;
            }
            stream_chunk_min = std::max(1, std::stoi(argv[i]));
        } else if (arg == "--stream-chunk-max") {
            if (++i >= argc) {
                fprintf(stderr, "Error: missing --stream-chunk-max value\n");
                return 1;
            }
            stream_chunk_max = std::max(1, std::stoi(argv[i]));
        } else if (arg == "--stream-left-context-min") {
            if (++i >= argc) {
                fprintf(stderr, "Error: missing --stream-left-context-min value\n");
                return 1;
            }
            stream_left_context_min = std::max(0, std::stoi(argv[i]));
        } else if (arg == "--stream-adapt-high") {
            if (++i >= argc) {
                fprintf(stderr, "Error: missing --stream-adapt-high value\n");
                return 1;
            }
            stream_adapt_high = std::stof(argv[i]);
        } else if (arg == "--stream-adapt-low") {
            if (++i >= argc) {
                fprintf(stderr, "Error: missing --stream-adapt-low value\n");
                return 1;
            }
            stream_adapt_low = std::stof(argv[i]);
        } else if (arg == "-j" || arg == "--threads") {
            if (++i >= argc) {
                fprintf(stderr, "Error: missing --threads value\n");
                return 1;
            }
            params.n_threads = std::stoi(argv[i]);
        } else {
            fprintf(stderr, "Error: unknown argument: %s\n", arg.c_str());
            print_usage(argv[0]);
            return 1;
        }
    }

    params.talker_attention_window = talker_attention_window;

    if (stream_realtime) {
        // Realtime preset: prioritize stable speech and RTF.
        // Do not force greedy decoding because it can collapse into long noisy tails.
        if (!temperature_explicit) {
            params.temperature = 0.6f;
        }
        if (!top_k_explicit) {
            params.top_k = 30;
        }
        if (!top_p_explicit) {
            params.top_p = 0.95f;
        }
        if (!repetition_penalty_explicit) {
            params.repetition_penalty = 1.05f;
        }
        stream_chunk_frames = std::max(stream_chunk_frames, 16);
        stream_chunk_min = std::max(stream_chunk_min, 8);
        stream_chunk_max = std::max(stream_chunk_max, 32);
        if (!stream_left_context_explicit) {
            stream_decoder_left_context = 0;
        }
        if (!stream_lookahead_explicit) {
            stream_decoder_lookahead = 0;
        }
        stream_left_context_min = std::min(stream_left_context_min, stream_decoder_left_context);
        stream_queue_cap = std::max(stream_queue_cap, 24);
        stream_parallel_decode = true;
        stream_drop_oldest_on_overflow = false;
        stream_adaptive_tuning = false;

        if (stream_dbg) {
            fprintf(stderr,
                    "[stream-realtime] final params: temp=%.2f top_k=%d top_p=%.2f rep=%.2f chunk=%d ctx=%d lookahead=%d adaptive=%d talker_window=%d\n",
                    params.temperature,
                    params.top_k,
                    params.top_p,
                    params.repetition_penalty,
                    stream_chunk_frames,
                    stream_decoder_left_context,
                    stream_decoder_lookahead,
                    stream_adaptive_tuning ? 1 : 0,
                    stream_talker_attention_window);
        }

        fprintf(stderr,
                "[stream-realtime] preset applied: temp=%.2f top_k=%d top_p=%.2f rep=%.2f max_tokens=%d chunk=%d ctx=%d queue=%d talker_window=%d\n",
                params.temperature,
                params.top_k,
                params.top_p,
                params.repetition_penalty,
                params.max_audio_tokens,
                stream_chunk_frames,
                stream_decoder_left_context,
                stream_queue_cap,
                stream_talker_attention_window);
    }

    if (stream_mode && stream_no_playback_requested && output_file.empty()) {
        fprintf(
            stderr,
            "[stream] --stream-no-playback is enabled: audio playback is disabled (benchmark mode). Use --stream-playback or remove --stream-no-playback to hear audio.\n"
        );
    }

    if (model_path_arg.empty()) {
        fprintf(stderr, "Error: --model is required\n");
        print_usage(argv[0]);
        return 1;
    }
    if (tokenizer_path_arg.empty()) {
        fprintf(stderr, "Error: --model-tokenizer is required\n");
        print_usage(argv[0]);
        return 1;
    }
    if (text_input.empty()) {
        fprintf(stderr, "Error: --text is required\n");
        print_usage(argv[0]);
        return 1;
    }

    if (!load_text_or_file(text_input, text) || text.empty()) {
        fprintf(stderr, "Error: failed to load text from --text argument\n");
        return 1;
    }

    std::string model_path;
    std::string tokenizer_path;
    std::string resolve_error;

    if (!resolve_gguf_file_path(model_path_arg, "--model", model_path, resolve_error)) {
        fprintf(stderr, "Error: %s\n", resolve_error.c_str());
        return 1;
    }
    if (!resolve_gguf_file_path(tokenizer_path_arg, "--model-tokenizer", tokenizer_path, resolve_error)) {
        fprintf(stderr, "Error: %s\n", resolve_error.c_str());
        return 1;
    }

    qwen3_tts::Qwen3TTS tts;
    fprintf(stderr, "Loading model: %s\n", model_path.c_str());
    fprintf(stderr, "Loading tokenizer: %s\n", tokenizer_path.c_str());

    if (!tts.load_models_from_files(model_path, tokenizer_path)) {
        fprintf(stderr, "Error: %s\n", tts.get_error().c_str());
        return 1;
    }

    qwen3_tts::tts_model_variant variant = tts.loaded_model_variant();
    if (variant == qwen3_tts::tts_model_variant::auto_variant) {
        variant = infer_model_variant_from_path(model_path);
    }
    fprintf(stderr, "Loaded variant: %s\n", variant_name(variant));

    if (stream_mode) {
        qwen3_tts::tts_stream_params stream_params;
        stream_params.tts = params;
        stream_params.stream_chunk_frames = stream_chunk_frames;
        stream_params.stream_queue_capacity = stream_queue_cap;
        stream_params.stream_poll_timeout_ms = stream_poll_timeout;
        stream_params.stream_full_flush = false;
        stream_params.stream_decoder_left_context_frames = stream_decoder_left_context;
        stream_params.stream_decoder_lookahead_frames = stream_decoder_lookahead;
        stream_params.stream_talker_attention_window = stream_talker_attention_window;
        stream_params.stream_adaptive_tuning = stream_adaptive_tuning;
        stream_params.stream_chunk_frames_min = stream_chunk_min;
        stream_params.stream_chunk_frames_max = std::max(stream_chunk_min, stream_chunk_max);
        stream_params.stream_left_context_frames_min = stream_left_context_min;
        stream_params.stream_adaptive_high_ratio = std::max(0.1f, stream_adapt_high);
        stream_params.stream_adaptive_low_ratio = std::max(0.05f, stream_adapt_low);
        stream_params.stream_parallel_decode = stream_parallel_decode;
        stream_params.stream_drop_oldest_on_overflow = stream_drop_oldest_on_overflow;
        if (stream_params.stream_adaptive_low_ratio > stream_params.stream_adaptive_high_ratio) {
            stream_params.stream_adaptive_low_ratio = stream_params.stream_adaptive_high_ratio;
        }

        std::shared_ptr<qwen3_tts::tts_stream_session> session;
        if (!reference_audio.empty()) {
            stream_params.tts.task_type = qwen3_tts::tts_task_type::voice_clone;
            stream_params.tts.reference_text = reference_text;
            if (!instruct.empty()) {
                fprintf(stderr, "Warning: --instruct is ignored in voice_clone mode (--reference provided).\n");
            }
            stream_params.tts.instruct.clear();
            stream_params.tts.speaker.clear();

            if (!stream_params.tts.x_vector_only_mode && stream_params.tts.reference_text.empty()) {
                fprintf(stderr, "Error: voice clone requires --ref-text unless --x-vector-only is set\n");
                return 1;
            }

            session = tts.synthesize_with_voice_stream(text, reference_audio, stream_params);
        } else if (!speaker.empty()) {
            stream_params.tts.task_type = qwen3_tts::tts_task_type::custom_voice;
            stream_params.tts.speaker = speaker;
            stream_params.tts.instruct = instruct;
            session = tts.synthesize_stream(text, stream_params);
        } else if (!instruct.empty()) {
            if (variant != qwen3_tts::tts_model_variant::voice_design) {
                fprintf(stderr, "Error: instruct-only mode requires a voice_design model, or provide --speaker\n");
                return 1;
            }
            stream_params.tts.task_type = qwen3_tts::tts_task_type::voice_design;
            stream_params.tts.instruct = instruct;
            session = tts.synthesize_stream(text, stream_params);
        } else {
            if (variant == qwen3_tts::tts_model_variant::custom_voice) {
                fprintf(stderr, "Error: custom_voice model requires --speaker\n");
                return 1;
            }
            if (variant == qwen3_tts::tts_model_variant::voice_design) {
                fprintf(stderr, "Error: voice_design model requires --instruct\n");
                return 1;
            }
            stream_params.tts.task_type = qwen3_tts::tts_task_type::auto_task;
            session = tts.synthesize_stream(text, stream_params);
        }

        if (!session) {
            fprintf(stderr, "Error: %s\n", tts.get_error().c_str());
            return 1;
        }

        const bool stream_playback_enabled = output_file.empty() ? stream_playback : false;

        StreamRunStats stats;
        std::string stream_error;
        if (!run_stream_session(session, output_file, stream_poll_timeout, stream_playback_enabled, stats, stream_error)) {
            fprintf(stderr, "Error: %s\n", stream_error.c_str());
            return 1;
        }

        if (!output_file.empty()) {
            fprintf(stderr, "Output saved to: %s\n", output_file.c_str());
        }

        const double stream_audio_sec =
            stats.sample_rate > 0 ? (double)stats.n_samples / (double)stats.sample_rate : 0.0;
        const double stream_rtf =
            stream_audio_sec > 0.0 ? ((double)stats.wall_ms / 1000.0) / stream_audio_sec : 0.0;
        fprintf(
            stderr,
            "Done(stream). sample_rate=%d, samples=%zu, audio_sec=%.2f, wall_sec=%.2f, rtf=%.3f, dropped_chunks=%d\n",
            stats.sample_rate,
            stats.n_samples,
            stream_audio_sec,
            (double)stats.wall_ms / 1000.0,
            stream_rtf,
            stats.dropped_chunks
        );
        return 0;
    }

    qwen3_tts::tts_result result;

    if (!reference_audio.empty()) {
        params.task_type = qwen3_tts::tts_task_type::voice_clone;
        params.reference_text = reference_text;
        if (!instruct.empty()) {
            fprintf(stderr, "Warning: --instruct is ignored in voice_clone mode (--reference provided).\n");
        }
        params.instruct.clear();
        params.speaker.clear();

        if (!params.x_vector_only_mode && params.reference_text.empty()) {
            fprintf(stderr, "Error: voice clone requires --ref-text unless --x-vector-only is set\n");
            return 1;
        }

        result = tts.synthesize_with_voice(text, reference_audio, params);
    } else if (!speaker.empty()) {
        params.task_type = qwen3_tts::tts_task_type::custom_voice;
        params.speaker = speaker;
        params.instruct = instruct;
        result = tts.synthesize(text, params);
    } else if (!instruct.empty()) {
        if (variant != qwen3_tts::tts_model_variant::voice_design) {
            fprintf(stderr, "Error: instruct-only mode requires a voice_design model, or provide --speaker\n");
            return 1;
        }
        params.task_type = qwen3_tts::tts_task_type::voice_design;
        params.instruct = instruct;
        result = tts.synthesize(text, params);
    } else {
        if (variant == qwen3_tts::tts_model_variant::custom_voice) {
            fprintf(stderr, "Error: custom_voice model requires --speaker\n");
            return 1;
        }
        if (variant == qwen3_tts::tts_model_variant::voice_design) {
            fprintf(stderr, "Error: voice_design model requires --instruct\n");
            return 1;
        }
        params.task_type = qwen3_tts::tts_task_type::auto_task;
        result = tts.synthesize(text, params);
    }

    if (!result.success) {
        fprintf(stderr, "Error: %s\n", result.error_msg.c_str());
        return 1;
    }

    if (output_file.empty()) {
        if (!qwen3_tts::play_audio(result.audio, result.sample_rate)) {
            fprintf(stderr, "Error: playback failed\n");
            return 1;
        }
    } else {
        if (!qwen3_tts::save_audio_file(output_file, result.audio, result.sample_rate)) {
            fprintf(stderr, "Error: failed to save output file: %s\n", output_file.c_str());
            return 1;
        }
        fprintf(stderr, "Output saved to: %s\n", output_file.c_str());
    }

    fprintf(
        stderr,
        "Done. sample_rate=%d, samples=%zu, audio_sec=%.2f\n",
        result.sample_rate,
        result.audio.size(),
        result.sample_rate > 0 ? (double)result.audio.size() / (double)result.sample_rate : 0.0
    );

    return 0;
}
