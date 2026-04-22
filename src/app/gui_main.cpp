#include "qwen3_tts.h"

#include "imgui.h"
#include "backends/imgui_impl_dx11.h"
#include "backends/imgui_impl_win32.h"

#include <SDL3/SDL.h>

#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif

#include <d3d11.h>
#include <windows.h>

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

extern IMGUI_IMPL_API LRESULT ImGui_ImplWin32_WndProcHandler(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam);

namespace {

namespace fs = std::filesystem;

static ID3D11Device *g_pd3d_device = nullptr;
static ID3D11DeviceContext *g_pd3d_device_context = nullptr;
static IDXGISwapChain *g_p_swap_chain = nullptr;
static ID3D11RenderTargetView *g_main_render_target_view = nullptr;

enum class ui_lang {
    zh,
    en,
};

enum class model_id {
    model_06b_base,
    model_06b_custom,
    model_17b_base,
    model_17b_custom,
    model_17b_design,
};

struct speaker_option {
    const char *id;
    const char *name_zh;
    const char *name_en;
};

static const speaker_option k_speakers[] = {
    {"Vivian", "Vivian（女）", "Vivian (Female)"},
    {"Serena", "Serena（女）", "Serena (Female)"},
    {"Uncle_Fu", "Uncle_Fu（男）", "Uncle_Fu (Male)"},
    {"Dylan", "Dylan（男）", "Dylan (Male)"},
    {"Eric", "Eric（男）", "Eric (Male)"},
    {"Ryan", "Ryan（男）", "Ryan (Male)"},
    {"Aiden", "Aiden（男）", "Aiden (Male)"},
    {"Ono_Anna", "Ono_Anna（女）", "Ono_Anna (Female)"},
    {"Sohee", "Sohee（女）", "Sohee (Female)"},
};

static const char *tr(ui_lang lang, const char *zh, const char *en) {
    return lang == ui_lang::zh ? zh : en;
}

static ui_lang detect_ui_lang() {
    const LANGID lid = GetUserDefaultUILanguage();
    return PRIMARYLANGID(lid) == LANG_CHINESE ? ui_lang::zh : ui_lang::en;
}

static std::string to_lower_copy(const std::string &s) {
    std::string out = s;
    std::transform(out.begin(), out.end(), out.begin(), [](unsigned char c) { return (char)std::tolower(c); });
    return out;
}

static bool parse_language_id_from_key(const std::string &lang, int32_t &out) {
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
    FILE *f = nullptr;
#ifdef _WIN32
    if (fopen_s(&f, input.c_str(), "rb") != 0) f = nullptr;
#else
    f = fopen(input.c_str(), "rb");
#endif
    if (!f) return true;

    if (fseek(f, 0, SEEK_END) != 0) {
        fclose(f);
        return false;
    }
    const long size = ftell(f);
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
        if (fread(&content[0], 1, (size_t)size, f) != (size_t)size) {
            fclose(f);
            return false;
        }
    }
    fclose(f);
    out_text = std::move(content);
    return true;
}

struct text_segment {
    std::string text;
    int word_unit_count;
};

static size_t utf8_len(char c) {
    const unsigned char uc = (unsigned char)c;
    if ((uc & 0x80u) == 0) return 1;
    if ((uc & 0xE0u) == 0xC0u) return 2;
    if ((uc & 0xF0u) == 0xE0u) return 3;
    if ((uc & 0xF8u) == 0xF0u) return 4;
    return 1;
}

static size_t utf8_decode(const std::string &text, size_t offset, uint32_t &codepoint) {
    if (offset >= text.size()) {
        codepoint = 0;
        return 0;
    }

    const unsigned char lead = (unsigned char)text[offset];
    size_t len = utf8_len((char)lead);
    if (offset + len > text.size()) {
        codepoint = lead;
        return 1;
    }

    if (len == 1) {
        codepoint = lead;
        return 1;
    }

    for (size_t i = 1; i < len; ++i) {
        const unsigned char uc = (unsigned char)text[offset + i];
        if ((uc & 0xC0u) != 0x80u) {
            codepoint = lead;
            return 1;
        }
    }

    if (len == 2) {
        codepoint = ((uint32_t)(lead & 0x1Fu) << 6) | (uint32_t)(text[offset + 1] & 0x3Fu);
    } else if (len == 3) {
        codepoint = ((uint32_t)(lead & 0x0Fu) << 12)
            | ((uint32_t)(text[offset + 1] & 0x3Fu) << 6)
            | (uint32_t)(text[offset + 2] & 0x3Fu);
    } else {
        codepoint = ((uint32_t)(lead & 0x07u) << 18)
            | ((uint32_t)(text[offset + 1] & 0x3Fu) << 12)
            | ((uint32_t)(text[offset + 2] & 0x3Fu) << 6)
            | (uint32_t)(text[offset + 3] & 0x3Fu);
    }

    return len;
}

static bool is_cjk_ideograph(uint32_t cp) {
    return (cp >= 0x4E00u && cp <= 0x9FFFu) || (cp >= 0x3400u && cp <= 0x4DBFu);
}

static bool is_cjk_punct(uint32_t cp) {
    return cp >= 0x3000u && cp <= 0x303Fu;
}

static bool is_latin_word_char(uint32_t cp) {
    return cp <= 0x7Fu && std::isalnum((unsigned char)cp);
}

static bool is_whitespace(uint32_t cp) {
    if (cp <= 0x7Fu) {
        return std::isspace((unsigned char)cp) != 0;
    }
    return cp == 0x3000u;
}

static bool is_cjk_sentence_punct(uint32_t cp) {
    switch (cp) {
        case 0x3002u: // 。
        case 0xFF01u: // ！
        case 0xFF1Fu: // ？
        case 0xFF1Bu: // ；
        case 0x2026u: // …
        case 0x3001u: // 、
        case 0x300Au: // 《
        case 0x300Bu: // 》
        case 0x3010u: // 【
        case 0x3011u: // 】
            return true;
        default:
            return false;
    }
}

static bool is_latin_sentence_punct(uint32_t cp) {
    switch (cp) {
        case '.':
        case '!':
        case '?':
        case ';':
        case ':':
        case 0x2014u: // —
            return true;
        default:
            return false;
    }
}

static bool is_secondary_break(uint32_t cp) {
    switch (cp) {
        case ',':
        case ';':
        case 0xFF0Cu: // ，
        case 0xFF1Bu: // ；
        case 0x3001u: // 、
            return true;
        default:
            return false;
    }
}

struct text_symbol {
    size_t byte_pos;
    size_t byte_len;
    uint32_t codepoint;
    bool cjk_ideograph;
    bool cjk_punct;
    bool latin_word;
    bool whitespace;
    bool primary_break;
    bool secondary_break;
};

static std::vector<text_segment> segment_text(const std::string &input, int max_units = 80) {
    std::vector<text_segment> segments;
    if (input.empty()) return segments;

    if (max_units <= 0) {
        int units = 0;
        bool in_word = false;
        size_t i = 0;
        while (i < input.size()) {
            uint32_t cp = 0;
            size_t len = utf8_decode(input, i, cp);
            if (len == 0) break;
            if (is_cjk_ideograph(cp)) {
                units += 1;
                in_word = false;
            } else if (is_latin_word_char(cp)) {
                if (!in_word) {
                    units += 1;
                    in_word = true;
                }
            } else {
                in_word = false;
            }
            i += len;
        }
        segments.push_back({input, units});
        return segments;
    }

    std::vector<text_symbol> symbols;
    symbols.reserve(input.size());
    size_t offset = 0;
    while (offset < input.size()) {
        uint32_t cp = 0;
        size_t len = utf8_decode(input, offset, cp);
        if (len == 0) break;
        text_symbol sym{};
        sym.byte_pos = offset;
        sym.byte_len = len;
        sym.codepoint = cp;
        sym.cjk_ideograph = is_cjk_ideograph(cp);
        sym.cjk_punct = is_cjk_punct(cp);
        sym.latin_word = is_latin_word_char(cp);
        sym.whitespace = is_whitespace(cp);
        sym.primary_break = is_cjk_sentence_punct(cp) || is_latin_sentence_punct(cp);
        sym.secondary_break = is_secondary_break(cp);
        symbols.push_back(sym);
        offset += len;
    }

    if (symbols.empty()) return segments;

    // Post-pass: un-mark decimal points (e.g. "3.14") as primary breaks.
    // A '.' between two digit characters is a decimal separator, not a sentence break.
    for (size_t i = 1; i + 1 < symbols.size(); ++i) {
        if (symbols[i].codepoint == '.' && symbols[i].primary_break) {
            // Check if previous and next symbols are digits
            bool prev_digit = (symbols[i - 1].codepoint >= '0' && symbols[i - 1].codepoint <= '9');
            bool next_digit = (symbols[i + 1].codepoint >= '0' && symbols[i + 1].codepoint <= '9');
            if (prev_digit && next_digit) {
                symbols[i].primary_break = false;
                // Also check for numbers like "3.14.5.6" - mark all consecutive decimal dots as non-break
            }
        }
    }
    // Handle edge case: number followed by period+digit at index 0
    // and period at penultimate position before a digit
    // Also handle: "3." at end (not a decimal) - leave as break
    // ".5" at start - not a decimal point sentence break either
    if (symbols.size() >= 2) {
        if (symbols[0].codepoint == '.' && symbols[1].codepoint >= '0' && symbols[1].codepoint <= '9') {
            symbols[0].primary_break = false;
        }
    }

    const int extra_units = (max_units * 3) / 10;
    size_t start_index = 0;
    while (start_index < symbols.size()) {
        int units = 0;
        bool in_word = false;
        size_t index = start_index;
        bool reached_limit = false;
        for (; index < symbols.size(); ++index) {
            const text_symbol &sym = symbols[index];
            if (sym.cjk_ideograph) {
                units += 1;
                in_word = false;
            } else if (sym.latin_word) {
                if (!in_word) {
                    units += 1;
                    in_word = true;
                }
            } else {
                in_word = false;
            }

            if (units >= max_units) {
                reached_limit = true;
                break;
            }
        }

        if (!reached_limit || index >= symbols.size()) {
            const text_symbol &last = symbols.back();
            size_t start_byte = symbols[start_index].byte_pos;
            size_t end_byte = last.byte_pos + last.byte_len;
            segments.push_back({input.substr(start_byte, end_byte - start_byte), units});
            break;
        }

        size_t segment_end = index;
        int segment_units = units;

        size_t primary_break = symbols.size();
        size_t secondary_break = symbols.size();
        int primary_units = 0;
        int secondary_units = 0;
        int scan_units = units;
        bool scan_in_word = in_word;
        const int limit_units = max_units + extra_units;

        for (size_t scan_index = index; scan_index < symbols.size(); ++scan_index) {
            if (scan_index > index) {
                const text_symbol &scan_sym = symbols[scan_index];
                if (scan_sym.cjk_ideograph) {
                    scan_units += 1;
                    scan_in_word = false;
                } else if (scan_sym.latin_word) {
                    if (!scan_in_word) {
                        scan_units += 1;
                        scan_in_word = true;
                    }
                } else {
                    scan_in_word = false;
                }
                if (scan_units > limit_units) {
                    break;
                }
            }

            const text_symbol &scan_sym = symbols[scan_index];
            if (scan_sym.primary_break) {
                primary_break = scan_index;
                primary_units = scan_units;
                break;
            }
            if (secondary_break == symbols.size() && scan_sym.secondary_break) {
                secondary_break = scan_index;
                secondary_units = scan_units;
            }
        }

        if (primary_break != symbols.size()) {
            segment_end = primary_break;
            segment_units = primary_units;
        } else if (secondary_break != symbols.size()) {
            segment_end = secondary_break;
            segment_units = secondary_units;
        }

        const text_symbol &end_sym = symbols[segment_end];
        size_t start_byte = symbols[start_index].byte_pos;
        size_t end_byte = end_sym.byte_pos + end_sym.byte_len;
        segments.push_back({input.substr(start_byte, end_byte - start_byte), segment_units});

        start_index = segment_end + 1;
    }

    return segments;
}

class WavStreamWriter {
public:
    ~WavStreamWriter() { close(); }

    bool open(const std::string &path, int32_t sample_rate, std::string &error_msg) {
        close();
        if (sample_rate <= 0) {
            error_msg = "Invalid sample rate";
            return false;
        }
        if (fopen_s(&file_, path.c_str(), "wb") != 0) file_ = nullptr;
        if (!file_) {
            error_msg = "Failed to open output: " + path;
            return false;
        }
        path_ = path;
        sample_rate_ = sample_rate;
        data_bytes_ = 0;
        return write_header(0, error_msg);
    }

    bool append(const std::vector<float> &samples, std::string &error_msg) {
        if (!file_) return true;
        for (float s : samples) {
            if (s > 1.0f) s = 1.0f;
            if (s < -1.0f) s = -1.0f;
            const int16_t pcm = (int16_t)(s * 32767.0f);
            if (fwrite(&pcm, sizeof(int16_t), 1, file_) != 1) {
                error_msg = "Failed writing wav payload";
                return false;
            }
            data_bytes_ += sizeof(int16_t);
        }
        return true;
    }

    bool close() {
        if (!file_) return true;
        std::string err;
        write_header(data_bytes_, err);
        fclose(file_);
        file_ = nullptr;
        return true;
    }

private:
    bool write_header(uint32_t data_size, std::string &error_msg) {
        if (!file_) return true;
        if (fseek(file_, 0, SEEK_SET) != 0) {
            error_msg = "Failed seek";
            return false;
        }
        const uint16_t num_channels = 1;
        const uint16_t bits_per_sample = 16;
        const uint16_t block_align = (uint16_t)(num_channels * bits_per_sample / 8);
        const uint32_t byte_rate = (uint32_t)(sample_rate_ * block_align);
        const uint32_t file_size = 36u + data_size;
        const uint16_t audio_format = 1;
        const uint32_t fmt_size = 16;
        if (fwrite("RIFF", 1, 4, file_) != 4 || fwrite(&file_size, 4, 1, file_) != 1 ||
            fwrite("WAVE", 1, 4, file_) != 4 || fwrite("fmt ", 1, 4, file_) != 4 ||
            fwrite(&fmt_size, 4, 1, file_) != 1 || fwrite(&audio_format, 2, 1, file_) != 1 ||
            fwrite(&num_channels, 2, 1, file_) != 1 || fwrite(&sample_rate_, 4, 1, file_) != 1 ||
            fwrite(&byte_rate, 4, 1, file_) != 1 || fwrite(&block_align, 2, 1, file_) != 1 ||
            fwrite(&bits_per_sample, 2, 1, file_) != 1 || fwrite("data", 1, 4, file_) != 4 ||
            fwrite(&data_size, 4, 1, file_) != 1) {
            error_msg = "Failed writing wav header";
            return false;
        }
        fseek(file_, 0, SEEK_END);
        return true;
    }

    FILE *file_ = nullptr;
    std::string path_;
    int32_t sample_rate_ = 24000;
    uint32_t data_bytes_ = 0;
};

class StreamPlayer {
public:
    ~StreamPlayer() { close(); }

    bool open(int32_t sample_rate, std::string &error_msg) {
        if (stream_) return true;
        if ((SDL_WasInit(SDL_INIT_AUDIO) & SDL_INIT_AUDIO) == 0) {
            if (!SDL_InitSubSystem(SDL_INIT_AUDIO)) {
                error_msg = std::string("SDL init failed: ") + SDL_GetError();
                return false;
            }
            inited_ = true;
        }
        SDL_AudioSpec spec = {};
        spec.format = SDL_AUDIO_F32;
        spec.channels = 1;
        spec.freq = sample_rate;
        stream_ = SDL_OpenAudioDeviceStream(SDL_AUDIO_DEVICE_DEFAULT_PLAYBACK, &spec, nullptr, nullptr);
        if (!stream_) {
            error_msg = std::string("Open stream failed: ") + SDL_GetError();
            close();
            return false;
        }
        if (!SDL_ResumeAudioStreamDevice(stream_)) {
            error_msg = std::string("Resume stream failed: ") + SDL_GetError();
            close();
            return false;
        }
        return true;
    }

    bool push(const std::vector<float> &samples, std::string &error_msg) {
        if (!stream_ || cancelled_) return true;
        if (!SDL_PutAudioStreamData(stream_, samples.data(), (int)(samples.size() * sizeof(float)))) {
            error_msg = std::string("Push stream failed: ") + SDL_GetError();
            return false;
        }
        return true;
    }

    bool finish(std::string &error_msg) {
        if (!stream_ || cancelled_) {
            close();
            return true;
        }
        if (!SDL_FlushAudioStream(stream_)) {
            error_msg = std::string("Flush stream failed: ") + SDL_GetError();
            close();
            return false;
        }
        // Drain with cancellation support: check cancel flag every 10ms
        while (SDL_GetAudioStreamQueued(stream_) > 0) {
            if (cancelled_) break;
            SDL_Delay(10);
        }
        close();
        return true;
    }

    void cancel() {
        cancelled_ = true;
    }

    bool is_cancelled() const { return cancelled_; }

private:
    void close() {
        if (stream_) {
            SDL_DestroyAudioStream(stream_);
            stream_ = nullptr;
        }
        if (inited_) {
            SDL_QuitSubSystem(SDL_INIT_AUDIO);
            inited_ = false;
        }
    }

    SDL_AudioStream *stream_ = nullptr;
    bool inited_ = false;
    std::atomic<bool> cancelled_{false};
};

struct app_state {
    ui_lang lang = detect_ui_lang();

    int model_index = 0;
    int speaker_index = 0;
    int synthesis_language_index = 0;

    char text_input[8192] = {};
    char reference_audio[1024] = {};
    char reference_text_input[4096] = {};
    char instruct_input[4096] = {};
    char output_path[1024] = {};

    bool x_vector_only = false;
    bool save_output = false;
    bool play_audio = true;
    bool stream = false;
    bool stream_realtime = false;

    float temperature = 0.9f;
    int top_k = 50;
    float top_p = 1.0f;
    int max_tokens = 4096;
    float repetition_penalty = 1.05f;
    int seed = -1;
    int threads = 4;

    std::atomic<bool> model_setup_running{false};
    std::atomic<bool> models_ready{false};
    std::atomic<bool> generation_running{false};
    std::thread model_setup_thread;
    std::thread generation_thread;

    std::mutex log_mutex;
    std::string log_text;

    bool request_validation_popup = false;
    std::string validation_message;
    
    std::atomic<int> segment_total{0};
    std::atomic<int> segment_completed{0};
    std::atomic<int> progress_percent{0};
    std::atomic<bool> cancel_requested{false};

    bool config_dirty = false;
};

struct synthesis_language_option {
    const char *key;
    const char *name_zh;
    const char *name_en;
};

static const synthesis_language_option k_synthesis_languages[] = {
    {"auto", "自动", "Auto"},
    {"en", "英语", "English"},
    {"ru", "俄语", "Russian"},
    {"zh", "中文", "Chinese"},
    {"ja", "日语", "Japanese"},
    {"ko", "韩语", "Korean"},
    {"de", "德语", "German"},
    {"fr", "法语", "French"},
    {"es", "西班牙语", "Spanish"},
    {"it", "意大利语", "Italian"},
    {"pt", "葡萄牙语", "Portuguese"},
    {"beijing_dialect", "北京话", "Beijing Dialect"},
    {"sichuan_dialect", "四川话", "Sichuan Dialect"},
};

static void append_log(app_state &state, const std::string &msg) {
    std::lock_guard<std::mutex> lock(state.log_mutex);
    state.log_text += msg;
    state.log_text += "\n";
}

static bool contains_any_gguf(const fs::path &dir) {
    std::error_code ec;
    if (!fs::exists(dir, ec) || !fs::is_directory(dir, ec)) return false;
    for (const auto &entry : fs::directory_iterator(dir, ec)) {
        if (!entry.is_regular_file()) continue;
        if (to_lower_copy(entry.path().extension().string()) == ".gguf") return true;
    }
    return false;
}

static bool check_models_layout() {
    const fs::path models = fs::current_path() / "models";
    const fs::path gguf = models / "gguf";
    const std::vector<fs::path> dirs = {
        gguf / "0.6b-base",
        gguf / "0.6b-custom-voice",
        gguf / "1.7b-base",
        gguf / "1.7b-custom-voice",
        gguf / "1.7b-voice-design",
        gguf / "tokenizer",
        models / "Qwen3-TTS-12Hz-0.6B-Base",
        models / "Qwen3-TTS-12Hz-0.6B-CustomVoice",
        models / "Qwen3-TTS-12Hz-1.7B-Base",
        models / "Qwen3-TTS-12Hz-1.7B-CustomVoice",
        models / "Qwen3-TTS-12Hz-1.7B-VoiceDesign",
        models / "Qwen3-TTS-Tokenizer-12Hz",
    };
    std::error_code ec;
    for (const auto &d : dirs) {
        if (!fs::exists(d, ec) || !fs::is_directory(d, ec)) return false;
    }
    return contains_any_gguf(gguf / "0.6b-base") && contains_any_gguf(gguf / "0.6b-custom-voice") &&
           contains_any_gguf(gguf / "1.7b-base") && contains_any_gguf(gguf / "1.7b-custom-voice") &&
           contains_any_gguf(gguf / "1.7b-voice-design") && contains_any_gguf(gguf / "tokenizer");
}

static bool run_command(const std::string &command, app_state &state) {
    append_log(state, "[run] " + command);
    const int rc = std::system(command.c_str());
    if (rc != 0) {
        append_log(state, "[error] command failed, code=" + std::to_string(rc));
        return false;
    }
    return true;
}

static bool ensure_models_layout(app_state &state) {
    if (check_models_layout()) {
        append_log(state, "[ok] models layout valid");
        return true;
    }
    append_log(state, "[warn] models layout mismatch, repairing...");
    std::error_code ec;
    fs::remove_all(fs::current_path() / "models", ec);
    fs::create_directories(fs::current_path() / "models", ec);

    if (!run_command("python -m venv .venv", state)) return false;
    const std::string py = ".venv\\Scripts\\python.exe";
    if (!run_command("\"" + py + "\" -m pip install huggingface_hub gguf torch safetensors numpy tqdm coremltools", state)) return false;
    if (!run_command("\"" + py + "\" scripts\\setup_pipeline_models.py", state)) return false;

    if (!check_models_layout()) {
        append_log(state, "[error] models layout still invalid after setup");
        return false;
    }
    append_log(state, "[ok] models repaired");
    return true;
}

static std::string choose_best_gguf(const fs::path &dir) {
    std::vector<std::string> f16;
    std::vector<std::string> others;
    std::error_code ec;
    for (const auto &entry : fs::directory_iterator(dir, ec)) {
        if (!entry.is_regular_file()) continue;
        if (to_lower_copy(entry.path().extension().string()) != ".gguf") continue;
        const std::string file = entry.path().string();
        const std::string name = to_lower_copy(entry.path().filename().string());
        if (name.find("f16") != std::string::npos) f16.push_back(file);
        else others.push_back(file);
    }
    std::sort(f16.begin(), f16.end());
    std::sort(others.begin(), others.end());
    if (!f16.empty()) return f16.front();
    if (!others.empty()) return others.front();
    return {};
}

static std::string tokenizer_path() {
    return choose_best_gguf(fs::current_path() / "models" / "gguf" / "tokenizer");
}

static std::string model_path_for(model_id id) {
    const fs::path root = fs::current_path() / "models" / "gguf";
    switch (id) {
        case model_id::model_06b_base: return choose_best_gguf(root / "0.6b-base");
        case model_id::model_06b_custom: return choose_best_gguf(root / "0.6b-custom-voice");
        case model_id::model_17b_base: return choose_best_gguf(root / "1.7b-base");
        case model_id::model_17b_custom: return choose_best_gguf(root / "1.7b-custom-voice");
        case model_id::model_17b_design: return choose_best_gguf(root / "1.7b-voice-design");
    }
    return {};
}

static bool is_base(model_id id) {
    return id == model_id::model_06b_base || id == model_id::model_17b_base;
}

static bool is_custom(model_id id) {
    return id == model_id::model_06b_custom || id == model_id::model_17b_custom;
}

static bool is_design(model_id id) {
    return id == model_id::model_17b_design;
}

static bool is_custom_17b(model_id id) {
    return id == model_id::model_17b_custom;
}

static std::string save_config_path() {
    return (fs::current_path() / "gui_config.ini").string();
}

static std::string save_file_default() {
    return (fs::current_path() / "outputs" / "tts_output.wav").string();
}

static void reset_state_to_defaults(app_state &state) {
    state.model_index = 0;
    state.speaker_index = 0;
    state.synthesis_language_index = 0;
    memset(state.text_input, 0, sizeof(state.text_input));
    memset(state.reference_audio, 0, sizeof(state.reference_audio));
    memset(state.reference_text_input, 0, sizeof(state.reference_text_input));
    memset(state.instruct_input, 0, sizeof(state.instruct_input));
    strncpy_s(state.output_path, save_file_default().c_str(), _TRUNCATE);
    state.x_vector_only = false;
    state.save_output = false;
    state.play_audio = true;
    state.stream = false;
    state.stream_realtime = false;
    state.temperature = 0.9f;
    state.top_k = 50;
    state.top_p = 1.0f;
    state.max_tokens = 4096;
    state.repetition_penalty = 1.05f;
    state.seed = -1;
    state.threads = 4;
    state.config_dirty = true;
}

bool save_config(const app_state &state, const std::string &path) {
    std::ofstream out(path, std::ios::out | std::ios::trunc);
    if (!out.is_open()) return false;

    out << "model_index=" << state.model_index << "\n";
    out << "speaker_index=" << state.speaker_index << "\n";
    out << "synthesis_language_index=" << state.synthesis_language_index << "\n";

    out << "text_input=" << state.text_input << "\n";
    out << "reference_audio=" << state.reference_audio << "\n";
    out << "reference_text_input=" << state.reference_text_input << "\n";
    out << "instruct_input=" << state.instruct_input << "\n";
    out << "output_path=" << state.output_path << "\n";

    out << "x_vector_only=" << (state.x_vector_only ? 1 : 0) << "\n";
    out << "save_output=" << (state.save_output ? 1 : 0) << "\n";
    out << "play_audio=" << (state.play_audio ? 1 : 0) << "\n";
    out << "stream=" << (state.stream ? 1 : 0) << "\n";
    out << "stream_realtime=" << (state.stream_realtime ? 1 : 0) << "\n";

    out << "temperature=" << std::to_string(state.temperature) << "\n";
    out << "top_k=" << state.top_k << "\n";
    out << "top_p=" << std::to_string(state.top_p) << "\n";
    out << "max_tokens=" << state.max_tokens << "\n";
    out << "repetition_penalty=" << std::to_string(state.repetition_penalty) << "\n";
    out << "seed=" << state.seed << "\n";
    out << "threads=" << state.threads << "\n";

    return out.good();
}

bool load_config(app_state &state, const std::string &path) {
    std::ifstream in(path);
    if (!in.is_open()) return false;

    const auto parse_int = [](const std::string &s, int &out) -> bool {
        char *end = nullptr;
        const long v = std::strtol(s.c_str(), &end, 10);
        if (end == s.c_str() || *end != '\0') return false;
        out = (int)v;
        return true;
    };

    const auto parse_float = [](const std::string &s, float &out) -> bool {
        char *end = nullptr;
        const float v = std::strtof(s.c_str(), &end);
        if (end == s.c_str() || *end != '\0') return false;
        out = v;
        return true;
    };

    std::string line;
    while (std::getline(in, line)) {
        const size_t sep = line.find('=');
        if (sep == std::string::npos) continue;

        std::string key = line.substr(0, sep);
        std::string value = line.substr(sep + 1);
        if (!value.empty() && value.back() == '\r') value.pop_back();

        int int_value = 0;
        float float_value = 0.0f;
        if (key == "model_index") {
            if (!parse_int(value, state.model_index)) return false;
        } else if (key == "speaker_index") {
            if (!parse_int(value, state.speaker_index)) return false;
        } else if (key == "synthesis_language_index") {
            if (!parse_int(value, state.synthesis_language_index)) return false;
        } else if (key == "text_input") {
            strncpy_s(state.text_input, value.c_str(), _TRUNCATE);
        } else if (key == "reference_audio") {
            strncpy_s(state.reference_audio, value.c_str(), _TRUNCATE);
        } else if (key == "reference_text_input") {
            strncpy_s(state.reference_text_input, value.c_str(), _TRUNCATE);
        } else if (key == "instruct_input") {
            strncpy_s(state.instruct_input, value.c_str(), _TRUNCATE);
        } else if (key == "output_path") {
            strncpy_s(state.output_path, value.c_str(), _TRUNCATE);
        } else if (key == "x_vector_only") {
            if (!parse_int(value, int_value)) return false;
            state.x_vector_only = (bool)int_value;
        } else if (key == "save_output") {
            if (!parse_int(value, int_value)) return false;
            state.save_output = (bool)int_value;
        } else if (key == "play_audio") {
            if (!parse_int(value, int_value)) return false;
            state.play_audio = (bool)int_value;
        } else if (key == "stream") {
            if (!parse_int(value, int_value)) return false;
            state.stream = (bool)int_value;
        } else if (key == "stream_realtime") {
            if (!parse_int(value, int_value)) return false;
            state.stream_realtime = (bool)int_value;
        } else if (key == "temperature") {
            if (!parse_float(value, float_value)) return false;
            state.temperature = float_value;
        } else if (key == "top_k") {
            if (!parse_int(value, state.top_k)) return false;
        } else if (key == "top_p") {
            if (!parse_float(value, float_value)) return false;
            state.top_p = float_value;
        } else if (key == "max_tokens") {
            if (!parse_int(value, state.max_tokens)) return false;
        } else if (key == "repetition_penalty") {
            if (!parse_float(value, float_value)) return false;
            state.repetition_penalty = float_value;
        } else if (key == "seed") {
            if (!parse_int(value, state.seed)) return false;
        } else if (key == "threads") {
            if (!parse_int(value, state.threads)) return false;
        }
    }

    if (state.threads < 1) state.threads = 1;
    return true;
}

static bool run_stream_session(
    const std::shared_ptr<qwen3_tts::tts_stream_session> &session,
    const std::string &output_file,
    bool playback_enabled,
    std::string &error_msg,
    app_state &state
) {
    if (!session) {
        error_msg = "Stream session is null";
        return false;
    }
    StreamPlayer player;
    WavStreamWriter writer;
    bool player_opened = false;
    bool writer_opened = false;
    int32_t sample_rate = 24000;

    while (true) {
        // Check for cancellation before polling
        if (state.cancel_requested.load()) {
            session->cancel();
            // Drain remaining audio quickly without playback
            qwen3_tts::tts_audio_chunk drain_chunk;
            while (session->poll(drain_chunk, 10) == qwen3_tts::tts_stream_poll_status::chunk) {
                // Just drain, don't play
            }
            error_msg = "cancelled";
            return false;
        }

        qwen3_tts::tts_audio_chunk chunk;
        const auto status = session->poll(chunk, 100);
        if (status == qwen3_tts::tts_stream_poll_status::timeout) continue;
        if (status == qwen3_tts::tts_stream_poll_status::error) {
            error_msg = session->get_error();
            if (error_msg.empty()) error_msg = "stream error";
            return false;
        }
        if (status == qwen3_tts::tts_stream_poll_status::finished) break;
        if (chunk.sample_rate > 0) sample_rate = chunk.sample_rate;
        if (!output_file.empty() && !writer_opened) {
            if (!writer.open(output_file, sample_rate, error_msg)) return false;
            writer_opened = true;
        }
        if (!chunk.audio.empty()) {
            if (playback_enabled && !player_opened) {
                if (!player.open(sample_rate, error_msg)) return false;
                player_opened = true;
            }
            if (playback_enabled && !player.push(chunk.audio, error_msg)) return false;
            if (writer_opened && !writer.append(chunk.audio, error_msg)) return false;
        }
    }

    if (player_opened && !player.finish(error_msg)) return false;
    if (writer_opened && !writer.close()) {
        error_msg = "finalize wav failed";
        return false;
    }
    return true;
}

static bool create_device_d3d(HWND h_wnd) {
    DXGI_SWAP_CHAIN_DESC sd = {};
    sd.BufferCount = 2;
    sd.BufferDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    sd.BufferDesc.RefreshRate.Numerator = 60;
    sd.BufferDesc.RefreshRate.Denominator = 1;
    sd.Flags = DXGI_SWAP_CHAIN_FLAG_ALLOW_MODE_SWITCH;
    sd.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
    sd.OutputWindow = h_wnd;
    sd.SampleDesc.Count = 1;
    sd.Windowed = TRUE;
    sd.SwapEffect = DXGI_SWAP_EFFECT_DISCARD;

    D3D_FEATURE_LEVEL feature_level;
    const D3D_FEATURE_LEVEL levels[2] = {D3D_FEATURE_LEVEL_11_0, D3D_FEATURE_LEVEL_10_0};
    if (D3D11CreateDeviceAndSwapChain(
            nullptr,
            D3D_DRIVER_TYPE_HARDWARE,
            nullptr,
            0,
            levels,
            2,
            D3D11_SDK_VERSION,
            &sd,
            &g_p_swap_chain,
            &g_pd3d_device,
            &feature_level,
            &g_pd3d_device_context
        ) != S_OK) {
        return false;
    }

    ID3D11Texture2D *p_back_buffer;
    g_p_swap_chain->GetBuffer(0, IID_PPV_ARGS(&p_back_buffer));
    g_pd3d_device->CreateRenderTargetView(p_back_buffer, nullptr, &g_main_render_target_view);
    p_back_buffer->Release();
    return true;
}

static void cleanup_device_d3d() {
    if (g_main_render_target_view) { g_main_render_target_view->Release(); g_main_render_target_view = nullptr; }
    if (g_p_swap_chain) { g_p_swap_chain->Release(); g_p_swap_chain = nullptr; }
    if (g_pd3d_device_context) { g_pd3d_device_context->Release(); g_pd3d_device_context = nullptr; }
    if (g_pd3d_device) { g_pd3d_device->Release(); g_pd3d_device = nullptr; }
}

LRESULT WINAPI wnd_proc(HWND h_wnd, UINT msg, WPARAM w_param, LPARAM l_param) {
    if (ImGui_ImplWin32_WndProcHandler(h_wnd, msg, w_param, l_param)) return true;
    switch (msg) {
        case WM_SIZE:
            if (g_pd3d_device != nullptr && w_param != SIZE_MINIMIZED) {
                if (g_main_render_target_view) {
                    g_main_render_target_view->Release();
                    g_main_render_target_view = nullptr;
                }
                g_p_swap_chain->ResizeBuffers(0, (UINT)LOWORD(l_param), (UINT)HIWORD(l_param), DXGI_FORMAT_UNKNOWN, 0);
                ID3D11Texture2D *p_back_buffer;
                g_p_swap_chain->GetBuffer(0, IID_PPV_ARGS(&p_back_buffer));
                g_pd3d_device->CreateRenderTargetView(p_back_buffer, nullptr, &g_main_render_target_view);
                p_back_buffer->Release();
            }
            return 0;
        case WM_DESTROY:
            PostQuitMessage(0);
            return 0;
        default:
            return DefWindowProc(h_wnd, msg, w_param, l_param);
    }
}

static void setup_style(ui_lang lang) {
    ImGuiStyle &style = ImGui::GetStyle();
    style.WindowRounding = 10.0f;
    style.ChildRounding = 8.0f;
    style.FrameRounding = 7.0f;
    style.PopupRounding = 8.0f;
    style.ScrollbarRounding = 8.0f;
    style.GrabRounding = 6.0f;
    style.FramePadding = ImVec2(10, 8);
    style.ItemSpacing = ImVec2(10, 8);
    style.WindowPadding = ImVec2(14, 14);

    ImVec4 *c = style.Colors;
    c[ImGuiCol_WindowBg] = ImVec4(0.07f, 0.08f, 0.10f, 1.00f);
    c[ImGuiCol_ChildBg] = ImVec4(0.10f, 0.11f, 0.14f, 0.85f);
    c[ImGuiCol_FrameBg] = ImVec4(0.15f, 0.17f, 0.20f, 1.00f);
    c[ImGuiCol_FrameBgHovered] = ImVec4(0.22f, 0.27f, 0.35f, 1.00f);
    c[ImGuiCol_FrameBgActive] = ImVec4(0.18f, 0.40f, 0.78f, 1.00f);
    c[ImGuiCol_Button] = ImVec4(0.19f, 0.36f, 0.64f, 1.00f);
    c[ImGuiCol_ButtonHovered] = ImVec4(0.25f, 0.46f, 0.78f, 1.00f);
    c[ImGuiCol_ButtonActive] = ImVec4(0.15f, 0.32f, 0.58f, 1.00f);
    c[ImGuiCol_Header] = ImVec4(0.17f, 0.30f, 0.53f, 1.00f);
    c[ImGuiCol_HeaderHovered] = ImVec4(0.24f, 0.41f, 0.69f, 1.00f);
    c[ImGuiCol_HeaderActive] = ImVec4(0.17f, 0.34f, 0.60f, 1.00f);
    c[ImGuiCol_TitleBg] = ImVec4(0.08f, 0.10f, 0.13f, 1.00f);
    c[ImGuiCol_TitleBgActive] = ImVec4(0.10f, 0.14f, 0.22f, 1.00f);
    c[ImGuiCol_Border] = ImVec4(0.25f, 0.30f, 0.38f, 0.45f);

    ImGuiIO &io = ImGui::GetIO();
    io.Fonts->Clear();
    ImFontConfig cfg;
    cfg.OversampleH = 3;
    cfg.OversampleV = 2;
    cfg.PixelSnapH = true;
    const float font_size = 18.0f;
    if (lang == ui_lang::zh) {
        if (io.Fonts->AddFontFromFileTTF("C:/Windows/Fonts/msyh.ttc", font_size, &cfg, io.Fonts->GetGlyphRangesChineseFull()) == nullptr) {
            io.Fonts->AddFontDefault();
        }
    } else {
        if (io.Fonts->AddFontFromFileTTF("C:/Windows/Fonts/segoeui.ttf", font_size, &cfg, io.Fonts->GetGlyphRangesDefault()) == nullptr) {
            io.Fonts->AddFontDefault();
        }
    }
}

static const char *model_label(ui_lang lang, model_id id) {
    switch (id) {
        case model_id::model_06b_base: return tr(lang, "音色克隆模型-0.6B", "Voice Clone Model - 0.6B");
        case model_id::model_06b_custom: return tr(lang, "预置音色模型-0.6B", "Custom Voice Model - 0.6B");
        case model_id::model_17b_base: return tr(lang, "音色克隆模型-1.7B", "Voice Clone Model - 1.7B");
        case model_id::model_17b_custom: return tr(lang, "预置音色模型-1.7B", "Custom Voice Model - 1.7B");
        case model_id::model_17b_design: return tr(lang, "音色设计模型-1.7B", "Voice Design Model - 1.7B");
    }
    return "";
}

static void validate_or_set_popup(app_state &state, model_id selected, bool save_output, const std::string &output_path) {
    if (std::string(state.text_input).empty()) {
        state.validation_message = tr(state.lang, "请输入要合成的文本。", "Please input text to synthesize.");
        state.request_validation_popup = true;
        return;
    }
    if (is_base(selected)) {
        if (std::string(state.reference_audio).empty()) {
            state.validation_message = tr(state.lang, "音色克隆模型必须提供参考音频。", "Voice clone model requires a reference audio file.");
            state.request_validation_popup = true;
            return;
        }
        if (!state.x_vector_only && std::string(state.reference_text_input).empty()) {
            state.validation_message = tr(state.lang, "请填写参考文本，或开启无需参考文本。", "Please provide reference text, or enable No ref text.");
            state.request_validation_popup = true;
            return;
        }
    }
    if (is_custom(selected)) {
        if (state.speaker_index < 0 || state.speaker_index >= (int)(sizeof(k_speakers) / sizeof(k_speakers[0]))) {
            state.validation_message = tr(state.lang, "请选择 Speaker。", "Please select a speaker.");
            state.request_validation_popup = true;
            return;
        }
    }
    if (is_design(selected) && std::string(state.instruct_input).empty()) {
        state.validation_message = tr(state.lang, "音色设计模型必须填写风格描述。", "Voice design model requires an instruction prompt.");
        state.request_validation_popup = true;
        return;
    }
    if (save_output && output_path.empty()) {
        state.validation_message = tr(state.lang, "请填写输出文件名。", "Please provide output file name.");
        state.request_validation_popup = true;
        return;
    }
}

static void run_generation(app_state &state, model_id selected) {
    const std::string tts_model = model_path_for(selected);
    const std::string tok_model = tokenizer_path();
    if (tts_model.empty() || tok_model.empty()) {
        append_log(state, "[error] built-in model files not found");
        return;
    }

    std::string text;
    std::string ref_text;
    std::string instruct;
    if (!load_text_or_file(state.text_input, text) || text.empty()) {
        append_log(state, "[error] text load failed");
        return;
    }
    if (!std::string(state.reference_text_input).empty() && !load_text_or_file(state.reference_text_input, ref_text)) {
        append_log(state, "[error] ref text load failed");
        return;
    }
    if (!std::string(state.instruct_input).empty() && !load_text_or_file(state.instruct_input, instruct)) {
        append_log(state, "[error] instruct load failed");
        return;
    }

    const auto segments = segment_text(text, 80);
    state.segment_total = (int)segments.size();
    state.segment_completed = 0;
    state.progress_percent = 0;

    if (segments.empty()) {
        append_log(state, "[error] text segmentation produced no segments");
        return;
    }

    if (segments.size() == 1) {
        append_log(state, "[run] single segment, generating...");
    } else {
        append_log(state, "[run] text split into " + std::to_string(segments.size()) + " segments");
    }

    qwen3_tts::tts_params params;
    params.temperature = state.temperature;
    params.top_k = state.top_k;
    params.top_p = state.top_p;
    params.max_audio_tokens = state.max_tokens;
    params.repetition_penalty = state.repetition_penalty;
    params.seed = state.seed;
    params.n_threads = state.threads;
    params.x_vector_only_mode = state.x_vector_only;
    if (state.synthesis_language_index < 0 ||
        state.synthesis_language_index >= (int)(sizeof(k_synthesis_languages) / sizeof(k_synthesis_languages[0])) ||
        !parse_language_id_from_key(k_synthesis_languages[state.synthesis_language_index].key, params.language_id)) {
        append_log(state, "[error] invalid synthesis language option");
        return;
    }

    if (state.stream_realtime) {
        params.temperature = 0.6f;
        params.top_k = 30;
        params.top_p = 0.95f;
        params.repetition_penalty = 1.05f;
        params.seed = 123;
    }

    qwen3_tts::Qwen3TTS tts;
    append_log(state, "[run] loading model");
    if (!tts.load_models_from_files(tts_model, tok_model)) {
        append_log(state, "[error] " + tts.get_error());
        return;
    }

    // Set progress callback for all modes (streaming and non-streaming)
    tts.set_progress_callback([&](int tok, int max_tok) {
        const size_t seg_idx_cur = (size_t)state.segment_completed.load();
        const int total_segs = state.segment_total.load();
        if (total_segs > 0 && max_tok > 0) {
            int seg_prog = (int)((seg_idx_cur * 100 + (size_t)tok * 100 / (size_t)max_tok) / (size_t)total_segs);
            state.progress_percent = std::min(seg_prog, 100);
        }
    });

    const std::string output = state.save_output ? std::string(state.output_path) : std::string();

    if (state.stream) {
        for (size_t seg_idx = 0; seg_idx < segments.size(); ++seg_idx) {
            if (state.cancel_requested.load()) {
                tts.request_cancel();
                append_log(state, "[cancelled] stream generation stopped");
                break;
            }

            append_log(state, "[run] segment " + std::to_string(seg_idx + 1) + "/" + std::to_string(segments.size()));

            qwen3_tts::tts_stream_params stream_params;
            stream_params.tts = params;
            if (state.stream_realtime) {
                stream_params.stream_chunk_frames = std::max(stream_params.stream_chunk_frames, 16);
                stream_params.stream_decoder_left_context_frames = 1;
                stream_params.stream_decoder_lookahead_frames = 2;
                stream_params.stream_queue_capacity = std::max(stream_params.stream_queue_capacity, 24);
            }

            std::shared_ptr<qwen3_tts::tts_stream_session> session;
            if (is_base(selected)) {
                stream_params.tts.task_type = qwen3_tts::tts_task_type::voice_clone;
                stream_params.tts.reference_text = ref_text;
                session = tts.synthesize_with_voice_stream(segments[seg_idx].text, std::string(state.reference_audio), stream_params);
            } else if (is_custom(selected)) {
                stream_params.tts.task_type = qwen3_tts::tts_task_type::custom_voice;
                stream_params.tts.speaker = k_speakers[state.speaker_index].id;
                stream_params.tts.instruct = is_custom_17b(selected) ? instruct : std::string();
                session = tts.synthesize_stream(segments[seg_idx].text, stream_params);
            } else {
                stream_params.tts.task_type = qwen3_tts::tts_task_type::voice_design;
                stream_params.tts.instruct = instruct;
                session = tts.synthesize_stream(segments[seg_idx].text, stream_params);
            }

            state.segment_completed = (int)seg_idx;

            if (!session) {
                append_log(state, "[error] " + tts.get_error());
                return;
            }
            std::string err;
            if (!run_stream_session(session, output, state.play_audio, err, state)) {
                if (state.cancel_requested.load()) {
                    append_log(state, "[cancelled] stream generation stopped");
                } else {
                    append_log(state, "[error] " + err);
                }
                return;
            }

            state.segment_completed = (int)seg_idx + 1;
            state.progress_percent = (int)((seg_idx + 1) * 100 / segments.size());
        }
        tts.set_progress_callback(nullptr);
        append_log(state, "[ok] stream generation done");
        state.progress_percent = 0;
        return;
    }

    std::vector<float> all_audio;
    int32_t sample_rate = 24000;

    for (size_t seg_idx = 0; seg_idx < segments.size(); ++seg_idx) {
        if (state.cancel_requested.load()) {
            tts.request_cancel();
            append_log(state, "[cancelled] generation stopped");
            break;
        }

        append_log(state, "[run] segment " + std::to_string(seg_idx + 1) + "/" + std::to_string(segments.size()));

        qwen3_tts::tts_result result;
        if (is_base(selected)) {
            params.task_type = qwen3_tts::tts_task_type::voice_clone;
            params.reference_text = ref_text;
            result = tts.synthesize_with_voice(segments[seg_idx].text, std::string(state.reference_audio), params);
        } else if (is_custom(selected)) {
            params.task_type = qwen3_tts::tts_task_type::custom_voice;
            params.speaker = k_speakers[state.speaker_index].id;
            params.instruct = is_custom_17b(selected) ? instruct : std::string();
            result = tts.synthesize(segments[seg_idx].text, params);
        } else {
            params.task_type = qwen3_tts::tts_task_type::voice_design;
            params.instruct = instruct;
            result = tts.synthesize(segments[seg_idx].text, params);
        }

        if (!result.success) {
            if (state.cancel_requested.load()) {
                append_log(state, "[cancelled] generation stopped");
            } else {
                append_log(state, "[error] " + result.error_msg);
            }
            return;
        }

        if (result.sample_rate > 0) sample_rate = result.sample_rate;
        all_audio.insert(all_audio.end(), result.audio.begin(), result.audio.end());

        state.segment_completed = (int)seg_idx + 1;
        state.progress_percent = (int)((seg_idx + 1) * 100 / segments.size());
    }

    tts.set_progress_callback(nullptr);

    if (!all_audio.empty() && !state.cancel_requested.load()) {
        if (state.save_output && !qwen3_tts::save_audio_file(output, all_audio, sample_rate)) {
            append_log(state, "[error] save output failed");
            state.progress_percent = 0;
            return;
        }
        if (state.play_audio) {
            // Play audio with cancellation support
            StreamPlayer player;
            std::string play_err;
            if (!player.open(sample_rate, play_err)) {
                append_log(state, "[error] " + play_err);
                state.progress_percent = 0;
                return;
            }
            // Push audio in chunks for cancellable playback
            const size_t chunk_size = 2400; // ~100ms at 24kHz
            size_t offset = 0;
            while (offset < all_audio.size()) {
                if (state.cancel_requested.load()) {
                    player.cancel();
                    append_log(state, "[cancelled] playback stopped");
                    break;
                }
                size_t end = std::min(offset + chunk_size, all_audio.size());
                std::vector<float> chunk(all_audio.begin() + offset, all_audio.begin() + end);
                if (!player.push(chunk, play_err)) {
                    append_log(state, "[error] " + play_err);
                    break;
                }
                offset = end;
            }
            if (!player.is_cancelled()) {
                player.finish(play_err);
            }
        }
    }

    if (state.cancel_requested.load()) {
        append_log(state, "[cancelled] generation stopped");
    } else {
        append_log(state, "[ok] generation done");
    }
    state.progress_percent = 0;
}

static fs::path locate_repo_root() {
    fs::path cur = fs::current_path();
    for (int i = 0; i < 8; ++i) {
        if (fs::exists(cur / "scripts" / "setup_pipeline_models.py") && fs::exists(cur / "CMakeLists.txt")) return cur;
        if (!cur.has_parent_path()) break;
        cur = cur.parent_path();
    }
    return fs::current_path();
}

} // namespace

int WINAPI WinMain(HINSTANCE, HINSTANCE, LPSTR, int) {
    fs::current_path(locate_repo_root());

    ImGui_ImplWin32_EnableDpiAwareness();

    WNDCLASSEXA wc = {sizeof(WNDCLASSEXA), CS_CLASSDC, wnd_proc, 0L, 0L, GetModuleHandle(nullptr), nullptr, nullptr, nullptr, nullptr, "qwen3_tts_gui", nullptr};
    RegisterClassExA(&wc);
    HWND hwnd = CreateWindowA(wc.lpszClassName, "Qwen3-tts.cpp", WS_OVERLAPPEDWINDOW, 80, 80, 1280, 860, nullptr, nullptr, wc.hInstance, nullptr);

    if (!create_device_d3d(hwnd)) {
        cleanup_device_d3d();
        UnregisterClassA(wc.lpszClassName, wc.hInstance);
        return 1;
    }

    ShowWindow(hwnd, SW_SHOWDEFAULT);
    UpdateWindow(hwnd);

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO &io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;

    app_state state;
    setup_style(state.lang);

    ImGui_ImplWin32_Init(hwnd);
    ImGui_ImplDX11_Init(g_pd3d_device, g_pd3d_device_context);

    append_log(state, std::string("[cwd] ") + fs::current_path().string());
    strncpy_s(state.output_path, save_file_default().c_str(), _TRUNCATE);

    const std::string config_path = save_config_path();
    if (fs::exists(config_path)) {
        if (load_config(state, config_path)) {
            append_log(state, tr(state.lang, "[ok] 已加载配置", "[ok] config loaded"));
        } else {
            append_log(state, tr(state.lang, "[warn] 配置加载失败，使用默认值", "[warn] config load failed, using defaults"));
        }
    } else {
        if (save_config(state, config_path)) {
            append_log(state, tr(state.lang, "[ok] 已创建默认配置: ", "[ok] created default config: ") + config_path);
        } else {
            append_log(state, tr(state.lang, "[warn] 创建默认配置失败: ", "[warn] failed to create default config: ") + config_path);
        }
    }

    state.model_setup_running = true;
    state.model_setup_thread = std::thread([&state]() {
        const bool ok = ensure_models_layout(state);
        state.models_ready = ok;
        state.model_setup_running = false;
    });

    bool done = false;
    while (!done) {
        MSG msg;
        while (PeekMessage(&msg, nullptr, 0U, 0U, PM_REMOVE)) {
            TranslateMessage(&msg);
            DispatchMessage(&msg);
            if (msg.message == WM_QUIT) done = true;
        }
        if (done) break;

        if (!state.model_setup_running && state.model_setup_thread.joinable()) state.model_setup_thread.join();
        if (!state.generation_running && state.generation_thread.joinable()) state.generation_thread.join();

        ImGui_ImplDX11_NewFrame();
        ImGui_ImplWin32_NewFrame();
        ImGui::NewFrame();

        ImGui::SetNextWindowPos(ImVec2(0, 0), ImGuiCond_Always);
        ImGui::SetNextWindowSize(io.DisplaySize, ImGuiCond_Always);
        ImGui::Begin(
            "Qwen3 TTS 全系列模型",
            nullptr,
            ImGuiWindowFlags_NoResize |
            ImGuiWindowFlags_NoMove |
            ImGuiWindowFlags_NoCollapse |
            ImGuiWindowFlags_NoScrollbar
        );

        if (state.model_setup_running) {
            ImGui::TextUnformatted(tr(state.lang, "正在准备内置模型，请稍候...", "Preparing built-in models, please wait..."));
        } else if (!state.models_ready) {
            ImGui::TextColored(ImVec4(1.0f, 0.4f, 0.4f, 1.0f), "%s", tr(state.lang, "模型准备失败，请查看日志。", "Model setup failed. Check logs."));
        } else {
            ImGui::TextUnformatted(tr(state.lang, "内置模型已就绪", "Built-in models ready"));
        }

        if (ImGui::Button(tr(state.lang, "重置参数", "Reset Parameters"))) {
            reset_state_to_defaults(state);
            if (save_config(state, save_config_path())) {
                append_log(state, tr(state.lang, "[ok] 参数已重置并保存", "[ok] parameters reset and saved"));
            }
        }
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("%s", tr(state.lang, "重置所有参数为默认值", "Reset all parameters to defaults"));
        }

        model_id selected = (model_id)state.model_index;
        const char *model_items[5] = {
            model_label(state.lang, model_id::model_06b_base),
            model_label(state.lang, model_id::model_06b_custom),
            model_label(state.lang, model_id::model_17b_base),
            model_label(state.lang, model_id::model_17b_custom),
            model_label(state.lang, model_id::model_17b_design),
        };

        ImGui::SeparatorText(tr(state.lang, "模型", "Model"));
        if (ImGui::Combo(tr(state.lang, "模型选择", "Model Selection"), &state.model_index, model_items, 5)) {
            state.config_dirty = true;
        }
        selected = (model_id)state.model_index;

        ImGui::SeparatorText(tr(state.lang, "基础输入", "Input"));
        if (state.text_input[0] == '\0') {
            ImGui::TextDisabled("%s", tr(state.lang, "输入需要合成的文本...", "Input text to synthesize..."));
        }
        if (ImGui::InputTextMultiline(
            "##synthesis_text",
            state.text_input,
            sizeof(state.text_input),
            ImVec2(-1.0f, 180),
            ImGuiInputTextFlags_NoHorizontalScroll
        )) {
            state.config_dirty = true;
        }

        std::vector<const char *> lang_items;
        lang_items.reserve(sizeof(k_synthesis_languages) / sizeof(k_synthesis_languages[0]));
        for (const auto &opt : k_synthesis_languages) {
            lang_items.push_back(state.lang == ui_lang::zh ? opt.name_zh : opt.name_en);
        }
        if (ImGui::Combo(
            tr(state.lang, "合成语言", "Synthesis Language"),
            &state.synthesis_language_index,
            lang_items.data(),
            (int)lang_items.size()
        )) {
            state.config_dirty = true;
        }

        if (ImGui::Checkbox(tr(state.lang, "流式输出", "Streaming"), &state.stream)) { state.config_dirty = true; }
        ImGui::SameLine();
        if (ImGui::Checkbox(tr(state.lang, "实时优先", "Realtime Priority"), &state.stream_realtime)) { state.config_dirty = true; }
        ImGui::SameLine();
        if (ImGui::Checkbox(tr(state.lang, "播放音频", "Play Audio"), &state.play_audio)) { state.config_dirty = true; }

        if (is_base(selected)) {
            ImGui::SeparatorText(tr(state.lang, "音色克隆参数", "Voice Clone"));
            if (ImGui::InputText(tr(state.lang, "参考音频路径", "Reference Audio Path"), state.reference_audio, sizeof(state.reference_audio))) { state.config_dirty = true; }
            if (ImGui::InputText(tr(state.lang, "参考文本", "Reference Text"), state.reference_text_input, sizeof(state.reference_text_input))) { state.config_dirty = true; }
            if (ImGui::Checkbox(tr(state.lang, "无需参考文本", "No ref text"), &state.x_vector_only)) { state.config_dirty = true; }
        }

        if (is_custom(selected)) {
            ImGui::SeparatorText(tr(state.lang, "Speaker", "Speaker"));
            std::vector<const char *> sp;
            for (const auto &x : k_speakers) {
                sp.push_back(state.lang == ui_lang::zh ? x.name_zh : x.name_en);
            }
            if (ImGui::Combo(tr(state.lang, "声音", "Voice"), &state.speaker_index, sp.data(), (int)sp.size())) { state.config_dirty = true; }
            if (is_custom_17b(selected)) {
                if (ImGui::InputText(tr(state.lang, "风格描述（可选）", "Style Prompt (Optional)"), state.instruct_input, sizeof(state.instruct_input))) { state.config_dirty = true; }
            }
        }

        if (is_design(selected)) {
            ImGui::SeparatorText(tr(state.lang, "音色设计参数", "Voice Design"));
            if (ImGui::InputText(tr(state.lang, "风格描述", "Style Prompt"), state.instruct_input, sizeof(state.instruct_input))) { state.config_dirty = true; }
        }

        ImGui::SeparatorText(tr(state.lang, "输出", "Output"));
        if (ImGui::Checkbox(tr(state.lang, "保存到文件", "Save to File"), &state.save_output)) { state.config_dirty = true; }
        if (state.save_output) {
            if (ImGui::InputText(tr(state.lang, "输出文件", "Output File"), state.output_path, sizeof(state.output_path))) { state.config_dirty = true; }
        }

        if (ImGui::CollapsingHeader(tr(state.lang, "高级参数", "Advanced"))) {
            if (ImGui::SliderFloat(tr(state.lang, "温度", "Temperature"), &state.temperature, 0.0f, 2.0f, "%.2f")) { state.config_dirty = true; }
            if (ImGui::SliderInt("Top-k", &state.top_k, 0, 200)) { state.config_dirty = true; }
            if (ImGui::SliderFloat("Top-p", &state.top_p, 0.0f, 1.0f, "%.2f")) { state.config_dirty = true; }
            if (ImGui::SliderInt(tr(state.lang, "最大 Token", "Max Tokens"), &state.max_tokens, 32, 8192)) { state.config_dirty = true; }
            if (ImGui::SliderFloat(tr(state.lang, "重复惩罚", "Repetition Penalty"), &state.repetition_penalty, 1.0f, 2.0f, "%.2f")) { state.config_dirty = true; }
            if (ImGui::InputInt("Seed", &state.seed)) { state.config_dirty = true; }
            if (ImGui::InputInt(tr(state.lang, "线程数", "Threads"), &state.threads)) { state.config_dirty = true; }
            if (state.threads < 1) state.threads = 1;
        }

        if (state.generation_running) {
            ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.8f, 0.15f, 0.15f, 1.0f));
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.9f, 0.25f, 0.25f, 1.0f));
            ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.7f, 0.1f, 0.1f, 1.0f));
            if (ImGui::Button(tr(state.lang, "停止", "Stop"), ImVec2(180, 42))) {
                state.cancel_requested = true;
            }
            ImGui::PopStyleColor(3);
        } else {
            bool can_generate = state.models_ready && !state.generation_running;
            if (!can_generate) ImGui::BeginDisabled();
            if (ImGui::Button(tr(state.lang, "开始生成", "Generate"), ImVec2(180, 42))) {
                state.cancel_requested = false;
                state.segment_completed = 0;
                state.progress_percent = 0;
                validate_or_set_popup(state, selected, state.save_output, state.output_path);
                if (!state.request_validation_popup) {
                    state.generation_running = true;
                    state.generation_thread = std::thread([&state, selected]() {
                        run_generation(state, selected);
                        state.generation_running = false;
                    });
                }
            }
            if (!can_generate) ImGui::EndDisabled();
        }

        if (state.request_validation_popup) {
            ImGui::OpenPopup(tr(state.lang, "参数错误", "Parameter Error"));
            state.request_validation_popup = false;
        }
        if (ImGui::BeginPopupModal(tr(state.lang, "参数错误", "Parameter Error"), nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
            ImGui::TextWrapped("%s", state.validation_message.c_str());
            if (ImGui::Button("OK", ImVec2(120, 0))) {
                ImGui::CloseCurrentPopup();
            }
            ImGui::EndPopup();
        }

        if (state.generation_running) {
            float progress = (float)state.progress_percent / 100.0f;
            progress = (progress < 0.0f) ? 0.0f : (progress > 1.0f) ? 1.0f : progress;
            char label[96];
            if (state.segment_total > 1) {
                snprintf(label, sizeof(label), "%s %d/%d  %d%%",
                    state.lang == ui_lang::zh ? "段" : "seg",
                    (int)state.segment_completed, (int)state.segment_total,
                    (int)state.progress_percent);
            } else {
                snprintf(label, sizeof(label), "%d%%", (int)state.progress_percent);
            }
            ImGui::ProgressBar(progress, ImVec2(-1.0f, 0), label);
        }

        ImGui::SeparatorText(tr(state.lang, "日志", "Logs"));
        std::string logs;
        {
            std::lock_guard<std::mutex> lock(state.log_mutex);
            logs = state.log_text;
        }
        ImGui::BeginChild("##logs", ImVec2(-1.0f, 220), true);
        ImGui::TextUnformatted(logs.c_str());
        if (ImGui::GetScrollY() >= ImGui::GetScrollMaxY()) ImGui::SetScrollHereY(1.0f);
        ImGui::EndChild();

        ImGui::End();

        // Auto-save config when parameters change
        if (state.config_dirty && !state.generation_running) {
            state.config_dirty = false;
            save_config(state, save_config_path());
        }

        ImGui::Render();
        const float clear_color[4] = {0.06f, 0.07f, 0.09f, 1.00f};
        g_pd3d_device_context->OMSetRenderTargets(1, &g_main_render_target_view, nullptr);
        g_pd3d_device_context->ClearRenderTargetView(g_main_render_target_view, clear_color);
        ImGui_ImplDX11_RenderDrawData(ImGui::GetDrawData());
        g_p_swap_chain->Present(1, 0);
    }

    if (state.model_setup_thread.joinable()) state.model_setup_thread.join();
    if (state.generation_thread.joinable()) state.generation_thread.join();

    ImGui_ImplDX11_Shutdown();
    ImGui_ImplWin32_Shutdown();
    ImGui::DestroyContext();

    cleanup_device_d3d();
    DestroyWindow(hwnd);
    UnregisterClassA(wc.lpszClassName, wc.hInstance);
    return 0;
}
