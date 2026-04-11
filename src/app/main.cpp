#include "qwen3_tts.h"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

namespace {

namespace fs = std::filesystem;

static std::string to_lower_copy(const std::string &s) {
    std::string out = s;
    std::transform(out.begin(), out.end(), out.begin(), [](unsigned char c) { return (char)std::tolower(c); });
    return out;
}

static std::string pick_first_sorted(std::vector<std::string> &v) {
    if (v.empty()) {
        return {};
    }
    std::sort(v.begin(), v.end());
    return v.front();
}

static bool resolve_gguf_path(
    const std::string &path_or_dir,
    bool expect_tokenizer,
    std::string &resolved,
    std::string &error_msg
) {
    resolved.clear();
    error_msg.clear();

    std::error_code ec;
    if (fs::exists(path_or_dir, ec) && fs::is_regular_file(path_or_dir, ec)) {
        if (fs::path(path_or_dir).extension().string() != ".gguf") {
            error_msg = "Path is not a GGUF file: " + path_or_dir;
            return false;
        }
        resolved = path_or_dir;
        return true;
    }

    if (!fs::exists(path_or_dir, ec) || !fs::is_directory(path_or_dir, ec)) {
        error_msg = "Path does not exist: " + path_or_dir;
        return false;
    }

    std::vector<std::string> tokenizer_candidates;
    std::vector<std::string> q8_candidates;
    std::vector<std::string> f16_candidates;
    std::vector<std::string> fallback_candidates;

    for (const auto &entry : fs::directory_iterator(path_or_dir, ec)) {
        if (!entry.is_regular_file()) {
            continue;
        }
        if (entry.path().extension().string() != ".gguf") {
            continue;
        }

        const std::string path = entry.path().string();
        const std::string name = to_lower_copy(entry.path().filename().string());
        const bool is_tokenizer = (name.find("tokenizer") != std::string::npos);

        if (expect_tokenizer) {
            if (is_tokenizer) {
                tokenizer_candidates.push_back(path);
            } else {
                fallback_candidates.push_back(path);
            }
            continue;
        }

        if (is_tokenizer) {
            continue;
        }

        if (name.find("q8_0") != std::string::npos) {
            q8_candidates.push_back(path);
        } else if (name.find("f16") != std::string::npos) {
            f16_candidates.push_back(path);
        } else {
            fallback_candidates.push_back(path);
        }
    }

    if (expect_tokenizer) {
        if (!tokenizer_candidates.empty()) {
            resolved = pick_first_sorted(tokenizer_candidates);
        } else {
            resolved = pick_first_sorted(fallback_candidates);
        }
        if (resolved.empty()) {
            error_msg = "No tokenizer GGUF found in: " + path_or_dir;
            return false;
        }
        return true;
    }

    if (!q8_candidates.empty()) {
        resolved = pick_first_sorted(q8_candidates);
    } else if (!f16_candidates.empty()) {
        resolved = pick_first_sorted(f16_candidates);
    } else {
        resolved = pick_first_sorted(fallback_candidates);
    }

    if (resolved.empty()) {
        error_msg = "No functional model GGUF found in: " + path_or_dir;
        return false;
    }

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

void print_usage(const char *program) {
    fprintf(stderr, "Usage: %s -m <tts_model_path_or_dir> -mt <tokenizer_path_or_dir> -t <text> [options]\n", program);
    fprintf(stderr, "\n");
    fprintf(stderr, "Required:\n");
    fprintf(stderr, "  -m,  --model <path|dir>           Functional TTS model GGUF (or directory)\n");
    fprintf(stderr, "  -mt, --model-tokenizer <path|dir> Tokenizer model GGUF (or directory)\n");
    fprintf(stderr, "  -t,  --text <text|file>           Input text (or a file path)\n");
    fprintf(stderr, "\n");
    fprintf(stderr, "Optional:\n");
    fprintf(stderr, "  -o, --output <wav>                Output WAV file; omit to play audio\n");
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
    fprintf(stderr, "  %s -m models/gguf/0.6b-base -mt models/gguf/tokenizer -t \"Hello\" -r ref.wav --ref-text \"Hello\" -o clone.wav\n", program);
    fprintf(stderr, "  %s -m models/gguf/1.7b-custom-voice -mt models/gguf/tokenizer -t \"Hello\" --speaker ryan --instruct \"happy\" -o custom.wav\n", program);
    fprintf(stderr, "  %s -m models/gguf/1.7b-voice-design -mt models/gguf/tokenizer -t \"Hello\" --instruct \"female voice, warm tone\" -o design.wav\n", program);
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
        } else if (arg == "--top-k") {
            if (++i >= argc) {
                fprintf(stderr, "Error: missing --top-k value\n");
                return 1;
            }
            params.top_k = std::stoi(argv[i]);
        } else if (arg == "--top-p") {
            if (++i >= argc) {
                fprintf(stderr, "Error: missing --top-p value\n");
                return 1;
            }
            params.top_p = std::stof(argv[i]);
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

    if (!resolve_gguf_path(model_path_arg, false, model_path, resolve_error)) {
        fprintf(stderr, "Error: %s\n", resolve_error.c_str());
        return 1;
    }
    if (!resolve_gguf_path(tokenizer_path_arg, true, tokenizer_path, resolve_error)) {
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
