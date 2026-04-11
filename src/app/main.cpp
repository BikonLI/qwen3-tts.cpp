#include "qwen3_tts.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <string>
#include <vector>

namespace {

using qwen3_tts::Qwen3TTSModelHub;
using qwen3_tts::tts_common_params;
using qwen3_tts::tts_custom_voice_params;
using qwen3_tts::tts_model_id;
using qwen3_tts::tts_result;
using qwen3_tts::tts_voice_clone_params;
using qwen3_tts::tts_voice_design_params;

enum class synth_mode {
    auto_mode = 0,
    voice_clone,
    custom_voice,
    custom_voice_instruct,
    voice_design,
};

const char *model_id_name(tts_model_id model_id) {
    switch (model_id) {
        case tts_model_id::model_06b_base: return "06b_base";
        case tts_model_id::model_06b_custom_voice: return "06b_custom_voice";
        case tts_model_id::model_1_7b_base: return "1.7b_base";
        case tts_model_id::model_1_7b_custom_voice: return "1.7b_custom_voice";
        case tts_model_id::model_1_7b_voice_design: return "1.7b_voice_design";
        default: return "unknown";
    }
}

bool parse_model_id(const std::string &s, tts_model_id &out) {
    if (s == "06b_base") {
        out = tts_model_id::model_06b_base;
        return true;
    }
    if (s == "06b_custom_voice") {
        out = tts_model_id::model_06b_custom_voice;
        return true;
    }
    if (s == "1.7b_base") {
        out = tts_model_id::model_1_7b_base;
        return true;
    }
    if (s == "1.7b_custom_voice") {
        out = tts_model_id::model_1_7b_custom_voice;
        return true;
    }
    if (s == "1.7b_voice_design") {
        out = tts_model_id::model_1_7b_voice_design;
        return true;
    }
    return false;
}

bool parse_mode(const std::string &s, synth_mode &out) {
    if (s == "auto") {
        out = synth_mode::auto_mode;
        return true;
    }
    if (s == "voice" || s == "voice_clone") {
        out = synth_mode::voice_clone;
        return true;
    }
    if (s == "custom" || s == "custom_voice") {
        out = synth_mode::custom_voice;
        return true;
    }
    if (s == "custom_instruct") {
        out = synth_mode::custom_voice_instruct;
        return true;
    }
    if (s == "design" || s == "voice_design") {
        out = synth_mode::voice_design;
        return true;
    }
    return false;
}

bool parse_language_id(const std::string &lang, int32_t &out) {
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

bool load_text_or_file(const std::string &input, std::string &out_text) {
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

void print_loaded_models(const Qwen3TTSModelHub &hub) {
    std::vector<tts_model_id> ids = hub.loaded_models();
    fprintf(stderr, "Loaded models (%zu):\n", ids.size());
    for (tts_model_id id : ids) {
        fprintf(stderr, "  - %s\n", model_id_name(id));
    }
}

void print_usage(const char *program) {
    fprintf(stderr, "Usage: %s [load options] --model <model-id> --text <text> [run options]\n", program);
    fprintf(stderr, "\n");
    fprintf(stderr, "Load options (you can specify multiple):\n");
    fprintf(stderr, "  --tokenizer-12hz <path|dir>       Shared Qwen3-TTS-Tokenizer-12Hz GGUF (optional)\n");
    fprintf(stderr, "  --load-06b-base <dir>             Load Qwen3-TTS-12Hz-0.6B-Base\n");
    fprintf(stderr, "  --load-06b-custom-voice <dir>     Load Qwen3-TTS-12Hz-0.6B-CustomVoice\n");
    fprintf(stderr, "  --load-17b-base <dir>             Load Qwen3-TTS-12Hz-1.7B-Base\n");
    fprintf(stderr, "  --load-17b-custom-voice <dir>     Load Qwen3-TTS-12Hz-1.7B-CustomVoice\n");
    fprintf(stderr, "  --load-17b-voice-design <dir>     Load Qwen3-TTS-12Hz-1.7B-VoiceDesign\n");
    fprintf(stderr, "\n");
    fprintf(stderr, "Run options:\n");
    fprintf(stderr, "  --model <id>                      06b_base | 06b_custom_voice | 1.7b_base | 1.7b_custom_voice | 1.7b_voice_design\n");
    fprintf(stderr, "  --mode <name>                     auto | voice | custom | custom_instruct | design\n");
    fprintf(stderr, "  -t, --text <text|file>            Input text (or a file path)\n");
    fprintf(stderr, "  -o, --output <wav>                Output WAV file; omit to play audio\n");
    fprintf(stderr, "  -r, --reference <wav>             Reference audio for voice clone\n");
    fprintf(stderr, "  --ref-text <text>                 Reference transcript for voice clone\n");
    fprintf(stderr, "  --x-vector-only                   Voice clone without reference_text\n");
    fprintf(stderr, "  --speaker <name>                  CustomVoice speaker\n");
    fprintf(stderr, "  --instruct <text>                 Instruct text (1.7B CustomVoice / VoiceDesign)\n");
    fprintf(stderr, "  -l, --language <lang>             auto,en,ru,zh,ja,ko,de,fr,es,it,pt,beijing_dialect,sichuan_dialect\n");
    fprintf(stderr, "  --temperature <v>                 Sampling temperature (default: 0.9)\n");
    fprintf(stderr, "  --top-k <n>                       Top-k (default: 50)\n");
    fprintf(stderr, "  --top-p <v>                       Top-p (default: 1.0)\n");
    fprintf(stderr, "  --max-tokens <n>                  Max audio tokens (default: 4096)\n");
    fprintf(stderr, "  --repetition-penalty <v>          Repetition penalty (default: 1.05)\n");
    fprintf(stderr, "  -j, --threads <n>                 Number of threads\n");
    fprintf(stderr, "  --list-models                     Print loaded model IDs and exit\n");
    fprintf(stderr, "  -h, --help                        Show this help\n");
    fprintf(stderr, "\n");
    fprintf(stderr, "Examples:\n");
    fprintf(stderr, "  %s --tokenizer-12hz models/qwen3-tts-tokenizer-f16.gguf --load-17b-voice-design models-1.7B-v2/Qwen3-TTS-12Hz-1.7B-VoiceDesign --model 1.7b_voice_design --mode design -t \"Please speak slowly\" --instruct \"female voice, warm tone, calm pace\" -o design.wav\n", program);
    fprintf(stderr, "  %s --tokenizer-12hz models/qwen3-tts-tokenizer-f16.gguf --load-06b-base models/Qwen3-TTS-12Hz-0.6B-Base --model 06b_base --mode voice -t \"Hello\" -r clone.wav --ref-text \"Hello\" -o clone_out.wav\n", program);
}

} // namespace

int main(int argc, char **argv) {
    std::string tokenizer_12hz_path;

    std::string load_06b_base_dir;
    std::string load_06b_custom_dir;
    std::string load_17b_base_dir;
    std::string load_17b_custom_dir;
    std::string load_17b_design_dir;

    std::string selected_model_name;
    synth_mode mode = synth_mode::auto_mode;

    std::string text_input;
    std::string text;
    std::string output_file;
    std::string reference_audio;
    std::string reference_text;
    std::string speaker;
    std::string instruct;
    bool x_vector_only_mode = false;
    bool list_models_only = false;

    tts_common_params common_params;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            return 0;
        } else if (arg == "--tokenizer-12hz") {
            if (++i >= argc) {
                fprintf(stderr, "Error: missing tokenizer-12hz value\n");
                return 1;
            }
            tokenizer_12hz_path = argv[i];
        } else if (arg == "--load-06b-base") {
            if (++i >= argc) {
                fprintf(stderr, "Error: missing --load-06b-base value\n");
                return 1;
            }
            load_06b_base_dir = argv[i];
        } else if (arg == "--load-06b-custom-voice") {
            if (++i >= argc) {
                fprintf(stderr, "Error: missing --load-06b-custom-voice value\n");
                return 1;
            }
            load_06b_custom_dir = argv[i];
        } else if (arg == "--load-17b-base") {
            if (++i >= argc) {
                fprintf(stderr, "Error: missing --load-17b-base value\n");
                return 1;
            }
            load_17b_base_dir = argv[i];
        } else if (arg == "--load-17b-custom-voice") {
            if (++i >= argc) {
                fprintf(stderr, "Error: missing --load-17b-custom-voice value\n");
                return 1;
            }
            load_17b_custom_dir = argv[i];
        } else if (arg == "--load-17b-voice-design") {
            if (++i >= argc) {
                fprintf(stderr, "Error: missing --load-17b-voice-design value\n");
                return 1;
            }
            load_17b_design_dir = argv[i];
        } else if (arg == "--model") {
            if (++i >= argc) {
                fprintf(stderr, "Error: missing --model value\n");
                return 1;
            }
            selected_model_name = argv[i];
        } else if (arg == "--mode") {
            if (++i >= argc) {
                fprintf(stderr, "Error: missing --mode value\n");
                return 1;
            }
            if (!parse_mode(argv[i], mode)) {
                fprintf(stderr, "Error: invalid mode '%s'\n", argv[i]);
                return 1;
            }
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
            x_vector_only_mode = true;
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
            common_params.language_id = language_id;
        } else if (arg == "--temperature") {
            if (++i >= argc) {
                fprintf(stderr, "Error: missing --temperature value\n");
                return 1;
            }
            common_params.temperature = std::stof(argv[i]);
        } else if (arg == "--top-k") {
            if (++i >= argc) {
                fprintf(stderr, "Error: missing --top-k value\n");
                return 1;
            }
            common_params.top_k = std::stoi(argv[i]);
        } else if (arg == "--top-p") {
            if (++i >= argc) {
                fprintf(stderr, "Error: missing --top-p value\n");
                return 1;
            }
            common_params.top_p = std::stof(argv[i]);
        } else if (arg == "--max-tokens") {
            if (++i >= argc) {
                fprintf(stderr, "Error: missing --max-tokens value\n");
                return 1;
            }
            common_params.max_audio_tokens = std::stoi(argv[i]);
        } else if (arg == "--repetition-penalty") {
            if (++i >= argc) {
                fprintf(stderr, "Error: missing --repetition-penalty value\n");
                return 1;
            }
            common_params.repetition_penalty = std::stof(argv[i]);
        } else if (arg == "-j" || arg == "--threads") {
            if (++i >= argc) {
                fprintf(stderr, "Error: missing --threads value\n");
                return 1;
            }
            common_params.n_threads = std::stoi(argv[i]);
        } else if (arg == "--list-models") {
            list_models_only = true;
        } else {
            fprintf(stderr, "Error: unknown argument: %s\n", arg.c_str());
            print_usage(argv[0]);
            return 1;
        }
    }

    if (load_06b_base_dir.empty() &&
        load_06b_custom_dir.empty() &&
        load_17b_base_dir.empty() &&
        load_17b_custom_dir.empty() &&
        load_17b_design_dir.empty()) {
        fprintf(stderr, "Error: at least one --load-* option is required\n");
        print_usage(argv[0]);
        return 1;
    }

    Qwen3TTSModelHub hub;
    if (!tokenizer_12hz_path.empty()) {
        fprintf(stderr, "Using shared Tokenizer-12Hz from: %s\n", tokenizer_12hz_path.c_str());
        if (!hub.set_shared_tokenizer_12hz(tokenizer_12hz_path)) {
            fprintf(stderr, "Error: %s\n", hub.get_error().c_str());
            return 1;
        }
    }

    if (!load_06b_base_dir.empty()) {
        fprintf(stderr, "Loading 06b_base from: %s\n", load_06b_base_dir.c_str());
        if (!hub.load_models_06b_base(load_06b_base_dir)) {
            fprintf(stderr, "Error: %s\n", hub.get_error().c_str());
            return 1;
        }
    }
    if (!load_06b_custom_dir.empty()) {
        fprintf(stderr, "Loading 06b_custom_voice from: %s\n", load_06b_custom_dir.c_str());
        if (!hub.load_models_06b_custom_voice(load_06b_custom_dir)) {
            fprintf(stderr, "Error: %s\n", hub.get_error().c_str());
            return 1;
        }
    }
    if (!load_17b_base_dir.empty()) {
        fprintf(stderr, "Loading 1.7b_base from: %s\n", load_17b_base_dir.c_str());
        if (!hub.load_models_1_7b_base(load_17b_base_dir)) {
            fprintf(stderr, "Error: %s\n", hub.get_error().c_str());
            return 1;
        }
    }
    if (!load_17b_custom_dir.empty()) {
        fprintf(stderr, "Loading 1.7b_custom_voice from: %s\n", load_17b_custom_dir.c_str());
        if (!hub.load_models_1_7b_custom_voice(load_17b_custom_dir)) {
            fprintf(stderr, "Error: %s\n", hub.get_error().c_str());
            return 1;
        }
    }
    if (!load_17b_design_dir.empty()) {
        fprintf(stderr, "Loading 1.7b_voice_design from: %s\n", load_17b_design_dir.c_str());
        if (!hub.load_models_1_7b_voice_design(load_17b_design_dir)) {
            fprintf(stderr, "Error: %s\n", hub.get_error().c_str());
            return 1;
        }
    }

    print_loaded_models(hub);
    if (list_models_only) {
        return 0;
    }

    if (selected_model_name.empty()) {
        std::vector<tts_model_id> loaded = hub.loaded_models();
        if (loaded.size() == 1) {
            selected_model_name = model_id_name(loaded.front());
            fprintf(stderr, "Auto-selected model: %s\n", selected_model_name.c_str());
        } else {
            fprintf(stderr, "Error: --model is required when multiple models are loaded\n");
            return 1;
        }
    }

    tts_model_id model_id = tts_model_id::model_06b_base;
    if (!parse_model_id(selected_model_name, model_id)) {
        fprintf(stderr, "Error: unsupported --model value '%s'\n", selected_model_name.c_str());
        return 1;
    }
    if (!hub.is_model_loaded(model_id)) {
        fprintf(stderr, "Error: selected model is not loaded: %s\n", selected_model_name.c_str());
        return 1;
    }

    if (text_input.empty()) {
        fprintf(stderr, "Error: --text is required\n");
        return 1;
    }
    if (!load_text_or_file(text_input, text) || text.empty()) {
        fprintf(stderr, "Error: failed to load text from --text argument\n");
        return 1;
    }

    if (mode == synth_mode::auto_mode) {
        switch (model_id) {
            case tts_model_id::model_06b_base:
            case tts_model_id::model_1_7b_base:
                mode = synth_mode::voice_clone;
                break;
            case tts_model_id::model_06b_custom_voice:
            case tts_model_id::model_1_7b_custom_voice:
                mode = instruct.empty() ? synth_mode::custom_voice : synth_mode::custom_voice_instruct;
                break;
            case tts_model_id::model_1_7b_voice_design:
                mode = synth_mode::voice_design;
                break;
        }
    }

    tts_result result;
    if (mode == synth_mode::voice_clone) {
        if (reference_audio.empty()) {
            fprintf(stderr, "Error: voice mode requires --reference\n");
            return 1;
        }
        tts_voice_clone_params p;
        p.common = common_params;
        p.reference_text = reference_text;
        p.x_vector_only_mode = x_vector_only_mode;

        result = hub.synthesize_with_voice(model_id, text, reference_audio, p);
    } else if (mode == synth_mode::custom_voice) {
        tts_custom_voice_params p;
        p.common = common_params;
        p.speaker = speaker;

        if (model_id == tts_model_id::model_06b_custom_voice) {
            result = hub.synthesize_06b_custom_voice(text, p);
        } else if (model_id == tts_model_id::model_1_7b_custom_voice) {
            result = hub.synthesize_1_7b_custom_voice(text, p);
        } else {
            fprintf(stderr, "Error: custom mode only supports CustomVoice models\n");
            return 1;
        }
    } else if (mode == synth_mode::custom_voice_instruct) {
        if (instruct.empty()) {
            fprintf(stderr, "Error: custom_instruct mode requires --instruct\n");
            return 1;
        }
        tts_custom_voice_params p;
        p.common = common_params;
        p.speaker = speaker;
        result = hub.synthesize_with_instruct(model_id, text, instruct, p);
    } else if (mode == synth_mode::voice_design) {
        if (model_id != tts_model_id::model_1_7b_voice_design) {
            fprintf(stderr, "Error: design mode only supports 1.7b_voice_design\n");
            return 1;
        }
        tts_voice_design_params p;
        p.common = common_params;
        p.instruct = instruct;
        result = hub.synthesize_1_7b_voice_design(text, p);
    } else {
        fprintf(stderr, "Error: unknown mode\n");
        return 1;
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
        "Done. model=%s, sample_rate=%d, samples=%zu, audio_sec=%.2f\n",
        model_id_name(model_id),
        result.sample_rate,
        result.audio.size(),
        result.sample_rate > 0 ? (double)result.audio.size() / (double)result.sample_rate : 0.0
    );

    return 0;
}