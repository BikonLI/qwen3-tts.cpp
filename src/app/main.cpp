#include "qwen3_tts.h"

#include <cstdio>
#include <cstring>
#include <string>

void print_usage(const char *program) {
    fprintf(stderr, "Usage: %s [options] -m <model_dir> -t <text>\n", program);
    fprintf(stderr, "\n");
    fprintf(stderr, "Options:\n");
    fprintf(stderr, "  -m, --model <dir>      Model directory (required)\n");
    fprintf(stderr, "  -t, --text <text>      Text to synthesize (required)\n");
    fprintf(stderr, "  -o, --output <file>    Output WAV file (default: output.wav)\n");
    fprintf(stderr, "  -r, --reference <file> Reference audio for voice cloning\n");
    fprintf(stderr, "  --task <name>          Task: auto,voice_clone,custom_voice,voice_design\n");
    fprintf(stderr, "  --model-variant <name> Variant: auto,base,custom_voice,voice_design\n");
    fprintf(stderr, "  --speaker <name>       Preset speaker name for custom_voice\n");
    fprintf(stderr, "  --instruct <text>      Natural-language instruction\n");
    fprintf(stderr, "  --ref-text <text>      Reference transcript for voice_clone\n");
    fprintf(stderr, "  --x-vector-only        Voice clone without reference text\n");
    fprintf(stderr, "  --temperature <val>    Sampling temperature (default: 0.9, 0=greedy)\n");
    fprintf(stderr, "  --top-k <n>            Top-k sampling (default: 50, 0=disabled)\n");
    fprintf(stderr, "  --top-p <val>          Top-p sampling (default: 1.0)\n");
    fprintf(stderr, "  --max-tokens <n>       Maximum audio tokens (default: 4096)\n");
    fprintf(stderr, "  --repetition-penalty <val> Repetition penalty (default: 1.05)\n");
    fprintf(stderr, "  -l, --language <lang>  Language: auto,en,ru,zh,ja,ko,de,fr,es,it,pt (default: auto)\n");
    fprintf(stderr, "  -j, --threads <n>      Number of threads (default: 4)\n");
    fprintf(stderr, "  -h, --help             Show this help\n");
    fprintf(stderr, "\n");
    fprintf(stderr, "Example:\n");
    fprintf(stderr, "  %s -m ./models -t \"Hello, world!\" -o hello.wav\n", program);
    fprintf(stderr, "  %s -m ./models -t \"Hello!\" -r reference.wav -o cloned.wav\n", program);
}

int main(int argc, char **argv) {
    std::string model_dir;
    std::string text;
    std::string output_file = "";
    std::string reference_audio;

    qwen3_tts::tts_params params;

    // Parse arguments
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];

        if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            return 0;
        } else if (arg == "-m" || arg == "--model") {
            if (++i >= argc) {
                fprintf(stderr, "Error: missing model directory\n");
                return 1;
            }
            model_dir = argv[i];
        } else if (arg == "-t" || arg == "--text") {
            if (++i >= argc) {
                fprintf(stderr, "Error: missing text\n");
                return 1;
            }
            text = argv[i];
            // if the text is a path to a file, load the file content as text
            if (FILE *f = fopen(text.c_str(), "r")) {
                fseek(f, 0, SEEK_END);
                size_t size = ftell(f);
                fseek(f, 0, SEEK_SET);
                std::string file_content(size, '\0');
                fread(&file_content[0], 1, size, f);
                fclose(f);
                text = file_content;
            }

        } else if (arg == "-o" || arg == "--output") {
            if (++i >= argc) {
                fprintf(stderr, "Error: missing output file\n");
                return 1;
            }
            output_file = argv[i];
        } else if (arg == "-r" || arg == "--reference") {
            if (++i >= argc) {
                fprintf(stderr, "Error: missing reference audio\n");
                return 1;
            }
            reference_audio = argv[i];
        } else if (arg == "--task") {
            if (++i >= argc) {
                fprintf(stderr, "Error: missing task value\n");
                return 1;
            }
            std::string task = argv[i];
            if (task == "auto")
                params.task_type = qwen3_tts::tts_task_type::auto_task;
            else if (task == "voice_clone" || task == "clone" || task == "base")
                params.task_type = qwen3_tts::tts_task_type::voice_clone;
            else if (task == "custom_voice" || task == "custom")
                params.task_type = qwen3_tts::tts_task_type::custom_voice;
            else if (task == "voice_design" || task == "design")
                params.task_type = qwen3_tts::tts_task_type::voice_design;
            else {
                fprintf(stderr, "Error: unknown task '%s'\n", task.c_str());
                return 1;
            }
        } else if (arg == "--model-variant") {
            if (++i >= argc) {
                fprintf(stderr, "Error: missing model-variant value\n");
                return 1;
            }
            std::string variant = argv[i];
            if (variant == "auto")
                params.model_variant = qwen3_tts::tts_model_variant::auto_variant;
            else if (variant == "base")
                params.model_variant = qwen3_tts::tts_model_variant::base;
            else if (variant == "custom_voice" || variant == "custom")
                params.model_variant = qwen3_tts::tts_model_variant::custom_voice;
            else if (variant == "voice_design" || variant == "design")
                params.model_variant = qwen3_tts::tts_model_variant::voice_design;
            else {
                fprintf(stderr, "Error: unknown model variant '%s'\n", variant.c_str());
                return 1;
            }
        } else if (arg == "--speaker") {
            if (++i >= argc) {
                fprintf(stderr, "Error: missing speaker value\n");
                return 1;
            }
            params.speaker = argv[i];
        } else if (arg == "--instruct") {
            if (++i >= argc) {
                fprintf(stderr, "Error: missing instruct value\n");
                return 1;
            }
            params.instruct = argv[i];
        } else if (arg == "--ref-text") {
            if (++i >= argc) {
                fprintf(stderr, "Error: missing ref-text value\n");
                return 1;
            }
            params.reference_text = argv[i];
        } else if (arg == "--x-vector-only") {
            params.x_vector_only_mode = true;
        } else if (arg == "--temperature") {
            if (++i >= argc) {
                fprintf(stderr, "Error: missing temperature value\n");
                return 1;
            }
            params.temperature = std::stof(argv[i]);
        } else if (arg == "--top-k") {
            if (++i >= argc) {
                fprintf(stderr, "Error: missing top-k value\n");
                return 1;
            }
            params.top_k = std::stoi(argv[i]);
        } else if (arg == "--top-p") {
            if (++i >= argc) {
                fprintf(stderr, "Error: missing top-p value\n");
                return 1;
            }
            params.top_p = std::stof(argv[i]);
        } else if (arg == "--max-tokens") {
            if (++i >= argc) {
                fprintf(stderr, "Error: missing max-tokens value\n");
                return 1;
            }
            params.max_audio_tokens = std::stoi(argv[i]);
        } else if (arg == "--repetition-penalty") {
            if (++i >= argc) {
                fprintf(stderr, "Error: missing repetition-penalty value\n");
                return 1;
            }
            params.repetition_penalty = std::stof(argv[i]);
        } else if (arg == "-l" || arg == "--language") {
            if (++i >= argc) {
                fprintf(stderr, "Error: missing language value\n");
                return 1;
            }
            std::string lang = argv[i];
            if (lang == "auto")
                params.language_id = -1;
            else if (lang == "en" || lang == "english")
                params.language_id = 2050;
            else if (lang == "ru" || lang == "russian")
                params.language_id = 2069;
            else if (lang == "zh" || lang == "chinese")
                params.language_id = 2055;
            else if (lang == "ja" || lang == "japanese")
                params.language_id = 2058;
            else if (lang == "ko" || lang == "korean")
                params.language_id = 2064;
            else if (lang == "de" || lang == "german")
                params.language_id = 2053;
            else if (lang == "fr" || lang == "french")
                params.language_id = 2061;
            else if (lang == "es" || lang == "spanish")
                params.language_id = 2054;
            else if (lang == "it" || lang == "italian")
                params.language_id = 2070;
            else if (lang == "pt" || lang == "portuguese")
                params.language_id = 2071;
            else {
                fprintf(
                    stderr, "Error: unknown language '%s'. Supported: auto,en,ru,zh,ja,ko,de,fr,es,it,pt\n", lang.c_str()
                );
                return 1;
            }
        } else if (arg == "-j" || arg == "--threads") {
            if (++i >= argc) {
                fprintf(stderr, "Error: missing threads value\n");
                return 1;
            }
            params.n_threads = std::stoi(argv[i]);
        } else {
            fprintf(stderr, "Error: unknown argument: %s\n", arg.c_str());
            print_usage(argv[0]);
            return 1;
        }
    }

    // Validate required arguments
    if (model_dir.empty()) {
        fprintf(stderr, "Error: model directory is required\n");
        print_usage(argv[0]);
        return 1;
    }

    if (text.empty()) {
        fprintf(stderr, "Error: text is required\n");
        print_usage(argv[0]);
        return 1;
    }

    if (params.task_type == qwen3_tts::tts_task_type::auto_task) {
        if (!reference_audio.empty()) {
            params.task_type = qwen3_tts::tts_task_type::voice_clone;
        } else if (!params.speaker.empty()) {
            params.task_type = qwen3_tts::tts_task_type::custom_voice;
        } else if (!params.instruct.empty()) {
            params.task_type = qwen3_tts::tts_task_type::voice_design;
        }
    }

    if (params.task_type == qwen3_tts::tts_task_type::voice_clone) {
        if (reference_audio.empty()) {
            fprintf(stderr, "Error: voice_clone task requires --reference\n");
            return 1;
        }
        if (!params.x_vector_only_mode && params.reference_text.empty()) {
            fprintf(stderr, "Error: voice_clone task requires --ref-text unless --x-vector-only is set\n");
            return 1;
        }
    }

    if (params.task_type == qwen3_tts::tts_task_type::voice_design && params.instruct.empty()) {
        fprintf(stderr, "Error: voice_design task requires --instruct\n");
        return 1;
    }

    // Initialize TTS
    qwen3_tts::Qwen3TTS tts;

    fprintf(stderr, "Loading models from: %s\n", model_dir.c_str());
    if (!tts.load_models(model_dir)) {
        fprintf(stderr, "Error: %s\n", tts.get_error().c_str());
        return 1;
    }

    // Set progress callback
    tts.set_progress_callback([](int tokens, int max_tokens) {
        fprintf(stderr, "\rGenerating: %d/%d tokens", tokens, max_tokens);
    });

    // Generate speech
    qwen3_tts::tts_result result;

    if (params.task_type == qwen3_tts::tts_task_type::voice_clone) {
        fprintf(stderr, "Synthesizing with voice cloning: \"%s\"\n", text.c_str());
        fprintf(stderr, "Reference audio: %s\n", reference_audio.c_str());
        result = tts.synthesize_with_voice(text, reference_audio, params);
    } else if (reference_audio.empty()) {
        fprintf(stderr, "Synthesizing: \"%s\"\n", text.c_str());
        result = tts.synthesize(text, params);
    } else {
        fprintf(stderr, "Synthesizing with optional voice prompt: \"%s\"\n", text.c_str());
        fprintf(stderr, "Reference audio: %s\n", reference_audio.c_str());
        result = tts.synthesize_with_voice(text, reference_audio, params);
    }

    if (!result.success) {
        fprintf(stderr, "\nError: %s\n", result.error_msg.c_str());
        return 1;
    }

    fprintf(stderr, "\n");

    // Save output
    if (output_file.empty()) {
        qwen3_tts::play_audio(result.audio, result.sample_rate);
    } else {
        if (!qwen3_tts::save_audio_file(output_file, result.audio, result.sample_rate)) {
            fprintf(stderr, "Error: failed to save output file: %s\n", output_file.c_str());
            return 1;
        }

        fprintf(stderr, "Output saved to: %s\n", output_file.c_str());
        fprintf(stderr, "Audio duration: %.2f seconds\n", (float)result.audio.size() / result.sample_rate);
    }

    // Print timing
    if (params.print_timing) {
        fprintf(stderr, "\nTiming:\n");
        fprintf(stderr, "  Load:      %6lld ms\n", (long long)result.t_load_ms);
        fprintf(stderr, "  Tokenize:  %6lld ms\n", (long long)result.t_tokenize_ms);
        fprintf(stderr, "  Encode:    %6lld ms\n", (long long)result.t_encode_ms);
        fprintf(stderr, "  Generate:  %6lld ms\n", (long long)result.t_generate_ms);
        fprintf(stderr, "  Decode:    %6lld ms\n", (long long)result.t_decode_ms);
        fprintf(stderr, "  Total:     %6lld ms\n", (long long)result.t_total_ms);
    }

    return 0;
}
