# PROJECT KNOWLEDGE BASE

**Generated:** 2026-04-22
**Commit:** 539c865
**Branch:** fix-gui-bugs

## OVERVIEW

Pure C++17 implementation of Qwen3-TTS text-to-speech pipeline using GGML. Converts text → tokens → speaker embedding → codec codes → audio waveform. Supports 5 model variants (0.6B/1.7B × Base/CustomVoice + 1.7B VoiceDesign), streaming, DLL/CLI/GUI.

## STRUCTURE

```
qwen3-tts.cpp/
├── src/
│   ├── core/                        # Inference engine core
│   │   ├── tts_transformer.{h,cpp}  # Talker + code predictor (~2300 lines, THE core file)
│   │   ├── text_tokenizer.{h,cpp}   # BPE tokenizer
│   │   ├── gguf_loader.{h,cpp}      # GGUF weight loading + backend selection
│   │   ├── coreml_code_predictor.{h,mm,cpp}  # CoreML bridge (macOS) / stub (other)
│   ├── audio/                       # Audio I/O
│   │   ├── audio_tokenizer_encoder.{h,cpp}  # ECAPA-TDNN speaker encoder
│   │   ├── audio_tokenizer_decoder.{h,cpp}   # WavTokenizer vocoder
│   ├── api/                         # Public C API
│   │   └── qwen3tts_c_api.cpp       # C FFI wrapper (for DLL/Nim FFI)
│   ├── app/                         # Application entry points
│   │   ├── main.cpp                 # CLI (qwen3-tts-cli)
│   │   └── gui_main.cpp             # GUI (qwen3-tts-gui, Win32+DX11+ImGui)
│   └── qwen3_tts.cpp                # Qwen3TTS + Qwen3TTSModelHub orchestration
├── include/                         # Public headers
│   ├── qwen3_tts.h                  # C++ API (Qwen3TTS, Qwen3TTSModelHub, all param/result types)
│   └── qwen3tts_c_api.h             # C API (Qwen3Tts opaque handle, C-linkage functions)
├── tests/                           # Component tests
├── scripts/                         # Python utilities (convert, verify, benchmark)
├── reference/                       # Deterministic reference data (*.bin gitignored)
├── models/                          # GGUF models (gitignored)
├── ggml/                            # Vendored GGML (submodule)
├── imgui/                           # Vendored ImGui (Windows GUI only)
├── EUI/                             # EUI demo app (submodule)
└── CMakeLists.txt
```

## WHERE TO LOOK

| Task | Location | Notes |
|------|----------|-------|
| Add new generation mode | `src/qwen3_tts.cpp` | Orchestration lives here; add method to Qwen3TTS class |
| Modify transformer forward pass | `src/core/tts_transformer.cpp` | ~2300 lines; all graph builders + generate + prefill/step |
| Change token IDs / prefill format | `src/core/tts_transformer.h` | Special token constants defined in header |
| Add C API function | `src/api/qwen3tts_c_api.cpp` + `include/qwen3tts_c_api.h` | Must keep C-linkage, add to DLL exports |
| Add streaming feature | `src/qwen3_tts.cpp` | `synthesize_stream_internal()` handles chunked output |
| Change model weight loading | `src/core/gguf_loader.cpp` | `init_preferred_backend()` selects IGPU→GPU→ACCEL→CPU |
| Modify vocoder | `src/audio/audio_tokenizer_decoder.{h,cpp}` | WavTokenizer graph + weight loading |
| Modify speaker encoder | `src/audio/audio_tokenizer_encoder.{h,cpp}` | ECAPA-TDNN |
| GUI changes | `src/app/gui_main.cpp` | ImGui + Win32 DX11; Windows-only |
| CLI argument parsing | `src/app/main.cpp` | argparse-style CLI |
| CoreML code predictor | `src/core/coreml_code_predictor.mm` | macOS only; stub on other platforms |
| Convert HF models to GGUF | `scripts/convert_tts_to_gguf.py` / `convert_tts_to_gguf_v2.py` | Two versions exist; v2 is newer |
| Model download/setup | `scripts/setup_pipeline_models.py` | Automates HF download + GGUF conversion |
| Benchmark | `scripts/benchmark_pytorch_vs_cpp.py` | PyTorch vs C++ comparison |

## CODE MAP

| Symbol | Type | Location | Role |
|--------|------|----------|------|
| `Qwen3TTS` | class | `include/qwen3_tts.h` | Main pipeline: load, synthesize, stream |
| `Qwen3TTSModelHub` | class | `include/qwen3_tts.h` | Multi-model manager with typed load/synth APIs |
| `TTSTransformer` | class | `src/core/tts_transformer.h` | Talker + code predictor forward passes |
| `TextTokenizer` | class | `src/core/text_tokenizer.h` | BPE tokenizer |
| `AudioTokenizerEncoder` | class | `src/audio/audio_tokenizer_encoder.h` | ECAPA-TDNN speaker embedding |
| `AudioTokenizerDecoder` | class | `src/audio/audio_tokenizer_decoder.h` | WavTokenizer vocoder |
| `tts_params` | struct | `include/qwen3_tts.h` | Generation params (temp, top-p, top-k, seed, etc.) |
| `tts_stream_session` | class | `include/qwen3_tts.h` | Streaming poll-based session |
| `tts_result` | struct | `include/qwen3_tts.h` | Synthesis result (audio + timing + error) |
| `Qwen3Tts` (C API) | opaque | `include/qwen3tts_c_api.h` | C FFI handle for DLL consumption |
| `CoreMLCodePredictor` | class | `src/core/coreml_code_predictor.h` | Apple CoreML bridge (macOS only) |

## CONVENTIONS

- C++17, no exceptions, no RTTI
- `fprintf(stderr, ...)` for logging (never `std::cerr`)
- Error handling: methods return `bool`, details in `error_msg_` member
- Memory: GGML contexts own tensors; `ggml_free()` for cleanup
- Naming: `snake_case` functions/vars, `PascalCase` classes, `UPPER_CASE` macros
- Headers: `#pragma once`
- Public types in `qwen3_tts` namespace
- C API: `qwen3_tts_` prefix, C-linkage, `Qwen3Tts` opaque struct
- DLL exports: `WINDOWS_EXPORT_ALL_SYMBOLS ON` — all C++ public methods + C functions auto-exported

## ANTI-PATTERNS (THIS PROJECT)

- **NEVER** use `as any`, `@ts-ignore` (N/A, C++ project), or suppress compiler warnings
- **NEVER** commit `*.gguf` model files or `*.bin` reference data
- **NEVER** use `std::cerr` — always `fprintf(stderr, ...)`
- **NEVER** skip `ggml_cast` to F32 before `ggml_mul_mat` with F16 weights (causes precision bugs)
- **NEVER** forget to `ggml_backend_sched_reset()` after each forward pass
- **NEVER** forget CPU fallback backend when using non-CPU scheduler (Metal/GPU needs CPU fallback)
- Prefill embedding structure MUST mirror Python pipeline exactly (positions 0–9 mapping)

## UNIQUE STYLES

- Dual API surface: C++ class (`Qwen3TTS`) + C FFI (`qwen3_tts_*` functions) for DLL consumption
- `Qwen3TTSModelHub` provides model-ID-based typed APIs per model variant (06b_base, 1_7b_custom_voice, etc.)
- Streaming: `tts_stream_session` with poll-based `poll()` + callbacks (`on_audio_chunk`, `on_codes_frame`, `on_finish`, `on_error`)
- Platform-gated CoreML code predictor: `.mm` on macOS, `.cpp` stub elsewhere
- Windows GUI via vendor ImGui + DX11 backend (conditional `WIN32` in CMake)
- Build produces DLL + import LIB + CLI EXE + GUI EXE; install target for `dist/` layout

## COMMANDS

```bash
# Build GGML ( prerequisite)
cmake -S ggml -B ggml/build -G "Visual Studio 17 2022"
cmake --build ggml/build --config Release -j8

# Build project (MSVC on Windows)
cmake -S . -B build -G "Visual Studio 17 2022" -DSDL3_DIR="path/to/SDL3/cmake"
cmake --build build --config Release -j8

# Build with timing instrumentation
cmake -S . -B build -DQWEN3_TTS_TIMING=ON

# Install to dist/
cmake --install build --config Release --prefix dist

# Run tests
bash scripts/run_all_tests.sh
./build/test_transformer --ref-dir reference/

# Model setup
python scripts/setup_pipeline_models.py
```

## NOTES

- F16 weights cause autoregressive divergence vs Python float32 — results differ but audio is perceptually equivalent
- M-RoPE uses 1D positions (single-batch only; batched inference not yet supported)
- `--top-p` is parsed in CLI but not used in transformer sampling yet
- Top-level CMake expects vendored GGML at `./ggml` and SDL3 via `find_package(SDL3 REQUIRED)`
- The code predictor is ~71% of generation time (15 sequential forward passes per frame)
- Two GGUF converter scripts exist: `convert_tts_to_gguf.py` (original) and `convert_tts_to_gguf_v2.py` (newer)
- `build/` and `ggml/build/` are Windows MSVC CMake build dirs (in `.gitignore`)
