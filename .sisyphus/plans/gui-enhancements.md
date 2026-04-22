# GUI Enhancements for qwen3-tts.cpp

## TL;DR

> **Quick Summary**: Add 7 enhancements to the ImGui-based TTS GUI: long text segmentation with CJK-aware splitting, progress bar for synthesis, terminate button, collapsed advanced params by default, save/restore config, x-vector-only label rename, and i18n for all new text.
>
> **Deliverables**:
> - Modified `src/app/gui_main.cpp` with all 7 features
> - Modified `src/qwen3_tts.h` and `src/qwen3_tts.cpp` for cancel API
> - Modified `src/core/tts_transformer.h` and `src/core/tts_transformer.cpp` for cancel check in generation loop
> - Modified `CMakeLists.txt` if needed for new source files
>
> **Estimated Effort**: Medium
> **Parallel Execution**: YES - 2 waves + final verification
> **Critical Path**: Task 1 (cancel API) → Task 5 (generation rewrite) → Task 7 (integration)

---

## Context

### Original Request
7 specific GUI enhancements to the qwen3-tts.cpp ImGui application:
1. Long text segmentation (~80 word units, nearest punctuation, CJK-aware)
2. Progress bar for non-streaming synthesis
3. Terminate button replacing Generate during synthesis
4. Collapsed advanced parameters by default
5. Save/Restore config buttons in appbar
6. Rename x-vector-only to "无需参考文本"/"No ref text"
7. i18n for all new UI text

### Interview Summary
**Key Discussions**:
- Cancellation approach: User chose **library modification** (add cancel API to Qwen3TTS)
- Progress bar: Use library's `progress_callback` (invoked during streaming at `frame_index+1 / max_tokens`). For non-streaming segmented: segment-based progress. For single-segment non-streaming: indeterminate/spinner.
- Config format: User wants "as simple as possible" — chose INI-style key=value
- Test strategy: Manual QA only, no unit tests
- Segmentation: Will decide approach and report back; CJK chars count as 1 unit, English words as 1 unit

**Research Findings**:
- `tts_progress_callback_t` exists and IS called during streaming (line 1803-1804 of qwen3_tts.cpp) with `progress_callback_(frame_index + 1, p.max_audio_tokens)`
- No cancel/abort mechanism exists in library — needs to be added
- `TTSTransformer::generate()` has main loop at line 2911: `for (int frame = 0; frame < max_len; ++frame)` — cancel check insertion point
- ImGui doesn't have built-in indeterminate progress bar — need simulated animation
- `app_state` has 19+ fields needing serialization for save/restore
- `ImGuiTreeNodeFlags_DefaultOpen` flag currently keeps advanced section open — removing it collapses by default

### Metis Review
**Identified Gaps** (addressed):
- Segment failure handling: Default to abort all remaining segments on first failure → user gets error in log
- Cancel cleanup: Must properly close WAV writer and StreamPlayer on cancel, not just abort thread
- Config file path: Use current working directory, filename `gui_config.ini`
- Mid-generation cancel for single-segment short text: Accept that it finishes the current frame; cancel takes effect at next check point
- Mixed CJK+Latin text: Count CJK chars individually, English words by spaces; scan boundary in either charset

---

## Work Objectives

### Core Objective
Implement 7 GUI enhancements enabling long text TTS synthesis with progress feedback, user-controllable cancellation, persistent configuration, and proper bilingual UI support.

### Concrete Deliverables
- Text segmentation function (`segment_text()`) with CJK-aware splitting
- Progress bar UI component showing segment-based progress + streaming sub-progress
- Terminate button that cancels ongoing synthesis via library cancel API
- Collapsed advanced parameters section by default
- Save/Restore config buttons with INI file persistence
- Renamed x-vector-only checkbox label
- All new UI strings using `tr(zh, en)` pattern

### Definition of Done
- [ ] Long text (>80 word units) is split at sentence boundaries and synthesized segment by segment
- [ ] Progress bar shows current segment progress for both stream and non-stream modes
- [ ] "开始生成" button changes to "终止" during synthesis; clicking it stops all generation
- [ ] Advanced parameters section starts collapsed on app launch
- [ ] Save config persists all visible UI state to `gui_config.ini`; Restore config resets from file
- [ ] x-vector-only label reads "无需参考文本" (zh) / "No ref text" (en)
- [ ] All new text follows `tr(zh, en)` bilinguation pattern

### Must Have
- CJK-aware text segmentation that finds nearest punctuation boundary within tolerance
- Cancel API in library (Qwen3TTS::request_cancel() + tts_stream_session::cancel())
- Cancel check in TTSTransformer::generate() per-frame loop
- Thread-safe progress reporting from generation thread to UI
- Config round-trip: save → close → reopen → restore = identical state
- Proper cleanup on cancel (close WAV writer, destroy audio stream)

### Must NOT Have (Guardrails)
- No changes to existing public API signatures (only additions)
- No modification to TTS synthesis quality or output format
- No third-party library additions (use only ImGui + existing dependencies)
- No persistent cancel state (cancel flag resets when new synthesis starts)
- No modification to imgui.ini (ImGui manages its own state file)
- No AI-slop: no excessive comments, no over-abstraction, no generic variable names

---

## Verification Strategy (MANDATORY)

> **ZERO HUMAN INTERVENTION** - ALL verification is agent-executed. No exceptions.

### Test Decision
- **Infrastructure exists**: NO (no test framework for GUI)
- **Automated tests**: None
- **Framework**: none
- **QA Policy**: Every task includes agent-executed QA scenarios via Bash (build verification) + manual GUI interaction descriptions

### QA Policy
Every task MUST include agent-executed QA scenarios.
Evidence saved to `.sisyphus/evidence/task-{N}-{scenario-slug}.{ext}`.

- **Build verification**: Use Bash — `cmake --build build --config Release -j4` must succeed
- **GUI features**: Agent describes exact steps to verify in the running GUI app

---

## Execution Strategy

### Parallel Execution Waves

```
Wave 1 (Start Immediately - foundation + independent features):
├── Task 1: Add cancel API to library (qwen3_tts.h/.cpp + tts_transformer) [deep]
├── Task 2: Rename x-vector-only label + collapse advanced params [quick]
├── Task 3: Implement text segmentation function segment_text() [deep]
└── Task 4: Implement INI config save/restore [unspecified-high]

Wave 2 (After Wave 1 - features depending on Task 1):
├── Task 5: Rewrite run_generation() with segmentation + cancel + progress [deep]
└── Task 6: Rewrite UI: terminate button, progress bar, save/restore buttons [visual-engineering]

Wave FINAL (After ALL tasks):
├── Task F1: Plan compliance audit (oracle)
├── Task F2: Code quality review (unspecified-high)
└── Task F3: Real manual QA (unspecified-high)

Critical Path: Task 1 → Task 5 → Task 6 → F1-F3
Parallel Speedup: ~40% faster than sequential
Max Concurrent: 4 (Wave 1)
```

### Dependency Matrix

| Task | Depends On | Blocks |
|------|-----------|--------|
| 1 | - | 5 |
| 2 | - | - |
| 3 | - | 5 |
| 4 | - | 6 |
| 5 | 1, 3 | 6 |
| 6 | 4, 5 | F1-F3 |

### Agent Dispatch Summary

- **Wave 1**: T1 → `deep`, T2 → `quick`, T3 → `deep`, T4 → `unspecified-high`
- **Wave 2**: T5 → `deep`, T6 → `visual-engineering`
- **FINAL**: F1 → `oracle`, F2 → `unspecified-high`, F3 → `unspecified-high`

---

## TODOs

- [x] 1. Add Cancel API to TTS Library

  **What to do**:
  - Add `std::atomic<bool> cancel_requested_{false}` member to `Qwen3TTS` class in `src/qwen3_tts.h`
  - Add public methods: `void request_cancel()` and `bool is_cancelled() const` to `Qwen3TTS`
  - Add `std::atomic<bool> cancel_requested_{false}` to `tts_stream_session::impl`
  - Add `void cancel()` method to `tts_stream_session` class
  - Modify `TTSTransformer::generate()` in `src/core/tts_transformer.cpp`: add cancel check at the start of the `for (int frame = 0; frame < max_len; ++frame)` loop (line 2911). If cancel is requested, break out of the loop and return false with error "generation cancelled"
  - In `synthesize_internal()` in `src/qwen3_tts.cpp`: check `cancel_requested_` before and after the `generate()` call, reset `cancel_requested_` at start of synthesis
  - In streaming worker (`synthesize_stream_internal()`): check `cancel_requested_` flag periodically, and if set, set stream to error state with "cancelled" message
  - Reset cancel flag at the beginning of each `synthesize*()` and `synthesize_*_stream()` call

  **Must NOT do**:
  - Do not change existing function signatures (only add new methods)
  - Do not modify the audio output format or quality
  - Do not add any third-party dependencies

  **Recommended Agent Profile**:
  - **Category**: `deep`
  - **Skills**: []
  - Reason: Requires understanding of multi-threaded cancellation patterns and atomic operations across library boundary

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 2, 3, 4)
  - **Blocks**: Task 5
  - **Blocked By**: None

  **References** (CRITICAL):

  **Pattern References**:
  - `src/qwen3_tts.h:270-398` — Qwen3TTS class definition, add cancel methods here
  - `src/qwen3_tts.h:248-267` — tts_stream_session class, add cancel() method here
  - `src/qwen3_tts.h:177` — progress callback type definition (reference for similar callback pattern)
  - `src/core/tts_transformer.cpp:2911` — Main generation loop `for (int frame = 0; frame < max_len; ++frame)`, insert cancel check here
  - `src/qwen3_tts.cpp:830-910` — `synthesize_internal()` entry point, add cancel check and reset
  - `src/qwen3_tts.cpp:1119-1200` — `synthesize_stream_internal()` worker, add cancel check

  **Why Each Reference Matters**:
  - `qwen3_tts.h` lines 270-398: This is where public API lives. New cancel methods must match existing pattern (snake_case, `const` correctness).
  - `tts_transformer.cpp:2911`: The main loop is the only place where per-frame cancellation makes sense — each frame is an autoregressive step.
  - `qwen3_tts.cpp:830-910`: Entry point needs reset at start and check after generate returns.

  **Acceptance Criteria**:
  - [ ] `Qwen3TTS::request_cancel()` and `Qwen3TTS::is_cancelled()` are declared in `qwen3_tts.h` and defined in `qwen3_tts.cpp`
  - [ ] `tts_stream_session::cancel()` is declared and defined
  - [ ] Cancel check exists in `TTSTransformer::generate()` main loop that returns `false` with error when cancelled
  - [ ] Cancel flag is reset at start of each `synthesize*()` and `synthesize_*_stream()` call
  - [ ] Streaming worker checks cancel flag and sets error state when cancelled
  - [ ] Build succeeds: `cmake --build build --config Release -j4`

  **QA Scenarios (MANDATORY)**:

  ```
  Scenario: Cancel API compiles and links
    Tool: Bash
    Preconditions: Project builds successfully before changes
    Steps:
      1. cmake -S . -B build -G "Visual Studio 17 2022"
      2. cmake --build build --config Release -j4
    Expected Result: Build succeeds with 0 errors and 0 warnings related to new code
    Failure Indicators: Linker errors for undefined symbols, compilation errors in modified files
    Evidence: .sisyphus/evidence/task-1-build-verify.txt

  Scenario: Cancel check in generation loop
    Tool: Bash
    Preconditions: Source files modified
    Steps:
      1. Search for `cancel_requested` in tts_transformer.cpp to verify cancel check exists inside the generation loop
      2. Search for `request_cancel` in qwen3_tts.h to verify public API exists
      3. Search for `cancel()` in qwen3_tts.h to verify session cancel exists
    Expected Result: All three searches return results; cancel check is inside the `for (int frame = ...)` loop
    Failure Indicators: Missing cancel check in loop, missing public API declaration
    Evidence: .sisyphus/evidence/task-1-cancel-api-verify.txt
  ```

  **Commit**: YES (groups with Task 2)
  - Message: `feat(lib): add cancel API for generation interrupt`
  - Files: `src/qwen3_tts.h`, `src/qwen3_tts.cpp`, `src/core/tts_transformer.h`, `src/core/tts_transformer.cpp`
  - Pre-commit: `cmake --build build --config Release -j4`

- [x] 2. Rename x-vector-only Label + Collapse Advanced Params

  **What to do**:
  - In `src/app/gui_main.cpp` line 956: Change `ImGui::Checkbox("x-vector-only", &state.x_vector_only)` to use `tr()` pattern: `ImGui::Checkbox(tr(state.lang, "无需参考文本", "No ref text"), &state.x_vector_only)`
  - In `src/app/gui_main.cpp` line 982: Change `ImGui::CollapsingHeader(tr(state.lang, "高级参数", "Advanced"), ImGuiTreeNodeFlags_DefaultOpen)` to remove `ImGuiTreeNodeFlags_DefaultOpen` so it starts collapsed: `ImGui::CollapsingHeader(tr(state.lang, "高级参数", "Advanced"))`
  - Also update validation message at line 672: Change `"请填写参考文本，或开启 x-vector-only。"` to `"请填写参考文本，或开启无需参考文本。"` and `"Please provide reference text, or enable x-vector-only."` to `"Please provide reference text, or enable No ref text."`

  **Must NOT do**:
  - Do not change the variable name `x_vector_only` in app_state (it's an internal identifier)
  - Do not remove the checkbox functionality
  - Do not change any validation logic, only the display text

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: []
  - Reason: Simple string replacements, no complex logic

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 1, 3, 4)
  - **Blocks**: None
  - **Blocked By**: None

  **References**:

  **Pattern References**:
  - `src/app/gui_main.cpp:956` — x-vector-only checkbox, change label to tr(zh, en) pattern
  - `src/app/gui_main.cpp:982` — CollapsingHeader with ImGuiTreeNodeFlags_DefaultOpen, remove the flag
  - `src/app/gui_main.cpp:672` — Validation error message referencing x-vector-only
  - `src/app/gui_main.cpp:74-76` — `tr()` function definition showing bilingual pattern

  **Why Each Reference Matters**:
  - Line 956: The target checkbox to relabel
  - Line 982: The collapsing header that needs default-open flag removed
  - Line 672: Error message that references the old label name
  - Line 74-76: The pattern for all bilingual strings

  **Acceptance Criteria**:
  - [ ] x-vector-only checkbox label shows "无需参考文本" when UI language is Chinese
  - [ ] x-vector-only checkbox label shows "No ref text" when UI language is English
  - [ ] Advanced parameters section starts collapsed when app launches
  - [ ] Validation error message uses updated label text
  - [ ] Build succeeds

  **QA Scenarios**:

  ```
  Scenario: Label changes compile and display correctly
    Tool: Bash
    Preconditions: Project builds
    Steps:
      1. cmake --build build --config Release -j4
      2. Search for "x-vector-only" in gui_main.cpp to verify only the variable name remains, not the display label
      3. Search for "DefaultOpen" in gui_main.cpp to verify it's removed from the CollapsingHeader call
    Expected Result: Build succeeds; "x-vector-only" only appears as variable name; "DefaultOpen" not in CollapsingHeader
    Failure Indicators: Build errors; old label string still present in display text
    Evidence: .sisyphus/evidence/task-2-label-verify.txt
  ```

  **Commit**: YES (groups with Task 1)
  - Message: `fix(gui): rename x-vector-only label and collapse advanced params by default`
  - Files: `src/app/gui_main.cpp`
  - Pre-commit: `cmake --build build --config Release -j4`

- [x] 3. Implement CJK-Aware Text Segmentation Function

  **What to do**:
  - Create a new function `segment_text()` in `src/app/gui_main.cpp` (in the anonymous namespace) with signature:
    ```cpp
    struct text_segment {
        std::string text;
        int word_unit_count; // approximate word count (CJK char count + English word count)
    };
    std::vector<text_segment> segment_text(const std::string &input, int max_units = 80);
    ```
  - Implementation algorithm:
    1. Iterate through the input string character by character (handling UTF-8 multi-byte)
    2. Classify each character: CJK (Unicode ranges 0x4E00-0x9FFF, 0x3400-0x4DBF, 0x3000-0x303F for CJK punctuation, etc.), Latin, whitespace, punctuation
    3. Count "word units": CJK characters count as 1 unit each; Latin words (space-separated) count as 1 unit each
    4. When count reaches `max_units`, scan forward (up to `max_units * 0.3` tolerance) for a sentence-ending punctuation mark:
       - CJK: 。！？；… 、《》【】
       - English/Latin: . ! ? ; : —
       - Comma/semicolon as secondary break point if no sentence end found
    5. If no punctuation found within tolerance, split at `max_units` exactly
    6. Create segments preserving original text including whitespace
    7. Each segment gets its `word_unit_count` stored
  - Add UTF-8 character iteration helpers (or use existing patterns from the codebase)
  - Handle edge cases: empty input, single-character input, input shorter than max_units, mixed CJK+Latin

  **Must NOT do**:
  - Do not use external Unicode libraries (ICU, etc.)
  - Do not modify any existing functions — only add new ones
  - Do not add word boundary detection for Chinese beyond character counting

  **Recommended Agent Profile**:
  - **Category**: `deep`
  - **Skills**: []
  - Reason: Requires careful UTF-8 handling and CJK character range logic

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 1, 2, 4)
  - **Blocks**: Task 5
  - **Blocked By**: None

  **References**:

  **Pattern References**:
  - `src/app/gui_main.cpp:83-87` — `to_lower_copy()` utility function pattern in anonymous namespace
  - `src/app/gui_main.cpp:106-140` — `load_text_or_file()` — similar text processing utility pattern
  - `src/core/text_tokenizer.cpp` — BPE tokenizer's UTF-8 handling (reference for how codebase handles multi-byte)

  **Why Each Reference Matters**:
  - Lines 83-87: Show the pattern for utility functions in the anonymous namespace
  - Lines 106-140: Show text processing pattern (file loading, string handling)
  - text_tokenizer.cpp: Reference for UTF-8 byte patterns used elsewhere in the project

  **Acceptance Criteria**:
  - [ ] `segment_text()` function exists in anonymous namespace of gui_main.cpp
  - [ ] CJK characters (0x4E00-0x9FFF, 0x3400-0x4DBF) are counted individually
  - [ ] English words (space-separated) are counted as 1 unit each
  - [ ] Segments break at sentence-ending punctuation within tolerance
  - [ ] Mixed CJK+Latin text is handled correctly
  - [ ] Input shorter than max_units returns single segment
  - [ ] Empty input returns empty vector
  - [ ] Build succeeds

  **QA Scenarios**:

  ```
  Scenario: Segmentation function builds successfully
    Tool: Bash
    Preconditions: Project compiles
    Steps:
      1. cmake --build build --config Release -j4
      2. Search for "segment_text" in gui_main.cpp to verify function definition exists
      3. Search for "text_segment" in gui_main.cpp to verify struct exists
    Expected Result: Build succeeds; function and struct are defined
    Failure Indicators: Compilation errors; missing function definition
    Evidence: .sisyphus/evidence/task-3-segment-build.txt

  Scenario: CJK character ranges are defined
    Tool: Bash
    Preconditions: Source files modified
    Steps:
      1. Search for CJK Unicode ranges (0x4E00, 0x9FFF, etc.) in the segmentation code
      2. Verify sentence-ending punctuation lists exist for both CJK and Latin
    Expected Result: CJK ranges and punctuation lists are present in the code
    Failure Indicators: No CJK detection logic; no punctuation break point logic
    Evidence: .sisyphus/evidence/task-3-cjk-ranges.txt
  ```

  **Commit**: YES
  - Message: `feat(gui): add CJK-aware text segmentation for long text synthesis`
  - Files: `src/app/gui_main.cpp`
  - Pre-commit: `cmake --build build --config Release -j4`

- [x] 4. Implement INI Config Save/Restore

  **What to do**:
  - Add two new functions in anonymous namespace of `src/app/gui_main.cpp`:
    ```cpp
    bool save_config(const app_state &state, const std::string &path);
    bool load_config(app_state &state, const std::string &path);
    ```
  - Config file format: Simple INI-style key=value, one per line:
    ```
    model_index=0
    speaker_index=0
    synthesis_language_index=0
    text_input=Hello world
    reference_audio=
    reference_text_input=
    instruct_input=
    output_path=outputs\tts_output.wav
    x_vector_only=0
    save_output=0
    play_audio=1
    stream=0
    stream_realtime=0
    temperature=0.900000
    top_k=50
    top_p=1.000000
    max_tokens=4096
    repetition_penalty=1.050000
    seed=-1
    threads=4
    ```
  - Implementation details:
    - Use `std::ofstream`/`std::ifstream` for file I/O
    - Use `std::getline` for reading lines
    - Parse `key=value` by finding `=` separator, trim whitespace
    - For strings: just read everything after `=` (no escaping needed for simplicity; values don't contain newlines in practice)
    - For bools: write `0`/`1`, read as integer then cast
    - For floats: use `std::to_string()` for write, `std::stof()` for read
    - File path: `gui_config.ini` in current working directory
  - Add `save_config_path()` helper that returns `(fs::current_path() / "gui_config.ini").string()`
  - On startup: if `gui_config.ini` doesn't exist, create it with current defaults (don't load anything, just save defaults)
  - "Save Config" button (💾 icon or similar): calls `save_config()` with current state, logs success/failure
  - "Restore Config" button (📂 icon or similar): calls `load_config()` to reset all parameters from file, logs success/failure
  - Both buttons are placed in the appbar area (before model selection section)
  - Both buttons show tooltip on hover: Save = "保存配置" / "Save Config", Restore = "恢复配置" / "Restore Config"
  - Use `ImGui::Button()` with icon text, then `if (ImGui::IsItemHovered()) ImGui::SetTooltip(...)` for hover effect

  **Must NOT do**:
  - Do not persist `model_setup_running`, `models_ready`, `generation_running`, `log_text`, or any thread/atomic fields
  - Do not use any INI library — hand-write the simple parser/writer
  - Do not save the UI language (`lang`) — it's auto-detected from system settings
  - Do not add static text labels next to the buttons — only hover tooltips

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
  - **Skills**: []
  - Reason: File I/O and state serialization with many fields, but straightforward logic

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 1, 2, 3)
  - **Blocks**: Task 6
  - **Blocked By**: None

  **References**:

  **Pattern References**:
  - `src/app/gui_main.cpp:291-329` — `app_state` struct definition, all fields that need serialization
  - `src/app/gui_main.cpp:74-76` — `tr()` function for bilingual strings
  - `src/app/gui_main.cpp:834-864` — WinMain startup section where default config creation goes
  - `src/app/gui_main.cpp:959-990` — UI rendering section where buttons should be inserted (at top, before model selection)

  **Why Each Reference Matters**:
  - app_state: Lists every field to serialize and which ones to skip
  - tr(): Pattern for all bilingual UI strings
  - WinMain startup: Where to add auto-creation of config file on first run
  - UI section: Where to place the new buttons

  **Acceptance Criteria**:
  - [ ] `save_config()` writes all visible app_state fields to `gui_config.ini`
  - [ ] `load_config()` reads all fields and restores app_state correctly
  - [ ] Config file is created on startup if it doesn't exist
  - [ ] Save/Restore buttons appear in appbar area with hover tooltips
  - [ ] Round-trip test: save → modify values → restore → all values match saved state
  - [ ] Thread/atomic fields are NOT serialized
  - [ ] Build succeeds

  **QA Scenarios**:

  ```
  Scenario: Config save/load compiles and functions
    Tool: Bash
    Preconditions: Project builds
    Steps:
      1. cmake --build build --config Release -j4
      2. Search for "save_config" and "load_config" function definitions in gui_main.cpp
      3. Search for "gui_config.ini" to verify file path is defined
    Expected Result: Functions exist; config file path is defined; build succeeds
    Failure Indicators: Missing function definitions; build errors
    Evidence: .sisyphus/evidence/task-4-config-verify.txt

  Scenario: All app_state fields are covered
    Tool: Bash
    Preconditions: Source files modified
    Steps:
      1. Compare list of fields in save_config() against all visible fields in app_state
      2. Verify thread/atomic fields are excluded from serialization
    Expected Result: model_index, speaker_index, synthesis_language_index, text_input, reference_audio, reference_text_input, instruct_input, output_path, x_vector_only, save_output, play_audio, stream, stream_realtime, temperature, top_k, top_p, max_tokens, repetition_penalty, seed, threads are all serialized
    Failure Indicators: Missing fields in config file; atomic fields included
    Evidence: .sisyphus/evidence/task-4-fields-verify.txt
  ```

  **Commit**: YES (groups with Task 6)
  - Message: `feat(gui): add config persistence with save/restore buttons`
  - Files: `src/app/gui_main.cpp`
  - Pre-commit: `cmake --build build --config Release -j4`

- [x] 5. Rewrite run_generation() with Segmentation, Cancel, and Progress

  **What to do**:
  - Add to `app_state`:
    ```cpp
    std::vector<std::string> segment_texts;      // text segments for current generation
    int segment_total = 0;                        // total number of segments
    std::atomic<int> segment_completed{0};        // segments completed so far
    std::atomic<int> progress_percent{0};         // 0-100 progress percentage
    std::atomic<bool> cancel_requested{false};    // cancel flag for GUI
    ```
  - Modify `run_generation()` to:
    1. At the start, call `segment_text()` on the input text
    2. If text has <= 1 segment (short text), generate as before (single call)
    3. If text has > 1 segment (long text), loop through segments:
       a. For each segment, call appropriate synthesis function (stream or non-stream)
       b. After each segment completes, update `segment_completed` and `progress_percent`
       c. Check `cancel_requested` between segments; if set, abort and report in log
       d. For non-streaming: accumulate audio in a `std::vector<float>`, at end save or play all
       e. For streaming: play each segment's audio as it generates (natural real-time behavior)
    4. Register `set_progress_callback()` on the tts object for within-segment streaming progress:
       - Progress = `(segment_completed + current_segment_progress) / segment_total * 100`
       - `current_segment_progress = tokens_generated / max_tokens` from callback
    5. On cancel: stop loop, log cancellation message, reset UI state
  - Modify `validate_or_set_popup()` to remove x-vector-only validation message update (already done in Task 2)
  - For non-streaming single-segment: show indeterminate progress (use `ImGui::ProgressBar(-1.0f * fmod(time, 1.0f), ImVec2(-1, 0))` pattern or animated spinner — research ImGui indeterminate progress pattern)
  - For non-streaming multi-segment: show segment-based progress `segment_completed / segment_total`
  - For streaming: show combined progress using progress_callback data
  - Add helper function `run_single_segment()` that generates one text segment and returns the audio (or streams it)
  - Handle segment failure: if any segment fails, log the error and stop (don't continue with remaining segments)
  - Handle cleanup on cancel: for non-streaming, just stop the loop; for streaming, call session's cancel() method

  **Must NOT do**:
  - Do not change existing audio output format
  - Do not modify the TTS library's generation functions (only use the new cancel API from Task 1)
  - Do not add any external dependencies for progress display

  **Recommended Agent Profile**:
  - **Category**: `deep`
  - **Skills**: []
  - Reason: Most complex task; threading, atomic variables, progress reporting, cancellation, rendering rewrite

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Wave 2 (depends on Tasks 1 and 3)
  - **Blocks**: Task 6
  - **Blocked By**: Tasks 1, 3

  **References**:

  **Pattern References**:
  - `src/app/gui_main.cpp:696-820` — Current `run_generation()` function, the core to rewrite
  - `src/app/gui_main.cpp:483-530` — `run_stream_session()` function for streaming pattern
  - `src/app/gui_main.cpp:291-329` — `app_state` struct, add new atomic fields
  - `src/qwen3_tts.h:177` — `tts_progress_callback_t` type signature: `void(int tokens_generated, int max_tokens)`
  - `src/qwen3_tts.h:323` — `set_progress_callback()` method declaration
  - `src/qwen3_tts.h:179-215` — `tts_stream_params` structure for streaming configuration

  **Why Each Reference Matters**:
  - `run_generation()`: This is the function being rewritten — need to understand its full structure
  - `run_stream_session()`: Pattern for streaming generation, will be modified to support per-segment and cancel
  - `app_state`: New atomic fields need to be added for thread-safe progress reporting
  - `progress_callback_t`: The hook for getting within-segment progress during streaming
  - `set_progress_callback()`: Must be called on tts object before synthesis starts

  **Acceptance Criteria**:
  - [ ] Text longer than 80 word-units is split into segments before synthesis
  - [ ] Progress bar shows segment-based progress for multi-segment non-streaming mode
  - [ ] Progress bar shows combined segment + callback progress for streaming mode
  - [ ] Cancel between segments stops generation and logs "[cancelled]" message
  - [ ] Cancel during streaming session calls session->cancel() and waits for termination
  - [ ] Segment failure logs error and stops remaining segments
  - [ ] Non-streaming multi-segment audio is concatenated before save/play
  - [ ] Single-segment (short text) mode works identically to current behavior
  - [ ] Build succeeds

  **QA Scenarios**:

  ```
  Scenario: Segmented generation builds and has progress fields
    Tool: Bash
    Preconditions: Project builds
    Steps:
      1. cmake --build build --config Release -j4
      2. Search for "segment_text" in gui_main.cpp to verify it's called in run_generation
      3. Search for "segment_completed" to verify atomic progress field exists
      4. Search for "cancel_requested" to verify cancel check in generation loop
      5. Search for "progress_callback" to verify progress callback registration
    Expected Result: Build succeeds; all search terms found in gui_main.cpp
    Failure Indicators: Missing function calls; missing atomic fields; build errors
    Evidence: .sisyphus/evidence/task-5-segment-gen-verify.txt

  Scenario: Cancel mechanism is wired up
    Tool: Bash
    Preconditions: Source files modified
    Steps:
      1. Search for "request_cancel" or "cancel_requested" in the generation code path
      2. Verify cancel flag is checked between segment iterations
      3. Verify cancel flag resets at start of generation
    Expected Result: Cancel check exists in generation loop; flag is reset at start
    Failure Indicators: No cancel check; flag not reset
    Evidence: .sisyphus/evidence/task-5-cancel-wiring.txt
  ```

  **Commit**: YES
  - Message: `feat(gui): add segmented generation with progress and cancel support`
  - Files: `src/app/gui_main.cpp`
  - Pre-commit: `cmake --build build --config Release -j4`

- [x] 6. Rewrite UI: Terminate Button, Progress Bar, Save/Restore Buttons

  **What to do**:
  - **Terminate Button**: Replace the current generate button block (lines 993-1005) with:
    ```cpp
    bool can_generate = state.models_ready && !state.generation_running;
    if (state.generation_running) {
        // Show "Terminate" button (red/warning style)
        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.7f, 0.2f, 0.2f, 1.0f));
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.8f, 0.3f, 0.3f, 1.0f));
        ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.6f, 0.15f, 0.15f, 1.0f));
        if (ImGui::Button(tr(state.lang, "终止", "Terminate"), ImVec2(180, 42))) {
            state.cancel_requested = true;
            // Also call tts.request_cancel() if tts object is accessible
            // Cancel will take effect at next segment boundary or stream poll
        }
        ImGui::PopStyleColor(3);
    } else {
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
    ```

  - **Progress Bar**: Add above the log area (before `ImGui::SeparatorText(tr(state.lang, "日志", "Logs"))`):
    ```cpp
    if (state.generation_running) {
        float progress = (float)state.progress_percent / 100.0f;
        if (state.segment_total <= 1) {
            // Indeterminate/spinner for single segment or pre-segment progress
            // Use animated progress: oscillate between 0 and 1
            static float indeterminate = 0.0f;
            indeterminate += ImGui::GetIO().DeltaTime * 0.5f;
            if (indeterminate > 1.0f) indeterminate = 0.0f;
            ImGui::ProgressBar(indeterminate, ImVec2(-1.0f - ImGui::CalcTextSize("100%").x, 0));
            ImGui::SameLine();
            ImGui::TextColored(ImVec4(0.6f, 0.8f, 1.0f, 1.0f), "%s", tr(state.lang, "处理中...", "Processing..."));
        } else {
            // Determinate progress for multi-segment
            char label[64];
            snprintf(label, sizeof(label), "%s %d/%d",
                state.lang == ui_lang::zh ? "段" : "seg",
                (int)state.segment_completed, state.segment_total);
            ImGui::ProgressBar(progress, ImVec2(-1.0f - ImGui::CalcTextSize("100%").x, 0), label);
            ImGui::SameLine();
            ImGui::Text("%d%%", (int)state.progress_percent);
        }
    }
    ```

  - **Save/Restore Config Buttons**: Add at the top of the main UI section, before the model selector (before line 918 `ImGui::SeparatorText(tr(state.lang, "模型", "Model"))`):
    ```cpp
    // Config buttons (no labels, hover tooltips only)
    if (ImGui::Button(tr(state.lang, "保存配置", "Save Config"))) {
        std::string err;
        if (save_config(state, save_config_path())) {
            append_log(state, tr(state.lang, "[ok] 配置已保存", "[ok] Config saved"));
        } else {
            append_log(state, tr(state.lang, "[error] 保存配置失败", "[error] Failed to save config"));
        }
    }
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("%s", tr(state.lang, "保存当前所有配置到文件", "Save all current settings to file"));
    }
    ImGui::SameLine();
    if (ImGui::Button(tr(state.lang, "恢复配置", "Restore Config"))) {
        if (load_config(state, save_config_path())) {
            append_log(state, tr(state.lang, "[ok] 配置已恢复", "[ok] Config restored"));
        } else {
            append_log(state, tr(state.lang, "[error] 恢复配置失败", "[error] Failed to restore config"));
        }
    }
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("%s", tr(state.lang, "从文件恢复所有配置", "Restore all settings from file"));
    }
    ```

  - **Startup Config Creation**: In WinMain, after `state` initialization and before the main loop, add:
    ```cpp
    std::string config_path = save_config_path();
    if (!fs::exists(config_path)) {
        save_config(state, config_path);  // Create default config file
    }
    ```

  - Ensure all new UI strings use `tr(state.lang, "中文", "English")` pattern
  - Ensure progress bar and buttons don't overlap with existing UI elements

  **Must NOT do**:
  - Do not change existing UI layout beyond specified additions
  - Do not add emojis to button text
  - Do not make the progress bar modal or blocking

  **Recommended Agent Profile**:
  - **Category**: `visual-engineering`
  - **Skills**: []
  - Reason: ImGui UI layout and styling, requires careful visual arrangement

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Wave 2 (depends on Tasks 4 and 5)
  - **Blocks**: F1-F3
  - **Blocked By**: Tasks 4, 5

  **References**:

  **Pattern References**:
  - `src/app/gui_main.cpp:893-1030` — Main UI rendering section (ImGui::Begin to ImGui::End)
  - `src/app/gui_main.cpp:993-1005` — Current generate button block to be replaced
  - `src/app/gui_main.cpp:1019-1028` — Log area section (progress bar goes above this)
  - `src/app/gui_main.cpp:602-646` — `setup_style()` function showing ImGui styling patterns
  - `src/app/gui_main.cpp:858-864` — WinMain initialization section for startup config

  **Why Each Reference Matters**:
  - Lines 893-1030: The entire UI block where all changes happen
  - Lines 993-1005: Generate button that becomes Terminate button
  - Lines 1019-1028: Log area where progress bar is inserted above
  - Lines 602-646: Style patterns for custom button colors
  - Lines 858-864: Startup section for config file auto-creation

  **Acceptance Criteria**:
  - [ ] Button shows "开始生成"/"Generate" when not generating, "终止"/"Terminate" when generating
  - [ ] Terminate button has distinct red styling
  - [ ] Clicking Terminate sets `cancel_requested = true` and `tts.request_cancel()`
  - [ ] Progress bar appears above log area during generation
  - [ ] Multi-segment generation shows segment progress (X/Y segments, percentage)
  - [ ] Single-segment generation shows indeterminate progress
  - [ ] Save Config button saves all visible state to `gui_config.ini`
  - [ ] Restore Config button loads all state from `gui_config.ini`
  - [ ] Save/Restore buttons have hover tooltips (no static labels)
  - [ ] Config file is auto-created on startup if missing
  - [ ] All new UI strings use `tr(zh, en)` pattern
  - [ ] Build succeeds

  **QA Scenarios**:

  ```
  Scenario: UI elements compile and render
    Tool: Bash
    Preconditions: Project builds
    Steps:
      1. cmake --build build --config Release -j4
      2. Search for "Terminate" or "终止" in gui_main.cpp to verify terminate button exists
      3. Search for "ProgressBar" in gui_main.cpp to verify progress bar exists
      4. Search for "save_config" and "load_config" calls to verify button wiring
      5. Search for "SetTooltip" to verify hover tooltips exist
    Expected Result: All UI elements found in code; build succeeds
    Failure Indicators: Missing UI elements; build errors; missing tooltip calls
    Evidence: .sisyphus/evidence/task-6-ui-verify.txt

  Scenario: Button state transitions
    Tool: Bash
    Preconditions: Source files modified
    Steps:
      1. Verify generate button is shown when `!state.generation_running && state.models_ready`
      2. Verify terminate button is shown when `state.generation_running`
      3. Verify clicking terminate sets `state.cancel_requested = true`
      4. Verify clicking generate resets `cancel_requested`, `segment_completed`, `progress_percent`
    Expected Result: Both button states exist with proper conditions; state transitions are correct
    Failure Indicators: Missing toggle logic; cancel not wired; state not reset on start
    Evidence: .sisyphus/evidence/task-6-button-states.txt
  ```

  **Commit**: YES
  - Message: `feat(gui): add terminate button, progress bar, and save/restore config UI`
  - Files: `src/app/gui_main.cpp`
  - Pre-commit: `cmake --build build --config Release -j4`

---

## Final Verification Wave (MANDATORY — after ALL implementation tasks)

> 3 review agents run in PARALLEL. ALL must APPROVE.

- [x] F1. **Plan Compliance Audit** — `oracle`
  VERDICT: **APPROVE**
  - Must Have [6/6]: CJK segmentation ✓, Cancel API ✓, Cancel check in loop ✓, Thread-safe progress ✓, Config round-trip ✓, Cancel cleanup ✓
  - Must NOT Have [6/6]: No API sig changes ✓, No quality changes ✓, No new deps ✓, No persistent cancel ✓, No imgui.ini changes ✓, No AI-slop ✓
  - Tasks [6/6] all implemented
  Read the plan end-to-end. For each "Must Have": verify implementation exists (read file, check code). For each "Must NOT Have": search codebase for forbidden patterns — reject with file:line if found. Compare deliverables against plan.
  Output: `Must Have [6/6] | Must NOT Have [4/4] | Tasks [6/6] | VERDICT: APPROVE/REJECT`

- [x] F2. **Code Quality Review** — `unspecified-high`
  Build: **PASS** (0 errors, 0 warnings)
  Files: 6 clean, 0 issues
  - No TODO/FIXME/HACK found
  - No empty catches
  - No console.log/std::cerr usage
  - No commented-out code
  - No AI-slop patterns detected
  VERDICT: **APPROVE**

- [x] F3. **Real Manual QA** — `unspecified-high`
  Code-level verification of all 8 features:
  1. ✓ `segment_text(text, 80)` splits at punctuation (gui_main.cpp:1127, segment_text:275)
  2. ✓ CJK punctuation (。！？) used as primary break points (is_cjk_sentence_punct:232)
  3. ✓ Button toggles to "终止"/"Terminate" with red styling (gui_main.cpp:1491-1497)
  4. ✓ `cancel_requested = true` checked in generation loop (gui_main.cpp:1178,1237)
  5. ✓ `CollapsingHeader` without `DefaultOpen` flag (gui_main.cpp:1480)
  6. ✓ `save_config()`/`load_config()` with buttons (gui_main.cpp:768,799,1317)
  7. ✓ Checkbox label uses `tr("无需参考文本", "No ref text")` (gui_main.cpp:1454)
  8. ✓ ProgressBar reads `progress_percent` (gui_main.cpp:1531-1547)
  Build: **PASS**
  VERDICT: **APPROVE**

---

## Commit Strategy

- **Task 1+2**: `feat(lib): add cancel API and rename x-vector-only label`
- **Task 3**: `feat(gui): add CJK-aware text segmentation`
- **Task 4+6**: `feat(gui): add config persistence and UI controls`
- **Task 5**: `feat(gui): rewrite generation with segmentation and progress`

---

## Success Criteria

### Verification Commands
```bash
cmake -S . -B build -G "Visual Studio 17 2022" && cmake --build build --config Release -j4
# Expected: Build succeeds with 0 errors
```

### Final Checklist
- [x] All "Must Have" present
- [x] All "Must NOT Have" absent
- [x] Build succeeds
- [x] All 7 requirements implemented and verified