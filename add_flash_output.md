# qwen3-tts.cpp 流式生成开发方案（冻结版）

## 1. 目标与边界

### 1.1 本期目标
- 支持 **整段文本输入 + 语音流式输出**。
- 能做到边生成边播放、边生成边输出音频块。

### 1.2 非本期目标
- 不做文本增量输入（不做 token 级边输入边说）。
- 不改动模型语义，仅做推理链路与工程接口流式化。

## 2. 关键决策（已确认）

- 流式接口形态：**回调推送 + 阻塞超时 Poll** 两种同时提供。
- Decoder 路径：**直接做真增量（B 路径）**，不走前缀重解码过渡。
- 首期交付入口：**C++ API + CLI**（暂不做 C API）。
- 分块控制：**不硬编码**，由参数 `stream_chunk_frames` 控制（默认 4）。
- decoder 左上下文默认值：**4 帧**（优先保障实时性），可通过参数调大。
- 背压策略：有界队列满时 **丢最旧**（drop-oldest）。
- 收尾策略：默认 **轻量收尾（非完整 flush）**；仅在必要时开启 `stream_full_flush`。
- CLI 模式：`--stream` 默认边播，支持 `--stream-save` 边播边落盘。

## 3. 流式原理澄清

### 3.1 什么是本项目的“流式生成”
- 不是输入端的流式文本拼接。
- 是推理端在生成每个音频帧代码后，立即做增量 vocoder 解码并输出可播放音频块。

### 3.2 输入与输出行为
- 输入仍要求给定完整文本（一次性 tokenization + prefill）。
- 输出支持边生成边输出：每累计 `stream_chunk_frames` 帧（或到结束）即产出音频 chunk。

## 4. 架构改造方案

## 4.1 TTSTransformer（逐帧输出能力）

涉及文件：
- `src/core/tts_transformer.h`
- `src/core/tts_transformer.cpp`

改造点：
- 新增流式会话接口（示例命名）：
  - `start_stream(...)`
  - `step_stream(frame_codes_out, is_eos_out)`
  - `finish_stream()`
- 每次 `step_stream` 产出一帧 `frame_codes[16]`。
- 保留现有 `generate()`，内部复用流式接口并聚合，确保兼容旧调用。

## 4.2 AudioTokenizerDecoder（真增量解码）

涉及文件：
- `src/audio/audio_tokenizer_decoder.h`
- `src/audio/audio_tokenizer_decoder.cpp`

改造点：
- 新增流式接口：
  - `begin_stream(...)`
  - `decode_append(const int32_t * codes, int32_t n_new_frames, std::vector<float> & out_samples)`
  - `end_stream(std::vector<float> & tail_samples)`
- 维护增量状态（核心）：
  - pre-transformer 注意力层 KV cache；
  - 所有因果卷积需要的历史缓存（ring buffer / 滑窗缓存）；
  - 反卷积与后续残差块跨块边界需要的中间状态；
  - 输出拼接缓冲（避免 chunk 边界爆音/点击）。
- 保留 `decode()`，内部可通过 begin/append/end 路径实现全量兼容。

## 4.3 Qwen3TTS 编排层（统一流式管线）

涉及文件：
- `include/qwen3_tts.h`
- `src/qwen3_tts.cpp`

改造点：
- 新增流式参数结构（示例）：
  - `stream_chunk_frames`
  - `stream_queue_capacity`
  - `stream_poll_timeout_ms`
  - 其他必要开关（如 `stream_full_flush`）
- 新增流式句柄/会话对象（示例）：
  - 回调模式：注册 `on_audio_chunk`, `on_codes_frame`, `on_finish`, `on_error`；
  - Poll 模式：`poll(timeout_ms)` 阻塞超时获取下一块数据。
- 线程与队列：
  - 生产者线程：transformer 逐帧 + decoder 增量；
  - 消费者：回调线程或调用侧 poll；
  - 中间用有界队列承接。
- 背压策略：队列满时丢最旧，并统计 dropped chunks。

## 4.4 CLI 流式体验

涉及文件：
- `src/app/main.cpp`

新增参数建议：
- `--stream`
- `--stream-chunk-frames <n>`
- `--stream-queue-cap <n>`
- `--stream-poll-timeout <ms>`
- `--stream-save <wav>`

行为：
- `--stream` 时边推理边播放，不再等待整段音频。
- 指定 `--stream-save` 时，同时将每个 chunk 追加写入 WAV。
- 默认不做整段重解码 flush；采用增量解码 + 延迟补偿收尾。需要时可开启 `stream_full_flush`。

## 5. 数据结构与接口建议

## 5.1 音频块结构
- `chunk_id`
- `samples`（float32 mono）
- `sample_rate`
- `is_last`
- `dropped_before`（可选统计字段）

## 5.2 Poll 返回语义
- `ok + data`：取到块
- `timeout`：超时未取到
- `eos`：正常结束
- `error`：失败并携带错误信息

## 5.3 回调异常处理
- 回调内异常不影响核心推理状态机（C++ 层应捕获并标记错误）。
- 错误统一通过 `on_error` 与会话状态返回。

## 6. 测试计划

新增测试文件建议：
- `tests/test_transformer_stream.cpp`
- `tests/test_decoder_stream.cpp`
- `tests/test_e2e_stream.cpp`

测试要点：
- transformer 流式码流与离线 `generate()` 对齐。
- decoder 增量输出与全量 decode 对齐（MAE、相关性、边界无点击）。
- 长文本稳定性（30s+）、队列背压、默认轻量收尾与可选 `stream_full_flush` 行为。
- 回调与 poll 两模式一致性。

## 7. 验收标准

- 可边生成边播，主观无明显拼接噪声。
- `stream_chunk_frames` 参数可控且行为稳定。
- 队列满时按 drop-oldest 工作，系统不中断。
- 默认轻量收尾下结尾不明显截断；启用 `stream_full_flush` 时行为正确。
- 旧接口 `synthesize()` 行为不变，向后兼容。

## 8. 实施顺序

1. 冻结头文件流式 API 与数据结构。
2. 实现 transformer 流式会话。
3. 实现 decoder 真增量状态机。
4. 打通 Qwen3TTS 会话层（callback + poll + 队列背压）。
5. 接入 CLI `--stream` 与 `--stream-save`。
6. 补齐测试、调优延迟与边界音质。

## 9. 风险与对策

- 风险：decoder 增量状态不完整导致 chunk 接缝伪影。
  - 对策：逐层缓存建模 + 边界对齐单测 + 全量对比回归。
- 风险：回调阻塞导致吞吐抖动。
  - 对策：有界队列 + drop-oldest + 统计指标。
- 风险：多线程状态竞争。
  - 对策：会话对象内状态隔离、严格生命周期与同步策略。

---

本方案即当前版本的流式开发基线，可直接按该文档进入实现。
