# Qwen3-TTS-12Hz-1.7B-Base

## 1. 模型名称与定位
- 模型名: `Qwen3-TTS-12Hz-1.7B-Base`
- 规模: `1.7B` 级别
- 类型: `Base`（语音克隆模型）
- 能力: 参考音频驱动的 voice clone，支持 10 种语言
- 依赖: 内置 `speech_tokenizer`（即 `Qwen3-TTS-Tokenizer-12Hz`）

## 2. 模型输入（是否支持 instruct/ref.wav）
- `text`: 必填，目标合成文本
- `language`: 可选，默认 `Auto`，支持 `Chinese/English/Japanese/Korean/German/French/Russian/Portuguese/Spanish/Italian`
- `instruct`: Base 模型不使用 instruct（克隆模式不走 instruct 控制）
- `ref.wav`（支持）:
  - 支持本地路径 / URL / base64 / `(float32_waveform, sr)`
  - 两种克隆模式:
    1) `x_vector_only_mode=false`（默认）: 需要 `ref_audio + ref_text`，会同时使用参考语音码和说话人向量
    2) `x_vector_only_mode=true`: 只使用说话人向量，不强制 `ref_text`，音色相似度通常略低

- 参考音频的处理差异:
  - 给 tokenizer 编码 `ref_code` 时: 使用原始采样率（内部按输入 sr 处理）
  - 给 speaker encoder 提 embedding 时: 必须重采样到 `24kHz`
  - speaker embedding 提取使用梅尔参数:
    - `n_fft=1024`, `win=1024`, `hop=256`, `n_mels=128`, `fmin=0`, `fmax=12000`

- 文本模板（若你在 C++ 直连模型，需自己拼）:
  - 目标文本: `<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n`
  - 参考文本: `<|im_start|>assistant\n{ref_text}<|im_end|>\n`

## 3. 模型架构（层级与维度）
- 顶层: `Qwen3TTSForConditionalGeneration`
  - `talker`（主生成器）
  - `code_predictor`（子码本预测器）
  - `speaker_encoder`（Base 专有）
  - `speech_tokenizer`（12Hz codec）

- Talker 主干（先生成第 1 个 codebook）:
  - `num_hidden_layers=28`
  - `hidden_size=2048`
  - `intermediate_size=6144`
  - `num_attention_heads=16`, `num_key_value_heads=8`, `head_dim=128`
  - `rope_theta=1_000_000`
  - `rope_scaling.interleaved=true`, `mrope_section=[24,20,20]`
  - `text_embedding`: `[151936, 2048]`
  - `codec_embedding`: `[3072, 2048]`
  - `text_projection`: MLP `2048 -> 2048 -> 2048`
  - `codec_head`: `Linear(2048 -> 3072)`
  - 每层结构（28 层同构）:
    - `RMSNorm(2048)`
    - Self-Attn: `Q:2048->16*128`, `K:2048->8*128`, `V:2048->8*128`, `O:16*128->2048`
    - 残差
    - `RMSNorm(2048)`
    - SwiGLU MLP: `2048 -> 6144 -> 2048`
    - 残差

- Code Predictor（预测剩余 15 个 codebook）:
  - `num_code_groups=16`（总共 16 组码）
  - 预测器 Transformer:
    - `num_hidden_layers=5`
    - `hidden_size=1024`
    - `intermediate_size=3072`
    - `num_attention_heads=16`, `num_key_value_heads=8`, `head_dim=128`
    - `vocab_size=2048`
  - 输入嵌入:
    - 每个子码本一张 embedding 表，形状约 `[2048, 2048]`（共 15 张）
    - 因 `talker_hidden=2048` 与 predictor hidden 不同，存在投影 `2048 -> 1024`
  - 输出头:
    - 15 个 `Linear(1024 -> 2048)`（每个子码本一个头）

- Speaker Encoder（Base 专有，ECAPA-TDNN 变体）:
  - `enc_dim=2048`（输出说话人向量维度）
  - 输入梅尔维度 `128`
  - 通道配置: `enc_channels=[512,512,512,512,1536]`
  - kernel: `[5,3,3,3,1]`, dilation: `[1,2,3,4,1]`
  - 结构: TDNN -> 3x SE-Res2Net -> MFA -> ASP -> Conv1d 输出 embedding

- Speech Tokenizer（12Hz）:
  - 见独立文件 `Qwen3-TTS-Tokenizer-12Hz.md`
  - 本模型实际生成/解码使用 `16` 个 codebook，码字典大小 `2048`

- 控制 token（关键）:
  - `codec_pad_id=2148`, `codec_bos_id=2149`, `codec_eos_token_id=2150`
  - `codec_think_id=2154`, `codec_nothink_id=2155`, `codec_think_bos_id=2156`, `codec_think_eos_id=2157`
  - 语言 ID:
    - chinese=2055, english=2050, german=2053, italian=2070, portuguese=2071,
    - spanish=2054, japanese=2058, korean=2064, french=2061, russian=2069
  - 文本 special token:
    - `im_start=151644`, `im_end=151645`, `tts_pad=151671`, `tts_bos=151672`, `tts_eos=151673`

## 4. 可调参数（默认值）
- 采样参数（来自 `generation_config.json`）:
  - `do_sample=true`
  - `top_k=50`
  - `top_p=1.0`
  - `temperature=0.9`
  - `repetition_penalty=1.05`
  - `subtalker_dosample=true`
  - `subtalker_top_k=50`
  - `subtalker_top_p=1.0`
  - `subtalker_temperature=0.9`
  - `max_new_tokens=8192`

- 其他调用参数:
  - `non_streaming_mode`（`generate_voice_clone` 默认 `false`）
  - `x_vector_only_mode`（默认 `false`）
  - `language`（默认 `Auto`）

- 代码中固定逻辑（C++ 复现建议保持一致）:
  - `min_new_tokens=2`
  - `eos_token_id=2150`（除非显式覆盖）
  - `suppress_tokens = [2048..3071]`，但不抑制 `2150`
