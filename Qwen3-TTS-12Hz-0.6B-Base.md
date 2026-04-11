# Qwen3-TTS-12Hz-0.6B-Base

## 1. 模型名称与定位
- 模型名: `Qwen3-TTS-12Hz-0.6B-Base`
- 规模: `0.6B` 级别
- 类型: `Base`（语音克隆模型）
- 能力: 参考音频驱动的 voice clone，支持 10 种语言
- 依赖: 内置 `speech_tokenizer`（`Qwen3-TTS-Tokenizer-12Hz`）

## 2. 模型输入（是否支持 instruct/ref.wav）
- `text`: 必填
- `language`: 可选，默认 `Auto`
- `instruct`: Base 模型不使用 instruct
- `ref.wav`（支持）:
  - 支持本地路径 / URL / base64 / `(float32_waveform, sr)`
  - 模式与 1.7B Base 相同:
    1) `x_vector_only_mode=false`（默认）: 必须 `ref_audio + ref_text`
    2) `x_vector_only_mode=true`: 仅说话人向量

- 参考音频处理:
  - tokenizer 编码可用原始 sr
  - speaker embedding 提取前需重采样到 `24kHz`
  - 梅尔参数固定: `n_fft=1024, win=1024, hop=256, n_mels=128, fmin=0, fmax=12000`

- 文本模板（C++ 直连需自己拼）:
  - 目标文本: `<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n`
  - 参考文本: `<|im_start|>assistant\n{ref_text}<|im_end|>\n`

## 3. 模型架构（层级与维度）
- 顶层组件与 1.7B Base 相同: `talker + code_predictor + speaker_encoder + speech_tokenizer`

- Talker 主干（与 1.7B 的主要差异在 hidden 维度）:
  - `num_hidden_layers=28`
  - `hidden_size=1024`
  - `intermediate_size=3072`
  - `num_attention_heads=16`, `num_key_value_heads=8`, `head_dim=128`
  - `text_embedding`: `[151936, 2048]`
  - `codec_embedding`: `[3072, 1024]`
  - `text_projection`: MLP `2048 -> 2048 -> 1024`
  - `codec_head`: `Linear(1024 -> 3072)`
  - 每层结构（28 层同构）:
    - `RMSNorm(1024)`
    - Self-Attn: `Q:1024->16*128`, `K:1024->8*128`, `V:1024->8*128`, `O:16*128->1024`
    - `RMSNorm(1024)`
    - SwiGLU MLP: `1024 -> 3072 -> 1024`

- Code Predictor:
  - 配置与 1.7B 同: `5` 层, `hidden=1024`, `ffn=3072`, `heads=16`, `kv=8`, `vocab=2048`, `num_code_groups=16`
  - 由于 talker hidden 本身就是 `1024`，不需要 `2048->1024` 的额外降维投影（等价 Identity）

- Speaker Encoder（Base 专有）:
  - ECAPA-TDNN 变体，输出维度 `enc_dim=1024`
  - 其余骨干参数与 1.7B Base 相同:
    - `enc_channels=[512,512,512,512,1536]`
    - `kernel=[5,3,3,3,1]`, `dilation=[1,2,3,4,1]`
    - `sample_rate=24000`

- Token IDs（与 1.7B Base 相同）:
  - `codec_pad_id=2148`, `codec_bos_id=2149`, `codec_eos_token_id=2150`
  - `codec_think_id=2154`, `codec_nothink_id=2155`, `codec_think_bos_id=2156`, `codec_think_eos_id=2157`
  - 语言 ID: chinese=2055, english=2050, german=2053, italian=2070, portuguese=2071,
    spanish=2054, japanese=2058, korean=2064, french=2061, russian=2069

## 4. 可调参数（默认值）
- 默认采样参数与 1.7B Base 一致:
  - `do_sample=true`, `top_k=50`, `top_p=1.0`, `temperature=0.9`
  - `repetition_penalty=1.05`
  - `subtalker_dosample=true`, `subtalker_top_k=50`, `subtalker_top_p=1.0`, `subtalker_temperature=0.9`
  - `max_new_tokens=8192`

- 其他:
  - `non_streaming_mode` 默认 `false`
  - `x_vector_only_mode` 默认 `false`
  - `language` 默认 `Auto`
