# Qwen3-TTS-12Hz-1.7B-VoiceDesign

## 1. 模型名称与定位
- 模型名: `Qwen3-TTS-12Hz-1.7B-VoiceDesign`
- 规模: `1.7B` 级别
- 类型: `VoiceDesign`（文本 + 自然语言声音描述）
- 功能: 通过 `instruct` 设计音色/情绪/语速/语气等属性

## 2. 模型输入（是否支持 instruct/ref.wav）
- `text`: 必填
- `instruct`: 核心输入（建议必填；实现上允许空）
- `language`: 可选，默认 `Auto`
- `ref.wav`: 不支持（不是克隆模型）

- 文本模板（C++ 直连需自己拼）:
  - 目标文本: `<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n`
  - 指令文本: `<|im_start|>user\n{instruct}<|im_end|>\n`

## 3. 模型架构（层级与维度）
- 顶层: `Qwen3TTSForConditionalGeneration`
  - `talker + code_predictor + speech_tokenizer`
  - `speaker_encoder=None`（VoiceDesign 不需要）

- Talker 主干:
  - `num_hidden_layers=28`
  - `hidden_size=2048`
  - `intermediate_size=6144`
  - `num_attention_heads=16`, `num_key_value_heads=8`, `head_dim=128`
  - `text_embedding=[151936,2048]`, `codec_embedding=[3072,2048]`
  - `text_projection: 2048 -> 2048 -> 2048`
  - `codec_head: 2048 -> 3072`
  - 每层（28 层同构）:
    - `RMSNorm(2048)`
    - SelfAttn: `Q:2048->16*128`, `K:2048->8*128`, `V:2048->8*128`, `O:16*128->2048`
    - `RMSNorm(2048)`
    - SwiGLU MLP: `2048 -> 6144 -> 2048`

- Code Predictor:
  - `num_hidden_layers=5`
  - `hidden_size=1024`
  - `intermediate_size=3072`
  - `num_attention_heads=16`, `num_key_value_heads=8`, `head_dim=128`
  - `vocab_size=2048`, `num_code_groups=16`
  - 有 `2048->1024` 投影

- 关键 token 与语言 ID:
  - `codec_pad_id=2148`, `codec_bos_id=2149`, `codec_eos_token_id=2150`
  - `codec_think_id=2154`, `codec_nothink_id=2155`, `codec_think_bos_id=2156`, `codec_think_eos_id=2157`
  - 语言 ID:
    - chinese=2055, english=2050, german=2053, italian=2070, portuguese=2071,
      spanish=2054, japanese=2058, korean=2064, french=2061, russian=2069

## 4. 可调参数（默认值）
- 默认采样参数:
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

- 调用参数:
  - `non_streaming_mode` 默认 `true`
  - `language` 默认 `Auto`
  - `instruct` 建议始终提供（这是 VoiceDesign 的主要控制入口）
