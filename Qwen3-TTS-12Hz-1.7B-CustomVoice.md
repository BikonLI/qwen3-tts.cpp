# Qwen3-TTS-12Hz-1.7B-CustomVoice

## 1. 模型名称与定位
- 模型名: `Qwen3-TTS-12Hz-1.7B-CustomVoice`
- 规模: `1.7B` 级别
- 类型: `CustomVoice`（预置音色 + 可选指令控制）
- 功能: 9 个官方音色，支持 10 种语言，多语合成

## 2. 模型输入（是否支持 instruct/ref.wav）
- `text`: 必填
- `speaker`: 必填，必须是支持的说话人 ID
- `language`: 可选，默认 `Auto`
- `instruct`: 支持（可空），用于语气/风格控制
- `ref.wav`: 不支持（此模型不是克隆模型）

- 说话人列表（9 个）:
  - `vivian`, `serena`, `uncle_fu`, `dylan`, `eric`, `ryan`, `aiden`, `ono_anna`, `sohee`

- 特殊处理（方言）:
  - 若 `language` 为 `Chinese` 或 `Auto`，且 speaker 是方言音色:
    - `dylan` 自动映射到 `beijing_dialect`
    - `eric` 自动映射到 `sichuan_dialect`

- 文本模板（C++ 直连需自己拼）:
  - 目标文本: `<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n`
  - 指令文本: `<|im_start|>user\n{instruct}<|im_end|>\n`

## 3. 模型架构（层级与维度）
- 顶层: `Qwen3TTSForConditionalGeneration`
  - `talker` + `code_predictor` + `speech_tokenizer`
  - `speaker_encoder=None`（CustomVoice 不用 speaker encoder）

- Talker 主干:
  - `num_hidden_layers=28`
  - `hidden_size=2048`
  - `intermediate_size=6144`
  - `num_attention_heads=16`, `num_key_value_heads=8`, `head_dim=128`
  - `rope_theta=1_000_000`, `mrope_section=[24,20,20]`, `interleaved=true`
  - `text_embedding=[151936,2048]`, `codec_embedding=[3072,2048]`
  - `text_projection: 2048 -> 2048 -> 2048`
  - `codec_head: 2048 -> 3072`
  - 每层（28 层同构）:
    - `RMSNorm(2048)` -> SelfAttn(`Q:2048->16*128`, `K/V:2048->8*128`, `O:16*128->2048`) -> 残差
    - `RMSNorm(2048)` -> SwiGLU(`2048->6144->2048`) -> 残差

- Code Predictor:
  - `5` 层 Transformer，`hidden=1024`, `ffn=3072`, `heads=16`, `kv=8`, `vocab=2048`
  - `num_code_groups=16`（主干先出第 1 组，子预测器再补 15 组）
  - 有 `2048->1024` 投影（因为 talker hidden 为 2048）

- Speaker/Language 控制维度:
  - 说话人 ID 通过 `codec_embedding` 注入（不是独立 speaker encoder）
  - 语言 ID 通过 codec 前缀 token 注入

- 关键 token:
  - `codec_pad_id=2148`, `codec_bos_id=2149`, `codec_eos_token_id=2150`
  - `codec_think_id=2154`, `codec_nothink_id=2155`, `codec_think_bos_id=2156`, `codec_think_eos_id=2157`
  - 语言 ID:
    - chinese=2055, english=2050, german=2053, italian=2070, portuguese=2071,
      spanish=2054, japanese=2058, korean=2064, french=2061, russian=2069,
      beijing_dialect=2074, sichuan_dialect=2062

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
  - `speaker` 必填且需在支持列表内
  - `instruct` 可选（空字符串表示不启用风格控制）
