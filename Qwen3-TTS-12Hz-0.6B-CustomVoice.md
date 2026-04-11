# Qwen3-TTS-12Hz-0.6B-CustomVoice

## 1. 模型名称与定位
- 模型名: `Qwen3-TTS-12Hz-0.6B-CustomVoice`
- 规模: `0.6B` 级别
- 类型: `CustomVoice`（预置音色）
- 功能: 9 个官方音色，支持 10 种语言

## 2. 模型输入（是否支持 instruct/ref.wav）
- `text`: 必填
- `speaker`: 必填
- `language`: 可选，默认 `Auto`
- `instruct`: 官方接口中对 `0.6B` 会被忽略（可视为不支持指令控制）
- `ref.wav`: 不支持（非克隆模型）

- 说话人列表（9 个）:
  - `vivian`, `serena`, `uncle_fu`, `dylan`, `eric`, `ryan`, `aiden`, `ono_anna`, `sohee`

- 方言自动映射（与 1.7B CustomVoice 一致）:
  - `dylan -> beijing_dialect`
  - `eric -> sichuan_dialect`

- 文本模板（C++ 直连需自己拼）:
  - 目标文本: `<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n`
  - （如你强行传 instruct）指令模板仍是 `<|im_start|>user\n{instruct}<|im_end|>\n`，但 0.6B 默认不会使用

## 3. 模型架构（层级与维度）
- 顶层: `talker + code_predictor + speech_tokenizer`，无 speaker encoder

- Talker 主干:
  - `num_hidden_layers=28`
  - `hidden_size=1024`
  - `intermediate_size=3072`
  - `num_attention_heads=16`, `num_key_value_heads=8`, `head_dim=128`
  - `text_embedding=[151936,2048]`
  - `codec_embedding=[3072,1024]`
  - `text_projection: 2048 -> 2048 -> 1024`
  - `codec_head: 1024 -> 3072`
  - 每层结构:
    - `RMSNorm(1024)` -> SelfAttn(`Q:1024->16*128`, `K/V:1024->8*128`, `O:16*128->1024`) -> 残差
    - `RMSNorm(1024)` -> SwiGLU(`1024->3072->1024`) -> 残差

- Code Predictor:
  - `5` 层, `hidden=1024`, `ffn=3072`, `heads=16`, `kv=8`, `vocab=2048`, `num_code_groups=16`
  - 与 talker hidden 同为 1024，因此输入到 predictor 不需要额外降维

- 关键 token 与语言/说话人 ID:
  - 与 1.7B CustomVoice 相同
  - `codec_pad_id=2148`, `codec_bos_id=2149`, `codec_eos_token_id=2150`
  - 语言 ID含 `beijing_dialect=2074`, `sichuan_dialect=2062`

## 4. 可调参数（默认值）
- 默认采样参数:
  - `do_sample=true`, `top_k=50`, `top_p=1.0`, `temperature=0.9`
  - `repetition_penalty=1.05`
  - `subtalker_dosample=true`, `subtalker_top_k=50`, `subtalker_top_p=1.0`, `subtalker_temperature=0.9`
  - `max_new_tokens=8192`

- 调用参数:
  - `non_streaming_mode` 默认 `true`
  - `language` 默认 `Auto`
  - `speaker` 必填
  - `instruct` 对 0.6B 默认无效（实现层会清空）
