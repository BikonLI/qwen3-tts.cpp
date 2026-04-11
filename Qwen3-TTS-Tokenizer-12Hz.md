# Qwen3-TTS-Tokenizer-12Hz

## 1. 模型名称与定位
- 模型名: `Qwen3-TTS-Tokenizer-12Hz`
- 类型: 语音 codec tokenizer（把波形编码成离散语音 token，再解码回波形）
- 采样率: 输入/输出均为 `24kHz`
- 码率: `12.5 Hz` 帧率（每秒约 12.5 个时间步）

## 2. 模型输入（接口与处理）
- 编码输入: 单条或批量音频波形，形状通常是 `float32, [B, T]`
- 解码输入: 离散语音码 `audio_codes`，形状 `LongTensor [B, L, 16]`
- 编码输出: 每条样本输出 `16` 个 codebook 的 token 序列，单样本可视作 `[L, 16]`
- 解码输出: 波形列表 `List[float32]`，输出采样率 `24000`
- 关键长度换算:
  - 编码下采样率 `encode_downsample_rate = 1920`（24k / 1920 = 12.5Hz）
  - 解码上采样率 `decode_upsample_rate = 1920`

## 3. 模型架构（可直接据此实现 C++ 版加载/调用）
- 顶层结构: `Qwen3TTSTokenizerV2Model = Encoder(Mimi) + Decoder(AR Transformer + RVQ + Vocoder)`

- Encoder（Mimi）配置维度:
  - `hidden_size=512`, `num_hidden_layers=8`, `num_attention_heads=8`, `head_dim=64`
  - `intermediate_size=2048`, `num_quantizers=32`, `codebook_size=2048`, `codebook_dim=256`
  - 实际导出仅取前 `encoder_valid_num_quantizers=16` 组码

- Decoder（TokenizerV2Decoder）配置维度:
  - `num_quantizers=16`, `codebook_size=2048`, `semantic_codebook_size=4096`
  - `codebook_dim=512`, `latent_dim=1024`, `decoder_dim=1536`
  - Transformer 预解码器:
    - 层数 `8`
    - 每层: `RMSNorm(512) -> SelfAttn(16 heads, head_dim=64, kv_heads=16, sliding_window=72) -> RMSNorm(512) -> SwiGLU MLP(512->1024->512)`
    - LayerScale 初值 `0.01`
  - RVQ 解码:
    - Split RVQ: `1` 个 semantic quantizer + `15` 个 acoustic quantizer
    - 每个 codebook 大小 `2048`
  - 波形上采样链路:
    - `upsampling_ratios = [2, 2]`
    - `upsample_rates = [8, 5, 4, 3]`
    - 总上采样倍数 `2*2*8*5*4*3 = 1920`

- 关键实现点（C++）:
  - 解码前要把无效码位（若有）裁剪/填充到合法范围；官方实现中对负值做了 `clamp(min=0)`
  - 官方 chunked decode 默认窗口:
    - `chunk_size=300`（码帧）
    - `left_context_size=25`

## 4. 可调参数（默认值）
- Tokenizer 本身不使用 `top_k/top_p/temperature` 这类采样参数
- 常用可调项:
  - `chunk_size=300`（chunked 解码）
  - `left_context_size=25`（chunked 解码左上下文）
- 固定配置（来自 checkpoint）:
  - `input_sample_rate=24000`
  - `output_sample_rate=24000`
  - `encode_downsample_rate=1920`
  - `decode_upsample_rate=1920`
