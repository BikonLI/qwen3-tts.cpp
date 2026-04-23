#pragma once

#include "ggml.h"
#include "ggml-backend.h"
#include "gguf.h"

#include <string>
#include <map>
#include <vector>
#include <memory>

namespace qwen3_tts {

// Audio codec encoder configuration (Mimi-based encoder)
struct codec_encoder_config {
    int32_t sample_rate = 24000;
    int32_t frame_rate = 12;            // 12.5 Hz after 2x downsample from 25 Hz
    int32_t n_codebooks = 16;           // 1 semantic + 15 acoustic
    int32_t codebook_size = 2048;       // Entries per codebook
    int32_t codebook_dim = 256;         // Embedding dimension per codebook
    int32_t hidden_dim = 512;           // SEANet + transformer hidden dimension
    int32_t n_tfm_layers = 8;           // Encoder transformer layers
    int32_t n_heads = 8;                // Attention heads in encoder transformer
    int32_t ffn_dim = 2048;             // FFN intermediate dimension (4x hidden)
    int32_t n_seanet_res_blocks = 1;    // Residual blocks per SEANet layer (MimiConfig num_residual_layers)
    int32_t n_downsample_layers = 4;    // SEANet downsample stages
    int32_t n_quantizers = 32;          // Total quantizers stored (only first 1+15 used)
    int32_t valid_quantizers = 16;      // Actually used: 1 semantic + 15 acoustic
    float rms_norm_eps = 1e-5f;
    float rope_theta = 10000.0f;
    int32_t sliding_window = 250;       // Sliding window attention size (MimiConfig default)
    // SEANet downsample strides (reversed upsampling_ratios from MimiConfig):
    // [4, 5, 6, 8] -> total 960x from 24kHz -> 25 Hz
    // Then transformer at 25 Hz
    // Then 2x downsample conv -> 12.5 Hz (1920x from 24kHz)
    int32_t seanet_strides[4] = {4, 5, 6, 8};
    int32_t seanet_kernel_sizes[4] = {8, 10, 12, 16};  // kernel = 2*stride for downsample convs
};

// SEANet residual block weights (ELU + Conv1d(kernel=3) + ELU + Conv1d(kernel=1))
// Each block has 2 convs with a skip connection wrapping the entire block.
struct seanet_residual_block {
    struct ggml_tensor * conv1_w = nullptr;  // First conv (kernel=3, dilation from config)
    struct ggml_tensor * conv1_b = nullptr;
    struct ggml_tensor * conv2_w = nullptr;  // Second conv (kernel=1)
    struct ggml_tensor * conv2_b = nullptr;
};

// SEANet encoder layer weights (residual block + downsample conv)
struct seanet_encoder_layer {
    // Residual block (1 per layer, matching MimiConfig num_residual_layers=1)
    seanet_residual_block res_block;

    // Downsample convolution (stride > 1)
    struct ggml_tensor * conv_w = nullptr;
    struct ggml_tensor * conv_b = nullptr;
};

// Encoder transformer layer weights
struct enc_tfm_layer {
    // Attention
    struct ggml_tensor * attn_norm_w = nullptr;
    struct ggml_tensor * attn_norm_b = nullptr;
    struct ggml_tensor * attn_q_w = nullptr;
    struct ggml_tensor * attn_k_w = nullptr;
    struct ggml_tensor * attn_v_w = nullptr;
    struct ggml_tensor * attn_output_w = nullptr;
    struct ggml_tensor * attn_scale = nullptr;  // layer_scale for attention

    // FFN (SiLU MLP: up + down)
    struct ggml_tensor * ffn_norm_w = nullptr;
    struct ggml_tensor * ffn_norm_b = nullptr;
    struct ggml_tensor * ffn_up_w = nullptr;
    struct ggml_tensor * ffn_down_w = nullptr;
    struct ggml_tensor * ffn_scale = nullptr;   // layer_scale for FFN
};

// Audio codec encoder model weights
struct codec_encoder_model {
    codec_encoder_config config;

    // SEANet encoder: initial conv
    struct ggml_tensor * init_conv_w = nullptr;   // tok_enc.conv.0.weight
    struct ggml_tensor * init_conv_b = nullptr;   // tok_enc.conv.0.bias

    // SEANet encoder layers (4 downsample stages)
    // Each has 1 residual block + downsample conv
    // Tensors: tok_enc.conv.{3,6,9,12}.weight/bias (downsample convs)
    //          tok_enc.res.{0..3}.blk.{1,3}.weight/bias (residual block convs)
    seanet_encoder_layer seanet_layers[4];

    // SEANet encoder: final conv
    struct ggml_tensor * final_conv_w = nullptr;  // tok_enc.conv.14.weight
    struct ggml_tensor * final_conv_b = nullptr;  // tok_enc.conv.14.bias

    // Encoder transformer layers (8 layers)
    enc_tfm_layer tfm_layers[8];

    // Downsample conv after transformer (stride=2)
    struct ggml_tensor * downsample_w = nullptr;  // tok_enc.downsample.weight

    // VQ semantic quantizer
    struct ggml_tensor * vq_semantic_input_proj = nullptr;   // tok_enc.vq_semantic.input_proj.weight
    struct ggml_tensor * vq_semantic_output_proj = nullptr;  // tok_enc.vq_semantic.output_proj.weight
    struct ggml_tensor * vq_semantic_codebook[32] = {nullptr}; // tok_enc.vq_semantic.{i}.codebook

    // VQ acoustic quantizer
    struct ggml_tensor * vq_acoustic_input_proj = nullptr;   // tok_enc.vq_acoustic.input_proj.weight
    struct ggml_tensor * vq_acoustic_output_proj = nullptr;  // tok_enc.vq_acoustic.output_proj.weight
    struct ggml_tensor * vq_acoustic_codebook[32] = {nullptr}; // tok_enc.vq_acoustic.{i}.codebook

    // GGML context for tensor metadata
    struct ggml_context * ctx = nullptr;

    // Backend buffer for weights
    ggml_backend_buffer_t buffer = nullptr;

    // Tensor name to tensor mapping
    std::map<std::string, struct ggml_tensor *> tensors;
};

// Compute state for encoder
struct codec_encoder_state {
    ggml_backend_t backend = nullptr;
    ggml_backend_t backend_cpu = nullptr;
    ggml_backend_sched_t sched = nullptr;
    std::vector<uint8_t> compute_meta;

    // Precomputed sliding window causal mask [max_sliding_window, max_sliding_window]
    // Filled once at load time, reused for all forward passes.
    std::vector<float> sw_mask_data;
    int32_t sw_mask_size = 0;  // Actual dimension of the precomputed mask
};

// Audio codec encoder class (Mimi-based)
// Encodes audio waveform to discrete codes (audio -> codes)
class AudioCodecEncoder {
public:
    AudioCodecEncoder();
    ~AudioCodecEncoder();

    // Load model from GGUF file (tokenizer model)
    bool load_model(const std::string & model_path);

    // Release all model/runtime resources
    void unload_model();

    // Encode audio samples to discrete codes
    // samples: audio samples normalized to [-1, 1], 24kHz mono
    // n_samples: number of samples
    // codes: output codes [n_frames * n_codebooks] as int32_t (row-major)
    // n_frames: output number of frames
    bool encode(const float * samples, int32_t n_samples,
                std::vector<int32_t> & codes, int32_t & n_frames);

    const codec_encoder_config & get_config() const { return model_.config; }

    // Configure backend thread count when supported by selected backend.
    // n_threads <= 0 keeps backend defaults.
    bool set_n_threads(int32_t n_threads);

    const std::string & get_error() const { return error_msg_; }

private:
    // Build computation graph for SEANet encoder + transformer forward pass
    // Returns the embeddings tensor [hidden_dim, n_frames_down] after downsample
    struct ggml_cgraph * build_graph(struct ggml_context * ctx0,
                                      int32_t n_samples);

    // Apply LayerNorm (with optional bias) — MimiConfig uses LayerNorm in encoder
    struct ggml_tensor * apply_rms_norm(struct ggml_context * ctx,
                                         struct ggml_tensor * x,
                                         struct ggml_tensor * w,
                                         struct ggml_tensor * b,
                                         float eps);

    // Apply encoder transformer layer
    struct ggml_tensor * apply_tfm_layer(struct ggml_context * ctx,
                                          struct ggml_tensor * x,
                                          const enc_tfm_layer & layer,
                                          int32_t n_frames,
                                          struct ggml_tensor * positions);

    // Apply SEANet residual block (ELU + Conv1d(kernel=3) + ELU + Conv1d(kernel=1) + residual)
    struct ggml_tensor * apply_residual_block(struct ggml_context * ctx,
                                                 struct ggml_tensor * x,
                                                 const seanet_residual_block & block);

    // VQ quantization on CPU: find nearest codebook entry for each frame
    // embeddings: [n_frames, hidden_dim] float data
    // Returns codes [n_frames * n_codebooks] int32 row-major
    bool quantize_vq_cpu(const float * embeddings, int32_t n_frames,
                         std::vector<int32_t> & codes);

    codec_encoder_model model_;
    codec_encoder_state state_;
    std::string error_msg_;
};

// Free model resources
void free_codec_encoder_model(codec_encoder_model & model);

} // namespace qwen3_tts
