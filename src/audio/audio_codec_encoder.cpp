#include "audio_codec_encoder.h"
#include "gguf_loader.h"
#include "ggml-cpu.h"

#include <cmath>
#include <cstring>
#include <algorithm>
#include <numeric>
#include <cstdlib>
#include <limits>

#define QWEN3_TTS_ENC_MAX_NODES 32768

namespace qwen3_tts {

static bool encoder_debug_enabled() {
    const char * env = std::getenv("QWEN3_TTS_ENCODER_DEBUG");
    if (!env || env[0] == '\0') {
        return false;
    }
    std::string v = env;
    std::transform(v.begin(), v.end(), v.begin(), [](unsigned char c) { return (char)std::tolower(c); });
    return !(v == "0" || v == "false" || v == "off" || v == "no");
}

AudioCodecEncoder::AudioCodecEncoder() = default;

AudioCodecEncoder::~AudioCodecEncoder() {
    unload_model();
}

void AudioCodecEncoder::unload_model() {
    free_codec_encoder_model(model_);

    if (state_.sched) {
        ggml_backend_sched_free(state_.sched);
        state_.sched = nullptr;
    }
    if (state_.backend) {
        ggml_backend_free(state_.backend);
        state_.backend = nullptr;
    }
    if (state_.backend_cpu) {
        ggml_backend_free(state_.backend_cpu);
        state_.backend_cpu = nullptr;
    }
    state_.compute_meta.clear();
}

bool AudioCodecEncoder::load_model(const std::string & model_path) {
    unload_model();

    GGUFLoader loader;
    if (!loader.open(model_path)) {
        error_msg_ = loader.get_error();
        return false;
    }

    // Read encoder config from GGUF metadata
    auto & cfg = model_.config;
    cfg.sample_rate       = loader.get_u32("qwen3-tts.tokenizer.sample_rate", 24000);
    cfg.codebook_size     = loader.get_u32("qwen3-tts.tokenizer.codebook_size", 2048);
    cfg.hidden_dim        = loader.get_u32("qwen3-tts.encoder.hidden_size", 512);
    cfg.n_tfm_layers      = loader.get_u32("qwen3-tts.encoder.num_layers", 8);
    cfg.n_heads           = loader.get_u32("qwen3-tts.encoder.num_heads", 8);
    cfg.n_quantizers      = loader.get_u32("qwen3-tts.encoder.num_quantizers", 32);
    cfg.valid_quantizers  = loader.get_u32("qwen3-tts.encoder.valid_quantizers", 16);
    cfg.codebook_dim      = loader.get_u32("qwen3-tts.encoder.codebook_dim", 256);

    // Derived config
    cfg.ffn_dim = cfg.hidden_dim * 4;  // 2048
    cfg.n_codebooks = cfg.valid_quantizers;  // 16 = 1 semantic + 15 acoustic

    // Count encoder tensors (prefix "tok_enc.")
    int64_t n_tensors = loader.get_n_tensors();
    int enc_tensor_count = 0;
    for (int64_t i = 0; i < n_tensors; ++i) {
        const char * name = loader.get_tensor_name(i);
        if (name && strncmp(name, "tok_enc.", 8) == 0) {
            enc_tensor_count++;
        }
    }

    if (enc_tensor_count == 0) {
        error_msg_ = "No encoder tensors found in model";
        return false;
    }

    fprintf(stderr, "AudioCodecEncoder: found %d encoder tensors\n", enc_tensor_count);

    size_t ctx_size = ggml_tensor_overhead() * enc_tensor_count;
    struct ggml_init_params params = {
        /*.mem_size   =*/ ctx_size,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };

    model_.ctx = ggml_init(params);
    if (!model_.ctx) {
        error_msg_ = "Failed to initialize GGML context";
        return false;
    }

    struct gguf_context * gguf_ctx = loader.get_ctx();
    struct ggml_context * meta_ctx = loader.get_meta_ctx();

    // Map all tok_enc.* tensors to struct members
    for (int64_t i = 0; i < n_tensors; ++i) {
        const char * name = loader.get_tensor_name(i);
        if (!name || strncmp(name, "tok_enc.", 8) != 0) {
            continue;
        }

        struct ggml_tensor * meta_tensor = ggml_get_tensor(meta_ctx, name);
        if (!meta_tensor) {
            continue;
        }

        struct ggml_tensor * tensor = ggml_dup_tensor(model_.ctx, meta_tensor);
        ggml_set_name(tensor, name);

        model_.tensors[name] = tensor;

        std::string sname(name);

        // --- Direct mappings ---
        if (sname == "tok_enc.downsample.weight") {
            model_.downsample_w = tensor;
        }
        // VQ semantic
        else if (sname == "tok_enc.vq_semantic.input_proj.weight") {
            model_.vq_semantic_input_proj = tensor;
        }
        else if (sname == "tok_enc.vq_semantic.output_proj.weight") {
            model_.vq_semantic_output_proj = tensor;
        }
        // VQ acoustic
        else if (sname == "tok_enc.vq_acoustic.input_proj.weight") {
            model_.vq_acoustic_input_proj = tensor;
        }
        else if (sname == "tok_enc.vq_acoustic.output_proj.weight") {
            model_.vq_acoustic_output_proj = tensor;
        }
        // --- Pattern-based mappings ---
        else {
            int idx0 = -1, idx1 = -1, n = 0;
            char suffix[64] = {};

            #define MATCH1(fmt, var) (sscanf(name, fmt "%n", &var, &n) == 1 && (size_t)n == strlen(name))
            #define MATCH2(fmt, v1, v2) (sscanf(name, fmt "%n", &v1, &v2, &n) == 2 && (size_t)n == strlen(name))
            #define MATCH1S(fmt, var, suf) (sscanf(name, fmt, &var, suf) == 2)

            // SEANet conv layers: tok_enc.conv.{i}.weight/bias
            // Actual MimiEncoder conv layer indices: 0, 3, 6, 9, 12, 14
            // i=0: initial conv, i=3,6,9,12: downsample convs, i=14: final conv
            if (MATCH1S("tok_enc.conv.%d.%63s", idx0, suffix)) {
                if (idx0 == 0) {
                    if (strcmp(suffix, "weight") == 0) model_.init_conv_w = tensor;
                    else if (strcmp(suffix, "bias") == 0) model_.init_conv_b = tensor;
                } else if (idx0 == 3) {
                    if (strcmp(suffix, "weight") == 0) model_.seanet_layers[0].conv_w = tensor;
                    else if (strcmp(suffix, "bias") == 0) model_.seanet_layers[0].conv_b = tensor;
                } else if (idx0 == 6) {
                    if (strcmp(suffix, "weight") == 0) model_.seanet_layers[1].conv_w = tensor;
                    else if (strcmp(suffix, "bias") == 0) model_.seanet_layers[1].conv_b = tensor;
                } else if (idx0 == 9) {
                    if (strcmp(suffix, "weight") == 0) model_.seanet_layers[2].conv_w = tensor;
                    else if (strcmp(suffix, "bias") == 0) model_.seanet_layers[2].conv_b = tensor;
                } else if (idx0 == 12) {
                    if (strcmp(suffix, "weight") == 0) model_.seanet_layers[3].conv_w = tensor;
                    else if (strcmp(suffix, "bias") == 0) model_.seanet_layers[3].conv_b = tensor;
                } else if (idx0 == 14) {
                    if (strcmp(suffix, "weight") == 0) model_.final_conv_w = tensor;
                    else if (strcmp(suffix, "bias") == 0) model_.final_conv_b = tensor;
                }
            }
            // SEANet residual blocks: tok_enc.res.{i}.blk.{j}.weight/bias
            // MimiResnetBlock has 2 convs: block.1 (kernel=3) and block.3 (kernel=1)
            else if (MATCH2("tok_enc.res.%d.blk.%d.weight", idx0, idx1)) {
                if (idx0 >= 0 && idx0 < 4) {
                    if (idx1 == 1) {
                        model_.seanet_layers[idx0].res_block.conv1_w = tensor;
                    } else if (idx1 == 3) {
                        model_.seanet_layers[idx0].res_block.conv2_w = tensor;
                    }
                }
            }
            else if (MATCH2("tok_enc.res.%d.blk.%d.bias", idx0, idx1)) {
                if (idx0 >= 0 && idx0 < 4) {
                    if (idx1 == 1) {
                        model_.seanet_layers[idx0].res_block.conv1_b = tensor;
                    } else if (idx1 == 3) {
                        model_.seanet_layers[idx0].res_block.conv2_b = tensor;
                    }
                }
            }
            // Encoder transformer layers: tok_enc.blk.{i}.*
            else if (sscanf(name, "tok_enc.blk.%d.", &idx0) == 1 && idx0 >= 0 && idx0 < 8) {
                if (sname.find(".attn_norm.weight") != std::string::npos)
                    model_.tfm_layers[idx0].attn_norm_w = tensor;
                else if (sname.find(".attn_norm.bias") != std::string::npos)
                    model_.tfm_layers[idx0].attn_norm_b = tensor;
                else if (sname.find(".attn_q.weight") != std::string::npos)
                    model_.tfm_layers[idx0].attn_q_w = tensor;
                else if (sname.find(".attn_k.weight") != std::string::npos)
                    model_.tfm_layers[idx0].attn_k_w = tensor;
                else if (sname.find(".attn_v.weight") != std::string::npos)
                    model_.tfm_layers[idx0].attn_v_w = tensor;
                else if (sname.find(".attn_output.weight") != std::string::npos)
                    model_.tfm_layers[idx0].attn_output_w = tensor;
                else if (sname.find(".attn_scale") != std::string::npos)
                    model_.tfm_layers[idx0].attn_scale = tensor;
                else if (sname.find(".ffn_norm.weight") != std::string::npos)
                    model_.tfm_layers[idx0].ffn_norm_w = tensor;
                else if (sname.find(".ffn_norm.bias") != std::string::npos)
                    model_.tfm_layers[idx0].ffn_norm_b = tensor;
                else if (sname.find(".ffn_up.weight") != std::string::npos)
                    model_.tfm_layers[idx0].ffn_up_w = tensor;
                else if (sname.find(".ffn_down.weight") != std::string::npos)
                    model_.tfm_layers[idx0].ffn_down_w = tensor;
                else if (sname.find(".ffn_scale") != std::string::npos)
                    model_.tfm_layers[idx0].ffn_scale = tensor;
            }
            // VQ semantic codebooks: tok_enc.vq_semantic.{i}.codebook
            else if (MATCH1("tok_enc.vq_semantic.%d.codebook", idx0)) {
                if (idx0 >= 0 && idx0 < 32) {
                    model_.vq_semantic_codebook[idx0] = tensor;
                }
            }
            // VQ acoustic codebooks: tok_enc.vq_acoustic.{i}.codebook
            else if (MATCH1("tok_enc.vq_acoustic.%d.codebook", idx0)) {
                if (idx0 >= 0 && idx0 < 32) {
                    model_.vq_acoustic_codebook[idx0] = tensor;
                }
            }

            #undef MATCH1
            #undef MATCH2
            #undef MATCH1S
        }
    }

    // Load tensor data from file into backend buffer
    if (!load_tensor_data_from_file(model_path, gguf_ctx, model_.ctx,
                                     model_.tensors, model_.buffer, error_msg_,
                                     GGML_BACKEND_DEVICE_TYPE_IGPU)) {
        return false;
    }

    // Initialize backend
    state_.backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_IGPU, nullptr);
    if (!state_.backend) {
        state_.backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU, nullptr);
    }
    if (!state_.backend) {
        state_.backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_ACCEL, nullptr);
    }
    if (!state_.backend) {
        state_.backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    }
    if (!state_.backend) {
        error_msg_ = "Failed to initialize backend for AudioCodecEncoder";
        return false;
    }

    ggml_backend_dev_t device = ggml_backend_get_device(state_.backend);
    const char * device_name = device ? ggml_backend_dev_name(device) : "Unknown";
    fprintf(stderr, "  AudioCodecEncoder backend: %s\n", device_name);

    if (device && ggml_backend_dev_type(device) != GGML_BACKEND_DEVICE_TYPE_CPU) {
        state_.backend_cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
        if (!state_.backend_cpu) {
            error_msg_ = "Failed to initialize CPU fallback backend for AudioCodecEncoder";
            return false;
        }
    }

    std::vector<ggml_backend_t> backends;
    backends.push_back(state_.backend);
    if (state_.backend_cpu) {
        backends.push_back(state_.backend_cpu);
    }

    state_.sched = ggml_backend_sched_new(backends.data(), nullptr, (int)backends.size(),
                                            QWEN3_TTS_ENC_MAX_NODES, false, true);
    if (!state_.sched) {
        error_msg_ = "Failed to create backend scheduler";
        return false;
    }

    state_.compute_meta.resize(ggml_tensor_overhead() * QWEN3_TTS_ENC_MAX_NODES + ggml_graph_overhead());

    // Precompute sliding window causal mask up to a reasonable max size.
    // At 25Hz, 4096 frames = ~164s of audio. Most utterances are shorter.
    const int32_t max_mask_frames = 4096;
    if (cfg.sliding_window > 0) {
        state_.sw_mask_size = max_mask_frames;
        state_.sw_mask_data.resize((size_t)max_mask_frames * max_mask_frames);
        for (int q = 0; q < max_mask_frames; ++q) {
            for (int k = 0; k < max_mask_frames; ++k) {
                if (k > q - cfg.sliding_window && k <= q) {
                    state_.sw_mask_data[q * max_mask_frames + k] = 0.0f;
                } else {
                    state_.sw_mask_data[q * max_mask_frames + k] = -INFINITY;
                }
            }
        }
    }

    fprintf(stderr, "AudioCodecEncoder: model loaded successfully (hidden=%d, layers=%d, heads=%d, codebooks=%d)\n",
            cfg.hidden_dim, cfg.n_tfm_layers, cfg.n_heads, cfg.n_codebooks);

    return true;
}

bool AudioCodecEncoder::set_n_threads(int32_t n_threads) {
    if (n_threads <= 0) {
        return true;
    }

    if (!state_.backend) {
        error_msg_ = "Backend not initialized";
        return false;
    }

    if (!set_backend_n_threads(state_.backend, n_threads)) {
        error_msg_ = "Backend does not support thread configuration";
        return false;
    }

    if (state_.backend_cpu && !set_backend_n_threads(state_.backend_cpu, n_threads)) {
        error_msg_ = "CPU fallback backend does not support thread configuration";
        return false;
    }

    return true;
}

// Apply ELU activation: max(x,0) + min(alpha*(exp(x)-1), 0)
// GGML has ggml_elu which does exactly this with alpha=1.0
static struct ggml_tensor * apply_elu(struct ggml_context * ctx, struct ggml_tensor * x) {
    return ggml_elu(ctx, x);
}

struct ggml_tensor * AudioCodecEncoder::apply_rms_norm(struct ggml_context * ctx,
                                                        struct ggml_tensor * x,
                                                        struct ggml_tensor * w,
                                                        struct ggml_tensor * b,
                                                        float eps) {
    // MimiConfig uses LayerNorm (not RMSNorm) in the encoder transformer.
    // LayerNorm: (x - mean) / sqrt(var + eps) * w + b
    // GGML provides ggml_norm for LayerNorm.
    struct ggml_tensor * normed = ggml_norm(ctx, x, eps);
    normed = ggml_mul(ctx, normed, w);
    if (b) {
        normed = ggml_add(ctx, normed, b);
    }
    return normed;
}

struct ggml_tensor * AudioCodecEncoder::apply_residual_block(struct ggml_context * ctx,
                                                               struct ggml_tensor * x,
                                                               const seanet_residual_block & block) {
    // MimiResnetBlock: ELU → Conv1d(kernel=3) → ELU → Conv1d(kernel=1) + residual
    struct ggml_tensor * cur = apply_elu(ctx, x);

    // First conv: kernel=3, causal padding
    if (block.conv1_w) {
        int64_t kernel_size = block.conv1_w->ne[0];
        int pad_left = (int)(kernel_size - 1);
        if (pad_left > 0) {
            cur = ggml_pad_ext(ctx, cur, pad_left, 0, 0, 0, 0, 0, 0, 0);
        }
        cur = ggml_conv_1d(ctx, block.conv1_w, cur, 1, 0, 1);
        if (block.conv1_b) {
            int64_t oc = cur->ne[1];
            cur = ggml_add(ctx, cur, ggml_reshape_3d(ctx, block.conv1_b, 1, oc, 1));
        }
    }

    // Second conv: kernel=1, no padding needed
    cur = apply_elu(ctx, cur);
    if (block.conv2_w) {
        cur = ggml_conv_1d(ctx, block.conv2_w, cur, 1, 0, 1);
        if (block.conv2_b) {
            int64_t oc = cur->ne[1];
            cur = ggml_add(ctx, cur, ggml_reshape_3d(ctx, block.conv2_b, 1, oc, 1));
        }
    }

    // Residual connection (MimiConfig use_conv_shortcut=false, so Identity)
    return ggml_add(ctx, x, cur);
}

struct ggml_tensor * AudioCodecEncoder::apply_tfm_layer(struct ggml_context * ctx,
                                                          struct ggml_tensor * x,
                                                          const enc_tfm_layer & layer,
                                                          int32_t n_frames,
                                                          struct ggml_tensor * positions) {
    const auto & cfg = model_.config;
    const int n_heads = cfg.n_heads;
    const int hidden_dim = cfg.hidden_dim;
    const int head_dim = hidden_dim / n_heads;

    if (!layer.attn_norm_w || !layer.attn_q_w || !layer.attn_k_w || !layer.attn_v_w ||
        !layer.attn_output_w || !layer.ffn_norm_w || !layer.ffn_up_w || !layer.ffn_down_w) {
        return x;
    }

    struct ggml_tensor * residual = x;

    // Attention sub-layer
    struct ggml_tensor * normed = apply_rms_norm(ctx, x, layer.attn_norm_w, layer.attn_norm_b, cfg.rms_norm_eps);

    // Cast F16 weights to F32 before mul_mat (prevents precision bugs)
    struct ggml_tensor * attn_q_f32 = ggml_cast(ctx, layer.attn_q_w, GGML_TYPE_F32);
    struct ggml_tensor * attn_k_f32 = ggml_cast(ctx, layer.attn_k_w, GGML_TYPE_F32);
    struct ggml_tensor * attn_v_f32 = ggml_cast(ctx, layer.attn_v_w, GGML_TYPE_F32);

    struct ggml_tensor * Qcur = ggml_mul_mat(ctx, attn_q_f32, normed);
    struct ggml_tensor * Kcur = ggml_mul_mat(ctx, attn_k_f32, normed);
    struct ggml_tensor * Vcur = ggml_mul_mat(ctx, attn_v_f32, normed);

    Qcur = ggml_reshape_3d(ctx, Qcur, head_dim, n_heads, n_frames);
    Kcur = ggml_reshape_3d(ctx, Kcur, head_dim, n_heads, n_frames);
    Vcur = ggml_reshape_3d(ctx, Vcur, head_dim, n_heads, n_frames);

    // RoPE
    Qcur = ggml_rope_ext(ctx, Qcur, positions, nullptr,
                          head_dim, GGML_ROPE_TYPE_NEOX, 0,
                          cfg.rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    Kcur = ggml_rope_ext(ctx, Kcur, positions, nullptr,
                          head_dim, GGML_ROPE_TYPE_NEOX, 0,
                          cfg.rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

    struct ggml_tensor * Q = ggml_permute(ctx, Qcur, 0, 2, 1, 3);
    struct ggml_tensor * K = ggml_permute(ctx, Kcur, 0, 2, 1, 3);
    struct ggml_tensor * V = ggml_permute(ctx, Vcur, 0, 2, 1, 3);

    struct ggml_tensor * KQ = ggml_mul_mat(ctx, K, Q);
    KQ = ggml_scale(ctx, KQ, 1.0f / sqrtf((float)head_dim));
    // Sliding window causal mask: each position can only attend to previous
    // positions within the sliding_window (default 250, from MimiConfig)
    if (cfg.sliding_window > 0 && n_frames > cfg.sliding_window) {
        if (n_frames <= state_.sw_mask_size && !state_.sw_mask_data.empty()) {
            // Use precomputed mask from encoder state. Slice the top-left n_frames x n_frames submatrix.
            struct ggml_tensor * sw_mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_frames, n_frames);
            // For no_alloc contexts, we must set the tensor as an input and copy data after graph allocation.
            ggml_set_name(sw_mask, "sw_mask");
            ggml_set_input(sw_mask);
            // Store a pointer to the mask tensor so we can copy data after ggml_backend_sched_alloc_graph
            // We'll use a side-channel: the caller (encode()) will copy the mask data.
            KQ = ggml_add(ctx, KQ, sw_mask);
        } else {
            // Fallback: precomputed mask unavailable (n_frames > sw_mask_size).
            // Apply full causal mask to prevent attending to future positions.
            // This is slightly less restrictive than sliding window (allows attending
            // further back than sliding_window) but correctly prevents forward leakage.
            // For typical audio (<164s at 25Hz), the precomputed mask is always used.
            KQ = ggml_diag_mask_inf(ctx, KQ, 0);
        }
        KQ = ggml_soft_max(ctx, KQ);
    } else {
        // Full causal mask (sequence fits within window)
        KQ = ggml_diag_mask_inf(ctx, KQ, 0);
        KQ = ggml_soft_max(ctx, KQ);
    }

    V = ggml_cont(ctx, ggml_transpose(ctx, V));

    struct ggml_tensor * KQV = ggml_mul_mat(ctx, V, KQ);
    KQV = ggml_permute(ctx, KQV, 0, 2, 1, 3);
    struct ggml_tensor * attn_out = ggml_cont_2d(ctx, KQV, n_heads * head_dim, n_frames);

    attn_out = ggml_mul_mat(ctx, ggml_cast(ctx, layer.attn_output_w, GGML_TYPE_F32), attn_out);

    // Layer scale for attention
    if (layer.attn_scale) {
        attn_out = ggml_mul(ctx, attn_out, layer.attn_scale);
    }

    x = ggml_add(ctx, residual, attn_out);
    residual = x;

    // FFN sub-layer: standard 2-layer MLP with GELU activation
    // MimiConfig hidden_act="gelu" — NOT SiLU
    normed = apply_rms_norm(ctx, x, layer.ffn_norm_w, layer.ffn_norm_b, cfg.rms_norm_eps);

    struct ggml_tensor * up = ggml_mul_mat(ctx, ggml_cast(ctx, layer.ffn_up_w, GGML_TYPE_F32), normed);
    up = ggml_gelu(ctx, up);
    struct ggml_tensor * ffn_out = ggml_mul_mat(ctx, ggml_cast(ctx, layer.ffn_down_w, GGML_TYPE_F32), up);

    // Layer scale for FFN
    if (layer.ffn_scale) {
        ffn_out = ggml_mul(ctx, ffn_out, layer.ffn_scale);
    }

    return ggml_add(ctx, residual, ffn_out);
}

struct ggml_cgraph * AudioCodecEncoder::build_graph(struct ggml_context * ctx0, int32_t n_samples) {
    const auto & cfg = model_.config;

    if (!ctx0) {
        error_msg_ = "Encoder graph context is not initialized";
        return nullptr;
    }

    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, QWEN3_TTS_ENC_MAX_NODES, false);

    // Input: audio waveform [1, T] at 24kHz
    // GGML conv1d expects [T, C, B] layout
    struct ggml_tensor * audio = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, n_samples, 1, 1);
    ggml_set_name(audio, "audio");
    ggml_set_input(audio);

    struct ggml_tensor * cur = audio;

    // === SEANet Encoder ===

    // Initial conv: Conv1d(1, 64, kernel=7, stride=1) with causal padding
    // tok_enc.conv.0
    // Note: No ELU after initial conv in MimiEncoder
    {
        int kernel_size = (model_.init_conv_w) ? (int)model_.init_conv_w->ne[0] : 7;
        int pad_left = kernel_size - 1;
        if (pad_left > 0) {
            cur = ggml_pad_ext(ctx0, cur, pad_left, 0, 0, 0, 0, 0, 0, 0);
        }
        cur = ggml_conv_1d(ctx0, model_.init_conv_w, cur, 1, 0, 1);
        if (model_.init_conv_b) {
            int64_t oc = cur->ne[1];
            cur = ggml_add(ctx0, cur, ggml_reshape_3d(ctx0, model_.init_conv_b, 1, oc, 1));
        }
        ggml_set_name(cur, "init_conv_out");
    }

    // SEANet downsample layers: 4 stages
    // Each: residual block -> ELU -> downsample conv (with stride)
    // Matching MimiEncoder architecture:
    //   reversed(upsampling_ratios) = [4, 5, 6, 8]
    //   kernel_size = ratio * 2
    //   channels double after each downsample
    for (int i = 0; i < cfg.n_downsample_layers; ++i) {
        const auto & layer = model_.seanet_layers[i];

        // Residual block (1 per layer, matching MimiConfig num_residual_layers=1)
        if (layer.res_block.conv1_w) {
            cur = apply_residual_block(ctx0, cur, layer.res_block);
        }

        // ELU before downsample conv
        cur = apply_elu(ctx0, cur);

        // Downsample conv with stride
        if (layer.conv_w) {
            int kernel_size = (int)layer.conv_w->ne[0];
            int stride = cfg.seanet_strides[i];
            // Causal padding: pad left with (kernel_size - stride)
            int pad_left = kernel_size - stride;
            if (pad_left > 0) {
                cur = ggml_pad_ext(ctx0, cur, pad_left, 0, 0, 0, 0, 0, 0, 0);
            }
            cur = ggml_conv_1d(ctx0, layer.conv_w, cur, stride, 0, 1);
            if (layer.conv_b) {
                int64_t oc = cur->ne[1];
                cur = ggml_add(ctx0, cur, ggml_reshape_3d(ctx0, layer.conv_b, 1, oc, 1));
            }
        }

        char name[32];
        snprintf(name, sizeof(name), "seanet_layer%d_out", i);
        ggml_set_name(cur, name);
    }

    // Final conv: Conv1d(1024 -> 512, kernel=3, stride=1) with causal padding
    // tok_enc.conv.14
    // Note: ELU before final conv in MimiEncoder
    if (model_.final_conv_w) {
        cur = apply_elu(ctx0, cur);
        int kernel_size = (int)model_.final_conv_w->ne[0];
        int pad_left = kernel_size - 1;
        if (pad_left > 0) {
            cur = ggml_pad_ext(ctx0, cur, pad_left, 0, 0, 0, 0, 0, 0, 0);
        }
        cur = ggml_conv_1d(ctx0, model_.final_conv_w, cur, 1, 0, 1);
        if (model_.final_conv_b) {
            int64_t oc = cur->ne[1];
            cur = ggml_add(ctx0, cur, ggml_reshape_3d(ctx0, model_.final_conv_b, 1, oc, 1));
        }
        ggml_set_name(cur, "final_conv_out");
    }

    // After SEANet: cur is [T/960, 512, 1] at 25 Hz (4*5*6*8=960x downsampling)
    int32_t n_frames_25hz = (int32_t)cur->ne[0];

    // === Encoder Transformer ===
    // Reshape for transformer: [n_frames, hidden_dim] -> [hidden_dim, n_frames] (transpose)
    // cur is [T, C, B] = [n_frames, 512, 1]
    // For transformer, we need [n_frames, hidden_dim] as 2D
    cur = ggml_reshape_2d(ctx0, cur, n_frames_25hz, cfg.hidden_dim);
    // Transpose to [hidden_dim, n_frames] for mul_mat convention
    cur = ggml_cont(ctx0, ggml_transpose(ctx0, cur));
    ggml_set_name(cur, "tfm_input");

    // Position indices for RoPE
    struct ggml_tensor * positions = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_frames_25hz);
    ggml_set_name(positions, "positions");
    ggml_set_input(positions);

    // Apply transformer layers
    for (int i = 0; i < cfg.n_tfm_layers; ++i) {
        cur = apply_tfm_layer(ctx0, cur, model_.tfm_layers[i], n_frames_25hz, positions);
        char name[32];
        snprintf(name, sizeof(name), "tfm_layer%d_out", i);
        ggml_set_name(cur, name);
    }

    ggml_set_name(cur, "tfm_output");

    // === Downsample Conv (25Hz -> 12.5Hz) ===
    // tok_enc.downsample.weight: Conv1d stride=2
    if (model_.downsample_w) {
        // cur is [hidden_dim, n_frames_25hz] -> reshape to [n_frames_25hz, hidden_dim, 1] for conv1d
        cur = ggml_cont(ctx0, ggml_transpose(ctx0, cur));
        cur = ggml_reshape_3d(ctx0, cur, n_frames_25hz, cfg.hidden_dim, 1);

        int kernel_size = (int)model_.downsample_w->ne[0];
        int stride = 2;
        // Causal padding: pad left with (kernel_size - stride)
        int pad_left = kernel_size - stride;
        if (pad_left > 0) {
            cur = ggml_pad_ext(ctx0, cur, pad_left, 0, 0, 0, 0, 0, 0, 0);
        }
        cur = ggml_conv_1d(ctx0, model_.downsample_w, cur, stride, 0, 1);
        // No bias for downsample conv

        ggml_set_name(cur, "downsample_out");
    }

    // Output: embeddings tensor [n_frames_12.5hz, hidden_dim, 1]
    // Transpose to [hidden_dim, n_frames_12.5hz] for easier extraction
    int32_t n_frames_down = (int32_t)cur->ne[0];
    cur = ggml_cont(ctx0, ggml_transpose(ctx0, ggml_reshape_2d(ctx0, cur, n_frames_down, cfg.hidden_dim)));
    ggml_set_name(cur, "embeddings");
    ggml_set_output(cur);

    ggml_build_forward_expand(gf, cur);

    return gf;
}

bool AudioCodecEncoder::quantize_vq_cpu(const float * embeddings, int32_t n_frames,
                                         std::vector<int32_t> & codes) {
    const auto & cfg = model_.config;
    const int n_codebooks = cfg.n_codebooks;  // 16
    const int codebook_dim = cfg.codebook_dim;  // 256
    const int hidden_dim = cfg.hidden_dim;  // 512

    codes.resize((size_t)n_frames * n_codebooks);

    // Semantic VQ: project from hidden_dim -> codebook_dim, quantize with 1 codebook
    // Acoustic VQ: project from hidden_dim -> codebook_dim, residual quantize with 15 codebooks

    // Get codebook data (F16 on backend, need to read to host)
    // Note: Do NOT check t->data — backend-allocated tensors have data==NULL
    // but ggml_backend_tensor_get works correctly via the backend buffer.
    auto read_tensor_to_f32 = [](struct ggml_tensor * t, std::vector<float> & out) -> bool {
        if (!t) return false;
        int64_t n = ggml_nelements(t);
        out.resize(n);
        if (t->type == GGML_TYPE_F32) {
            ggml_backend_tensor_get(t, out.data(), 0, n * sizeof(float));
        } else if (t->type == GGML_TYPE_F16) {
            std::vector<ggml_fp16_t> f16_buf(n);
            ggml_backend_tensor_get(t, f16_buf.data(), 0, n * sizeof(ggml_fp16_t));
            for (int64_t i = 0; i < n; ++i) {
                out[i] = ggml_fp16_to_fp32(f16_buf[i]);
            }
        } else {
            return false;
        }
        return true;
    };

    // Read projection weights
    // input_proj: [codebook_dim, hidden_dim] - projects hidden -> codebook_dim
    // output_proj: [hidden_dim, codebook_dim] - projects codebook_dim -> hidden (for residual)
    std::vector<float> sem_input_proj, sem_output_proj;
    std::vector<float> acou_input_proj, acou_output_proj;

    if (!read_tensor_to_f32(model_.vq_semantic_input_proj, sem_input_proj)) {
        error_msg_ = "Failed to read semantic input projection weights";
        return false;
    }
    if (!read_tensor_to_f32(model_.vq_semantic_output_proj, sem_output_proj)) {
        error_msg_ = "Failed to read semantic output projection weights";
        return false;
    }
    if (!read_tensor_to_f32(model_.vq_acoustic_input_proj, acou_input_proj)) {
        error_msg_ = "Failed to read acoustic input projection weights";
        return false;
    }
    if (!read_tensor_to_f32(model_.vq_acoustic_output_proj, acou_output_proj)) {
        error_msg_ = "Failed to read acoustic output projection weights";
        return false;
    }

    // Read codebook data
    // Semantic: 1 codebook (index 0)
    // Acoustic: 15 codebooks (indices 0..14)
    std::vector<float> sem_cb_data;
    if (!read_tensor_to_f32(model_.vq_semantic_codebook[0], sem_cb_data)) {
        error_msg_ = "Failed to read semantic codebook 0";
        return false;
    }
    // sem_cb_data: [codebook_dim, codebook_size] = [256, 2048]

    std::vector<float> acou_cb_data[15];
    for (int cb = 0; cb < 15; ++cb) {
        if (!read_tensor_to_f32(model_.vq_acoustic_codebook[cb], acou_cb_data[cb])) {
            error_msg_ = "Failed to read acoustic codebook " + std::to_string(cb);
            return false;
        }
    }

    // Matrix-vector multiply helper: out = W * in, where W is [out_dim, in_dim] row-major
    // W[row * in_dim + col], in[col], out[row]
    auto mat_vec_mul = [](const float * W, const float * in, int out_dim, int in_dim, float * out) {
        for (int r = 0; r < out_dim; ++r) {
            float sum = 0.0f;
            const float * row = W + r * in_dim;
            for (int c = 0; c < in_dim; ++c) {
                sum += row[c] * in[c];
            }
            out[r] = sum;
        }
    };

    // Find nearest codebook entry: argmin ||z - cb[i]||
    // cb_data: [codebook_dim, codebook_size] row-major
    // z: [codebook_dim]
    auto find_nearest = [](const float * cb_data, const float * z,
                            int codebook_dim, int codebook_size) -> int32_t {
        int32_t best_idx = 0;
        float best_dist = std::numeric_limits<float>::max();
        for (int i = 0; i < codebook_size; ++i) {
            const float * entry = cb_data + i * codebook_dim;
            float dist = 0.0f;
            for (int d = 0; d < codebook_dim; ++d) {
                float diff = z[d] - entry[d];
                dist += diff * diff;
            }
            if (dist < best_dist) {
                best_dist = dist;
                best_idx = i;
            }
        }
        return best_idx;
    };

    // Process each frame
    std::vector<float> z_sem(codebook_dim);     // projected semantic embedding
    std::vector<float> z_acou(codebook_dim);    // projected acoustic embedding
    std::vector<float> residual(hidden_dim);     // residual in hidden_dim space
    std::vector<float> quantized(codebook_dim); // quantized vector in codebook_dim

    for (int32_t f = 0; f < n_frames; ++f) {
        const float * frame = embeddings + f * hidden_dim;

        // --- Semantic VQ (codebook 0) ---
        mat_vec_mul(sem_input_proj.data(), frame, codebook_dim, hidden_dim, z_sem.data());
        int32_t sem_code = find_nearest(sem_cb_data.data(), z_sem.data(), codebook_dim, cfg.codebook_size);
        codes[f * n_codebooks + 0] = sem_code;

        // Get quantized vector for residual computation
        for (int d = 0; d < codebook_dim; ++d) {
            quantized[d] = sem_cb_data[sem_code * codebook_dim + d];
        }

        // Compute residual: residual = frame - output_proj * quantized
        // output_proj: [hidden_dim, codebook_dim]
        std::vector<float> recon(hidden_dim);
        mat_vec_mul(sem_output_proj.data(), quantized.data(), hidden_dim, codebook_dim, recon.data());
        for (int d = 0; d < hidden_dim; ++d) {
            residual[d] = frame[d] - recon[d];
        }

        // --- Acoustic VQ (codebooks 1..15) ---
        // Residual quantization: project residual, find nearest, subtract reconstruction
        float * current_residual = residual.data();
        for (int cb = 0; cb < 15; ++cb) {
            // Project residual to codebook_dim
            mat_vec_mul(acou_input_proj.data(), current_residual, codebook_dim, hidden_dim, z_acou.data());

            // Find nearest codebook entry
            int32_t acou_code = find_nearest(acou_cb_data[cb].data(), z_acou.data(), codebook_dim, cfg.codebook_size);
            codes[f * n_codebooks + 1 + cb] = acou_code;

            // Get quantized vector
            for (int d = 0; d < codebook_dim; ++d) {
                quantized[d] = acou_cb_data[cb][acou_code * codebook_dim + d];
            }

            // Update residual: residual = residual - output_proj * quantized
            std::vector<float> recon_cb(hidden_dim);
            mat_vec_mul(acou_output_proj.data(), quantized.data(), hidden_dim, codebook_dim, recon_cb.data());
            for (int d = 0; d < hidden_dim; ++d) {
                current_residual[d] = current_residual[d] - recon_cb[d];
            }
        }
    }

    return true;
}

bool AudioCodecEncoder::encode(const float * samples, int32_t n_samples,
                                std::vector<int32_t> & codes, int32_t & n_frames) {
    if (!model_.ctx) {
        error_msg_ = "Model not loaded";
        return false;
    }

    if (!samples || n_samples <= 0) {
        error_msg_ = "Invalid input: null samples or non-positive sample count";
        return false;
    }

    const auto & cfg = model_.config;
    const bool enc_dbg = encoder_debug_enabled();

    // Ensure minimum audio length for the encoder pipeline
    // Total downsampling factor: 960 (SEANet) * 2 (post-transformer) = 1920
    // Need at least 1920 samples for 1 frame
    const int min_samples = 1920;
    if (n_samples < min_samples) {
        error_msg_ = "Audio too short: need at least " + std::to_string(min_samples) +
                     " samples, got " + std::to_string(n_samples);
        return false;
    }

    // Build graph
    struct ggml_init_params params = {
        /*.mem_size   =*/ state_.compute_meta.size(),
        /*.mem_buffer =*/ state_.compute_meta.data(),
        /*.no_alloc   =*/ true,
    };

    struct ggml_context * ctx0 = ggml_init(params);
    if (!ctx0) {
        error_msg_ = "Failed to initialize graph context";
        return false;
    }

    struct ggml_cgraph * gf = build_graph(ctx0, n_samples);
    if (!gf) {
        ggml_free(ctx0);
        return false;
    }

    // Allocate graph
    if (!ggml_backend_sched_alloc_graph(state_.sched, gf)) {
        error_msg_ = "Failed to allocate encoder graph";
        ggml_free(ctx0);
        return false;
    }

    // Set input: audio waveform
    struct ggml_tensor * audio_tensor = ggml_graph_get_tensor(gf, "audio");
    if (!audio_tensor) {
        error_msg_ = "Failed to find audio input tensor";
        ggml_backend_sched_reset(state_.sched);
        ggml_free(ctx0);
        return false;
    }
    ggml_backend_tensor_set(audio_tensor, samples, 0, n_samples * sizeof(float));

    // Set positions
    struct ggml_tensor * positions_tensor = ggml_graph_get_tensor(gf, "positions");
    if (positions_tensor) {
        int32_t n_pos = (int32_t)positions_tensor->ne[0];
        std::vector<int32_t> pos_buf(n_pos);
        for (int32_t i = 0; i < n_pos; ++i) {
            pos_buf[i] = i;
        }
        ggml_backend_tensor_set(positions_tensor, pos_buf.data(), 0, n_pos * sizeof(int32_t));
    }

    // Set sliding window mask if present
    struct ggml_tensor * sw_mask_tensor = ggml_graph_get_tensor(gf, "sw_mask");
    if (sw_mask_tensor && !state_.sw_mask_data.empty()) {
        int32_t mask_n = (int32_t)sw_mask_tensor->ne[0];  // n_frames x n_frames
        if (mask_n <= state_.sw_mask_size) {
            // Copy the top-left mask_n x mask_n submatrix from precomputed buffer
            size_t row_bytes = mask_n * sizeof(float);
            for (int r = 0; r < mask_n; ++r) {
                const float * src_row = state_.sw_mask_data.data() + r * state_.sw_mask_size;
                ggml_backend_tensor_set(sw_mask_tensor, src_row,
                                        r * mask_n * sizeof(float), row_bytes);
            }
        }
    }

    // Compute
    if (ggml_backend_sched_graph_compute(state_.sched, gf) != GGML_STATUS_SUCCESS) {
        error_msg_ = "Failed to compute encoder graph";
        ggml_backend_sched_reset(state_.sched);
        ggml_free(ctx0);
        return false;
    }

    // Extract embeddings
    struct ggml_tensor * emb_tensor = ggml_graph_get_tensor(gf, "embeddings");
    if (!emb_tensor) {
        error_msg_ = "Failed to find embeddings output tensor";
        ggml_backend_sched_reset(state_.sched);
        ggml_free(ctx0);
        return false;
    }

    // embeddings shape: [hidden_dim, n_frames_down]
    int32_t n_frames_down = (int32_t)emb_tensor->ne[1];
    int32_t emb_dim = (int32_t)emb_tensor->ne[0];
    n_frames = n_frames_down;

    if (enc_dbg) {
        fprintf(stderr, "[encoder-dbg] embeddings shape: [%d, %d], n_samples=%d\n",
                emb_dim, n_frames_down, n_samples);
    }

    // Read embeddings to host
    std::vector<float> embeddings((size_t)emb_dim * n_frames_down);
    ggml_backend_tensor_get(emb_tensor, embeddings.data(), 0, embeddings.size() * sizeof(float));

    // Reset scheduler
    ggml_backend_sched_reset(state_.sched);
    ggml_free(ctx0);

    // VQ quantization on CPU
    if (!quantize_vq_cpu(embeddings.data(), n_frames_down, codes)) {
        return false;
    }

    n_frames = n_frames_down;

    if (enc_dbg) {
        fprintf(stderr, "[encoder-dbg] encode done: n_frames=%d, n_codebooks=%d, total_codes=%lld\n",
                n_frames_down, cfg.n_codebooks, (long long)codes.size());
    }

    return true;
}

void free_codec_encoder_model(codec_encoder_model & model) {
    if (model.buffer) {
        ggml_backend_buffer_free(model.buffer);
        model.buffer = nullptr;
    }
    if (model.ctx) {
        ggml_free(model.ctx);
        model.ctx = nullptr;
    }
    model.tensors.clear();
}

} // namespace qwen3_tts
