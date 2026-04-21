#include "audio_tokenizer_decoder.h"
#include "gguf_loader.h"
#include "ggml-cpu.h"

#include <chrono>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <numeric>
#include <cstdlib>
#include <limits>

#define QWEN3_TTS_DEC_MAX_NODES 32768

namespace qwen3_tts {

static bool decoder_debug_enabled() {
    const char * env = std::getenv("QWEN3_TTS_DECODER_DEBUG");
    if (!env || env[0] == '\0') {
        return false;
    }
    std::string v = env;
    std::transform(v.begin(), v.end(), v.begin(), [](unsigned char c) { return (char)std::tolower(c); });
    return !(v == "0" || v == "false" || v == "off" || v == "no");
}

AudioTokenizerDecoder::AudioTokenizerDecoder() = default;

AudioTokenizerDecoder::~AudioTokenizerDecoder() {
    unload_model();
}

void AudioTokenizerDecoder::unload_model() {
    reset_decode_graph_cache();

    free_audio_decoder_model(model_);
    
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
}

void AudioTokenizerDecoder::normalize_codebooks() {
    const float epsilon = 1e-5f;
    
    auto normalize_codebook = [epsilon](struct ggml_tensor * codebook, struct ggml_tensor * usage, const char *) {
        if (!codebook || !usage || !codebook->data || !usage->data) return;
        
        int64_t codebook_dim = codebook->ne[0];
        int64_t codebook_size = codebook->ne[1];
        
        ggml_fp16_t * cb_data = (ggml_fp16_t *)codebook->data;
        float * usage_data = (float *)usage->data;
        
        for (int64_t emb_idx = 0; emb_idx < codebook_size; ++emb_idx) {
            float u = usage_data[emb_idx];
            if (u < epsilon) u = epsilon;
            float inv_u = 1.0f / u;
            
            for (int64_t dim_idx = 0; dim_idx < codebook_dim; ++dim_idx) {
                int64_t mem_idx = dim_idx + emb_idx * codebook_dim;
                float val = ggml_fp16_to_fp32(cb_data[mem_idx]);
                cb_data[mem_idx] = ggml_fp32_to_fp16(val * inv_u);
            }
        }
        
    };
    
    normalize_codebook(model_.vq_first_codebook, model_.vq_first_usage, "first");
    
    for (int i = 0; i < 15; ++i) {
        char name[16];
        snprintf(name, sizeof(name), "rest%d", i);
        normalize_codebook(model_.vq_rest_codebook[i], model_.vq_rest_usage[i], name);
    }
}

bool AudioTokenizerDecoder::load_model(const std::string & model_path) {
    unload_model();

    GGUFLoader loader;
    if (!loader.open(model_path)) {
        error_msg_ = loader.get_error();
        return false;
    }
    
    model_.config.sample_rate = loader.get_u32("qwen3-tts.tokenizer.sample_rate", 24000);
    model_.config.n_codebooks = loader.get_u32("qwen3-tts.tokenizer.num_codebooks", 16);
    model_.config.codebook_size = loader.get_u32("qwen3-tts.tokenizer.codebook_size", 2048);
    
    int64_t n_tensors = loader.get_n_tensors();
    int dec_tensor_count = 0;
    for (int64_t i = 0; i < n_tensors; ++i) {
        const char * name = loader.get_tensor_name(i);
        if (name && strncmp(name, "tok_dec.", 8) == 0) {
            dec_tensor_count++;
        }
    }
    
    if (dec_tensor_count == 0) {
        error_msg_ = "No decoder tensors found in model";
        return false;
    }
    
    size_t ctx_size = ggml_tensor_overhead() * dec_tensor_count;
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
    
    for (int64_t i = 0; i < n_tensors; ++i) {
        const char * name = loader.get_tensor_name(i);
        if (!name || strncmp(name, "tok_dec.", 8) != 0) {
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
        
        if (sname == "tok_dec.vq_first.input_proj.weight") model_.vq_first_input_proj = tensor;
        else if (sname == "tok_dec.vq_first.output_proj.weight") model_.vq_first_output_proj = tensor;
        else if (sname == "tok_dec.vq_first.0.codebook") model_.vq_first_codebook = tensor;
        else if (sname == "tok_dec.vq_first.0.usage") model_.vq_first_usage = tensor;
        else if (sname == "tok_dec.vq_rest.input_proj.weight") model_.vq_rest_input_proj = tensor;
        else if (sname == "tok_dec.vq_rest.output_proj.weight") model_.vq_rest_output_proj = tensor;
        else if (sname == "tok_dec.pre_conv.weight") model_.pre_conv_w = tensor;
        else if (sname == "tok_dec.pre_conv.bias") model_.pre_conv_b = tensor;
        else if (sname == "tok_dec.pre_tfm.input_proj.weight") model_.pre_tfm_input_proj_w = tensor;
        else if (sname == "tok_dec.pre_tfm.input_proj.bias") model_.pre_tfm_input_proj_b = tensor;
        else if (sname == "tok_dec.pre_tfm.norm.weight") model_.pre_tfm_norm_w = tensor;
        else if (sname == "tok_dec.pre_tfm.output_proj.weight") model_.pre_tfm_output_proj_w = tensor;
        else if (sname == "tok_dec.pre_tfm.output_proj.bias") model_.pre_tfm_output_proj_b = tensor;
        else if (sname == "tok_dec.dec.0.conv.weight") model_.dec0_conv_w = tensor;
        else if (sname == "tok_dec.dec.0.conv.bias") model_.dec0_conv_b = tensor;
        else if (sname == "tok_dec.dec.5.snake.alpha") model_.dec5_snake_alpha = tensor;
        else if (sname == "tok_dec.dec.5.snake.beta") model_.dec5_snake_beta = tensor;
        else if (sname == "tok_dec.dec.6.conv.weight") model_.dec6_conv_w = tensor;
        else if (sname == "tok_dec.dec.6.conv.bias") model_.dec6_conv_b = tensor;
        else if (sname.find("pre_tfm.blk.") != std::string::npos) {
            int blk_idx;
            if (sscanf(name, "tok_dec.pre_tfm.blk.%d.", &blk_idx) == 1 && blk_idx >= 0 && blk_idx < 8) {
                if (sname.find(".attn_v.weight") != std::string::npos) model_.pre_tfm_layers[blk_idx].attn_v_w = tensor;
                else if (sname.find(".ffn_gate.weight") != std::string::npos) model_.pre_tfm_layers[blk_idx].ffn_gate_w = tensor;
                else if (sname.find(".attn_norm.weight") != std::string::npos) model_.pre_tfm_layers[blk_idx].attn_norm_w = tensor;
                else if (sname.find(".attn_q.weight") != std::string::npos) model_.pre_tfm_layers[blk_idx].attn_q_w = tensor;
                else if (sname.find(".attn_k.weight") != std::string::npos) model_.pre_tfm_layers[blk_idx].attn_k_w = tensor;
                else if (sname.find(".attn_output.weight") != std::string::npos) model_.pre_tfm_layers[blk_idx].attn_output_w = tensor;
                else if (sname.find(".attn_scale") != std::string::npos) model_.pre_tfm_layers[blk_idx].attn_scale = tensor;
                else if (sname.find(".ffn_norm.weight") != std::string::npos) model_.pre_tfm_layers[blk_idx].ffn_norm_w = tensor;
                else if (sname.find(".ffn_up.weight") != std::string::npos) model_.pre_tfm_layers[blk_idx].ffn_up_w = tensor;
                else if (sname.find(".ffn_down.weight") != std::string::npos) model_.pre_tfm_layers[blk_idx].ffn_down_w = tensor;
                else if (sname.find(".ffn_scale") != std::string::npos) model_.pre_tfm_layers[blk_idx].ffn_scale = tensor;
            }
        }
        else {
            int blk_idx, res_idx, cb_idx, n = 0;
            char suffix[64];
            size_t name_len = strlen(name);
            

            
            #define MATCH1(fmt, var) (sscanf(name, fmt "%n", &var, &n) == 1 && (size_t)n == name_len)
            #define MATCH2(fmt, v1, v2) (sscanf(name, fmt "%n", &v1, &v2, &n) == 2 && (size_t)n == name_len)
            #define MATCH1S(fmt, var, suf) (sscanf(name, fmt, &var, suf) == 2)
            
            if (MATCH1("tok_dec.vq_rest.%d.codebook", cb_idx)) {
                if (cb_idx >= 0 && cb_idx < 15) {
                    model_.vq_rest_codebook[cb_idx] = tensor;
                }
            }
            else if (MATCH1("tok_dec.vq_rest.%d.usage", cb_idx)) {
                if (cb_idx >= 0 && cb_idx < 15) {
                    model_.vq_rest_usage[cb_idx] = tensor;
                }
            }
            else if (MATCH1S("tok_dec.upsample.%d.conv.%63s", blk_idx, suffix)) {
                if (blk_idx >= 0 && blk_idx < 2) {
                    if (strcmp(suffix, "weight") == 0) model_.upsample[blk_idx].conv_w = tensor;
                    else if (strcmp(suffix, "bias") == 0) model_.upsample[blk_idx].conv_b = tensor;
                }
            }
            else if (MATCH1S("tok_dec.upsample.%d.dwconv.%63s", blk_idx, suffix)) {
                if (blk_idx >= 0 && blk_idx < 2) {
                    if (strcmp(suffix, "weight") == 0) model_.upsample[blk_idx].dwconv_w = tensor;
                    else if (strcmp(suffix, "bias") == 0) model_.upsample[blk_idx].dwconv_b = tensor;
                }
            }
            else if (MATCH1S("tok_dec.upsample.%d.norm.%63s", blk_idx, suffix)) {
                if (blk_idx >= 0 && blk_idx < 2) {
                    if (strcmp(suffix, "weight") == 0) model_.upsample[blk_idx].norm_w = tensor;
                    else if (strcmp(suffix, "bias") == 0) model_.upsample[blk_idx].norm_b = tensor;
                }
            }
            else if (MATCH1S("tok_dec.upsample.%d.pwconv1.%63s", blk_idx, suffix)) {
                if (blk_idx >= 0 && blk_idx < 2) {
                    if (strcmp(suffix, "weight") == 0) model_.upsample[blk_idx].pwconv1_w = tensor;
                    else if (strcmp(suffix, "bias") == 0) model_.upsample[blk_idx].pwconv1_b = tensor;
                }
            }
            else if (MATCH1S("tok_dec.upsample.%d.pwconv2.%63s", blk_idx, suffix)) {
                if (blk_idx >= 0 && blk_idx < 2) {
                    if (strcmp(suffix, "weight") == 0) model_.upsample[blk_idx].pwconv2_w = tensor;
                    else if (strcmp(suffix, "bias") == 0) model_.upsample[blk_idx].pwconv2_b = tensor;
                }
            }
            else if (MATCH1("tok_dec.upsample.%d.gamma", blk_idx)) {
                if (blk_idx >= 0 && blk_idx < 2) model_.upsample[blk_idx].gamma = tensor;
            }
            else if (MATCH1("tok_dec.pre_tfm.blk.%d.attn_norm.weight", blk_idx)) {
                if (blk_idx >= 0 && blk_idx < 8) model_.pre_tfm_layers[blk_idx].attn_norm_w = tensor;
            }
            else if (MATCH1("tok_dec.pre_tfm.blk.%d.attn_q.weight", blk_idx)) {
                if (blk_idx >= 0 && blk_idx < 8) model_.pre_tfm_layers[blk_idx].attn_q_w = tensor;
            }
            else if (MATCH1("tok_dec.pre_tfm.blk.%d.attn_k.weight", blk_idx)) {
                if (blk_idx >= 0 && blk_idx < 8) model_.pre_tfm_layers[blk_idx].attn_k_w = tensor;
            }
            else if (MATCH1("tok_dec.pre_tfm.blk.%d.attn_v.weight", blk_idx)) {
                if (blk_idx >= 0 && blk_idx < 8) model_.pre_tfm_layers[blk_idx].attn_v_w = tensor;
            }
            else if (MATCH1("tok_dec.pre_tfm.blk.%d.attn_output.weight", blk_idx)) {
                if (blk_idx >= 0 && blk_idx < 8) model_.pre_tfm_layers[blk_idx].attn_output_w = tensor;
            }
            else if (MATCH1("tok_dec.pre_tfm.blk.%d.attn_scale", blk_idx)) {
                if (blk_idx >= 0 && blk_idx < 8) model_.pre_tfm_layers[blk_idx].attn_scale = tensor;
            }
            else if (MATCH1("tok_dec.pre_tfm.blk.%d.ffn_norm.weight", blk_idx)) {
                if (blk_idx >= 0 && blk_idx < 8) model_.pre_tfm_layers[blk_idx].ffn_norm_w = tensor;
            }
            else if (MATCH1("tok_dec.pre_tfm.blk.%d.ffn_gate.weight", blk_idx)) {
                if (blk_idx >= 0 && blk_idx < 8) model_.pre_tfm_layers[blk_idx].ffn_gate_w = tensor;
            }
            else if (MATCH1("tok_dec.pre_tfm.blk.%d.ffn_up.weight", blk_idx)) {
                if (blk_idx >= 0 && blk_idx < 8) model_.pre_tfm_layers[blk_idx].ffn_up_w = tensor;
            }
            else if (MATCH1("tok_dec.pre_tfm.blk.%d.ffn_down.weight", blk_idx)) {
                if (blk_idx >= 0 && blk_idx < 8) model_.pre_tfm_layers[blk_idx].ffn_down_w = tensor;
            }
            else if (MATCH1("tok_dec.pre_tfm.blk.%d.ffn_scale", blk_idx)) {
                if (blk_idx >= 0 && blk_idx < 8) model_.pre_tfm_layers[blk_idx].ffn_scale = tensor;
            }
            else if (MATCH1("tok_dec.dec.%d.snake.alpha", blk_idx)) {
                if (blk_idx >= 1 && blk_idx <= 4) model_.dec_blocks[blk_idx-1].snake_alpha = tensor;
            }
            else if (MATCH1("tok_dec.dec.%d.snake.beta", blk_idx)) {
                if (blk_idx >= 1 && blk_idx <= 4) model_.dec_blocks[blk_idx-1].snake_beta = tensor;
            }
            else if (MATCH1("tok_dec.dec.%d.conv_t.weight", blk_idx)) {
                if (blk_idx >= 1 && blk_idx <= 4) model_.dec_blocks[blk_idx-1].conv_t_w = tensor;
            }
            else if (MATCH1("tok_dec.dec.%d.conv_t.bias", blk_idx)) {
                if (blk_idx >= 1 && blk_idx <= 4) model_.dec_blocks[blk_idx-1].conv_t_b = tensor;
            }
            else if (MATCH2("tok_dec.dec.%d.res.%d.act1.alpha", blk_idx, res_idx)) {
                if (blk_idx >= 1 && blk_idx <= 4 && res_idx >= 2 && res_idx <= 4) {
                    model_.dec_blocks[blk_idx-1].res[res_idx-2].act1_alpha = tensor;
                }
            }
            else if (MATCH2("tok_dec.dec.%d.res.%d.act1.beta", blk_idx, res_idx)) {
                if (blk_idx >= 1 && blk_idx <= 4 && res_idx >= 2 && res_idx <= 4) {
                    model_.dec_blocks[blk_idx-1].res[res_idx-2].act1_beta = tensor;
                }
            }
            else if (MATCH2("tok_dec.dec.%d.res.%d.conv1.weight", blk_idx, res_idx)) {
                if (blk_idx >= 1 && blk_idx <= 4 && res_idx >= 2 && res_idx <= 4) {
                    model_.dec_blocks[blk_idx-1].res[res_idx-2].conv1_w = tensor;
                }
            }
            else if (MATCH2("tok_dec.dec.%d.res.%d.conv1.bias", blk_idx, res_idx)) {
                if (blk_idx >= 1 && blk_idx <= 4 && res_idx >= 2 && res_idx <= 4) {
                    model_.dec_blocks[blk_idx-1].res[res_idx-2].conv1_b = tensor;
                }
            }
            else if (MATCH2("tok_dec.dec.%d.res.%d.act2.alpha", blk_idx, res_idx)) {
                if (blk_idx >= 1 && blk_idx <= 4 && res_idx >= 2 && res_idx <= 4) {
                    model_.dec_blocks[blk_idx-1].res[res_idx-2].act2_alpha = tensor;
                }
            }
            else if (MATCH2("tok_dec.dec.%d.res.%d.act2.beta", blk_idx, res_idx)) {
                if (blk_idx >= 1 && blk_idx <= 4 && res_idx >= 2 && res_idx <= 4) {
                    model_.dec_blocks[blk_idx-1].res[res_idx-2].act2_beta = tensor;
                }
            }
            else if (MATCH2("tok_dec.dec.%d.res.%d.conv2.weight", blk_idx, res_idx)) {
                if (blk_idx >= 1 && blk_idx <= 4 && res_idx >= 2 && res_idx <= 4) {
                    model_.dec_blocks[blk_idx-1].res[res_idx-2].conv2_w = tensor;
                }
            }
            else if (MATCH2("tok_dec.dec.%d.res.%d.conv2.bias", blk_idx, res_idx)) {
                if (blk_idx >= 1 && blk_idx <= 4 && res_idx >= 2 && res_idx <= 4) {
                    model_.dec_blocks[blk_idx-1].res[res_idx-2].conv2_b = tensor;
                }
            }
            #undef MATCH1
            #undef MATCH2
            #undef MATCH1S
        }
    }
    
    if (!load_tensor_data_from_file(model_path, gguf_ctx, model_.ctx,
                                     model_.tensors, model_.buffer, error_msg_,
                                     GGML_BACKEND_DEVICE_TYPE_IGPU)) {
        return false;
    }
    
    for (int i = 0; i < 4; ++i) {
        model_.dec_blocks[i].res[0].dilation = 1;
        model_.dec_blocks[i].res[1].dilation = 3;
        model_.dec_blocks[i].res[2].dilation = 9;
    }
    
    normalize_codebooks();
    // Codebooks are normalized in host memory; sync once to backend tensors.
    auto upload_if_present = [](struct ggml_tensor * t) {
        if (t && t->data) {
            ggml_backend_tensor_set(t, t->data, 0, ggml_nbytes(t));
        }
    };
    upload_if_present(model_.vq_first_codebook);
    for (int i = 0; i < 15; ++i) {
        upload_if_present(model_.vq_rest_codebook[i]);
    }
    
    auto init_decoder_backend = [&]() -> bool {
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

        // Use a dedicated runtime backend for decoder to avoid cross-thread
        // contention with talker/code-predictor during streaming.
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
            error_msg_ = "Failed to initialize backend (IGPU/GPU/ACCEL/CPU) for AudioTokenizerDecoder";
            return false;
        }

        ggml_backend_dev_t device = ggml_backend_get_device(state_.backend);
        const char * device_name = device ? ggml_backend_dev_name(device) : "Unknown";
        fprintf(stderr, "  AudioTokenizerDecoder backend: %s\n", device_name);

        if (device && ggml_backend_dev_type(device) != GGML_BACKEND_DEVICE_TYPE_CPU) {
            state_.backend_cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
            if (!state_.backend_cpu) {
                error_msg_ = "Failed to initialize CPU fallback backend for AudioTokenizerDecoder";
                return false;
            }
        }

        std::vector<ggml_backend_t> backends;
        backends.push_back(state_.backend);
        if (state_.backend_cpu) {
            backends.push_back(state_.backend_cpu);
        }

        if (decoder_debug_enabled()) {
            fprintf(stderr, "[decoder-dbg] scheduler backends=%d cpu_fallback=%d\n",
                    (int)backends.size(), state_.backend_cpu ? 1 : 0);
        }

        state_.sched = ggml_backend_sched_new(backends.data(), nullptr, (int)backends.size(), QWEN3_TTS_DEC_MAX_NODES, false, true);
        if (!state_.sched) {
            error_msg_ = "Failed to create backend scheduler";
            return false;
        }
        return true;
    };

    if (!init_decoder_backend()) {
        return false;
    }
    
    return true;
}

bool AudioTokenizerDecoder::set_n_threads(int32_t n_threads) {
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

void AudioTokenizerDecoder::reset_decode_graph_cache() {
    if (state_.sched) {
        ggml_backend_sched_reset(state_.sched);
    }

    for (auto & entry : decode_graph_cache_) {
        free_decode_graph_entry(entry);
    }
    decode_graph_cache_.clear();
    active_decode_graph_frames_ = -1;
    decode_graph_use_tick_ = 0;
}

void AudioTokenizerDecoder::free_decode_graph_entry(decode_graph_cache_entry & entry) {
    if (entry.ctx) {
        ggml_free(entry.ctx);
    }
    entry = decode_graph_cache_entry();
}

void AudioTokenizerDecoder::apply_graph_backend_hints(decode_graph_cache_entry & entry) {
    (void)entry;
}

bool AudioTokenizerDecoder::ensure_decode_graph(int32_t n_frames, decode_graph_cache_entry *& entry_out) {
    entry_out = nullptr;

    if (n_frames <= 0) {
        error_msg_ = "Invalid frame count";
        return false;
    }

    if (!state_.sched) {
        error_msg_ = "Backend scheduler is not initialized";
        return false;
    }

    for (auto & entry : decode_graph_cache_) {
        if (entry.n_frames == n_frames && entry.gf && entry.ctx) {
            entry.use_tick = ++decode_graph_use_tick_;
            entry_out = &entry;
            return true;
        }
    }

    decode_graph_cache_entry fresh;
    fresh.n_frames = n_frames;
    fresh.use_tick = ++decode_graph_use_tick_;

    const size_t meta_size = ggml_tensor_overhead() * QWEN3_TTS_DEC_MAX_NODES + ggml_graph_overhead();
    fresh.meta.resize(meta_size);

    struct ggml_init_params ggml_params = {
        /*.mem_size   =*/ fresh.meta.size(),
        /*.mem_buffer =*/ fresh.meta.data(),
        /*.no_alloc   =*/ true,
    };

    fresh.ctx = ggml_init(ggml_params);
    if (!fresh.ctx) {
        error_msg_ = "Failed to initialize decode graph context";
        return false;
    }

    fresh.gf = build_graph(fresh.ctx, n_frames);
    if (!fresh.gf) {
        free_decode_graph_entry(fresh);
        error_msg_ = "Failed to build decode graph";
        return false;
    }

    for (int cb = 0; cb < 16; ++cb) {
        char name[32];
        snprintf(name, sizeof(name), "codes_cb%d", cb);
        fresh.cb_tensors[cb] = ggml_graph_get_tensor(fresh.gf, name);
        if (!fresh.cb_tensors[cb]) {
            free_decode_graph_entry(fresh);
            error_msg_ = "Failed to cache codes tensor for codebook " + std::to_string(cb);
            return false;
        }
        fresh.cb_codes[cb].assign(n_frames, 0);
    }

    fresh.positions = ggml_graph_get_tensor(fresh.gf, "positions");
    fresh.audio = ggml_graph_get_tensor(fresh.gf, "audio");

    if (!fresh.audio) {
        free_decode_graph_entry(fresh);
        error_msg_ = "Failed to cache audio tensor";
        return false;
    }

    fresh.positions_buf.resize(n_frames);
    for (int i = 0; i < n_frames; ++i) {
        fresh.positions_buf[i] = i;
    }

    if ((int32_t)decode_graph_cache_.size() >= DECODE_GRAPH_CACHE_MAX) {
        size_t victim = 0;
        bool has_victim = false;
        uint64_t min_tick = std::numeric_limits<uint64_t>::max();
        for (size_t i = 0; i < decode_graph_cache_.size(); ++i) {
            const auto & e = decode_graph_cache_[i];
            if (!has_victim || e.use_tick < min_tick) {
                min_tick = e.use_tick;
                victim = i;
                has_victim = true;
            }
        }

        if (!has_victim) {
            free_decode_graph_entry(fresh);
            error_msg_ = "Failed to select graph cache victim";
            return false;
        }

        if (decode_graph_cache_[victim].n_frames == active_decode_graph_frames_ && state_.sched) {
            ggml_backend_sched_reset(state_.sched);
            active_decode_graph_frames_ = -1;
            for (auto & e : decode_graph_cache_) {
                e.sched_allocated = false;
                e.positions_uploaded = false;
            }
        }

        free_decode_graph_entry(decode_graph_cache_[victim]);
        decode_graph_cache_[victim] = std::move(fresh);
        entry_out = &decode_graph_cache_[victim];
    } else {
        decode_graph_cache_.push_back(std::move(fresh));
        entry_out = &decode_graph_cache_.back();
    }

    if (!entry_out || !entry_out->gf || !entry_out->ctx) {
        error_msg_ = "Failed to cache decode graph entry";
        return false;
    }

    return true;
}

struct ggml_tensor * AudioTokenizerDecoder::apply_snake(struct ggml_context * ctx,
                                                         struct ggml_tensor * x,
                                                         struct ggml_tensor * alpha,
                                                         struct ggml_tensor * beta) {
    int64_t seq_len = x->ne[0];
    int64_t channels = x->ne[1];
    int64_t batch = x->ne[2];
    
    struct ggml_tensor * alpha_exp = ggml_exp(ctx, alpha);
    
    struct ggml_tensor * alpha_3d = ggml_reshape_3d(ctx, alpha_exp, 1, channels, 1);
    struct ggml_tensor * alpha_broad = ggml_repeat(ctx, alpha_3d, 
                                                    ggml_new_tensor_3d(ctx, GGML_TYPE_F32, seq_len, channels, batch));
    
    struct ggml_tensor * ax = ggml_mul(ctx, x, alpha_broad);
    struct ggml_tensor * sin_ax = ggml_sin(ctx, ax);
    struct ggml_tensor * sin_sq = ggml_sqr(ctx, sin_ax);
    
    struct ggml_tensor * neg_beta = ggml_scale(ctx, beta, -1.0f);
    struct ggml_tensor * inv_beta_exp = ggml_exp(ctx, neg_beta);
    struct ggml_tensor * inv_beta_3d = ggml_reshape_3d(ctx, inv_beta_exp, 1, channels, 1);
    struct ggml_tensor * inv_beta = ggml_repeat(ctx, inv_beta_3d, 
                                                 ggml_new_tensor_3d(ctx, GGML_TYPE_F32, seq_len, channels, batch));
    
    struct ggml_tensor * scaled_sin = ggml_mul(ctx, sin_sq, inv_beta);
    
    return ggml_add(ctx, x, scaled_sin);
}

struct ggml_tensor * AudioTokenizerDecoder::apply_rms_norm(struct ggml_context * ctx,
                                                            struct ggml_tensor * x,
                                                            struct ggml_tensor * w,
                                                            float eps) {
    struct ggml_tensor * normed = ggml_rms_norm(ctx, x, eps);
    return ggml_mul(ctx, normed, w);
}

struct ggml_tensor * AudioTokenizerDecoder::apply_pre_tfm_layer(struct ggml_context * ctx,
                                                                 struct ggml_tensor * x,
                                                                 const pre_tfm_layer & layer,
                                                                 int32_t n_frames,
                                                                 struct ggml_tensor * positions) {
    const auto & cfg = model_.config;
    const int n_heads = cfg.n_heads;
    const int qkv_dim = cfg.latent_dim;
    const int head_dim = qkv_dim / n_heads;
    
    if (!layer.attn_norm_w || !layer.attn_q_w || !layer.attn_k_w || !layer.attn_v_w ||
        !layer.attn_output_w || !layer.ffn_norm_w || !layer.ffn_gate_w || 
        !layer.ffn_up_w || !layer.ffn_down_w) {
        return x;
    }
    
    struct ggml_tensor * residual = x;
    
    struct ggml_tensor * normed = apply_rms_norm(ctx, x, layer.attn_norm_w, cfg.rms_norm_eps);
    
    struct ggml_tensor * Qcur = ggml_mul_mat(ctx, layer.attn_q_w, normed);
    struct ggml_tensor * Kcur = ggml_mul_mat(ctx, layer.attn_k_w, normed);
    struct ggml_tensor * Vcur = ggml_mul_mat(ctx, layer.attn_v_w, normed);
    
    Qcur = ggml_reshape_3d(ctx, Qcur, head_dim, n_heads, n_frames);
    Kcur = ggml_reshape_3d(ctx, Kcur, head_dim, n_heads, n_frames);
    Vcur = ggml_reshape_3d(ctx, Vcur, head_dim, n_heads, n_frames);
    
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
    // Apply causal mask (each position can only attend to itself and previous positions)
    KQ = ggml_diag_mask_inf(ctx, KQ, 0);
    KQ = ggml_soft_max(ctx, KQ);
    
    V = ggml_cont(ctx, ggml_transpose(ctx, V));
    
    struct ggml_tensor * KQV = ggml_mul_mat(ctx, V, KQ);
    KQV = ggml_permute(ctx, KQV, 0, 2, 1, 3);
    struct ggml_tensor * attn_out = ggml_cont_2d(ctx, KQV, n_heads * head_dim, n_frames);
    
    attn_out = ggml_mul_mat(ctx, layer.attn_output_w, attn_out);
    
    if (layer.attn_scale) {
        attn_out = ggml_mul(ctx, attn_out, layer.attn_scale);
    }
    
    x = ggml_add(ctx, residual, attn_out);
    residual = x;
    
    normed = apply_rms_norm(ctx, x, layer.ffn_norm_w, cfg.rms_norm_eps);
    
    struct ggml_tensor * gate = ggml_mul_mat(ctx, layer.ffn_gate_w, normed);
    struct ggml_tensor * up = ggml_mul_mat(ctx, layer.ffn_up_w, normed);
    
    gate = ggml_silu(ctx, gate);
    struct ggml_tensor * ffn_out = ggml_mul(ctx, gate, up);
    
    ffn_out = ggml_mul_mat(ctx, layer.ffn_down_w, ffn_out);
    
    if (layer.ffn_scale) {
        ffn_out = ggml_mul(ctx, ffn_out, layer.ffn_scale);
    }
    
    return ggml_add(ctx, residual, ffn_out);
}

struct ggml_tensor * AudioTokenizerDecoder::apply_upsample_block(struct ggml_context * ctx,
                                                                   struct ggml_tensor * x,
                                                                   const upsample_block & block,
                                                                   int block_idx) {
    int64_t seq_len = x->ne[0];
    int64_t channels = x->ne[1];
    
     struct ggml_tensor * x_2d = ggml_reshape_2d(ctx, x, seq_len, channels);
     x_2d = ggml_conv_transpose_1d(ctx, block.conv_w, x_2d, 2, 0, 1);
     
     int64_t new_seq_len = x_2d->ne[0];
     x = ggml_reshape_3d(ctx, x_2d, new_seq_len, channels, 1);
     
     if (block.conv_b) {
         x = ggml_add(ctx, x, ggml_reshape_3d(ctx, block.conv_b, 1, channels, 1));
     }
    
     struct ggml_tensor * residual = x;
     
     if (block.dwconv_w) {
         // Causal padding: pad left with 6 zeros (kernel_size - 1 = 7 - 1 = 6)
         x = ggml_pad_ext(ctx, x, 6, 0, 0, 0, 0, 0, 0, 0);  // left pad only
         x = ggml_conv_1d_dw(ctx, block.dwconv_w, x, 1, 0, 1);  // no padding in conv
         if (block.dwconv_b) {
             x = ggml_add(ctx, x, ggml_reshape_3d(ctx, block.dwconv_b, 1, channels, 1));
         }
     }
    
    x = ggml_permute(ctx, x, 1, 0, 2, 3);
    x = ggml_cont(ctx, x);
    
     if (block.norm_w && block.norm_b) {
         x = ggml_norm(ctx, x, 1e-6f);
         x = ggml_mul(ctx, x, block.norm_w);
         x = ggml_add(ctx, x, block.norm_b);
     }
    
     x = ggml_mul_mat(ctx, block.pwconv1_w, x);
     if (block.pwconv1_b) {
         x = ggml_add(ctx, x, block.pwconv1_b);
     }
    
     x = ggml_gelu(ctx, x);
    
     x = ggml_mul_mat(ctx, block.pwconv2_w, x);
     if (block.pwconv2_b) {
         x = ggml_add(ctx, x, block.pwconv2_b);
     }
    
    x = ggml_permute(ctx, x, 1, 0, 2, 3);
    x = ggml_cont(ctx, x);
    
     if (block.gamma) {
         struct ggml_tensor * gamma_3d = ggml_reshape_3d(ctx, block.gamma, 1, channels, 1);
         x = ggml_mul(ctx, x, ggml_repeat(ctx, gamma_3d, 
                                           ggml_new_tensor_3d(ctx, GGML_TYPE_F32, new_seq_len, channels, 1)));
     }
    
    return ggml_add(ctx, residual, x);
}

struct ggml_tensor * AudioTokenizerDecoder::apply_residual_block(struct ggml_context * ctx,
                                                                  struct ggml_tensor * x,
                                                                  const residual_block & block) {
    struct ggml_tensor * residual = x;
    
    if (block.act1_alpha) {
        x = apply_snake(ctx, x, block.act1_alpha, block.act1_beta);
    }
    
    int64_t out_channels = block.conv1_w->ne[2];
    int padding = 6 * block.dilation;
    x = ggml_pad_ext(ctx, x, padding, 0, 0, 0, 0, 0, 0, 0);
    x = ggml_conv_1d(ctx, block.conv1_w, x, 1, 0, block.dilation);
    if (block.conv1_b) {
        x = ggml_add(ctx, x, ggml_reshape_3d(ctx, block.conv1_b, 1, out_channels, 1));
    }
    
    if (block.act2_alpha) {
        x = apply_snake(ctx, x, block.act2_alpha, block.act2_beta);
    }
    
    out_channels = block.conv2_w->ne[2];
    x = ggml_conv_1d(ctx, block.conv2_w, x, 1, 0, 1);
    if (block.conv2_b) {
        x = ggml_add(ctx, x, ggml_reshape_3d(ctx, block.conv2_b, 1, out_channels, 1));
    }
    
    return ggml_add(ctx, residual, x);
}

struct ggml_tensor * AudioTokenizerDecoder::apply_decoder_block(struct ggml_context * ctx,
                                                                  struct ggml_tensor * x,
                                                                  const decoder_block & block,
                                                                  int upsample_rate,
                                                                  int block_idx) {
    if (block.snake_alpha && block.snake_beta) {
        x = apply_snake(ctx, x, block.snake_alpha, block.snake_beta);
    }
    
     int64_t seq_len = x->ne[0];
     int64_t in_channels = x->ne[1];
     int64_t out_channels = block.conv_t_w->ne[1];
     int kernel_size = block.conv_t_w->ne[0];
     
     struct ggml_tensor * x_2d = ggml_reshape_2d(ctx, x, seq_len, in_channels);
     x_2d = ggml_conv_transpose_1d(ctx, block.conv_t_w, x_2d, upsample_rate, 0, 1);
     
     int64_t new_seq_len = x_2d->ne[0];
     x = ggml_reshape_3d(ctx, x_2d, new_seq_len, out_channels, 1);
     
     // Python CausalTransConvNet: left_pad = right_pad = kernel_size - stride
     int pad = kernel_size - upsample_rate;
     int left_pad = pad;
     int right_pad = pad;
     int64_t out_seq_len = new_seq_len - left_pad - right_pad;
     
     x = ggml_view_3d(ctx, x, out_seq_len, out_channels, 1,
                      x->nb[1], x->nb[2], left_pad * x->nb[0]);
     x = ggml_cont(ctx, x);
     
     if (block.conv_t_b) {
         x = ggml_add(ctx, x, ggml_reshape_3d(ctx, block.conv_t_b, 1, out_channels, 1));
     }
    
    for (int i = 0; i < 3; ++i) {
        x = apply_residual_block(ctx, x, block.res[i]);
    }
    
    return x;
}

struct ggml_cgraph * AudioTokenizerDecoder::build_graph(struct ggml_context * ctx0, int32_t n_frames) {
    const auto & cfg = model_.config;

    if (!ctx0) {
        error_msg_ = "Decode graph context is not initialized";
        return nullptr;
    }
    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, QWEN3_TTS_DEC_MAX_NODES, false);
    
    static const char * cb_names[16] = {
        "codes_cb0", "codes_cb1", "codes_cb2", "codes_cb3",
        "codes_cb4", "codes_cb5", "codes_cb6", "codes_cb7",
        "codes_cb8", "codes_cb9", "codes_cb10", "codes_cb11",
        "codes_cb12", "codes_cb13", "codes_cb14", "codes_cb15"
    };
    
    struct ggml_tensor * cb_codes_tensors[16];
    for (int cb = 0; cb < 16; ++cb) {
        cb_codes_tensors[cb] = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_frames);
        ggml_set_name(cb_codes_tensors[cb], cb_names[cb]);
        ggml_set_input(cb_codes_tensors[cb]);
    }
    
    struct ggml_tensor * first_codes = cb_codes_tensors[0];
    
     struct ggml_tensor * first_emb = ggml_get_rows(ctx0, model_.vq_first_codebook, first_codes);
     ggml_set_name(first_emb, "first_emb_raw");
     
     struct ggml_tensor * rest_emb[15];
     for (int cb = 0; cb < 15; ++cb) {
         struct ggml_tensor * cb_codes = cb_codes_tensors[cb + 1];
         rest_emb[cb] = ggml_get_rows(ctx0, model_.vq_rest_codebook[cb], cb_codes);
         
         if (cb == 0) {
             ggml_set_name(rest_emb[cb], "rest_cb0_emb_raw");
         }
     }
    
     struct ggml_tensor * first_emb_2d = ggml_reshape_2d(ctx0, first_emb, cfg.codebook_dim, n_frames);
     ggml_set_name(first_emb_2d, "first_emb_2d");
     
     struct ggml_tensor * first_proj_weight_2d = ggml_reshape_2d(ctx0, model_.vq_first_output_proj, 
                                                                   cfg.codebook_dim, cfg.hidden_dim);
     struct ggml_tensor * first_proj_2d = ggml_mul_mat(ctx0, first_proj_weight_2d, first_emb_2d);
     ggml_set_name(first_proj_2d, "first_proj_2d");
    
    struct ggml_tensor * rest_proj_weight_2d = ggml_reshape_2d(ctx0, model_.vq_rest_output_proj,
                                                                 cfg.codebook_dim, cfg.hidden_dim);
    
     struct ggml_tensor * rest_proj_2d = nullptr;
     for (int cb = 0; cb < 15; ++cb) {
         struct ggml_tensor * cb_emb_2d = ggml_reshape_2d(ctx0, rest_emb[cb], cfg.codebook_dim, n_frames);
         
         if (cb == 0) {
             ggml_set_name(cb_emb_2d, "rest_cb0_emb_2d");
         }
         
         struct ggml_tensor * cb_proj_2d = ggml_mul_mat(ctx0, rest_proj_weight_2d, cb_emb_2d);
         
         if (rest_proj_2d == nullptr) {
             rest_proj_2d = cb_proj_2d;
         } else {
             rest_proj_2d = ggml_add(ctx0, rest_proj_2d, cb_proj_2d);
         }
     }
     ggml_set_name(rest_proj_2d, "rest_proj_2d");
    
     struct ggml_tensor * latent_2d = ggml_add(ctx0, first_proj_2d, rest_proj_2d);
     ggml_set_name(latent_2d, "latent_2d");
     
     struct ggml_tensor * latent_t = ggml_transpose(ctx0, latent_2d);
     ggml_set_name(latent_t, "latent_t");
     
     struct ggml_tensor * latent_cont = ggml_cont(ctx0, latent_t);
     ggml_set_name(latent_cont, "latent_cont");
     
     struct ggml_tensor * latent = ggml_reshape_3d(ctx0, latent_cont, n_frames, cfg.hidden_dim, 1);

     ggml_set_name(latent, "vq_output");
    
    struct ggml_tensor * latent_for_conv = ggml_cont(ctx0, latent);
    struct ggml_tensor * latent_padded = ggml_pad_ext(ctx0, latent_for_conv, 2, 0, 0, 0, 0, 0, 0, 0);
     struct ggml_tensor * cur = ggml_conv_1d(ctx0, model_.pre_conv_w, latent_padded, 1, 0, 1);
     if (model_.pre_conv_b) {
         cur = ggml_add(ctx0, cur, ggml_reshape_3d(ctx0, model_.pre_conv_b, 1, cfg.latent_dim, 1));
     }
     
     ggml_set_name(cur, "pre_conv_output");
     
     struct ggml_tensor * cur_2d = ggml_reshape_2d(ctx0, cur, n_frames, cfg.latent_dim);
     struct ggml_tensor * cur_t = ggml_transpose(ctx0, cur_2d);
     cur = ggml_cont(ctx0, cur_t);
     
     ggml_set_name(cur, "pre_conv_reshaped");
     
     cur = ggml_mul_mat(ctx0, model_.pre_tfm_input_proj_w, cur);
     if (model_.pre_tfm_input_proj_b) {
         cur = ggml_add(ctx0, cur, model_.pre_tfm_input_proj_b);
     }
     
     ggml_set_name(cur, "pre_tfm_input");
    
    struct ggml_tensor * positions = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_frames);
    ggml_set_name(positions, "positions");
    ggml_set_input(positions);
    
     for (int i = 0; i < cfg.n_pre_tfm_layers; ++i) {
         cur = apply_pre_tfm_layer(ctx0, cur, model_.pre_tfm_layers[i], n_frames, positions);
     }
     
     if (model_.pre_tfm_norm_w) {
         cur = apply_rms_norm(ctx0, cur, model_.pre_tfm_norm_w, cfg.rms_norm_eps);
     }
     
     cur = ggml_mul_mat(ctx0, model_.pre_tfm_output_proj_w, cur);
     if (model_.pre_tfm_output_proj_b) {
         cur = ggml_add(ctx0, cur, model_.pre_tfm_output_proj_b);
     }
     
     ggml_set_name(cur, "pre_tfm_output");
    
    cur = ggml_permute(ctx0, cur, 1, 0, 2, 3);
     cur = ggml_cont(ctx0, cur);
     cur = ggml_reshape_3d(ctx0, cur, n_frames, cfg.latent_dim, 1);
     
     ggml_set_name(cur, "pre_tfm_reshaped");
    
     for (int i = 0; i < 2; ++i) {
         cur = apply_upsample_block(ctx0, cur, model_.upsample[i], i);
     }
     
     ggml_set_name(cur, "upsample_output");
     
     // Causal padding: left pad with 6 (kernel_size - 1 = 7 - 1 = 6)
     cur = ggml_pad_ext(ctx0, cur, 6, 0, 0, 0, 0, 0, 0, 0);
     cur = ggml_conv_1d(ctx0, model_.dec0_conv_w, cur, 1, 0, 1);
     if (model_.dec0_conv_b) {
         cur = ggml_add(ctx0, cur, ggml_reshape_3d(ctx0, model_.dec0_conv_b, 1, cfg.decoder_dim, 1));
     }
     
     ggml_set_name(cur, "dec0_output");
     
     int upsample_rates[4] = {8, 5, 4, 3};
     for (int i = 0; i < 4; ++i) {
         cur = apply_decoder_block(ctx0, cur, model_.dec_blocks[i], upsample_rates[i], i);
         char name[32];
         snprintf(name, sizeof(name), "dec%d_output", i + 1);
         ggml_set_name(cur, name);
     }
     
     if (model_.dec5_snake_alpha) {
         cur = apply_snake(ctx0, cur, model_.dec5_snake_alpha, model_.dec5_snake_beta);
     }
     
     ggml_set_name(cur, "dec5_output");
     
     // Causal padding: left pad with 6 (kernel_size - 1 = 7 - 1 = 6)
     cur = ggml_pad_ext(ctx0, cur, 6, 0, 0, 0, 0, 0, 0, 0);
     cur = ggml_conv_1d(ctx0, model_.dec6_conv_w, cur, 1, 0, 1);
     if (model_.dec6_conv_b) {
         cur = ggml_add(ctx0, cur, ggml_reshape_3d(ctx0, model_.dec6_conv_b, 1, 1, 1));
     }
     
     ggml_set_name(cur, "dec6_output");
    
    cur = ggml_tanh(ctx0, cur);
    
    cur = ggml_reshape_1d(ctx0, cur, cur->ne[0]);
    
    ggml_set_name(cur, "audio");
    ggml_set_output(cur);
    
    ggml_build_forward_expand(gf, cur);

    return gf;
}

bool AudioTokenizerDecoder::decode(const int32_t * codes, int32_t n_frames,
                                    std::vector<float> & samples) {
    if (!model_.ctx) {
        error_msg_ = "Model not loaded";
        return false;
    }
    
    const auto & cfg = model_.config;
    
    const bool dec_dbg = decoder_debug_enabled();
    auto t0 = std::chrono::steady_clock::now();

    // Rebuild decode graph cache when frame shape switches. Reusing an old
    // shape entry (e.g. 16 -> 19 -> 16 in streaming tail) can become unstable.
    if (active_decode_graph_frames_ >= 0 && active_decode_graph_frames_ != n_frames) {
        reset_decode_graph_cache();
    }

    decode_graph_cache_entry * graph = nullptr;
    if (!ensure_decode_graph(n_frames, graph) || !graph) {
        return false;
    }

    auto t_build = std::chrono::steady_clock::now();

    if (active_decode_graph_frames_ != n_frames) {
        ggml_backend_sched_reset(state_.sched);
        for (auto & e : decode_graph_cache_) {
            e.sched_allocated = false;
            e.positions_uploaded = false;
        }
        active_decode_graph_frames_ = n_frames;
        graph->sched_allocated = false;
    }

    if (!graph->sched_allocated) {
        apply_graph_backend_hints(*graph);
        if (!ggml_backend_sched_alloc_graph(state_.sched, graph->gf)) {
            error_msg_ = "Failed to allocate decode graph";
            active_decode_graph_frames_ = -1;
            graph->sched_allocated = false;
            return false;
        }
        graph->sched_allocated = true;
    }

    auto t_alloc = std::chrono::steady_clock::now();

    auto t_pack = std::chrono::steady_clock::now();

    for (int cb = 0; cb < cfg.n_codebooks; ++cb) {
        struct ggml_tensor * cb_tensor = graph->cb_tensors[cb];
        if (!cb_tensor) {
            error_msg_ = "Failed to find codes tensor for codebook " + std::to_string(cb);
            active_decode_graph_frames_ = -1;
            graph->sched_allocated = false;
            return false;
        }

        std::vector<int32_t> & cb_codes = graph->cb_codes[cb];
        const int32_t codebook_size = std::max(1, cfg.codebook_size);
        const int32_t * src = codes + cb;
        int32_t * dst = cb_codes.data();
        for (int f = 0; f < n_frames; ++f, src += cfg.n_codebooks) {
            const int32_t code = *src;
            if (code < 0 || code >= codebook_size) {
                error_msg_ = "Audio code out of range at frame=" + std::to_string(f) +
                             ", codebook=" + std::to_string(cb) +
                             ", value=" + std::to_string(code) +
                             ", expected=[0," + std::to_string(codebook_size - 1) + "]";
                active_decode_graph_frames_ = -1;
                graph->sched_allocated = false;
                return false;
            }
            dst[f] = code;
        }

        ggml_backend_tensor_set(cb_tensor, cb_codes.data(), 0, n_frames * sizeof(int32_t));
    }

    if (!graph->positions_uploaded) {
        struct ggml_tensor * positions_tensor = graph->positions;
        if (positions_tensor) {
            ggml_backend_tensor_set(positions_tensor, graph->positions_buf.data(), 0,
                                    n_frames * sizeof(int32_t));
        }
        graph->positions_uploaded = true;
    }
    auto t_set = std::chrono::steady_clock::now();

    if (ggml_backend_sched_graph_compute(state_.sched, graph->gf) != GGML_STATUS_SUCCESS) {
        error_msg_ = "Failed to compute graph";
        ggml_backend_sched_reset(state_.sched);
        for (auto & e : decode_graph_cache_) {
            e.sched_allocated = false;
            e.positions_uploaded = false;
        }
        active_decode_graph_frames_ = -1;
        graph->sched_allocated = false;
        return false;
    }

    auto t_compute = std::chrono::steady_clock::now();

    struct ggml_tensor * audio_tensor = graph->audio;
    if (!audio_tensor) {
        error_msg_ = "Failed to find audio tensor";
        ggml_backend_sched_reset(state_.sched);
        for (auto & e : decode_graph_cache_) {
            e.sched_allocated = false;
            e.positions_uploaded = false;
        }
        active_decode_graph_frames_ = -1;
        graph->sched_allocated = false;
        return false;
    }

    int64_t n_samples = audio_tensor->ne[0];
    samples.resize((size_t)n_samples);
    ggml_backend_tensor_get(audio_tensor, samples.data(), 0, n_samples * sizeof(float));
    auto t_get = std::chrono::steady_clock::now();

    auto t_reset = std::chrono::steady_clock::now();

    if (dec_dbg) {
        auto ms = [](const std::chrono::steady_clock::time_point & a,
                     const std::chrono::steady_clock::time_point & b) -> long long {
            return (long long)std::chrono::duration_cast<std::chrono::milliseconds>(b - a).count();
        };
        fprintf(
            stderr,
            "[decoder-dbg] decode frames=%d samples=%lld total_ms=%lld pack=%lld build=%lld alloc=%lld set=%lld compute=%lld get=%lld reset=%lld\n",
            n_frames,
            (long long)n_samples,
            ms(t0, t_reset),
            ms(t0, t_pack),
            ms(t_pack, t_build),
            ms(t_build, t_alloc),
            ms(t_alloc, t_set),
            ms(t_set, t_compute),
            ms(t_compute, t_get),
            ms(t_get, t_reset)
        );
    }
    
    return true;
}

bool AudioTokenizerDecoder::begin_stream(int32_t left_context_frames,
                                         int32_t lookahead_frames) {
    if (!model_.ctx) {
        error_msg_ = "Model not loaded";
        return false;
    }

    const auto & cfg = model_.config;

    if (left_context_frames < 0) {
        left_context_frames = 0;
    }
    if (lookahead_frames < 0) {
        lookahead_frames = 0;
    }

    stream_state_.active = true;
    stream_state_.left_context_frames = left_context_frames;
    stream_state_.lookahead_frames = lookahead_frames;
    stream_state_.emitted_samples = 0;
    stream_state_.upsample_rate = 1920;
    stream_state_.model_delay_samples = -1;
    stream_state_.total_frames = 0;
    stream_state_.history_limit_frames = std::max(16, left_context_frames + lookahead_frames + 24);
    stream_state_.history_codes.clear();
    stream_state_.history_codes.reserve((size_t)stream_state_.history_limit_frames * (size_t)cfg.n_codebooks);
    stream_state_.pending_codes.clear();
    stream_state_.pending_codes.reserve((size_t)std::max(8, lookahead_frames + 8) * (size_t)cfg.n_codebooks);
    stream_state_.decode_codes.clear();
    stream_state_.seam_tail_samples.clear();
    stream_state_.start_fade_samples = 96;
    return true;
}

bool AudioTokenizerDecoder::decode_chunk_with_context(const int32_t * chunk_codes,
                                                      int32_t n_chunk_frames,
                                                      int32_t n_context_frames,
                                                      std::vector<float> & out_samples,
                                                      bool full_output) {
    out_samples.clear();

    if (!chunk_codes || n_chunk_frames <= 0) {
        return true;
    }

    std::vector<float> decoded;
    if (!decode(chunk_codes, n_chunk_frames, decoded)) {
        return false;
    }

    if (full_output || n_context_frames <= 0) {
        out_samples = std::move(decoded);
        return true;
    }

    if (stream_state_.upsample_rate <= 0) {
        stream_state_.upsample_rate = 1920;
    }

    const int64_t cut = (int64_t)n_context_frames * (int64_t)stream_state_.upsample_rate;
    if (cut <= 0) {
        out_samples = std::move(decoded);
        return true;
    }

    if ((size_t)cut >= decoded.size()) {
        out_samples.clear();
        return true;
    }

    out_samples.assign(decoded.begin() + cut, decoded.end());

    if (stream_state_.model_delay_samples < 0 && n_context_frames > 0) {
        const int64_t effective_frames = std::max<int64_t>(0, (int64_t)n_chunk_frames - (int64_t)n_context_frames);
        const int64_t expected = effective_frames * (int64_t)stream_state_.upsample_rate;
        const int64_t actual = (int64_t)out_samples.size();
        int64_t extra = actual - expected;
        if (extra > 0) {
            stream_state_.model_delay_samples = (int32_t)extra;
        } else {
            stream_state_.model_delay_samples = 0;
        }
    }
    return true;
}

void AudioTokenizerDecoder::blend_stream_audio(const std::vector<float> & chunk_samples,
                                               bool is_final,
                                               std::vector<float> & out_samples) {
    out_samples.clear();

    const int32_t crossfade = std::max(0, stream_state_.seam_crossfade_samples);
    if (crossfade == 0) {
        if (!stream_state_.seam_tail_samples.empty()) {
            out_samples.insert(out_samples.end(),
                               stream_state_.seam_tail_samples.begin(),
                               stream_state_.seam_tail_samples.end());
            stream_state_.seam_tail_samples.clear();
        }
        out_samples.insert(out_samples.end(), chunk_samples.begin(), chunk_samples.end());
        return;
    }

    if (!is_final) {
        if ((int32_t)chunk_samples.size() <= crossfade) {
            if (stream_state_.seam_tail_samples.empty()) {
                stream_state_.seam_tail_samples = chunk_samples;
            } else {
                const size_t old_sz = stream_state_.seam_tail_samples.size();
                stream_state_.seam_tail_samples.resize(old_sz + chunk_samples.size());
                std::copy(chunk_samples.begin(), chunk_samples.end(),
                          stream_state_.seam_tail_samples.begin() + (ptrdiff_t)old_sz);
                if (stream_state_.seam_tail_samples.size() > (size_t)crossfade) {
                    const size_t drop = stream_state_.seam_tail_samples.size() - (size_t)crossfade;
                    stream_state_.seam_tail_samples.erase(
                        stream_state_.seam_tail_samples.begin(),
                        stream_state_.seam_tail_samples.begin() + (ptrdiff_t)drop
                    );
                }
            }
            return;
        }

        const size_t body_len = chunk_samples.size() - (size_t)crossfade;
        out_samples.reserve((stream_state_.seam_tail_samples.empty() ? 0 : stream_state_.seam_tail_samples.size()) + body_len);

        if (!stream_state_.seam_tail_samples.empty()) {
            const size_t xfade = std::min(stream_state_.seam_tail_samples.size(), body_len);
            const size_t tail_head = stream_state_.seam_tail_samples.size() - xfade;

            out_samples.insert(out_samples.end(),
                               stream_state_.seam_tail_samples.begin(),
                               stream_state_.seam_tail_samples.begin() + (ptrdiff_t)tail_head);

            for (size_t i = 0; i < xfade; ++i) {
                const float t = (float)(i + 1) / (float)(xfade + 1);
                const float a = stream_state_.seam_tail_samples[tail_head + i];
                const float b = chunk_samples[i];
                out_samples.push_back(a * (1.0f - t) + b * t);
            }

            if (body_len > xfade) {
                out_samples.insert(out_samples.end(),
                                   chunk_samples.begin() + (ptrdiff_t)xfade,
                                   chunk_samples.begin() + (ptrdiff_t)body_len);
            }
        } else {
            out_samples.insert(out_samples.end(),
                               chunk_samples.begin(),
                               chunk_samples.begin() + (ptrdiff_t)body_len);
        }

        stream_state_.seam_tail_samples.assign(
            chunk_samples.begin() + (ptrdiff_t)body_len,
            chunk_samples.end()
        );
        return;
    }

    out_samples.reserve(stream_state_.seam_tail_samples.size() + chunk_samples.size());
    if (!stream_state_.seam_tail_samples.empty()) {
        const size_t xfade = std::min(stream_state_.seam_tail_samples.size(), chunk_samples.size());
        const size_t tail_head = stream_state_.seam_tail_samples.size() - xfade;

        out_samples.insert(out_samples.end(),
                           stream_state_.seam_tail_samples.begin(),
                           stream_state_.seam_tail_samples.begin() + (ptrdiff_t)tail_head);

        for (size_t i = 0; i < xfade; ++i) {
            const float t = (float)(i + 1) / (float)(xfade + 1);
            const float a = stream_state_.seam_tail_samples[tail_head + i];
            const float b = chunk_samples[i];
            out_samples.push_back(a * (1.0f - t) + b * t);
        }

        if (chunk_samples.size() > xfade) {
            out_samples.insert(out_samples.end(),
                               chunk_samples.begin() + (ptrdiff_t)xfade,
                               chunk_samples.end());
        }
        stream_state_.seam_tail_samples.clear();
    } else {
        out_samples.insert(out_samples.end(), chunk_samples.begin(), chunk_samples.end());
    }
}

void AudioTokenizerDecoder::apply_stream_start_fade(std::vector<float> & samples) {
    if (samples.empty()) {
        return;
    }
    if (stream_state_.start_fade_samples <= 0) {
        return;
    }

    const size_t n_fade = std::min<size_t>((size_t)stream_state_.start_fade_samples, samples.size());
    for (size_t i = 0; i < n_fade; ++i) {
        const float t = (float)(i + 1) / (float)(n_fade + 1);
        samples[i] *= t;
    }
    stream_state_.start_fade_samples -= (int32_t)n_fade;
}

bool AudioTokenizerDecoder::decode_append(const int32_t * codes, int32_t n_new_frames,
                                          std::vector<float> & out_samples) {
    out_samples.clear();

    if (!stream_state_.active) {
        error_msg_ = "Streaming session not started";
        return false;
    }
    if (!codes || n_new_frames <= 0) {
        return true;
    }

    const int32_t n_codebooks = model_.config.n_codebooks;
    const int32_t lookahead_frames = std::max(0, stream_state_.lookahead_frames);

    stream_state_.pending_codes.insert(
        stream_state_.pending_codes.end(),
        codes,
        codes + (size_t)n_new_frames * (size_t)n_codebooks
    );

    const int32_t pending_frames =
        (int32_t)(stream_state_.pending_codes.size() / (size_t)n_codebooks);
    if (pending_frames <= lookahead_frames) {
        if (decoder_debug_enabled()) {
            fprintf(
                stderr,
                "[decoder-dbg] append buffered_only new=%d pending=%d lookahead=%d total_frames=%d\n",
                n_new_frames,
                pending_frames,
                lookahead_frames,
                stream_state_.total_frames
            );
        }
        return true;
    }

    const int32_t emit_frames = pending_frames - lookahead_frames;
    const int32_t processed_frames = stream_state_.total_frames;
    const int32_t n_ctx = std::min(stream_state_.left_context_frames, processed_frames);
    const int32_t n_decode_frames = n_ctx + pending_frames;

    if (stream_state_.history_limit_frames < stream_state_.left_context_frames + lookahead_frames + emit_frames + 8) {
        stream_state_.history_limit_frames = stream_state_.left_context_frames + lookahead_frames + emit_frames + 8;
    }

    stream_state_.decode_codes.resize((size_t)n_decode_frames * (size_t)n_codebooks);

    if (n_ctx > 0) {
        const int32_t keep_frames = (int32_t)(stream_state_.history_codes.size() / (size_t)n_codebooks);
        if (keep_frames < n_ctx) {
            error_msg_ = "Streaming history underflow";
            return false;
        }
        const int32_t start_ctx_frame = keep_frames - n_ctx;
        const int32_t * src_ctx = stream_state_.history_codes.data() + (size_t)start_ctx_frame * (size_t)n_codebooks;
        std::copy(src_ctx, src_ctx + (size_t)n_ctx * (size_t)n_codebooks, stream_state_.decode_codes.begin());
    }

    std::copy(
        stream_state_.pending_codes.begin(),
        stream_state_.pending_codes.end(),
        stream_state_.decode_codes.begin() + (size_t)n_ctx * (size_t)n_codebooks
    );

    std::vector<float> decoded_pending;
    if (!decode_chunk_with_context(stream_state_.decode_codes.data(), n_decode_frames, n_ctx, decoded_pending, false)) {
        return false;
    }

    if (stream_state_.upsample_rate <= 0) {
        stream_state_.upsample_rate = 1920;
    }
    const int64_t expected_emit_samples = (int64_t)emit_frames * (int64_t)stream_state_.upsample_rate;
    const size_t emit_samples =
        (size_t)std::max<int64_t>(0, std::min<int64_t>(expected_emit_samples, (int64_t)decoded_pending.size()));

    if (emit_samples > 0) {
        std::vector<float> raw_emit(decoded_pending.begin(), decoded_pending.begin() + (ptrdiff_t)emit_samples);
        blend_stream_audio(raw_emit, false, out_samples);
        apply_stream_start_fade(out_samples);
    }

    const size_t emit_codes = (size_t)emit_frames * (size_t)n_codebooks;
    stream_state_.history_codes.insert(
        stream_state_.history_codes.end(),
        stream_state_.pending_codes.begin(),
        stream_state_.pending_codes.begin() + (ptrdiff_t)emit_codes
    );

    stream_state_.pending_codes.erase(
        stream_state_.pending_codes.begin(),
        stream_state_.pending_codes.begin() + (ptrdiff_t)emit_codes
    );

    const size_t keep_codes = (size_t)std::max(1, stream_state_.history_limit_frames) * (size_t)n_codebooks;
    if (stream_state_.history_codes.size() > keep_codes) {
        const size_t drop = stream_state_.history_codes.size() - keep_codes;
        stream_state_.history_codes.erase(
            stream_state_.history_codes.begin(),
            stream_state_.history_codes.begin() + (ptrdiff_t)drop
        );
    }

    stream_state_.total_frames += emit_frames;
    stream_state_.emitted_samples += (int64_t)out_samples.size();

    if (decoder_debug_enabled()) {
        fprintf(
            stderr,
            "[decoder-dbg] append new=%d emit=%d keep=%d ctx=%d decode_frames=%d out_samples=%zu total_frames=%d\n",
            n_new_frames,
            emit_frames,
            (int32_t)(stream_state_.pending_codes.size() / (size_t)n_codebooks),
            n_ctx,
            n_decode_frames,
            out_samples.size(),
            stream_state_.total_frames
        );
    }

    return true;
}

bool AudioTokenizerDecoder::end_stream(std::vector<float> & tail_samples, bool full_flush) {
    tail_samples.clear();
    if (!stream_state_.active) {
        return true;
    }

    const int32_t n_codebooks = model_.config.n_codebooks;
    const int32_t pending_frames =
        (int32_t)(stream_state_.pending_codes.size() / (size_t)n_codebooks);

    if (pending_frames > 0) {
        const int32_t processed_frames = stream_state_.total_frames;
        const int32_t n_ctx = std::min(stream_state_.left_context_frames, processed_frames);
        const int32_t n_decode_frames = n_ctx + pending_frames;

        stream_state_.decode_codes.resize((size_t)n_decode_frames * (size_t)n_codebooks);

        if (n_ctx > 0) {
            const int32_t keep_frames = (int32_t)(stream_state_.history_codes.size() / (size_t)n_codebooks);
            if (keep_frames < n_ctx) {
                error_msg_ = "Streaming history underflow in end_stream";
                return false;
            }
            const int32_t start_ctx_frame = keep_frames - n_ctx;
            const int32_t * src_ctx = stream_state_.history_codes.data() + (size_t)start_ctx_frame * (size_t)n_codebooks;
            std::copy(src_ctx, src_ctx + (size_t)n_ctx * (size_t)n_codebooks, stream_state_.decode_codes.begin());
        }

        std::copy(
            stream_state_.pending_codes.begin(),
            stream_state_.pending_codes.end(),
            stream_state_.decode_codes.begin() + (size_t)n_ctx * (size_t)n_codebooks
        );

        std::vector<float> final_chunk;
        if (!decode_chunk_with_context(stream_state_.decode_codes.data(), n_decode_frames, n_ctx, final_chunk, false)) {
            return false;
        }
        blend_stream_audio(final_chunk, true, tail_samples);

        stream_state_.history_codes.insert(
            stream_state_.history_codes.end(),
            stream_state_.pending_codes.begin(),
            stream_state_.pending_codes.end()
        );
        stream_state_.pending_codes.clear();
        stream_state_.total_frames += pending_frames;
    } else if (!stream_state_.seam_tail_samples.empty()) {
        std::vector<float> empty_chunk;
        blend_stream_audio(empty_chunk, true, tail_samples);
    }

    apply_stream_start_fade(tail_samples);

    if (full_flush && stream_state_.total_frames > 0 && stream_state_.model_delay_samples > 0) {
        const int32_t n_delay = std::min<int32_t>(stream_state_.model_delay_samples,
                                                  std::max<int32_t>(0, stream_state_.upsample_rate * 4));
        if (n_delay > 0) {
            const size_t base = tail_samples.size();
            tail_samples.resize(base + (size_t)n_delay, 0.0f);
        }
    }

    stream_state_.emitted_samples += (int64_t)tail_samples.size();

    stream_state_.active = false;
    stream_state_.left_context_frames = 4;
    stream_state_.lookahead_frames = 4;
    stream_state_.emitted_samples = 0;
    stream_state_.upsample_rate = 1;
    stream_state_.model_delay_samples = -1;
    stream_state_.total_frames = 0;
    stream_state_.history_limit_frames = std::max(16, stream_state_.left_context_frames + stream_state_.lookahead_frames + 16);
    stream_state_.start_fade_samples = 96;
    stream_state_.history_codes.clear();
    stream_state_.pending_codes.clear();
    stream_state_.decode_codes.clear();
    stream_state_.seam_tail_samples.clear();
    return true;
}

bool AudioTokenizerDecoder::set_stream_left_context(int32_t left_context_frames) {
    if (!stream_state_.active) {
        error_msg_ = "Streaming session not started";
        return false;
    }
    if (left_context_frames < 0) {
        left_context_frames = 0;
    }
    stream_state_.left_context_frames = left_context_frames;
    stream_state_.history_limit_frames = std::max(
        stream_state_.history_limit_frames,
        left_context_frames + stream_state_.lookahead_frames + 8
    );
    return true;
}

bool AudioTokenizerDecoder::set_stream_lookahead(int32_t lookahead_frames) {
    if (!stream_state_.active) {
        error_msg_ = "Streaming session not started";
        return false;
    }
    if (lookahead_frames < 0) {
        lookahead_frames = 0;
    }
    stream_state_.lookahead_frames = lookahead_frames;
    stream_state_.history_limit_frames = std::max(
        stream_state_.history_limit_frames,
        stream_state_.left_context_frames + lookahead_frames + 8
    );
    return true;
}

void free_audio_decoder_model(audio_decoder_model & model) {
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
