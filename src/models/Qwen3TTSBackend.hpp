#pragma once

#include "IModelBackend.hpp"
#include "../core/Context.hpp"
#include "../core/Tensor.hpp"
#include "../ops/Ops.hpp"
#include "../memory/KVCache.hpp"
#include "../utils/SafeTensors.hpp"
#include "engine/Types.hpp"
#include <vector>
#include <string>

using bf16 = sycl::ext::oneapi::bfloat16;

class Qwen3TTSBackend : public IModelBackend {
public:
    Qwen3TTSBackend() = default;
    ~Qwen3TTSBackend() override = default;

    bool load(Context& ctx, ModelWeights& weights, const ModelSpec& spec,
              int max_seq_len, std::string* error_message) override;
    
    Tensor& forward(Context& ctx, const int* token_ids_device, int seq_len) override;
    
    void reset() override;
    bool truncate_kv_cache(int new_len) override;
    int max_seq_len() const override { return max_seq_len_; }
    int vocab_size() const override { return talker_cfg_.vocab_size; }
    ModelFamily family() const override { return ModelFamily::Qwen3TTS; }

    // TTS 专有 C++ API：接收文本 tokens 和 预设/克隆的 speaker_embedding (可选)，直接自回归生成 discrete codes
    bool synthesize_codes(Context& ctx,
                          const std::vector<int>& text_tokens,
                          const std::vector<float>& speaker_embedding,  // Base: ECAPA-TDNN embedding (empty if not used)
                          int speaker_id,                                // CustomVoice: spk_id codec token (0 if not used)
                          const std::vector<int>& instruct_tokens,       // VoiceDesign/CustomVoice: instruct text tokens (empty if not used)
                          int language_id,                               // codec language token (0 = auto/nothink)
                          const GenerationConfig& gen_config,
                          std::vector<int32_t>& out_codes,
                          int& out_n_frames);

    bool load_mimi_vocoder(Context& ctx, const std::string& model_dir, std::string* error_message);
    bool decode_mimi_vocoder(Context& ctx,
                             const std::vector<int32_t>& codes,
                             int n_frames,
                             std::vector<float>& out_samples);

    // --- Mimi streaming state (for incremental decode) ---
    struct MimiStreamState {
        int total_frames = 0;
        int max_frames = 512;

        // Pre-transformer KV cache: 8 layers, per-layer [16 heads, max_frames, 64]
        std::vector<Tensor> k_cache;
        std::vector<Tensor> v_cache;

        // Accumulated buffers (full history for conv stages)
        Tensor latent_buffer;   // [total_frames, 512]
        Tensor preconv_buffer;  // [total_frames, 1024]

        void reset() {
            total_frames = 0;
            k_cache.clear();
            v_cache.clear();
            latent_buffer = Tensor();
            preconv_buffer = Tensor();
        }
    };

    // Initialize stream state: allocates KV cache for max_frames
    bool init_mimi_stream(Context& ctx, MimiStreamState& state, int max_frames);

    // Incremental Mimi decode: process new codec frames, append audio to out_samples
    bool decode_mimi_incremental(Context& ctx,
                                  const std::vector<int32_t>& codes,
                                  int new_frames,
                                  MimiStreamState& state,
                                  std::vector<float>& out_samples);

    // Flush remaining conv context into final audio samples
    bool decode_mimi_flush(Context& ctx, MimiStreamState& state,
                           std::vector<float>& out_samples);

private:
    void ensure_talker_runtime_buffers(Context& ctx, int seq_len);
    void ensure_talker_prefill_scores(Context& ctx, int seq_len);
    void ensure_talker_incr_prefill_scores(Context& ctx, int seq_len, int total_len);

    void ensure_predictor_runtime_buffers(Context& ctx, int seq_len);
    void ensure_predictor_prefill_scores(Context& ctx, int seq_len);

    // Qwen3-TTS 两个子模型的配置
    Qwen3Config talker_cfg_{};
    Qwen3Config predictor_cfg_{};
    int max_seq_len_ = 0;
    int current_talker_len_ = 0;

    // ==========================================
    // Talker 权重结构体
    // ==========================================
    struct TalkerLayer {
        Linear qkv_proj, o_proj;
        Linear gate_up_proj, down_proj;
        Tensor* input_ln_weight = nullptr;
        Tensor* post_attn_ln_weight = nullptr;
        Tensor* q_norm_weight = nullptr;
        Tensor* k_norm_weight = nullptr;
    };

    Tensor* talker_codec_embed_weight_ = nullptr; // [3072, H_talker]
    Tensor* talker_text_embed_weight_ = nullptr;  // [151936, 2048]
    
    // ResizeMLP (text_projection)
    Linear talker_text_proj_fc1_;
    Linear talker_text_proj_fc2_;
    Tensor* talker_text_proj_fc1_bias_ = nullptr;
    Tensor* talker_text_proj_fc2_bias_ = nullptr;

    std::vector<TalkerLayer> talker_layers_;
    std::vector<Tensor> talker_fused_weights_;
    Tensor* talker_final_norm_weight_ = nullptr;
    Linear talker_codec_head_;
    KVCache talker_kv_cache_;

    // ==========================================
    // Code Predictor 权重结构体
    // ==========================================
    struct PredictorLayer {
        Linear qkv_proj, o_proj;
        Linear gate_up_proj, down_proj;
        Tensor* input_ln_weight = nullptr;
        Tensor* post_attn_ln_weight = nullptr;
        Tensor* q_norm_weight = nullptr;
        Tensor* k_norm_weight = nullptr;
    };

    // small_to_mtp_projection (只在 1.7B 中非 identity，用于将 2048 维 Talker 状态投射到 1024 维)
    bool has_predictor_projection_ = false;
    Linear predictor_projection_linear_;
    Tensor* predictor_projection_bias_ = nullptr;

    // 15 个 Embedding 权重 ( ModuleList 键：talker.code_predictor.model.codec_embedding.i.weight )
    std::vector<Tensor*> predictor_embed_weights_; // 15 x [2048, H_talker]

    std::vector<PredictorLayer> predictor_layers_;
    std::vector<Tensor> predictor_fused_weights_;
    Tensor* predictor_final_norm_weight_ = nullptr;

    // 15 个 Output Head ( ModuleList 键：talker.code_predictor.lm_head.i.weight )
    std::vector<Linear> predictor_lm_heads_; // 15 x Linear(1024, 2048)

    KVCache predictor_kv_cache_;

    // ==========================================
    // 独立推理运行时缓存 (Talker & Predictor)
    // ==========================================
    struct TalkerBuffers {
        Tensor hidden;
        Tensor normed;
        Tensor qkv;
        Tensor q;
        Tensor k;
        Tensor v;
        Tensor attn_out;
        Tensor gate_up;
        Tensor gate;
        Tensor up;
        Tensor logits;
        Tensor decode_scores;
        Tensor scores;
        Tensor incr_scores;
        Tensor decode_attn_partials;
    } t_buf_;

    struct PredictorBuffers {
        Tensor hidden;       // [17, 1024]
        Tensor normed;       // [17, 1024]
        Tensor qkv;          // [17, qkv_dim]
        Tensor q;            // [17, QD]
        Tensor k;            // [17, KVD]
        Tensor v;            // [17, KVD]
        Tensor attn_out;     // [17, QD]
        Tensor gate_up;      // [17, 2*FF]
        Tensor gate;         // [17, FF]
        Tensor up;           // [17, FF]
        Tensor logits;       // [1, 2048]
        Tensor scores;       // [num_heads, 17, 17] for prefill
        Tensor decode_scores;// [num_heads, 17] for decode
        Tensor decode_attn_partials;
        Tensor pred_input_proj; // [2, 1024] 投影后的输入
    } p_buf_;

    int talker_runtime_seq_capacity_ = 0;
    int talker_prefill_scores_capacity_ = 0;
    int talker_incr_prefill_seq_cap_ = 0;
    int talker_incr_prefill_total_cap_ = 0;

    int predictor_runtime_seq_capacity_ = 0;
    int predictor_prefill_scores_capacity_ = 0;

    // Mimi Decoder (Speech Tokenizer) Vocoder
    bool mimi_loaded_ = false;
    ModelWeights mimi_weights_;

    // TTS model type (Base / CustomVoice / VoiceDesign)
    Qwen3TTSModelType tts_model_type_ = Qwen3TTSModelType::Base;

    // Pre-computed embeddings (bos/eos/pad, computed once during load)
    Tensor precomputed_tts_bos_; // [1, H_talker]
    Tensor precomputed_tts_eos_; // [1, H_talker]
    Tensor precomputed_tts_pad_; // [1, H_talker]
    Tensor precomputed_codec_pad_; // [1, H_talker]
    Tensor precomputed_codec_bos_; // [1, H_talker]
};
