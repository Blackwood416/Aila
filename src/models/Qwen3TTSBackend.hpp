#pragma once

#include "IModelBackend.hpp"
#include "Qwen3TtsTextLayout.hpp"
#include "../core/Context.hpp"
#include "../core/Tensor.hpp"
#include "../ops/Ops.hpp"
#include "../memory/KVCache.hpp"
#include "../utils/SafeTensors.hpp"
#include "engine/Types.hpp"
#include <array>
#include <memory>
#include <vector>
#include <string>
#include <functional>

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
    int get_current_context_len() const override { return current_talker_len_; }

    // TTS 专有 C++ API：接收文本 tokens 和 预设/克隆的 speaker_embedding (可选)，直接自回归生成 discrete codes
    using CodeFrameCallback = std::function<bool(const std::vector<int32_t>& codes,
                                                 int n_frames)>;

    bool synthesize_codes(Context& ctx,
                          const std::vector<int>& text_tokens,
                          const std::vector<float>& speaker_embedding,  // Base: ECAPA-TDNN embedding (empty if not used)
                          int speaker_id,                                // CustomVoice: spk_id codec token (0 if not used)
                          const std::vector<int>& instruct_tokens,       // VoiceDesign/CustomVoice: instruct text tokens (empty if not used)
                          int language_id,                               // codec language token (0 = auto/nothink)
                          const GenerationConfig& gen_config,
                          std::vector<int32_t>& out_codes,
                          int& out_n_frames,
                          std::function<bool()> should_cancel = {},
                          CodeFrameCallback frame_callback = {},
                          int frame_callback_batch_frames = 0);

    // Streaming synthesis: calls audio_callback for each batch of generated audio
    using AudioChunkCallback = std::function<void(const std::vector<float>& samples)>;

    enum class TtsStreamStepResult {
        Continue,
        Eos,
        Error,
    };

    bool synthesize_codes_stream(Context& ctx,
                                  const std::vector<int>& text_tokens,
                                  const std::vector<float>& speaker_embedding,
                                  int speaker_id,
                                  const std::vector<int>& instruct_tokens,
                                  int language_id,
                                  const GenerationConfig& gen_config,
                                  int stream_batch_frames,
                                  AudioChunkCallback audio_callback,
                                  std::function<bool()> should_cancel = {});

    bool synthesize_tts_stream(
        Context& ctx,
        const std::vector<int>& text_tokens,
        const GenerationConfig& gen_config,
        int stream_batch_frames,
        std::function<void(const std::vector<float>&)> audio_callback,
        std::string* error_message = nullptr,
        std::function<bool()> should_cancel = {}) override;
    TtsBackendTiming last_tts_backend_timing() const override { return last_tts_timing_; }

    bool load_mimi_vocoder(Context& ctx, const std::string& model_dir, std::string* error_message);
    bool decode_mimi_vocoder(Context& ctx,
                             const std::vector<int32_t>& codes,
                             int n_frames,
                             std::vector<float>& out_samples);

    // --- Mimi streaming state (for incremental decode) ---
    struct MimiStreamState {
        int total_frames = 0;
        int max_frames = 512;

        // Accumulated latent history for full-history conv/pre-transformer stages.
        Tensor latent_buffer;   // [total_frames, 512]
        Tensor pre_tfm_out_buffer; // [total_frames, 1024]
        std::array<Tensor, 8> pre_tfm_k_cache; // [16, max_frames, 64]
        std::array<Tensor, 8> pre_tfm_v_cache; // [16, max_frames, 64]

        // Track previous audio output position for incremental slicing
        int last_audio_sample_count = 0;

        void reset() {
            total_frames = 0;
            latent_buffer = Tensor();
            pre_tfm_out_buffer = Tensor();
            for (auto& cache : pre_tfm_k_cache) cache = Tensor();
            for (auto& cache : pre_tfm_v_cache) cache = Tensor();
            last_audio_sample_count = 0;
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

    // Stateful text-chunk streaming session. The session owns Talker KV,
    // trailing text hidden, codec history, and Mimi state so that text chunks
    // appended after begin() continue the same utterance instead of resetting.
    bool tts_stream_begin(Context& ctx,
                          const std::vector<int>& text_tokens,
                          const std::vector<float>& speaker_embedding,
                          int speaker_id,
                          const std::vector<int>& instruct_tokens,
                          int language_id,
                          const GenerationConfig& gen_config,
                          int stream_batch_frames);
    bool tts_stream_append_text(Context& ctx,
                                const std::vector<int>& body_tokens);
    TtsStreamStepResult tts_stream_step(Context& ctx,
                                        std::vector<float>& audio_out);
    bool tts_stream_finish(Context& ctx);
    void tts_stream_reset();
    bool tts_stream_session_active() const {
        return stream_session_ != nullptr && stream_session_->active;
    }
    int tts_stream_session_eos_suppressed() const {
        return stream_session_ ? stream_session_->eos_suppressed_count : 0;
    }
    bool tts_stream_session_has_unconsumed_text() const;

private:
    struct TtsStreamSession {
        bool active = false;
        bool finishing = false;
        bool eos_reached = false;
        int gen_step = 0;
        int trailing_len = 0;
        int max_tokens = 0;
        int token = -1;
        int eos_id = 0;
        int stream_batch_frames = 4;
        int initial_callback_batch_frames = 4;
        int steady_callback_batch_frames = 4;
        int pending_callback_frames = 0;
        int pending_callback_real_frames = 0;
        int callback_batches_emitted = 0;
        int eos_suppressed_count = 0;
        bool playback_aware_steady_batch = false;
        bool use_steady_callback_batch = false;
        GenerationConfig gen_config{};
        aila::Qwen3TtsStreamTextState text_state;
        std::vector<int> generated_cb0_tokens;
        std::vector<int32_t> pending_callback_codes;
        std::vector<int> token_upload_storage;
        size_t token_upload_index = 0;
        Tensor past_hidden_talker;
        Tensor trailing_text_hidden;
        Tensor pred_input;
        Tensor token_dev;
        Tensor frame_codes_dev;
        Tensor last_id_hidden;
        Tensor predictor_final_hidden;
        Tensor predictor_final_normed;
        Tensor predictor_emb_h;
        Tensor predictor_emb_pred;
        Tensor predictor_step_normed;
        Tensor sum_emb;
        Tensor single_emb;
        Tensor step_normed_talker;
        Tensor final_normed_talker;
        MimiStreamState mimi_state;
    };
    std::unique_ptr<TtsStreamSession> stream_session_;

    // Conv stages of Mimi decoder (shared by full and incremental paths).
    // Takes pre-transformer output [n_frames, 1024], produces float PCM audio.
    bool mimi_conv_stages(Context& ctx, Tensor& pre_tfm_out, int n_frames,
                          std::vector<float>& out_samples,
                          int tail_samples = -1);
    void init_mimi_runtime_linears(Context& ctx);
    void ensure_talker_runtime_buffers(Context& ctx, int seq_len);
    void ensure_talker_prefill_scores(Context& ctx, int seq_len);
    void ensure_talker_incr_prefill_scores(Context& ctx, int seq_len, int total_len);
    bool project_text_tokens(Context& ctx,
                             const std::vector<int>& tokens,
                             Tensor& out);

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
    bool mimi_runtime_linears_initialized_ = false;
    Tensor mimi_first_proj_weight_view_;
    Tensor mimi_rest_proj_weight_view_;
    Linear mimi_first_proj_;
    Linear mimi_rest_proj_;
    Linear mimi_pre_tfm_in_proj_;
    Linear mimi_pre_tfm_out_proj_;
    struct MimiPreTransformerLayerLinears {
        Linear q_proj;
        Linear k_proj;
        Linear v_proj;
        Linear o_proj;
        Linear gate_proj;
        Linear up_proj;
        Linear down_proj;
    };
    std::array<MimiPreTransformerLayerLinears, 8> mimi_pre_tfm_linears_;
    struct MimiUpsampleLinears {
        Linear pwconv1;
        Linear pwconv2;
    };
    std::array<MimiUpsampleLinears, 2> mimi_upsample_linears_;

    // TTS model type (Base / CustomVoice / VoiceDesign)
    Qwen3TTSModelType tts_model_type_ = Qwen3TTSModelType::Base;

    // Pre-computed embeddings (bos/eos/pad, computed once during load)
    Tensor precomputed_tts_bos_; // [1, H_talker]
    Tensor precomputed_tts_eos_; // [1, H_talker]
    Tensor precomputed_tts_pad_; // [1, H_talker]
    Tensor precomputed_codec_pad_; // [1, H_talker]
    Tensor precomputed_codec_bos_; // [1, H_talker]
    TtsBackendTiming last_tts_timing_;
};
