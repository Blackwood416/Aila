#include "models/cpu/CpuQ35HybridModel.hpp"
#include "utils/EnvUtils.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <utility>

namespace {

void set_error(std::string* error, const std::string& message) {
    if (error) {
        *error = message;
    }
}

int64_t tensor_numel(const CpuTensorView& tensor) {
    int64_t result = 1;
    for (int64_t dim : tensor.shape) {
        result *= dim;
    }
    return result;
}

float bf16_to_float(const uint8_t* data) {
    uint16_t raw = 0;
    std::memcpy(&raw, data, sizeof(raw));
    const uint32_t bits = static_cast<uint32_t>(raw) << 16;
    float out = 0.0f;
    std::memcpy(&out, &bits, sizeof(out));
    return out;
}

float f16_to_float(const uint8_t* data) {
    uint16_t h = 0;
    std::memcpy(&h, data, sizeof(h));
    const uint32_t sign = (static_cast<uint32_t>(h & 0x8000u)) << 16;
    uint32_t exp = (h >> 10) & 0x1fu;
    uint32_t mant = h & 0x03ffu;
    uint32_t bits = 0;
    if (exp == 0) {
        if (mant == 0) {
            bits = sign;
        } else {
            exp = 1;
            while ((mant & 0x0400u) == 0) {
                mant <<= 1;
                --exp;
            }
            mant &= 0x03ffu;
            bits = sign | ((exp + 127 - 15) << 23) | (mant << 13);
        }
    } else if (exp == 0x1fu) {
        bits = sign | 0x7f800000u | (mant << 13);
    } else {
        bits = sign | ((exp + 127 - 15) << 23) | (mant << 13);
    }
    float out = 0.0f;
    std::memcpy(&out, &bits, sizeof(out));
    return out;
}

float tensor_value_as_float(const CpuTensorView& tensor, int64_t index) {
    switch (tensor.dtype) {
        case CpuDataType::F32:
            return tensor.f32_data()[static_cast<size_t>(index)];
        case CpuDataType::BF16:
            return bf16_to_float(tensor.data + static_cast<size_t>(index) * 2);
        case CpuDataType::F16:
            return f16_to_float(tensor.data + static_cast<size_t>(index) * 2);
        default:
            throw std::runtime_error("CPU Qwen3.5 tensor dtype is not float-compatible: " +
                                     tensor.name);
    }
}

bool load_float_vector(const CpuSafetensorsStore& store,
                       const std::string& name,
                       int64_t expected_numel,
                       std::vector<float>& out,
                       std::string* error) {
    if (!store.has(name)) {
        set_error(error, "Missing required CPU Qwen3.5 tensor: " + name);
        return false;
    }
    const CpuTensorView& tensor = store.get(name);
    const int64_t actual_numel = tensor_numel(tensor);
    if (actual_numel != expected_numel) {
        set_error(error, "CPU Qwen3.5 tensor has unexpected shape: " + name);
        return false;
    }
    out.resize(static_cast<size_t>(expected_numel));
    try {
        for (int64_t i = 0; i < expected_numel; ++i) {
            out[static_cast<size_t>(i)] = tensor_value_as_float(tensor, i);
        }
    } catch (const std::exception& e) {
        set_error(error, e.what());
        return false;
    }
    return true;
}

bool load_bnb_ref(const CpuSafetensorsStore& store,
                  const std::string& name,
                  CpuBnb4WeightRef& out,
                  CpuBnb4CacheMode cache_mode,
                  std::string* error) {
    if (!load_cpu_bnb4_weight_ref(store, name, out, error, cache_mode)) {
        return false;
    }
    return true;
}

void add_residual_and_q35_norm(std::vector<float>& hidden,
                               const std::vector<float>& residual,
                               const std::vector<float>& raw_weight,
                               float eps,
                               std::vector<float>& output) {
    const int n = static_cast<int>(hidden.size());
    for (int i = 0; i < n; ++i) {
        hidden[static_cast<size_t>(i)] += residual[static_cast<size_t>(i)];
    }
    cpu_q35::q35_rms_norm(hidden.data(), raw_weight.data(), n, eps, output.data());
}

void head_l2_norm(std::vector<float>& x,
                  int num_heads,
                  int head_dim,
                  float eps) {
    for (int h = 0; h < num_heads; ++h) {
        float* v = x.data() + static_cast<size_t>(h) * head_dim;
        float sum_sq = 0.0f;
        for (int d = 0; d < head_dim; ++d) {
            sum_sq += v[d] * v[d];
        }
        const float inv = 1.0f / std::sqrt(sum_sq + eps);
        for (int d = 0; d < head_dim; ++d) {
            v[d] *= inv;
        }
    }
}

void head_q35_rms_norm(std::vector<float>& x,
                       const std::vector<float>& raw_weight,
                       int num_heads,
                       int head_dim,
                       float eps) {
    for (int h = 0; h < num_heads; ++h) {
        float* v = x.data() + static_cast<size_t>(h) * head_dim;
        cpu_q35::q35_rms_norm(v, raw_weight.data(), head_dim, eps, v);
    }
}

int linear_delta_gqa_group_size(int q_heads, int kv_heads) {
    if (q_heads <= 0 || kv_heads <= 0 || (kv_heads % q_heads) != 0) {
        return 0;
    }
    return kv_heads / q_heads;
}

int linear_delta_key_head_for_value_head(int value_head, int q_heads, int kv_heads) {
    const int group_size = linear_delta_gqa_group_size(q_heads, kv_heads);
    if (group_size <= 0) {
        return std::clamp(value_head, 0, std::max(0, q_heads - 1));
    }
    return std::min(value_head / group_size, q_heads - 1);
}

float softplus(float x) {
    if (x > 20.0f) {
        return x;
    }
    if (x < -20.0f) {
        return std::exp(x);
    }
    return std::log1p(std::exp(x));
}

void apply_rope_partial_one(std::vector<float>& q,
                            std::vector<float>& k,
                            int q_heads,
                            int kv_heads,
                            int head_dim,
                            int rotary_dim,
                            int position,
                            float theta) {
    int rot = std::max(2, std::min(head_dim, rotary_dim));
    if (rot & 1) {
        --rot;
    }
    const int half_dim = rot / 2;
    for (int h = 0; h < q_heads; ++h) {
        float* qh = q.data() + static_cast<size_t>(h) * head_dim;
        float* kh = (h < kv_heads) ? k.data() + static_cast<size_t>(h) * head_dim
                                   : nullptr;
        for (int d = 0; d < half_dim; ++d) {
            const float freq =
                1.0f / std::pow(theta, (2.0f * static_cast<float>(d)) /
                                            static_cast<float>(rot));
            const float angle = static_cast<float>(position) * freq;
            const float c = std::cos(angle);
            const float s = std::sin(angle);

            const int i0 = d;
            const int i1 = d + half_dim;
            const float q0 = qh[i0];
            const float q1 = qh[i1];
            qh[i0] = q0 * c - q1 * s;
            qh[i1] = q1 * c + q0 * s;

            if (kh) {
                const float k0 = kh[i0];
                const float k1 = kh[i1];
                kh[i0] = k0 * c - k1 * s;
                kh[i1] = k1 * c + k0 * s;
            }
        }
    }
}

}  // namespace

namespace cpu_q35 {

float silu(float x) {
    return x / (1.0f + std::exp(-x));
}

void q35_rms_norm(const float* input,
                  const float* raw_weight,
                  int n,
                  float eps,
                  float* output) {
    float sum_sq = 0.0f;
    for (int i = 0; i < n; ++i) {
        sum_sq += input[i] * input[i];
    }
    const float inv_rms = 1.0f / std::sqrt(sum_sq / static_cast<float>(n) + eps);
    for (int i = 0; i < n; ++i) {
        output[i] = input[i] * inv_rms * (1.0f + raw_weight[i]);
    }
}

void sigmoid_gate(const float* input,
                  const float* gate,
                  int n,
                  float* output) {
    for (int i = 0; i < n; ++i) {
        output[i] = input[i] / (1.0f + std::exp(-gate[i]));
    }
}

}  // namespace cpu_q35

bool CpuQ35HybridModel::load(const std::string& model_dir,
                             const ModelSpec& spec,
                             int max_seq_len,
                             std::string* error) {
    auto store = std::make_unique<CpuSafetensorsStore>();
    std::string store_error;
    if (!store->load_from_dir(model_dir, &store_error)) {
        clear_loaded();
        set_error(error, store_error.empty()
                             ? "CPU Qwen3.5 safetensors load failed"
                             : "CPU Qwen3.5 safetensors load failed: " + store_error);
        return false;
    }

    CpuSafetensorsStore* store_ptr = store.get();
    if (!load_from_store(*store_ptr, spec, max_seq_len, error)) {
        return false;
    }
    owned_store_ = std::move(store);
    store_ = store_ptr;
    return true;
}

bool CpuQ35HybridModel::load_from_store(const CpuSafetensorsStore& store,
                                        const ModelSpec& spec,
                                        int max_seq_len,
                                        std::string* error) {
    clear_loaded();
    if (error) {
        error->clear();
    }

    if (spec.family != ModelFamily::Qwen35Hybrid ||
        !is_supported_qwen35_hybrid_0p8b_spec(spec.qwen35_text)) {
        set_error(error, "CPU Qwen3.5 backend supports only 0.8B Qwen3.5 hybrid text specs");
        return false;
    }
    if (!spec.is_bitsandbytes_4bit()) {
        set_error(error, "CPU Qwen3.5 backend requires a bitsandbytes 4-bit checkpoint");
        return false;
    }
    if (max_seq_len <= 0) {
        set_error(error, "CPU Qwen3.5 max_seq_len must be positive");
        return false;
    }

    constexpr const char* kEmbedWeight = "model.language_model.embed_tokens.weight";
    if (!store.has(kEmbedWeight)) {
        set_error(error, std::string("Missing required CPU Qwen3.5 tensor: ") + kEmbedWeight);
        return false;
    }

    store_ = &store;
    embed_weight_ = &store.get(kEmbedWeight);
    spec_ = spec;
    cfg_ = spec.qwen35_text;
    max_seq_len_ = max_seq_len;
    weight_cache_mode_ = parse_cpu_bnb4_cache_mode(
        aila::env::read_string("AILA_CPU_Q35_WEIGHT_CACHE", "fp16"));

    hidden_size_ = cfg_.hidden_size;
    ff_dim_ = cfg_.intermediate_size;
    full_q_heads_ = cfg_.num_attention_heads;
    full_kv_heads_ = cfg_.num_key_value_heads;
    full_head_dim_ = cfg_.head_dim;
    full_q_dim_ = full_q_heads_ * full_head_dim_;
    full_kv_dim_ = full_kv_heads_ * full_head_dim_;
    full_q_proj_dim_ = cfg_.attn_output_gate ? (2 * full_q_dim_) : full_q_dim_;

    linear_q_heads_ = cfg_.linear_num_key_heads;
    linear_kv_heads_ = cfg_.linear_num_value_heads;
    linear_head_dim_ = cfg_.linear_key_head_dim;
    linear_value_head_dim_ = cfg_.linear_value_head_dim;
    linear_q_dim_ = linear_q_heads_ * linear_head_dim_;
    linear_kv_dim_ = linear_kv_heads_ * linear_value_head_dim_;
    linear_qkv_dim_ = linear_q_dim_ + linear_q_dim_ + linear_kv_dim_;
    linear_z_dim_ = linear_kv_dim_;
    linear_conv_kernel_dim_ = std::max(1, cfg_.linear_conv_kernel_dim);
    linear_conv_channels_ = linear_qkv_dim_;

    full_rotary_dim_ =
        std::max(2, static_cast<int>(std::floor(full_head_dim_ * cfg_.rope.partial_rotary_factor)));
    full_rotary_dim_ = std::min(full_head_dim_, full_rotary_dim_);
    if (full_rotary_dim_ & 1) {
        --full_rotary_dim_;
    }

    if (embed_weight_->shape.size() != 2 ||
        embed_weight_->shape[0] != cfg_.vocab_size ||
        embed_weight_->shape[1] != hidden_size_) {
        set_error(error, "CPU Qwen3.5 embedding tensor has unexpected shape");
        return false;
    }

    const int64_t embed_numel =
        static_cast<int64_t>(cfg_.vocab_size) * static_cast<int64_t>(hidden_size_);
    embedding_f16_.resize(static_cast<size_t>(embed_numel));
    try {
        for (int64_t i = 0; i < embed_numel; ++i) {
            embedding_f16_[static_cast<size_t>(i)] =
                cpu_float_to_f16(tensor_value_as_float(*embed_weight_, i));
        }
    } catch (const std::exception& e) {
        set_error(error, e.what());
        return false;
    }

    if (!load_layers(error)) {
        clear_loaded();
        return false;
    }

    hidden_.assign(static_cast<size_t>(hidden_size_), 0.0f);
    normed_.assign(static_cast<size_t>(hidden_size_), 0.0f);
    mixer_out_.assign(static_cast<size_t>(hidden_size_), 0.0f);
    mlp_out_.assign(static_cast<size_t>(hidden_size_), 0.0f);
    loaded_ = true;
    reset_runtime_state();
    return true;
}

void CpuQ35HybridModel::reset() {
    if (loaded_) {
        reset_runtime_state();
    }
}

size_t CpuQ35HybridModel::dense_weight_cache_bytes() const {
    return embedding_cache_bytes() + projection_cache_bytes();
}

size_t CpuQ35HybridModel::embedding_cache_bytes() const {
    return embedding_f16_.size() * sizeof(uint16_t);
}

size_t CpuQ35HybridModel::projection_cache_bytes() const {
    size_t bytes = 0;
    const auto add_weight = [&bytes](const CpuBnb4WeightRef& weight) {
        bytes += weight.cache_bytes();
    };
    for (const CpuQ35Layer& layer : layers_) {
        if (layer.is_linear) {
            add_weight(layer.linear_qkv_proj);
            add_weight(layer.linear_z_proj);
            add_weight(layer.linear_a_proj);
            add_weight(layer.linear_b_proj);
            add_weight(layer.linear_o_proj);
        } else {
            add_weight(layer.full_q_proj);
            add_weight(layer.full_k_proj);
            add_weight(layer.full_v_proj);
            add_weight(layer.full_o_proj);
        }
        add_weight(layer.mlp_gate_proj);
        add_weight(layer.mlp_up_proj);
        add_weight(layer.mlp_down_proj);
    }
    return bytes;
}

void CpuQ35HybridModel::clear_loaded() {
    loaded_ = false;
    weight_cache_mode_ = CpuBnb4CacheMode::Fp16;
    store_ = nullptr;
    owned_store_.reset();
    embed_weight_ = nullptr;
    spec_ = {};
    cfg_ = {};
    max_seq_len_ = 0;
    current_len_ = 0;
    hidden_size_ = 0;
    ff_dim_ = 0;
    full_q_heads_ = 0;
    full_kv_heads_ = 0;
    full_head_dim_ = 0;
    full_q_dim_ = 0;
    full_kv_dim_ = 0;
    full_q_proj_dim_ = 0;
    linear_q_heads_ = 0;
    linear_kv_heads_ = 0;
    linear_head_dim_ = 0;
    linear_value_head_dim_ = 0;
    linear_q_dim_ = 0;
    linear_kv_dim_ = 0;
    linear_qkv_dim_ = 0;
    linear_z_dim_ = 0;
    linear_conv_kernel_dim_ = 0;
    linear_conv_channels_ = 0;
    full_rotary_dim_ = 0;
    layers_.clear();
    layer_caches_.clear();
    final_norm_weight_.clear();
    embedding_f16_.clear();
    hidden_.clear();
    normed_.clear();
    mixer_out_.clear();
    mlp_out_.clear();
    scratch_a_.clear();
    scratch_b_.clear();
    scratch_c_.clear();
    scratch_d_.clear();
    scratch_e_.clear();
}

void CpuQ35HybridModel::reset_runtime_state() {
    current_len_ = 0;
    for (CpuQ35LayerCache& cache : layer_caches_) {
        std::fill(cache.full_k.begin(), cache.full_k.end(), 0.0f);
        std::fill(cache.full_v.begin(), cache.full_v.end(), 0.0f);
        std::fill(cache.linear_state.begin(), cache.linear_state.end(), 0.0f);
        std::fill(cache.linear_conv_state.begin(), cache.linear_conv_state.end(), 0.0f);
    }
}

bool CpuQ35HybridModel::load_layers(std::string* error) {
    layers_.clear();
    layer_caches_.clear();
    layers_.resize(static_cast<size_t>(cfg_.num_hidden_layers));
    layer_caches_.resize(static_cast<size_t>(cfg_.num_hidden_layers));

    for (int i = 0; i < cfg_.num_hidden_layers; ++i) {
        CpuQ35Layer& layer = layers_[static_cast<size_t>(i)];
        CpuQ35LayerCache& cache = layer_caches_[static_cast<size_t>(i)];
        const std::string prefix =
            "model.language_model.layers." + std::to_string(i) + ".";

        if (!load_float_vector(*store_, prefix + "input_layernorm.weight",
                               hidden_size_, layer.input_ln_weight, error) ||
            !load_float_vector(*store_, prefix + "post_attention_layernorm.weight",
                               hidden_size_, layer.post_attn_ln_weight, error)) {
            return false;
        }

        bool is_linear = true;
        if (i < static_cast<int>(cfg_.layer_types.size())) {
            is_linear = (cfg_.layer_types[static_cast<size_t>(i)] == "linear_attention");
        } else {
            is_linear = ((i + 1) % 4 != 0);
        }
        layer.is_linear = is_linear;

        if (is_linear) {
            if (!load_bnb_ref(*store_, prefix + "linear_attn.in_proj_qkv.weight",
                              layer.linear_qkv_proj, weight_cache_mode_, error) ||
                !load_bnb_ref(*store_, prefix + "linear_attn.in_proj_z.weight",
                              layer.linear_z_proj, weight_cache_mode_, error) ||
                !load_bnb_ref(*store_, prefix + "linear_attn.in_proj_a.weight",
                              layer.linear_a_proj, weight_cache_mode_, error) ||
                !load_bnb_ref(*store_, prefix + "linear_attn.in_proj_b.weight",
                              layer.linear_b_proj, weight_cache_mode_, error) ||
                !load_bnb_ref(*store_, prefix + "linear_attn.out_proj.weight",
                              layer.linear_o_proj, weight_cache_mode_, error) ||
                !load_float_vector(*store_, prefix + "linear_attn.A_log",
                                   linear_kv_heads_, layer.linear_A_negexp, error) ||
                !load_float_vector(*store_, prefix + "linear_attn.dt_bias",
                                   linear_kv_heads_, layer.linear_dt_bias, error) ||
                !load_float_vector(*store_, prefix + "linear_attn.norm.weight",
                                   linear_value_head_dim_, layer.linear_norm_weight, error) ||
                !load_float_vector(*store_, prefix + "linear_attn.conv1d.weight",
                                   static_cast<int64_t>(linear_conv_channels_) *
                                       linear_conv_kernel_dim_,
                                   layer.linear_conv, error)) {
                return false;
            }
            for (float& value : layer.linear_A_negexp) {
                value = -std::exp(value);
            }

            cache.linear_state.assign(
                static_cast<size_t>(linear_kv_heads_) *
                    static_cast<size_t>(linear_head_dim_) *
                    static_cast<size_t>(linear_value_head_dim_),
                0.0f);
            cache.linear_conv_state.assign(
                static_cast<size_t>(std::max(0, linear_conv_kernel_dim_ - 1)) *
                    static_cast<size_t>(linear_conv_channels_),
                0.0f);
        } else {
            if (!load_bnb_ref(*store_, prefix + "self_attn.q_proj.weight",
                              layer.full_q_proj, weight_cache_mode_, error) ||
                !load_bnb_ref(*store_, prefix + "self_attn.k_proj.weight",
                              layer.full_k_proj, weight_cache_mode_, error) ||
                !load_bnb_ref(*store_, prefix + "self_attn.v_proj.weight",
                              layer.full_v_proj, weight_cache_mode_, error) ||
                !load_bnb_ref(*store_, prefix + "self_attn.o_proj.weight",
                              layer.full_o_proj, weight_cache_mode_, error) ||
                !load_float_vector(*store_, prefix + "self_attn.q_norm.weight",
                                   full_head_dim_, layer.q_norm_weight, error) ||
                !load_float_vector(*store_, prefix + "self_attn.k_norm.weight",
                                   full_head_dim_, layer.k_norm_weight, error)) {
                return false;
            }
            cache.full_k.assign(
                static_cast<size_t>(full_kv_heads_) *
                    static_cast<size_t>(max_seq_len_) *
                    static_cast<size_t>(full_head_dim_),
                0.0f);
            cache.full_v.assign(cache.full_k.size(), 0.0f);
        }

        if (!load_bnb_ref(*store_, prefix + "mlp.gate_proj.weight",
                          layer.mlp_gate_proj, weight_cache_mode_, error) ||
            !load_bnb_ref(*store_, prefix + "mlp.up_proj.weight",
                          layer.mlp_up_proj, weight_cache_mode_, error) ||
            !load_bnb_ref(*store_, prefix + "mlp.down_proj.weight",
                          layer.mlp_down_proj, weight_cache_mode_, error)) {
            return false;
        }
    }

    return load_float_vector(*store_,
                             "model.language_model.norm.weight",
                             hidden_size_,
                             final_norm_weight_,
                             error);
}

bool CpuQ35HybridModel::consume_one(int token_id, std::string* error) {
    return forward_one_impl(token_id, nullptr, error);
}

bool CpuQ35HybridModel::forward_one(int token_id,
                                    std::vector<float>& logits,
                                    std::string* error) {
    return forward_one_impl(token_id, &logits, error);
}

bool CpuQ35HybridModel::forward_one_impl(int token_id,
                                         std::vector<float>* logits,
                                         std::string* error) {
    if (!loaded_) {
        set_error(error, "CPU Qwen3.5 forward_one called before load");
        return false;
    }
    if (token_id < 0 || token_id >= cfg_.vocab_size) {
        set_error(error, "CPU Qwen3.5 token id is outside vocabulary");
        return false;
    }
    if (current_len_ >= max_seq_len_) {
        set_error(error, "CPU Qwen3.5 context window exceeded");
        return false;
    }

    try {
        const uint16_t* embed_row =
            embedding_f16_.data() + static_cast<size_t>(token_id) * hidden_size_;
        cpu_f16_to_f32(embed_row, hidden_.data(), hidden_size_);
        cpu_q35::q35_rms_norm(hidden_.data(),
                              layers_[0].input_ln_weight.data(),
                              hidden_size_,
                              cfg_.rms_norm_eps,
                              normed_.data());

        for (int i = 0; i < cfg_.num_hidden_layers; ++i) {
            CpuQ35Layer& layer = layers_[static_cast<size_t>(i)];
            CpuQ35LayerCache& cache = layer_caches_[static_cast<size_t>(i)];

            bool ok = layer.is_linear
                ? run_linear_attention(layer, cache, normed_, mixer_out_)
                : run_full_attention(layer, cache, normed_, mixer_out_);
            if (!ok) {
                set_error(error, "CPU Qwen3.5 layer mixer failed");
                return false;
            }

            add_residual_and_q35_norm(hidden_,
                                      mixer_out_,
                                      layer.post_attn_ln_weight,
                                      cfg_.rms_norm_eps,
                                      normed_);

            if (!run_mlp(layer, normed_, mlp_out_)) {
                set_error(error, "CPU Qwen3.5 MLP failed");
                return false;
            }

            const std::vector<float>& next_weight =
                (i < cfg_.num_hidden_layers - 1)
                    ? layers_[static_cast<size_t>(i + 1)].input_ln_weight
                    : final_norm_weight_;
            add_residual_and_q35_norm(hidden_,
                                      mlp_out_,
                                      next_weight,
                                      cfg_.rms_norm_eps,
                                      normed_);
        }

        if (logits) {
            if (!compute_logits(normed_, *logits, error)) {
                return false;
            }
        }
        ++current_len_;
        return true;
    } catch (const std::exception& e) {
        set_error(error, std::string("CPU Qwen3.5 forward_one failed: ") + e.what());
        return false;
    }
}

bool CpuQ35HybridModel::run_linear_attention(CpuQ35Layer& layer,
                                             CpuQ35LayerCache& cache,
                                             const std::vector<float>& input,
                                             std::vector<float>& output) {
    scratch_a_.assign(static_cast<size_t>(linear_qkv_dim_), 0.0f);
    scratch_b_.assign(static_cast<size_t>(linear_z_dim_), 0.0f);
    scratch_c_.assign(static_cast<size_t>(linear_kv_heads_), 0.0f);
    scratch_d_.assign(static_cast<size_t>(linear_kv_heads_), 0.0f);

    cpu_bnb4_matvec(layer.linear_qkv_proj, input.data(), scratch_a_.data());
    cpu_bnb4_matvec(layer.linear_z_proj, input.data(), scratch_b_.data());
    cpu_bnb4_matvec(layer.linear_a_proj, input.data(), scratch_c_.data());
    cpu_bnb4_matvec(layer.linear_b_proj, input.data(), scratch_d_.data());

    std::vector<float>& conv_out = scratch_e_;
    conv_out.assign(static_cast<size_t>(linear_qkv_dim_), 0.0f);
    for (int c = 0; c < linear_qkv_dim_; ++c) {
        const float* w = layer.linear_conv.data() +
                         static_cast<size_t>(c) * linear_conv_kernel_dim_;
        float v = 0.0f;
        for (int j = 0; j < linear_conv_kernel_dim_ - 1; ++j) {
            v += cache.linear_conv_state[static_cast<size_t>(j) * linear_qkv_dim_ + c] *
                 w[j];
        }
        v += scratch_a_[static_cast<size_t>(c)] * w[linear_conv_kernel_dim_ - 1];
        conv_out[static_cast<size_t>(c)] = cpu_q35::silu(v);
    }
    if (linear_conv_kernel_dim_ > 1) {
        const int rows = linear_conv_kernel_dim_ - 1;
        if (rows > 1) {
            std::memmove(cache.linear_conv_state.data(),
                         cache.linear_conv_state.data() + linear_qkv_dim_,
                         static_cast<size_t>(rows - 1) *
                             static_cast<size_t>(linear_qkv_dim_) *
                             sizeof(float));
        }
        std::memcpy(cache.linear_conv_state.data() +
                        static_cast<size_t>(rows - 1) * linear_qkv_dim_,
                    scratch_a_.data(),
                    static_cast<size_t>(linear_qkv_dim_) * sizeof(float));
    }

    std::vector<float> q(static_cast<size_t>(linear_q_dim_));
    std::vector<float> k(static_cast<size_t>(linear_q_dim_));
    std::vector<float> v(static_cast<size_t>(linear_kv_dim_));
    std::memcpy(q.data(), conv_out.data(), static_cast<size_t>(linear_q_dim_) * sizeof(float));
    std::memcpy(k.data(), conv_out.data() + linear_q_dim_,
                static_cast<size_t>(linear_q_dim_) * sizeof(float));
    std::memcpy(v.data(), conv_out.data() + linear_q_dim_ + linear_q_dim_,
                static_cast<size_t>(linear_kv_dim_) * sizeof(float));

    head_l2_norm(q, linear_q_heads_, linear_head_dim_, cfg_.rms_norm_eps);
    head_l2_norm(k, linear_q_heads_, linear_head_dim_, cfg_.rms_norm_eps);

    std::vector<float> attn_out(static_cast<size_t>(linear_kv_dim_), 0.0f);
    std::vector<float> sk(static_cast<size_t>(linear_value_head_dim_), 0.0f);
    std::vector<float> d(static_cast<size_t>(linear_value_head_dim_), 0.0f);
    const float scale = 1.0f / std::sqrt(static_cast<float>(linear_head_dim_));

    for (int hv = 0; hv < linear_kv_heads_; ++hv) {
        const int hk = linear_delta_key_head_for_value_head(hv, linear_q_heads_, linear_kv_heads_);
        const float* q_h = q.data() + static_cast<size_t>(hk) * linear_head_dim_;
        const float* k_h = k.data() + static_cast<size_t>(hk) * linear_head_dim_;
        const float* v_h = v.data() + static_cast<size_t>(hv) * linear_value_head_dim_;

        const float beta = 1.0f / (1.0f + std::exp(-scratch_d_[static_cast<size_t>(hv)]));
        const float alpha = scratch_c_[static_cast<size_t>(hv)];
        const float g_pre =
            softplus(alpha + layer.linear_dt_bias[static_cast<size_t>(hv)]) *
            layer.linear_A_negexp[static_cast<size_t>(hv)];
        const float decay = std::exp(g_pre);

        float* state = cache.linear_state.data() +
                       static_cast<size_t>(hv) *
                           static_cast<size_t>(linear_head_dim_) *
                           static_cast<size_t>(linear_value_head_dim_);
        for (int idx = 0; idx < linear_head_dim_ * linear_value_head_dim_; ++idx) {
            state[idx] *= decay;
        }

        for (int dv = 0; dv < linear_value_head_dim_; ++dv) {
            float acc = 0.0f;
            for (int j = 0; j < linear_head_dim_; ++j) {
                acc += state[static_cast<size_t>(j) * linear_value_head_dim_ + dv] *
                       k_h[j];
            }
            sk[static_cast<size_t>(dv)] = acc;
            d[static_cast<size_t>(dv)] = (v_h[dv] - acc) * beta;
        }

        for (int j = 0; j < linear_head_dim_; ++j) {
            float* row = state + static_cast<size_t>(j) * linear_value_head_dim_;
            const float kj = k_h[j];
            for (int dv = 0; dv < linear_value_head_dim_; ++dv) {
                row[dv] += kj * d[static_cast<size_t>(dv)];
            }
        }

        float* o_h = attn_out.data() + static_cast<size_t>(hv) * linear_value_head_dim_;
        for (int dv = 0; dv < linear_value_head_dim_; ++dv) {
            float acc = 0.0f;
            for (int j = 0; j < linear_head_dim_; ++j) {
                acc += state[static_cast<size_t>(j) * linear_value_head_dim_ + dv] *
                       q_h[j];
            }
            o_h[dv] = acc * scale;
        }
    }

    for (int hv = 0; hv < linear_kv_heads_; ++hv) {
        float* out_h = attn_out.data() + static_cast<size_t>(hv) * linear_value_head_dim_;
        const float* z_h = scratch_b_.data() + static_cast<size_t>(hv) * linear_value_head_dim_;
        float sum_sq = 0.0f;
        for (int d0 = 0; d0 < linear_value_head_dim_; ++d0) {
            sum_sq += out_h[d0] * out_h[d0];
        }
        const float inv =
            1.0f / std::sqrt(sum_sq / static_cast<float>(linear_value_head_dim_) +
                             cfg_.rms_norm_eps);
        for (int d0 = 0; d0 < linear_value_head_dim_; ++d0) {
            const float n =
                out_h[d0] * inv * layer.linear_norm_weight[static_cast<size_t>(d0)];
            out_h[d0] = n * cpu_q35::silu(z_h[d0]);
        }
    }

    output.assign(static_cast<size_t>(hidden_size_), 0.0f);
    cpu_bnb4_matvec(layer.linear_o_proj, attn_out.data(), output.data());
    return true;
}

bool CpuQ35HybridModel::run_full_attention(CpuQ35Layer& layer,
                                           CpuQ35LayerCache& cache,
                                           const std::vector<float>& input,
                                           std::vector<float>& output) {
    std::vector<float> q_proj(static_cast<size_t>(full_q_proj_dim_), 0.0f);
    std::vector<float> q(static_cast<size_t>(full_q_dim_), 0.0f);
    std::vector<float> gate(static_cast<size_t>(full_q_dim_), 0.0f);
    std::vector<float> k(static_cast<size_t>(full_kv_dim_), 0.0f);
    std::vector<float> v(static_cast<size_t>(full_kv_dim_), 0.0f);

    cpu_bnb4_matvec(layer.full_q_proj, input.data(), q_proj.data());
    cpu_bnb4_matvec(layer.full_k_proj, input.data(), k.data());
    cpu_bnb4_matvec(layer.full_v_proj, input.data(), v.data());

    if (cfg_.attn_output_gate) {
        std::memcpy(q.data(), q_proj.data(), static_cast<size_t>(full_q_dim_) * sizeof(float));
        std::memcpy(gate.data(), q_proj.data() + full_q_dim_,
                    static_cast<size_t>(full_q_dim_) * sizeof(float));
    } else {
        std::memcpy(q.data(), q_proj.data(), static_cast<size_t>(full_q_dim_) * sizeof(float));
    }

    head_q35_rms_norm(q, layer.q_norm_weight, full_q_heads_, full_head_dim_, cfg_.rms_norm_eps);
    head_q35_rms_norm(k, layer.k_norm_weight, full_kv_heads_, full_head_dim_, cfg_.rms_norm_eps);
    apply_rope_partial_one(q,
                           k,
                           full_q_heads_,
                           full_kv_heads_,
                           full_head_dim_,
                           full_rotary_dim_,
                           current_len_,
                           cfg_.rope.rope_theta);

    for (int kvh = 0; kvh < full_kv_heads_; ++kvh) {
        float* dst_k = cache.full_k.data() +
                       (static_cast<size_t>(kvh) * max_seq_len_ + current_len_) *
                           full_head_dim_;
        float* dst_v = cache.full_v.data() +
                       (static_cast<size_t>(kvh) * max_seq_len_ + current_len_) *
                           full_head_dim_;
        std::memcpy(dst_k,
                    k.data() + static_cast<size_t>(kvh) * full_head_dim_,
                    static_cast<size_t>(full_head_dim_) * sizeof(float));
        std::memcpy(dst_v,
                    v.data() + static_cast<size_t>(kvh) * full_head_dim_,
                    static_cast<size_t>(full_head_dim_) * sizeof(float));
    }

    std::vector<float> attn_out(static_cast<size_t>(full_q_dim_), 0.0f);
    std::vector<float> scores(static_cast<size_t>(current_len_ + 1), 0.0f);
    const int q_per_kv = std::max(1, full_q_heads_ / full_kv_heads_);
    const float scale = 1.0f / std::sqrt(static_cast<float>(full_head_dim_));

    for (int qh = 0; qh < full_q_heads_; ++qh) {
        const int kvh = std::min(full_kv_heads_ - 1, qh / q_per_kv);
        const float* q_h = q.data() + static_cast<size_t>(qh) * full_head_dim_;
        float max_score = -std::numeric_limits<float>::max();
        for (int pos = 0; pos <= current_len_; ++pos) {
            const float* k_h = cache.full_k.data() +
                               (static_cast<size_t>(kvh) * max_seq_len_ + pos) *
                                   full_head_dim_;
            float dot = 0.0f;
            for (int d = 0; d < full_head_dim_; ++d) {
                dot += q_h[d] * k_h[d];
            }
            scores[static_cast<size_t>(pos)] = dot * scale;
            max_score = std::max(max_score, scores[static_cast<size_t>(pos)]);
        }

        float denom = 0.0f;
        for (int pos = 0; pos <= current_len_; ++pos) {
            const float e = std::exp(scores[static_cast<size_t>(pos)] - max_score);
            scores[static_cast<size_t>(pos)] = e;
            denom += e;
        }
        const float inv_denom = 1.0f / std::max(denom, 1e-20f);
        float* out_h = attn_out.data() + static_cast<size_t>(qh) * full_head_dim_;
        for (int pos = 0; pos <= current_len_; ++pos) {
            const float prob = scores[static_cast<size_t>(pos)] * inv_denom;
            const float* v_h = cache.full_v.data() +
                               (static_cast<size_t>(kvh) * max_seq_len_ + pos) *
                                   full_head_dim_;
            for (int d = 0; d < full_head_dim_; ++d) {
                out_h[d] += prob * v_h[d];
            }
        }
    }

    if (cfg_.attn_output_gate) {
        cpu_q35::sigmoid_gate(attn_out.data(), gate.data(), full_q_dim_, attn_out.data());
    }

    output.assign(static_cast<size_t>(hidden_size_), 0.0f);
    cpu_bnb4_matvec(layer.full_o_proj, attn_out.data(), output.data());
    return true;
}

bool CpuQ35HybridModel::run_mlp(CpuQ35Layer& layer,
                                const std::vector<float>& input,
                                std::vector<float>& output) {
    scratch_a_.assign(static_cast<size_t>(ff_dim_), 0.0f);
    scratch_b_.assign(static_cast<size_t>(ff_dim_), 0.0f);
    scratch_c_.assign(static_cast<size_t>(ff_dim_), 0.0f);
    cpu_bnb4_matvec(layer.mlp_gate_proj, input.data(), scratch_a_.data());
    cpu_bnb4_matvec(layer.mlp_up_proj, input.data(), scratch_b_.data());
    for (int i = 0; i < ff_dim_; ++i) {
        scratch_c_[static_cast<size_t>(i)] =
            cpu_q35::silu(scratch_a_[static_cast<size_t>(i)]) *
            scratch_b_[static_cast<size_t>(i)];
    }
    output.assign(static_cast<size_t>(hidden_size_), 0.0f);
    cpu_bnb4_matvec(layer.mlp_down_proj, scratch_c_.data(), output.data());
    return true;
}

bool CpuQ35HybridModel::compute_logits(const std::vector<float>& hidden,
                                       std::vector<float>& logits,
                                       std::string* error) const {
    if (!cfg_.tie_word_embeddings) {
        set_error(error, "CPU Qwen3.5 currently supports only tied lm_head");
        return false;
    }
    logits.assign(static_cast<size_t>(cfg_.vocab_size), 0.0f);

    auto compute_rows = [&](int token_begin, int token_end) {
        for (int token = token_begin; token < token_end; ++token) {
            const uint16_t* row =
                embedding_f16_.data() + static_cast<size_t>(token) * hidden_size_;
            logits[static_cast<size_t>(token)] =
                cpu_f16_dot_f32(row, hidden.data(), hidden_size_);
        }
    };

    cpu_q35_parallel_rows(cfg_.vocab_size, hidden_size_, 4096, compute_rows);
    return true;
}
