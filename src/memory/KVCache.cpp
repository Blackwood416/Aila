#include "KVCache.hpp"
#include "profile/Profiling.hpp"
#include "../utils/EnvUtils.hpp"

void KVCache::init(Context& ctx, const Qwen3Config& config, int max_seq_len) {
    max_len_ = max_seq_len;
    num_kv_heads_ = config.num_key_value_heads;
    head_dim_ = config.head_dim;
    current_len_ = 0;
    quantized_ = aila::env::read_flag("AILA_KV_QUANT", false);

    layers_.resize(config.num_hidden_layers);

    auto dtype = quantized_ ? dnnl::memory::data_type::f8_e4m3 : dnnl::memory::data_type::bf16;
    size_t element_bytes = quantized_ ? 1 : 2;
    size_t per_tensor_bytes = (size_t)num_kv_heads_ * max_seq_len * head_dim_ * element_bytes;
    size_t total_bytes = 0;

    for (int i = 0; i < config.num_hidden_layers; i++) {
        std::vector<int64_t> shape = {num_kv_heads_, (int64_t)max_seq_len, head_dim_};
        layers_[i].k = Tensor::allocate(ctx, shape, dtype);
        layers_[i].v = Tensor::allocate(ctx, shape, dtype);
        total_bytes += 2 * per_tensor_bytes;
    }

    AILA_LOG_INFO("[KVCache] Allocated %d layers, max_seq=%d, quantized=%s, %.2f MB",
                  config.num_hidden_layers, max_seq_len, quantized_ ? "true" : "false",
                  total_bytes / (1024.0 * 1024.0));
}
