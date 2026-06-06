#pragma once

#include "../core/Context.hpp"
#include "../core/Tensor.hpp"
#include "engine/Types.hpp"
#include <vector>

// ============================================================
// KV Cache: 管理所有层的 Key/Value 缓存
// 布局: [num_kv_heads, max_seq_len, head_dim] per tensor
// ============================================================
class KVCache {
public:
    KVCache() = default;

    // 初始化: 为所有层预分配 GPU 内存
    void init(Context& ctx, const Qwen3Config& config, int max_seq_len);

    // 获取指定层的 K/V cache
    Tensor& k_cache(int layer_idx) { return layers_[layer_idx].k; }
    Tensor& v_cache(int layer_idx) { return layers_[layer_idx].v; }

    // 当前缓存长度
    int current_length() const { return current_len_; }

    // 增加缓存长度 (在 append 后调用)
    void advance(int seq_len) { current_len_ += seq_len; }

    // 重置 (新对话)
    void reset() { current_len_ = 0; }

    // 截断缓存长度 (用于剥离多轮对话中特定的思维链)
    void truncate(int new_len) {
        if (new_len < current_len_) {
            current_len_ = new_len;
        }
    }

    int max_length() const { return max_len_; }
    bool quantized() const { return quantized_; }

private:
    struct LayerCache {
        Tensor k;  // [num_kv_heads, max_seq_len, head_dim]
        Tensor v;  // [num_kv_heads, max_seq_len, head_dim]
    };

    std::vector<LayerCache> layers_;
    int current_len_ = 0;
    int max_len_ = 0;
    int num_kv_heads_ = 0;
    int head_dim_ = 0;
    bool quantized_ = false;
};
