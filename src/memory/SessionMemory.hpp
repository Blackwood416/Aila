#pragma once

#include "engine/Types.hpp"
#include "utils/EnvUtils.hpp"

#include <cstddef>
#include <limits>
#include <stdexcept>

namespace aila::memory {

// Payload reservations, independent of SYCL allocation and allocator metadata.
// Hybrid host bytes include a mirror and the maximum retained checkpoints.
struct SessionMemoryReservation {
    size_t deviceBytes = 0;
    size_t hostBytes = 0;
};

inline size_t checked_memory_product(size_t a, size_t b) {
    if (b != 0 && a > std::numeric_limits<size_t>::max() / b) {
        throw std::invalid_argument("Session memory reservation overflows size_t");
    }
    return a * b;
}

inline size_t checked_memory_sum(size_t a, size_t b) {
    if (a > std::numeric_limits<size_t>::max() - b) {
        throw std::invalid_argument("Session memory reservation overflows size_t");
    }
    return a + b;
}

inline size_t positive_memory_dimension(int dimension) {
    if (dimension <= 0) throw std::invalid_argument("Session cache dimensions must be positive");
    return static_cast<size_t>(dimension);
}

inline SessionMemoryReservation qwen3_session_memory(const Qwen3Config& config, int capacity) {
    size_t bytes = 4; // K and V, each BF16.
    for (int dimension : {config.num_hidden_layers, config.num_key_value_heads,
                          config.head_dim, capacity}) {
        bytes = checked_memory_product(bytes, positive_memory_dimension(dimension));
    }
    return {bytes, 0};
}

inline SessionMemoryReservation qwen35_session_memory(
    const Qwen35TextConfig& config, int capacity, size_t maxSnapshots) {
    const size_t layers = positive_memory_dimension(config.num_hidden_layers);
    positive_memory_dimension(capacity);
    if (config.layer_types.size() > layers) {
        throw std::invalid_argument("Qwen3.5 has more layer types than decoder layers");
    }
    size_t fullLayers = 0;
    size_t linearLayers = 0;
    for (const auto& type : config.layer_types) {
        if (type == "linear_attention") ++linearLayers;
        else if (type == "full_attention") ++fullLayers;
        else throw std::invalid_argument("Unknown Qwen3.5 attention layer type");
    }
    const size_t fallbackFull = layers / 4 - config.layer_types.size() / 4;
    fullLayers += fallbackFull;
    linearLayers += layers - config.layer_types.size() - fallbackFull;

    size_t fullBytes = 4; // BF16 Full Attention K and V.
    for (int dimension : {capacity, config.num_key_value_heads, config.head_dim}) {
        fullBytes = checked_memory_product(fullBytes, positive_memory_dimension(dimension));
    }
    fullBytes = checked_memory_product(fullBytes, fullLayers);

    const auto keyHeads = positive_memory_dimension(config.linear_num_key_heads);
    const auto valueHeads = positive_memory_dimension(config.linear_num_value_heads);
    const auto keyDim = positive_memory_dimension(config.linear_key_head_dim);
    const auto valueDim = positive_memory_dimension(config.linear_value_head_dim);
    const auto convRows = positive_memory_dimension(config.linear_conv_kernel_dim) - 1;
    const auto stateElements = checked_memory_product(checked_memory_product(valueHeads, keyDim), valueDim);
    const auto convChannels = checked_memory_sum(
        checked_memory_product(2, checked_memory_product(keyHeads, keyDim)),
        checked_memory_product(valueHeads, valueDim));
    const auto recurrentElements = checked_memory_sum(stateElements, checked_memory_product(convRows, convChannels));
    const auto recurrentBytes = checked_memory_product(
        checked_memory_product(recurrentElements, sizeof(float)), linearLayers);
    return {checked_memory_sum(fullBytes, recurrentBytes),
            checked_memory_product(recurrentBytes, checked_memory_sum(1, maxSnapshots))};
}

// Qwen4Exp (Qwen3.8-Flash-Next) is a fixed 48-layer architecture with a
// 12 Full / 36 Linear Attention split. The shared Session allocates its
// exclusive caches from hardcoded tile shapes (Qwen4ExpBnb4Backend::Session),
// so an unsupported geometry is rejected before any device reservation rather
// than reserving a cache the kernels could not index consistently.
//
// Device payload mirrors the per-session tensors: Full K/V plus QSA indexer
// and 4:1 compressed indexer keys (BF16), the DeltaNet recurrent matrix and
// Qwen4Exp always alternates three linear layers then one full-attention layer;
// the session builder hardcodes that interval.
inline constexpr size_t kQwen4ExpFullAttentionInterval = 4;

// Device bytes of the full-attention KV caches for one session. Single source
// for the session reservation and the backend expert-cache admission: the int8
// layout halves K/V and adds one F32 per-token scale for K and for V.
struct Qwen4ExpKvBytes {
    size_t kv = 0;
    size_t scales = 0;
    size_t total() const { return checked_memory_sum(kv, scales); }
};

inline Qwen4ExpKvBytes qwen4exp_kv_bytes(
    const Qwen4ExpTextConfig& config, size_t capacity, size_t kvElementBytes) {
    if (kvElementBytes != 1 && kvElementBytes != 2)
        throw std::invalid_argument("Qwen4Exp KV element width must be 1 or 2 bytes");
    const size_t layers = positive_memory_dimension(config.num_hidden_layers);
    const size_t fullLayers = layers / kQwen4ExpFullAttentionInterval;
    const auto kvHeads = positive_memory_dimension(config.num_key_value_heads);
    const auto headDim = positive_memory_dimension(config.head_dim);
    positive_memory_dimension(capacity);
    Qwen4ExpKvBytes bytes;
    bytes.kv = checked_memory_product(
        checked_memory_product(size_t{2} * kvElementBytes,
                               checked_memory_product(kvHeads, headDim)),
        checked_memory_product(capacity, fullLayers));
    if (kvElementBytes == 1) {
        bytes.scales = checked_memory_product(
            checked_memory_product(size_t{2} * sizeof(float), kvHeads),
            checked_memory_product(capacity, fullLayers));
    }
    return bytes;
}

// convolution history (F32), and the PLE convolution state (F32). The host
// payload is the persistent DeltaNet mirror plus retained snapshots; the PLE
// convolution state is device-only but every snapshot keeps a float copy.
inline SessionMemoryReservation qwen4exp_session_memory(
    const Qwen4ExpTextConfig& config, int capacity, size_t maxSnapshots,
    size_t kvElementBytes = 2) {
    if (!is_exact_qwen4exp_text_spec(config)) {
        throw std::invalid_argument("Unsupported Qwen4Exp text dimensions");
    }
    const size_t layers = positive_memory_dimension(config.num_hidden_layers);
    positive_memory_dimension(capacity);

    // The Session builder hardcodes these tile shapes instead of reading them
    // from config; a differing config would allocate a mismatched cache.
    constexpr int kIndexerHeadDim = 128;
    constexpr int kIndexerCompressRatio = 4;
    constexpr int kPleConvStateLen = 9;     // (kernel 4 - 1) * dilation 3
    constexpr int kPleConvChannels = 10240; // hc_count 4 * hidden_size 2560
    if (config.full_attention_interval != 4 ||
        config.indexer_kv_heads != 1 ||
        config.indexer_head_dim != kIndexerHeadDim ||
        config.indexer_compress_ratio != kIndexerCompressRatio ||
        config.linear_num_key_heads != 16 ||
        config.linear_num_value_heads != 48 ||
        config.linear_key_head_dim != 128 ||
        config.linear_value_head_dim != 128 ||
        config.linear_conv_kernel_dim != 4) {
        throw std::invalid_argument("Unsupported Qwen4Exp attention geometry");
    }
    if (!config.layer_types.empty()) {
        if (config.layer_types.size() > layers) {
            throw std::invalid_argument("Qwen4Exp has more layer types than decoder layers");
        }
        for (size_t i = 0; i < config.layer_types.size(); ++i) {
            const char* expected = ((i + 1) % 4 == 0) ? "full_attention" : "linear_attention";
            if (config.layer_types[i] != expected) {
                throw std::invalid_argument("Qwen4Exp layer types must match the fixed 4-layer interval");
            }
        }
    }
    const size_t fullLayers = layers / kQwen4ExpFullAttentionInterval;
    const size_t linearLayers = layers - fullLayers;

    const auto kvHeads = positive_memory_dimension(config.num_key_value_heads);
    const auto headDim = positive_memory_dimension(config.head_dim);
    const auto cached = positive_memory_dimension(capacity);

    // K and V, each `kvElementBytes` wide. The int8 layout also allocates a
    // per-token F32 scale for K and for V ({kvHeads, capacity} each).
    const size_t fullBytes = qwen4exp_kv_bytes(config, cached, kvElementBytes).total();

    const size_t compressed = cached / kIndexerCompressRatio > 0 ? cached / kIndexerCompressRatio : 1;
    size_t indexerBytes = checked_memory_product(
        checked_memory_product(checked_memory_sum(cached, compressed),
                               static_cast<size_t>(kIndexerHeadDim)), size_t{2});
    indexerBytes = checked_memory_product(indexerBytes, fullLayers);

    const auto keyHeads = positive_memory_dimension(config.linear_num_key_heads);
    const auto valueHeads = positive_memory_dimension(config.linear_num_value_heads);
    const auto keyDim = positive_memory_dimension(config.linear_key_head_dim);
    const auto valueDim = positive_memory_dimension(config.linear_value_head_dim);
    const auto convRows = positive_memory_dimension(config.linear_conv_kernel_dim) - 1;
    const auto stateElements = checked_memory_product(checked_memory_product(valueHeads, keyDim), valueDim);
    const auto convChannels = checked_memory_sum(
        checked_memory_product(2, checked_memory_product(keyHeads, keyDim)),
        checked_memory_product(valueHeads, valueDim));
    const auto recurrentElements = checked_memory_sum(stateElements, checked_memory_product(convRows, convChannels));
    const auto recurrentBytes = checked_memory_product(
        checked_memory_product(recurrentElements, sizeof(float)), linearLayers);

    const auto pleBytes = checked_memory_product(
        checked_memory_product(static_cast<size_t>(kPleConvStateLen),
                               static_cast<size_t>(kPleConvChannels)), sizeof(float));

    size_t deviceBytes = checked_memory_sum(
        checked_memory_sum(fullBytes, indexerBytes),
        checked_memory_sum(recurrentBytes, pleBytes));
    size_t hostBytes = checked_memory_sum(
        checked_memory_product(recurrentBytes, checked_memory_sum(1, maxSnapshots)),
        checked_memory_product(pleBytes, maxSnapshots));
    if (aila::env::read_flag("AILA_Q4E_GRAPH_GDN_STATE", false)) {
        const size_t headBytes = checked_memory_product(
            checked_memory_product(layers, size_t{3}), sizeof(int));
        deviceBytes = checked_memory_sum(deviceBytes, headBytes);
        hostBytes = checked_memory_sum(hostBytes, headBytes);
    }
    return {deviceBytes, hostBytes};
}

} // namespace aila::memory
