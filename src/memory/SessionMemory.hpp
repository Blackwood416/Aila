#pragma once

#include "engine/Types.hpp"

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

} // namespace aila::memory
