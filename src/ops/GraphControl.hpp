#pragma once

#include <sycl/sycl.hpp>

namespace ops {

// Fixed 16-word device scalar table shared by whole-token graph variants.
// Pointers are baked at capture time; values are updated before replay.
namespace graph_control {

inline constexpr int words = 16;
inline constexpr int start_pos = 0;
inline constexpr int compressed_blocks = 1;
inline constexpr int abort = 2;
inline constexpr int max_full_layers = 12;
// Slot 3 remains available for existing layer-0 test diagnostics.
inline constexpr int first_layer_blocks_slot = 4;

} // namespace graph_control

// Mutable because compressed_blocks advances inside the graph.
using GraphControl = int*;

// layer is a 0-based full-attention ordinal. Layer 0 keeps the original
// compressed_blocks slot; later layers use independent count slots so one
// executable can advance every full-attention layer without sharing state.
// Invalid ordinals fail closed at runtime.
inline constexpr int graph_control_layer_blocks_slot(int layer) {
    if (layer < 0 || layer >= graph_control::max_full_layers) return -1;
    return layer == 0
               ? graph_control::compressed_blocks
               : graph_control::first_layer_blocks_slot + layer - 1;
}

static_assert(
    graph_control_layer_blocks_slot(graph_control::max_full_layers - 1) <
        graph_control::words,
    "per-layer graph control slots must fit the fixed control table");

inline int graph_control_load(const GraphControl control, int slot) {
    return control ? control[slot] : 0;
}

inline bool graph_control_aborted(GraphControl control) {
    if (!control) return true;
    // Multiple work-groups can report failure concurrently.
    sycl::atomic_ref<
        int, sycl::memory_order::relaxed, sycl::memory_scope::device,
        sycl::access::address_space::global_space>
        abort_flag(control[graph_control::abort]);
    return abort_flag.load(sycl::memory_order::relaxed) != 0;
}

inline void graph_control_mark_abort(GraphControl control) {
    if (!control) return;
    sycl::atomic_ref<
        int, sycl::memory_order::relaxed, sycl::memory_scope::device,
        sycl::access::address_space::global_space>
        abort_flag(control[graph_control::abort]);
    abort_flag.store(1, sycl::memory_order::relaxed);
}

inline bool graph_control_consistent_blocks(
    GraphControl control, int max_blocks, int& visible_len,
    int& compressed_blocks, int layer = 0) {
    visible_len = 0;
    compressed_blocks = 0;
    const int blocks_slot = graph_control_layer_blocks_slot(layer);
    if (!control || blocks_slot < 0 || graph_control_aborted(control)) {
        graph_control_mark_abort(control);
        return false;
    }
    const int start_pos = control[graph_control::start_pos];
    const int blocks = control[blocks_slot];
    const long long visible = static_cast<long long>(start_pos) + 1;
    if (start_pos < 0 || visible <= 0 || visible > 0x7fffffffLL ||
        blocks < 0 || blocks > max_blocks ||
        blocks != static_cast<int>(visible / 4)) {
        graph_control_mark_abort(control);
        return false;
    }
    visible_len = static_cast<int>(visible);
    compressed_blocks = blocks;
    return true;
}

// Producer-side variant for nodes that write the raw K row before the
// completion node publishes compressed_blocks. Exactly one pending block is
// valid; older/stale counts fail closed before any cache write.
inline bool graph_control_pending_blocks(
    GraphControl control, int max_blocks, int& visible_len,
    int& compressed_blocks, int layer = 0) {
    visible_len = 0;
    compressed_blocks = 0;
    const int blocks_slot = graph_control_layer_blocks_slot(layer);
    if (!control || blocks_slot < 0 || graph_control_aborted(control)) {
        graph_control_mark_abort(control);
        return false;
    }
    const int start_pos = control[graph_control::start_pos];
    const int blocks = control[blocks_slot];
    const long long visible = static_cast<long long>(start_pos) + 1;
    if (start_pos < 0 || visible <= 0 || visible > 0x7fffffffLL) {
        graph_control_mark_abort(control);
        return false;
    }
    const int target_blocks = static_cast<int>(visible / 4);
    if (blocks < 0 || blocks > max_blocks || blocks > target_blocks ||
        target_blocks - blocks > 1) {
        graph_control_mark_abort(control);
        return false;
    }
    visible_len = static_cast<int>(visible);
    compressed_blocks = blocks;
    return true;
}

} // namespace ops
