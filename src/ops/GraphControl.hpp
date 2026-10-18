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

} // namespace graph_control

// Mutable because compressed_blocks advances inside the graph.
using GraphControl = int*;

inline int graph_control_load(const GraphControl control, int slot) {
    return control ? control[slot] : 0;
}

inline bool graph_control_aborted(GraphControl control) {
    if (!control) return true;
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
    int& compressed_blocks) {
    visible_len = 0;
    compressed_blocks = 0;
    if (!control || graph_control_aborted(control)) return false;
    const int start_pos = control[graph_control::start_pos];
    const int blocks = control[graph_control::compressed_blocks];
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

} // namespace ops
