#pragma once

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

} // namespace ops
