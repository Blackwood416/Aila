#pragma once

#include <cstdint>

namespace aila {
namespace vision {

inline bool qwen35_visual_temporal_patch_weight_can_release(
    int64_t numel,
    int hidden_size,
    int patch_size) {
    if (numel <= 0 || hidden_size <= 0 || patch_size <= 0) {
        return false;
    }

    const int64_t patch_area =
        static_cast<int64_t>(patch_size) * static_cast<int64_t>(patch_size);
    const int64_t spatial_patch_numel =
        static_cast<int64_t>(hidden_size) * 3 * patch_area;
    return numel == spatial_patch_numel * 2;
}

}  // namespace vision
}  // namespace aila
