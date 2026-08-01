#pragma once

#include <vector>

namespace ops {

inline bool is_token_suppressed(int token_id,
                                const std::vector<int>* suppress_tokens) {
    if (suppress_tokens == nullptr) {
        return false;
    }
    for (int suppressed : *suppress_tokens) {
        if (token_id == suppressed) {
            return true;
        }
    }
    return false;
}

}  // namespace ops
