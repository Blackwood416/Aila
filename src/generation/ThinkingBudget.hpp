#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

namespace aila::generation {

inline constexpr size_t kMaxThinkingCloseTokens = 16;

// Token metadata is supplied by the caller's trusted tokenizer/template. This
// host-only contract is shared by the step driver and optional IPC extension.
// Content tokens inside thinking count toward maxTokens; markers do not.
struct ThinkingBudgetOptions {
    uint32_t maxTokens = 0;
    int32_t openTokenId = -1;
    int32_t closeTokenId = -1;
    bool promptOpensThink = false;
    std::vector<int32_t> closeTokenIds;
    // Compatibility with one-shot reasoning budgets: after a forced close,
    // a later model-generated opening marker does not trigger another close.
    bool closeOnce = false;
};

inline const char* ValidateThinkingBudget(const ThinkingBudgetOptions& options,
                                         uint32_t vocabSize,
                                         const std::vector<int32_t>& stopTokenIds) {
    const auto validId = [vocabSize](int32_t id) {
        return id >= 0 && static_cast<uint32_t>(id) < vocabSize;
    };
    const auto isStop = [&stopTokenIds](int32_t id) {
        return std::find(stopTokenIds.begin(), stopTokenIds.end(), id) != stopTokenIds.end();
    };
    if (options.maxTokens > static_cast<uint32_t>((std::numeric_limits<int32_t>::max)()))
        return "Thinking budget exceeds int32 range";
    if (!validId(options.openTokenId) || !validId(options.closeTokenId) || options.openTokenId == options.closeTokenId)
        return "Thinking markers must be distinct valid token IDs";
    if (isStop(options.openTokenId) || isStop(options.closeTokenId))
        return "Thinking markers must not be stop tokens";
    if (options.closeTokenIds.empty() || options.closeTokenIds.size() > kMaxThinkingCloseTokens ||
        options.closeTokenIds.front() != options.closeTokenId)
        return "Thinking close sequence must start with the closing marker and contain 1..16 tokens";
    for (size_t i = 0; i < options.closeTokenIds.size(); ++i) {
        const auto token = options.closeTokenIds[i];
        if (!validId(token) || isStop(token) || token == options.openTokenId || (i && token == options.closeTokenId))
            return "Thinking close sequence contains an invalid, stop or repeated marker token";
    }
    return nullptr;
}

} // namespace aila::generation
