#pragma once
#include "ops/GreedyOps.hpp"

namespace mtp_probe {
using ops::GreedyResult;
using ops::GreedyGroups;
inline sycl::event Greedy(Context& ctx, const sycl::ext::oneapi::bfloat16* logits, int vocab,
                          GreedyResult* partials, GreedyResult* result, int rows = 1) {
    return ops::greedy_bf16(ctx, logits, vocab, partials, result, rows);
}
} // namespace mtp_probe
