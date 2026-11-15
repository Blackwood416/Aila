#include "memory/SessionMemory.hpp"
#include "memory/AllocationBudget.hpp"

#include <climits>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

namespace {

void Expect(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

template <typename Fn>
void ExpectInvalid(Fn&& fn) {
    try { fn(); }
    catch (const std::invalid_argument&) { return; }
    throw std::runtime_error("Invalid/overflowed memory reservation was accepted");
}

void TestKnownModelShapes() {
    Qwen3Config q3;
    q3.num_hidden_layers = 28;
    q3.num_key_value_heads = 8;
    q3.head_dim = 128;
    const auto dense = aila::memory::qwen3_session_memory(q3, 512);
    Expect(dense.deviceBytes == 58720256 && dense.hostBytes == 0,
           "Qwen3 0.6B BF16 cache must reserve 56 MiB per session");

    Qwen35TextConfig hybrid;
    const auto small = aila::memory::qwen35_session_memory(hybrid, 512, 4);
    // 6 Full KV layers: 6,291,456 B. 18 recurrent layers: 20,201,472 B.
    Expect(small.deviceBytes == 26492928 && small.hostBytes == 101007360,
           "Qwen3.5 0.8B cache/mirror/checkpoint payload differs from known dimensions");
    const auto noSnapshots = aila::memory::qwen35_session_memory(hybrid, 512, 0);
    Expect(noSnapshots.deviceBytes == small.deviceBytes && noSnapshots.hostBytes == 20201472,
           "Disabling checkpoints must retain the host mirror and all device states");
    const auto longer = aila::memory::qwen35_session_memory(hybrid, 1024, 4);
    Expect(longer.deviceBytes == small.deviceBytes + 6291456 && longer.hostBytes == small.hostBytes,
           "Longer context should grow Full KV only, not recurrent states or host snapshots");

    hybrid.num_hidden_layers = 32;
    hybrid.num_key_value_heads = 4;
    hybrid.linear_num_value_heads = 32;
    const auto grouped = aila::memory::qwen35_session_memory(hybrid, 512, 4);
    Expect(grouped.deviceBytes == 69468160 && grouped.hostBytes == 263454720,
           "Grouped DeltaNet heads must size state by value heads and convolution by Q/K/V channels");
}

void TestLayerTypesAndConvolution() {
    Qwen35TextConfig config;
    config.num_hidden_layers = 5;
    config.num_key_value_heads = 2;
    config.head_dim = 3;
    config.linear_num_key_heads = 2;
    config.linear_num_value_heads = 4;
    config.linear_key_head_dim = 3;
    config.linear_value_head_dim = 5;
    config.linear_conv_kernel_dim = 2;
    config.layer_types = {"full_attention", "linear_attention"};
    const auto partial = aila::memory::qwen35_session_memory(config, 7, 2);
    Expect(partial.deviceBytes == 1440 && partial.hostBytes == 3312,
           "Explicit layer types and fallback fourth-layer Full Attention must agree");
    config.layer_types = {"full_attention", "linear_attention", "linear_attention",
                          "full_attention", "linear_attention"};
    const auto explicitPlan = aila::memory::qwen35_session_memory(config, 7, 2);
    Expect(explicitPlan.deviceBytes == partial.deviceBytes && explicitPlan.hostBytes == partial.hostBytes,
           "Explicit plan must match equivalent fallback layers");
    config.linear_conv_kernel_dim = 1;
    const auto noHistory = aila::memory::qwen35_session_memory(config, 7, 2);
    Expect(noHistory.deviceBytes == 1056 && noHistory.hostBytes == 2160,
           "Kernel size one must not reserve convolution history");
}

void TestInvalidAndOverflow() {
    Qwen3Config q3;
    ExpectInvalid([&] { aila::memory::qwen3_session_memory(q3, 0); });
    q3.head_dim = -1;
    ExpectInvalid([&] { aila::memory::qwen3_session_memory(q3, 512); });
    q3.num_hidden_layers = q3.num_key_value_heads = q3.head_dim = INT_MAX;
    ExpectInvalid([&] { aila::memory::qwen3_session_memory(q3, INT_MAX); });

    Qwen35TextConfig hybrid;
    hybrid.layer_types = {"unexpected_attention"};
    ExpectInvalid([&] { aila::memory::qwen35_session_memory(hybrid, 512, 4); });
    hybrid.layer_types.assign(25, "linear_attention");
    ExpectInvalid([&] { aila::memory::qwen35_session_memory(hybrid, 512, 4); });
    hybrid.layer_types.clear();
    hybrid.linear_conv_kernel_dim = 0;
    ExpectInvalid([&] { aila::memory::qwen35_session_memory(hybrid, 512, 4); });
    hybrid.linear_conv_kernel_dim = 4;
    const size_t maximum = std::numeric_limits<size_t>::max();
    ExpectInvalid([&] { aila::memory::qwen35_session_memory(hybrid, 512, maximum); });
    const size_t mirrorBytes = 20201472;
    const size_t safeSnapshots = maximum / mirrorBytes - 1;
    const auto boundary = aila::memory::qwen35_session_memory(hybrid, 512, safeSnapshots);
    Expect(boundary.hostBytes == (safeSnapshots + 1) * mirrorBytes,
           "Largest representable snapshot reservation should remain exact");
    ExpectInvalid([&] { aila::memory::qwen35_session_memory(hybrid, 512, safeSnapshots + 1); });
    hybrid.num_hidden_layers = hybrid.num_key_value_heads = hybrid.head_dim = INT_MAX;
    ExpectInvalid([&] { aila::memory::qwen35_session_memory(hybrid, INT_MAX, 4); });
}

void TestAllocationBudget() {
    using aila::memory::AllocationBudget;
    using aila::memory::AllocationLimitExceeded;
    AllocationBudget budget;
    budget.SetLimit(100);
    budget.Acquire(40, "Device");
    budget.Acquire(60, "Device");
    Expect(budget.Current() == 100 && budget.Peak() == 100 && budget.Count() == 2,
           "Exact allocation budget boundary must be admissible");
    bool rejected = false;
    try { budget.Acquire(1, "Device"); } catch (const AllocationLimitExceeded&) { rejected = true; }
    Expect(rejected && budget.Current() == 100 && budget.Count() == 2 && budget.Peak() == 100,
           "Rejected allocation changed the ledger");
    rejected = false;
    try { budget.SetLimit(99); } catch (const AllocationLimitExceeded&) { rejected = true; }
    Expect(rejected && budget.Limit() == 100, "Failed budget shrink changed the limit");
    budget.Release(60);
    budget.SetLimit(50);
    budget.Acquire(10, "Device");
    budget.Release(40);
    budget.Release(10);
    Expect(budget.Current() == 0 && budget.Count() == 0 && budget.Peak() == 100,
           "Allocation release/reuse lost accounting or its peak");
    budget.SetLimit(0);
    budget.Acquire((std::numeric_limits<size_t>::max)(), "Device");
    rejected = false;
    try { budget.Acquire(1, "Device"); } catch (const AllocationLimitExceeded&) { rejected = true; }
    Expect(rejected && budget.Current() == (std::numeric_limits<size_t>::max)(),
           "Unlimited budget allowed integer overflow");
    budget.Release((std::numeric_limits<size_t>::max)());
}

void TestQwen4ExpSessionMemory() {
    Qwen4ExpTextConfig config; // default constructor is the supported 48-layer geometry
    const auto fixed = aila::memory::qwen4exp_session_memory(config, 2048, 0);
    Expect(fixed.deviceBytes == 176234496 && fixed.hostBytes == 117669888,
           "Qwen4Exp device payload or IPC host mirror differs from the fixed geometry");
    const auto snapshots = aila::memory::qwen4exp_session_memory(config, 2048, 4);
    Expect(snapshots.deviceBytes == fixed.deviceBytes && snapshots.hostBytes == 589824000,
           "Qwen4Exp snapshots must add one PLE copy each without changing the device payload");
    const auto longer = aila::memory::qwen4exp_session_memory(config, 4096, 0);
    Expect(longer.deviceBytes > fixed.deviceBytes && longer.hostBytes == fixed.hostBytes,
           "Longer Qwen4Exp context must grow Full KV/indexer, not the recurrent host mirror");

    // The Session builder hardcodes the 4:1 interval and these head shapes, so
    // a differing config must be rejected before any device reservation.
    Qwen4ExpTextConfig wrongInterval;
    wrongInterval.full_attention_interval = 3;
    ExpectInvalid([&] { aila::memory::qwen4exp_session_memory(wrongInterval, 2048, 0); });
    Qwen4ExpTextConfig wrongPlan;
    wrongPlan.layer_types.assign(48, "linear_attention");
    ExpectInvalid([&] { aila::memory::qwen4exp_session_memory(wrongPlan, 2048, 0); });
    Qwen4ExpTextConfig unsupported;
    unsupported.num_hidden_layers = 24;
    ExpectInvalid([&] { aila::memory::qwen4exp_session_memory(unsupported, 2048, 0); });
    ExpectInvalid([&] { aila::memory::qwen4exp_session_memory(config, 0, 0); });
    ExpectInvalid([&] {
        aila::memory::qwen4exp_session_memory(config, 2048, (std::numeric_limits<size_t>::max)());
    });

    auto set_env = [](const char* k, const char* v) {
#ifdef _WIN32
        _putenv_s(k, v ? v : "");
#else
        if (v) setenv(k, v, 1); else unsetenv(k);
#endif
    };
    set_env("AILA_Q4E_KV_STREAM", "1");
    set_env("AILA_Q4E_KV_RESIDENT_TOKENS", "8208");
    const auto planSmall = aila::memory::qwen4exp_kv_stream_plan(4096);
    Expect(!planSmall.enabled, "KV stream must stay off when capacity fits resident tokens");
    const auto planLarge = aila::memory::qwen4exp_kv_stream_plan(32768);
    Expect(planLarge.enabled && planLarge.residentTokens == 8208 && planLarge.slotCount == 2052 && planLarge.blockCount == 8192,
           "KV stream plan mismatch at 32768 capacity");
    const auto streamedBf16 = aila::memory::qwen4exp_session_memory(config, 32768, 0, 2);
    const auto streamedInt8 = aila::memory::qwen4exp_session_memory(config, 32768, 0, 1);
    set_env("AILA_Q4E_KV_STREAM", nullptr);
    set_env("AILA_Q4E_KV_RESIDENT_TOKENS", nullptr);
    const auto unstreamedBf16 = aila::memory::qwen4exp_session_memory(config, 32768, 0, 2);
    const size_t expectedIndexerTailDevice = size_t{12} * 8 * 128 * 2;
    Expect(streamedBf16.deviceBytes < unstreamedBf16.deviceBytes,
           "Streamed 32K BF16 device reservation must be smaller than unstreamed");
    Expect(streamedBf16.hostBytes == unstreamedBf16.hostBytes + aila::memory::qwen4exp_kv_bytes(config, 32768, 2).total(),
           "Streamed 32K BF16 host reservation must include full host KV without full raw indexer_k");
    Expect(streamedInt8.deviceBytes < streamedBf16.deviceBytes,
           "Streamed 32K INT8 device reservation must be smaller than BF16");
    Expect(streamedInt8.hostBytes == unstreamedBf16.hostBytes + aila::memory::qwen4exp_kv_bytes(config, 32768, 1).total(),
           "Streamed 32K INT8 host reservation must include full int8 host KV without full raw indexer_k");
    (void)expectedIndexerTailDevice;
    Expect(aila::memory::qwen4exp_kv_stream_select_auto_resident_tokens(config, 262144, size_t{500} << 20, 1) == 32768,
           "Auto resident tier should pick 32768 for 500 MiB INT8 262K budget");
    Expect(aila::memory::qwen4exp_kv_stream_select_auto_resident_tokens(config, 262144, size_t{120} << 20, 1) == 8208,
           "Auto resident tier should fall back to 8208 for tight 120 MiB budget");
    Expect(!aila::memory::qwen4exp_kv_stream_plan(32768, 0, 2).enabled,
           "Auto KV stream stays off at 32K BF16");
    Expect(aila::memory::qwen4exp_kv_stream_plan(65536, 0, 2).enabled,
           "Auto KV stream turns on at 64K BF16");
    Expect(!aila::memory::qwen4exp_kv_stream_plan(65536, 0, 1).enabled,
           "Auto KV stream stays off at 64K INT8");
    Expect(aila::memory::qwen4exp_kv_stream_plan(131072, 0, 1).enabled,
           "Auto KV stream turns on at 128K INT8");
}

} // namespace

int main() {
    try {
        TestKnownModelShapes();
        TestLayerTypesAndConvolution();
        TestInvalidAndOverflow();
        TestAllocationBudget();
        TestQwen4ExpSessionMemory();
        std::cout << "Session memory reservation tests passed (known models, layer plans, overflow, allocation budgets).\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
