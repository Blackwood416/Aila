#include "memory/SessionMemory.hpp"

#include <climits>
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

} // namespace

int main() {
    try {
        TestKnownModelShapes();
        TestLayerTypesAndConvolution();
        TestInvalidAndOverflow();
        std::cout << "Session memory reservation tests passed (known models, layer plans, overflow).\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
