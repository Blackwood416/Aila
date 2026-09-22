#include "core/Context.hpp"
#include "core/Tensor.hpp"
#include "inference/Model.hpp"
#include "memory/AllocationBudget.hpp"
#include "ops/Ops.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <tuple>
#include <vector>

namespace {

using bf16 = sycl::ext::oneapi::bfloat16;
using DataType = dnnl::memory::data_type;
using Format = dnnl::memory::format_tag;
using LimitExceeded = aila::memory::AllocationLimitExceeded;

void Expect(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

template <typename Exception, typename Function>
void ExpectThrows(Function&& function, const std::string& label) {
    try {
        function();
    } catch (const Exception&) {
        return;
    } catch (const std::exception& error) {
        throw std::runtime_error(label + ": wrong exception: " + error.what());
    }
    throw std::runtime_error(label + ": operation unexpectedly succeeded");
}

// Declare after tensors/primitive owners, so failures drain before their USM
// is destroyed. Successful paths also synchronize explicitly before checking.
struct DrainQueue {
    Context& context;
    ~DrainQueue() { try { context.synchronize(); } catch (...) {} }
};

struct Usage {
    size_t deviceBytes, pinnedBytes, deviceCount, pinnedCount;
    size_t devicePeak, pinnedPeak, deviceLimit, pinnedLimit;

    bool operator==(const Usage& other) const {
        return std::tie(deviceBytes, pinnedBytes, deviceCount, pinnedCount,
                        devicePeak, pinnedPeak, deviceLimit, pinnedLimit) ==
               std::tie(other.deviceBytes, other.pinnedBytes, other.deviceCount,
                        other.pinnedCount, other.devicePeak, other.pinnedPeak,
                        other.deviceLimit, other.pinnedLimit);
    }
};

Usage ReadUsage(const Context& context) {
    return {context.current_allocated_bytes(), context.current_pinned_host_bytes(),
            context.device_allocation_count(), context.pinned_host_allocation_count(),
            context.peak_allocated_bytes(), context.peak_pinned_host_bytes(),
            context.device_allocation_limit(), context.pinned_host_allocation_limit()};
}

void ExpectUsage(const Context& context, size_t deviceBytes, size_t pinnedBytes,
                 size_t deviceCount, size_t pinnedCount, const std::string& label) {
    const auto actual = ReadUsage(context);
    Expect(actual.deviceBytes == deviceBytes && actual.pinnedBytes == pinnedBytes &&
           actual.deviceCount == deviceCount && actual.pinnedCount == pinnedCount,
           label + ": device=" + std::to_string(actual.deviceBytes) + "/" +
           std::to_string(actual.deviceCount) + " pinned=" +
           std::to_string(actual.pinnedBytes) + "/" + std::to_string(actual.pinnedCount));
}

class UsmAllocation final {
public:
    UsmAllocation(Context& context, size_t bytes, bool pinned = false)
        : context_(context), pinned_(pinned),
          data_(pinned ? context.alloc_host(bytes) : context.alloc_device(bytes)) {}
    ~UsmAllocation() {
        try { context_.synchronize(); } catch (...) {}
        Free();
    }
    UsmAllocation(const UsmAllocation&) = delete;
    UsmAllocation& operator=(const UsmAllocation&) = delete;
    void* Data() const { return data_; }
    void Reset() { context_.synchronize(); Free(); }

private:
    void Free() {
        if (pinned_) context_.free_host(data_);
        else context_.free_device(data_);
        data_ = nullptr;
    }
    Context& context_;
    bool pinned_;
    void* data_;
};

void TestContextBudgetAndOrder() {
    Context context;
    Expect(context.queue().has_property<sycl::property::queue::in_order>(),
           "Execution Context must retain an in-order queue");
    ExpectUsage(context, 0, 0, 0, 0, "New Context");
    context.set_allocation_limits(257, 129);

    UsmAllocation device(context, 73), pinned(context, 33, true);
    UsmAllocation deviceTail(context, 184), pinnedTail(context, 96, true);
    ExpectUsage(context, 257, 129, 2, 2, "Exact payload budget boundary");
    Expect(context.peak_allocated_bytes() == 257 && context.peak_pinned_host_bytes() == 129,
           "Peaks must count requested payload bytes");
    Expect(sycl::get_pointer_type(device.Data(), context.queue().get_context()) == sycl::usm::alloc::device &&
           sycl::get_pointer_type(pinned.Data(), context.queue().get_context()) == sycl::usm::alloc::host,
           "Device and pinned allocations must use the owning SYCL context");

    const auto full = ReadUsage(context);
    context.check_device_allocation(0);
    ExpectThrows<LimitExceeded>([&] { context.check_device_allocation(1); }, "Device preflight budget");
    ExpectThrows<LimitExceeded>([&] { UsmAllocation extra(context, 1); }, "Device budget");
    ExpectThrows<LimitExceeded>([&] { UsmAllocation extra(context, 1, true); }, "Pinned budget");
    ExpectThrows<std::invalid_argument>([&] { UsmAllocation zero(context, 0); }, "Zero device allocation");
    ExpectThrows<std::invalid_argument>([&] { UsmAllocation zero(context, 0, true); }, "Zero pinned allocation");
    // A failure in either half must not partially change the other limit.
    ExpectThrows<LimitExceeded>([&] { context.set_allocation_limits(258, 128); }, "Lower pinned limit");
    ExpectThrows<LimitExceeded>([&] { context.set_allocation_limits(256, 130); }, "Lower device limit");
    Expect(ReadUsage(context) == full, "Rejected allocations/preflights/limits changed the ledger");

    auto* value = static_cast<int*>(device.Data());
    auto* host = static_cast<int*>(pinned.Data());
    *host = 11;
    context.queue().memcpy(value, host, sizeof(int));
    context.queue().single_task([=] { *value = *value * 3 + 2; });
    context.queue().memcpy(host, value, sizeof(int));
    context.synchronize();
    Expect(*host == 35, "In-order copy/kernel/copy dependencies were lost");

    deviceTail.Reset();
    pinnedTail.Reset();
    ExpectUsage(context, 73, 33, 1, 1, "Partial release");
    const auto partial = ReadUsage(context);
    context.check_device_allocation(184);
    ExpectThrows<LimitExceeded>([&] { context.check_device_allocation(185); }, "Remaining device credit");
    Expect(ReadUsage(context) == partial, "Successful preflight reserved memory");
    {
        UsmAllocation reusedDevice(context, 184), reusedPinned(context, 96, true);
        ExpectUsage(context, 257, 129, 2, 2, "Released budget is reusable");
    }

    context.set_allocation_limits(0, 0);
    const auto unlimited = ReadUsage(context);
    const size_t maximum = std::numeric_limits<size_t>::max();
    ExpectThrows<LimitExceeded>([&] { context.check_device_allocation(maximum); }, "Device size overflow");
    ExpectThrows<LimitExceeded>([&] { UsmAllocation extra(context, maximum); }, "Device allocation overflow");
    ExpectThrows<LimitExceeded>([&] { UsmAllocation extra(context, maximum, true); }, "Pinned allocation overflow");
    Expect(ReadUsage(context) == unlimited, "Overflow rejection changed usage or peaks");
    device.Reset();
    pinned.Reset();
    context.release_execution_resources();
    ExpectUsage(context, 0, 0, 0, 0, "All explicit allocations released");
    Expect(context.peak_allocated_bytes() == 257 && context.peak_pinned_host_bytes() == 129,
           "Releasing allocations must retain lifetime peaks");
}

struct ProbeWorkspace {
    explicit ProbeWorkspace(Context& context) : device(context, 96), pinned(context, 32, true) {}
    UsmAllocation device, pinned;
    int marker = 0;
};

void TestTypedWorkspaceOwnership() {
    Context first, second;
    auto& a = first.workspace<ProbeWorkspace>();
    auto& b = second.workspace<ProbeWorkspace>();
    a.marker = 7;
    b.marker = 19;
    Expect(&a == &first.workspace<ProbeWorkspace>(), "Same Context did not reuse its typed workspace");
    Expect(&a != &b && a.device.Data() != b.device.Data() && a.pinned.Data() != b.pinned.Data(),
           "Different Contexts shared a workspace or live USM allocation");
    ExpectUsage(first, 96, 32, 1, 1, "First workspace owner");
    ExpectUsage(second, 96, 32, 1, 1, "Second workspace owner");
    const auto secondUsage = ReadUsage(second);
    first.synchronize();
    first.release_execution_resources();
    ExpectUsage(first, 0, 0, 0, 0, "Workspace release");
    Expect(ReadUsage(second) == secondUsage && b.marker == 19,
           "Releasing one Context disturbed another workspace");
    Expect(first.workspace<ProbeWorkspace>().marker == 0, "Released workspace retained stale state");
    first.synchronize();
    first.release_execution_resources();
    second.synchronize();
    second.release_execution_resources();
    ExpectUsage(first, 0, 0, 0, 0, "Recreated workspace release");
    ExpectUsage(second, 0, 0, 0, 0, "Second workspace release");
}

GenerationConfig SamplingConfig(int topK) {
    GenerationConfig config;
    config.do_sample = true;
    config.temperature = 1.0f;
    config.top_k = topK;
    config.top_p = 1.0f;
    return config;
}

void RunSampling(Context& context, Tensor& logits, Tensor& result, int vocab, int salt) {
    std::vector<bf16> hostLogits(static_cast<size_t>(vocab));
    for (int i = 0; i < vocab; ++i) hostLogits[i] = bf16(-20.0f + ((i + salt) % 13) * 0.125f);
    const int winner = (salt * 197 + 17) % vocab;
    const int runnerUp = (winner + vocab - 3) % vocab;
    hostLogits[winner] = bf16(8.0f);
    hostLogits[runnerUp] = bf16(7.5f);
    DrainQueue drain{context};
    context.memcpy_h2d(logits.data(), hostLogits.data(), hostLogits.size() * sizeof(bf16));
    auto hostConfig = SamplingConfig(1);
    std::mt19937 rng(12345);
    Expect(ops::sample_with_config(context, logits, vocab, hostConfig, {}, rng) == winner,
           "Host sampling returned stale logits");
    Expect(ops::topk_sample(context, logits, vocab, 1.0f, 1) == winner,
           "Legacy top-k sampling did not reuse the correct host buffer");
    hostConfig.presence_penalty = 4.0f;
    Expect(ops::sample_with_config(context, logits, vocab, hostConfig, {winner}, rng) == runnerUp,
           "Penalty sampling used stale host workspace contents");
    // Exercise the reusable heap/sorted/probability vectors as well as the
    // <=64 stack path. A small top-p keeps this independent of RNG details.
    hostConfig.top_k = std::min(vocab, 128);
    hostConfig.top_p = 0.01f;
    Expect(ops::sample_with_config(context, logits, vocab, hostConfig, {winner}, rng) == runnerUp,
           "Large top-k sampling used stale Context-owned candidate vectors");
    Expect(ops::argmax(context, logits, vocab) == winner, "Host-returning argmax changed");
    int token = -1;
    ops::argmax(context, logits, vocab, result.data_as<int>());
    context.memcpy_d2h(&token, result.data(), sizeof(token));
    Expect(token == winner, "Device-result argmax changed");

    const auto deviceConfig = SamplingConfig(2);
    Expect(ops::can_use_device_sampling(vocab, deviceConfig), "Fixture must exercise device sampling");
    // Softmax([8, 7.5]) gives about 0.622/0.378. Both values are far from
    // the CDF boundary, and the second draw detects an accidental argmax path.
    for (const float uniform : {0.1f, 0.9f}) {
        ops::sample_with_config_device(context, logits, vocab, deviceConfig, uniform, result.data_as<int>());
        context.memcpy_d2h(&token, result.data(), sizeof(token));
        Expect(token == (uniform < 0.5f ? winner : runnerUp),
               "Device sampling returned stale candidates or used the wrong Context");
    }
    context.synchronize();
}

void TestSamplingIsolationAndReuse() {
    constexpr int smallVocab = 513;
    constexpr int largeVocab = 8193;
    Context first, second;
    {
        Tensor firstLogits = Tensor::allocate(first, {largeVocab});
        Tensor firstResult = Tensor::allocate(first, {1}, DataType::s32);
        Tensor secondLogits = Tensor::allocate(second, {largeVocab});
        Tensor secondResult = Tensor::allocate(second, {1}, DataType::s32);
        DrainQueue drainFirst{first}, drainSecond{second};
        const auto firstBase = ReadUsage(first), secondBase = ReadUsage(second);
        RunSampling(first, firstLogits, firstResult, smallVocab, 1);
        RunSampling(first, firstLogits, firstResult, smallVocab, 2);
        const auto firstWarm = ReadUsage(first);
        Expect(firstWarm.deviceBytes > firstBase.deviceBytes && firstWarm.deviceCount > firstBase.deviceCount &&
               firstWarm.pinnedBytes == smallVocab * sizeof(bf16) && firstWarm.pinnedCount == 1,
               "Sampling workspaces were not charged to the first Context");

        RunSampling(second, secondLogits, secondResult, smallVocab, 3);
        RunSampling(second, secondLogits, secondResult, smallVocab, 4);
        const auto secondSmall = ReadUsage(second);
        Expect(secondSmall.deviceBytes > secondBase.deviceBytes && secondSmall.deviceCount > secondBase.deviceCount &&
               secondSmall.pinnedBytes == smallVocab * sizeof(bf16) && secondSmall.pinnedCount == 1,
               "Second Context reused another Context's sampling USM");
        RunSampling(second, secondLogits, secondResult, largeVocab, 5);
        RunSampling(second, secondLogits, secondResult, largeVocab, 6);
        const auto secondWarm = ReadUsage(second);
        Expect(secondWarm.deviceBytes > secondSmall.deviceBytes &&
               secondWarm.deviceCount == secondSmall.deviceCount &&
               secondWarm.pinnedBytes == largeVocab * sizeof(bf16) && secondWarm.pinnedCount == secondSmall.pinnedCount,
               "Growing sampling workspace leaked old buffers or was not accounted");
        Expect(ReadUsage(first) == firstWarm, "Growing the second Context changed the first Context's usage");

        // Leave room only for the temporary argmax result. Sequential calls
        // from new threads must reuse the Context's existing host/device USM.
        first.set_allocation_limits(firstWarm.deviceBytes + sizeof(int), firstWarm.pinnedBytes);
        const auto bounded = ReadUsage(first);
        for (int salt = 7; salt < 11; ++salt) {
            std::exception_ptr failure;
            std::thread worker([&, salt] {
                try { RunSampling(first, firstLogits, firstResult, smallVocab, salt); }
                catch (...) { failure = std::current_exception(); }
            });
            worker.join();
            if (failure) std::rethrow_exception(failure);
            Expect(ReadUsage(first) == bounded, "Sequential thread change grew or leaked sampling workspace");
        }
        ExpectThrows<LimitExceeded>([&] {
            RunSampling(first, firstLogits, firstResult, largeVocab, 11);
        }, "Pinned sampling growth budget");
        const auto config = SamplingConfig(2);
        ExpectThrows<LimitExceeded>([&] {
            ops::sample_with_config_device(first, firstLogits, largeVocab, config, 0.1f, firstResult.data_as<int>());
        }, "Device sampling growth budget");
        first.synchronize();
        Expect(ReadUsage(first) == bounded, "Rejected sampling growth changed existing allocations");
        RunSampling(first, firstLogits, firstResult, smallVocab, 12);
        first.synchronize();
        first.release_execution_resources();
        ExpectUsage(first, firstBase.deviceBytes, 0, firstBase.deviceCount, 0, "Sampling workspace release");
        Expect(ReadUsage(second) == secondWarm, "First sampling teardown invalidated the second Context");
        RunSampling(second, secondLogits, secondResult, largeVocab, 13);
        RunSampling(first, firstLogits, firstResult, smallVocab, 14);
        first.synchronize();
        first.release_execution_resources();
        second.synchronize();
        second.release_execution_resources();
        ExpectUsage(first, firstBase.deviceBytes, 0, firstBase.deviceCount, 0, "Recreated sampling workspace release");
        ExpectUsage(second, secondBase.deviceBytes, 0, secondBase.deviceCount, 0, "Second sampling workspace release");
    }
    ExpectUsage(first, 0, 0, 0, 0, "First sampling tensors destroyed");
    ExpectUsage(second, 0, 0, 0, 0, "Second sampling tensors destroyed");
}

dnnl::memory::desc ScratchDescriptor(size_t bytes) {
    return dnnl::memory::desc({static_cast<int64_t>(bytes)}, DataType::u8, Format::a);
}

void TestSharedScratchBudget() {
    Context context;
    DrainQueue drain{context};
    context.scratchpad_memory(ScratchDescriptor(0));
    ExpectUsage(context, 0, 0, 0, 0, "Zero-sized scratch");
    auto small = context.scratchpad_memory(ScratchDescriptor(257));
    const auto oldAddress = reinterpret_cast<uintptr_t>(small.get_data_handle());
    ExpectUsage(context, 257, 0, 1, 0, "Initial shared scratch");
    context.queue().memset(small.get_data_handle(), 0x5a, 257);
    context.synchronize();
    auto same = context.scratchpad_memory(ScratchDescriptor(257));
    auto smaller = context.scratchpad_memory(ScratchDescriptor(96));
    Expect(same.get_data_handle() == small.get_data_handle() && smaller.get_data_handle() == small.get_data_handle(),
           "Same/smaller scratch descriptors failed to reuse storage");
    Expect(smaller.get_desc() == ScratchDescriptor(96), "Scratch handle retained an old descriptor");

    // Replacement temporarily owns both allocations. Reject one byte below
    // that peak, preserving the old usable buffer and every ledger field.
    context.set_allocation_limits(257 + 769 - 1, 0);
    const auto beforeFailure = ReadUsage(context);
    ExpectThrows<LimitExceeded>([&] { context.scratchpad_memory(ScratchDescriptor(769)); }, "Scratch growth budget");
    Expect(ReadUsage(context) == beforeFailure, "Failed scratch growth changed accounting");
    auto retained = context.scratchpad_memory(ScratchDescriptor(257));
    std::vector<uint8_t> bytes(257);
    context.memcpy_d2h(bytes.data(), retained.get_data_handle(), bytes.size());
    Expect(std::all_of(bytes.begin(), bytes.end(), [](uint8_t value) { return value == 0x5a; }),
           "Failed scratch growth destroyed the previous scratch contents");

    context.set_allocation_limits(257 + 769, 0);
    auto grown = context.scratchpad_memory(ScratchDescriptor(769));
    Expect(reinterpret_cast<uintptr_t>(grown.get_data_handle()) != oldAddress,
           "Growing live scratch did not replace its backing allocation");
    ExpectUsage(context, 769, 0, 1, 0, "Shared scratch growth");
    Expect(context.peak_allocated_bytes() == 257 + 769, "Transient scratch replacement was omitted from the peak");
    context.set_allocation_limits(769, 0);
    Expect(context.scratchpad_memory(ScratchDescriptor(96)).get_data_handle() == grown.get_data_handle(),
           "Smaller scratch descriptor retained a freed address");
    context.synchronize();
    context.release_execution_resources();
    ExpectUsage(context, 0, 0, 0, 0, "Shared scratch release");
    context.scratchpad_memory(ScratchDescriptor(257));
    ExpectUsage(context, 257, 0, 1, 0, "Shared scratch recreation");
    context.synchronize();
    context.release_execution_resources();
    ExpectUsage(context, 0, 0, 0, 0, "Recreated shared scratch release");
}

void ForceScratchGrowth(Context& context) {
    auto prior = context.scratchpad_memory(ScratchDescriptor(1));
    const auto oldAddress = reinterpret_cast<uintptr_t>(prior.get_data_handle());
    const auto before = ReadUsage(context);
    // Total current payload is an upper bound on scratch capacity, so this
    // guarantees replacement without depending on oneDNN's chosen algorithm.
    Expect(before.deviceBytes < 64 * 1024 * 1024, "Small-shape fixture exceeded its scratch growth bound");
    auto replacement = context.scratchpad_memory(ScratchDescriptor(before.deviceBytes + 4096));
    Expect(reinterpret_cast<uintptr_t>(replacement.get_data_handle()) != oldAddress &&
           context.device_allocation_count() == before.deviceCount,
           "Forced shared scratch growth did not replace and release storage");
}

void ExpectNear(float actual, float expected, float absolute, float relative,
                const std::string& label, size_t index) {
    Expect(std::isfinite(actual) && std::abs(actual - expected) <= absolute + relative * std::abs(expected),
           label + " index=" + std::to_string(index) + " actual=" + std::to_string(actual) +
           " expected=" + std::to_string(expected));
}

void TestLinearScratchRebinding() {
    constexpr int inputs = 64, outputs = 48, maximumRows = 65;
    Context context;
    {
        Tensor weights = Tensor::allocate(context, {inputs, outputs});
        Tensor input = Tensor::allocate(context, {maximumRows, inputs});
        Tensor output = Tensor::allocate(context, {maximumRows, outputs});
        Tensor outputFloat = Tensor::allocate(context, {maximumRows, outputs}, DataType::f32);
        Linear linear;
        DrainQueue drain{context};
        linear.init(context, weights, inputs, outputs, true, false);
        auto run = [&](int rows, int salt) {
            std::vector<bf16> hostInput(static_cast<size_t>(rows) * inputs);
            std::vector<bf16> hostWeights(static_cast<size_t>(inputs) * outputs);
            for (size_t i = 0; i < hostInput.size(); ++i)
                hostInput[i] = bf16((static_cast<int>((i * 3 + salt) % 13) - 6) / 16.0f);
            for (size_t i = 0; i < hostWeights.size(); ++i)
                hostWeights[i] = bf16((static_cast<int>((i * 5 + salt) % 11) - 5) / 32.0f);
            context.memcpy_h2d(input.data(), hostInput.data(), hostInput.size() * sizeof(bf16));
            context.memcpy_h2d(weights.data(), hostWeights.data(), hostWeights.size() * sizeof(bf16));
            linear.forward(context, input, output, rows);
            linear.forward_f32(context, input, outputFloat, rows);
            context.synchronize();
            std::vector<bf16> actual(static_cast<size_t>(rows) * outputs);
            std::vector<float> actualFloat(actual.size());
            context.memcpy_d2h(actual.data(), output.data(), actual.size() * sizeof(bf16));
            context.memcpy_d2h(actualFloat.data(), outputFloat.data(), actualFloat.size() * sizeof(float));
            for (int row = 0; row < rows; ++row) {
                for (int column = 0; column < outputs; ++column) {
                    float expected = 0;
                    for (int k = 0; k < inputs; ++k)
                        expected += static_cast<float>(hostInput[row * inputs + k]) *
                                    static_cast<float>(hostWeights[k * outputs + column]);
                    const size_t index = static_cast<size_t>(row) * outputs + column;
                    ExpectNear(actualFloat[index], expected, 2e-5f, 2e-5f, "Linear f32", index);
                    ExpectNear(static_cast<float>(actual[index]), static_cast<float>(bf16(expected)),
                               0.002f, 0.002f, "Linear bf16", index);
                }
            }
        };
        run(7, 1);
        run(7, 2);
        ForceScratchGrowth(context);
        run(7, 3);
        run(maximumRows, 4);
        ForceScratchGrowth(context);
        run(7, 5);
        context.synchronize();
        context.release_execution_resources();
        // The Linear owner survives this release. Cached primitives must bind
        // the next Context scratch allocation, not their prior freed handle.
        run(7, 6);
        context.synchronize();
    }
    context.synchronize();
    context.release_execution_resources();
    ExpectUsage(context, 0, 0, 0, 0, "Linear owners and shared scratch destroyed");
}

void TestMaskedAttentionCache() {
    constexpr int maximumRows = 24, kvLength = 17, heads = 4, kvHeads = 2, headSize = 16;
    constexpr int qWidth = heads * headSize, kvWidth = kvHeads * headSize;
    Context context;
    {
        Tensor q = Tensor::allocate(context, {maximumRows, qWidth});
        Tensor k = Tensor::allocate(context, {kvLength, kvWidth});
        Tensor v = Tensor::allocate(context, {kvLength, kvWidth});
        Tensor mask = Tensor::allocate(context, {maximumRows, kvLength}, DataType::f32);
        Tensor output = Tensor::allocate(context, {maximumRows, qWidth});
        Tensor scores = Tensor::allocate(context, {heads, maximumRows, kvLength}, DataType::f32);
        DrainQueue drain{context};
        const auto tensorsOnly = ReadUsage(context);
        // Ensure a shared scratch allocation is present before counting cache
        // entries; algorithm-dependent scratch growth then keeps one allocation.
        context.scratchpad_memory(ScratchDescriptor(1));
        auto run = [&](int rows, int salt) {
            std::vector<bf16> hq(static_cast<size_t>(rows) * qWidth);
            std::vector<bf16> hk(static_cast<size_t>(kvLength) * kvWidth), hv(hk.size());
            std::vector<float> hm(static_cast<size_t>(rows) * kvLength);
            for (size_t i = 0; i < hq.size(); ++i)
                hq[i] = bf16((static_cast<int>((i * 7 + salt) % 17) - 8) / 16.0f);
            for (size_t i = 0; i < hk.size(); ++i) {
                hk[i] = bf16((static_cast<int>((i * 3 + salt) % 13) - 6) / 16.0f);
                hv[i] = bf16((static_cast<int>((i * 5 + salt) % 19) - 9) / 16.0f);
            }
            for (int row = 0; row < rows; ++row) {
                const int visible = 1 + (row + salt) % kvLength;
                for (int token = 0; token < kvLength; ++token)
                    hm[row * kvLength + token] = token < visible ? -0.125f * (token % 5) : -10000.0f;
            }
            context.memcpy_h2d(q.data(), hq.data(), hq.size() * sizeof(bf16));
            context.memcpy_h2d(k.data(), hk.data(), hk.size() * sizeof(bf16));
            context.memcpy_h2d(v.data(), hv.data(), hv.size() * sizeof(bf16));
            context.memcpy_h2d(mask.data(), hm.data(), hm.size() * sizeof(float));
            // This small head configuration always selects the fallback,
            // independently of AILA_DG_FUSED_ATTN.
            ops::attention_prefill_masked(context, q, k, v, mask, output, scores,
                                          rows, kvLength, heads, kvHeads, headSize);
            context.synchronize();
            std::vector<bf16> actual(hq.size());
            context.memcpy_d2h(actual.data(), output.data(), actual.size() * sizeof(bf16));
            for (int row = 0; row < rows; ++row) {
                for (int head = 0; head < heads; ++head) {
                    const int kvHead = head / (heads / kvHeads);
                    float logits[kvLength], probabilities[kvLength];
                    float maximum = -std::numeric_limits<float>::infinity();
                    for (int token = 0; token < kvLength; ++token) {
                        float dot = 0;
                        for (int d = 0; d < headSize; ++d)
                            dot += static_cast<float>(hq[row * qWidth + head * headSize + d]) *
                                   static_cast<float>(hk[token * kvWidth + kvHead * headSize + d]);
                        // The public op expects pre-scaled Q: no extra sqrt(D).
                        // Its observable contract rounds QK and probabilities
                        // to bf16 before the next operation.
                        logits[token] = static_cast<float>(bf16(dot)) + hm[row * kvLength + token];
                        maximum = std::max(maximum, logits[token]);
                    }
                    float sum = 0;
                    for (int token = 0; token < kvLength; ++token) {
                        probabilities[token] = std::exp(logits[token] - maximum);
                        sum += probabilities[token];
                    }
                    for (float& probability : probabilities)
                        probability = static_cast<float>(bf16(probability / sum));
                    for (int d = 0; d < headSize; ++d) {
                        float expected = 0;
                        for (int token = 0; token < kvLength; ++token)
                            expected += probabilities[token] * static_cast<float>(hv[token * kvWidth + kvHead * headSize + d]);
                        const size_t index = static_cast<size_t>(row) * qWidth + head * headSize + d;
                        // A few bf16 ULPs cover exp/rounding differences while
                        // retaining sensitivity to masks, grouping and stale data.
                        ExpectNear(static_cast<float>(actual[index]), static_cast<float>(bf16(expected)),
                                   0.003f, 0.005f, "Masked attention rows=" + std::to_string(rows), index);
                    }
                }
            }
        };

        size_t firstShapeCount = 0, boundedCount = 0;
        for (int rows = 1; rows <= maximumRows; ++rows) {
            run(rows, rows);
            if (rows == 1) firstShapeCount = context.device_allocation_count();
            if (rows == 7) {
                ForceScratchGrowth(context);
                run(rows, 31);
            }
            if (rows == 16) {
                boundedCount = context.device_allocation_count();
                Expect(boundedCount > firstShapeCount, "Fixture did not exercise multiple attention cache entries");
            }
            if (rows > 16)
                Expect(context.device_allocation_count() <= boundedCount,
                       "Attention shape cache kept growing after 16 entries");
        }
        run(1, 37);
        Expect(context.device_allocation_count() <= boundedCount,
               "Revisiting an old attention shape leaked a cache entry");
        context.synchronize();
        context.release_execution_resources();
        ExpectUsage(context, tensorsOnly.deviceBytes, 0, tensorsOnly.deviceCount, 0,
                    "Attention cache and scratch release");
        run(7, 41);
        context.synchronize();
        context.release_execution_resources();
        ExpectUsage(context, tensorsOnly.deviceBytes, 0, tensorsOnly.deviceCount, 0,
                    "Recreated attention cache release");
    }
    ExpectUsage(context, 0, 0, 0, 0, "Attention tensors destroyed");
}

void TestRuntimeMemorySnapshot() {
    using aila::inference::DeviceRuntime;
    auto runtime = DeviceRuntime::Create();
    auto shared = DeviceRuntime::Create();
    const auto empty = runtime->Memory();
    Expect(empty.currentBytes == 0 && empty.pinnedHostBytes == 0 && empty.deviceAllocations == 0 &&
           empty.pinnedHostAllocations == 0 && empty.peakBytes == 0 && empty.peakPinnedHostBytes == 0,
           "Empty runtime did not report zero tracked USM");
    runtime->SetAllocationLimits(4096, 2048);
    const auto limited = shared->Memory();
    Expect(limited.deviceLimitBytes == 4096 && limited.pinnedHostLimitBytes == 2048 &&
           limited.currentBytes == 0 && limited.pinnedHostBytes == 0,
           "Shared runtime handles did not report configured budgets");
    shared->SetAllocationLimits(0, 0);
    Expect(runtime->Memory().deviceLimitBytes == 0 && runtime->Memory().pinnedHostLimitBytes == 0,
           "Zero runtime limits must disable both ceilings");
}

} // namespace

int main() {
    try {
        // This executable owns its environment and runs GPU work serially.
        // Set before can_use_device_sampling's first cached environment read.
#ifdef _WIN32
        Expect(_putenv_s("AILA_DEVICE_SAMPLING", "1") == 0, "Cannot enable device sampling fixture");
#else
        Expect(setenv("AILA_DEVICE_SAMPLING", "1", 1) == 0, "Cannot enable device sampling fixture");
#endif
        const struct { const char* name; void (*run)(); } cases[] = {
            {"context-budget-and-order", TestContextBudgetAndOrder},
            {"typed-workspace-ownership", TestTypedWorkspaceOwnership},
            {"sampling-isolation-and-reuse", TestSamplingIsolationAndReuse},
            {"shared-scratch-budget", TestSharedScratchBudget},
            {"linear-scratch-rebinding", TestLinearScratchRebinding},
            {"masked-attention-cache", TestMaskedAttentionCache},
            {"runtime-memory-snapshot", TestRuntimeMemorySnapshot},
        };
        for (const auto& test : cases) {
            std::cout << "[RUN] " << test.name << std::endl;
            test.run();
            std::cout << "[PASS] " << test.name << std::endl;
        }
        std::cout << "Execution resource tests passed (tracked Context USM only)." << std::endl;
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Execution resource test failed: " << error.what() << std::endl;
        return 1;
    }
}
