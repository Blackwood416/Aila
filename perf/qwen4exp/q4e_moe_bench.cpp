#include "core/Context.hpp"
#include "ops/MoeExpertOps.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;
using bf16 = sycl::ext::oneapi::bfloat16;

constexpr int kIn = 2560;
constexpr int kIntermediate = 640;
constexpr int kBlock = 64;
constexpr size_t kGateUpPacked = static_cast<size_t>(2 * kIntermediate) * (kIn / 2);
constexpr size_t kGateUpScales = static_cast<size_t>(2 * kIntermediate) * (kIn / kBlock);
constexpr size_t kDownPacked = static_cast<size_t>(kIn) * (kIntermediate / 2);
constexpr size_t kDownScales = static_cast<size_t>(kIn) * (kIntermediate / kBlock);
constexpr size_t kSlotBytes = kGateUpPacked + kGateUpScales * sizeof(float) +
                              kDownPacked + kDownScales * sizeof(float);
constexpr size_t kCompactSlotBytes = kGateUpPacked + kGateUpScales +
                                     kDownPacked + kDownScales;
constexpr int kNestedBlock = 256;
constexpr int kCheckpointExperts = 512;
constexpr size_t kGuNestedPerExpert = kGateUpScales / kNestedBlock;
constexpr size_t kDnNestedPerExpert = kDownScales / kNestedBlock;
constexpr size_t kLayerNestedFloats = kCheckpointExperts *
    (kGuNestedPerExpert + kDnNestedPerExpert) + 2 * 256;

struct Sample {
    double wall_ms = 0;
    double submit_ms = 0;
    double gate_device_ms = 0;
    double down_device_ms = 0;
    double device_span_ms = 0;
};

double elapsed_ms(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

uint64_t start_ns(const sycl::event& e) {
    return e.get_profiling_info<sycl::info::event_profiling::command_start>();
}

uint64_t end_ns(const sycl::event& e) {
    return e.get_profiling_info<sycl::info::event_profiling::command_end>();
}

double quantile(std::vector<double> values, double q) {
    std::sort(values.begin(), values.end());
    const double pos = q * static_cast<double>(values.size() - 1);
    const size_t lo = static_cast<size_t>(pos);
    const size_t hi = std::min(lo + 1, values.size() - 1);
    return values[lo] + (values[hi] - values[lo]) * (pos - static_cast<double>(lo));
}

void report(const char* label, const std::vector<Sample>& samples, double Sample::*field) {
    std::vector<double> values;
    values.reserve(samples.size());
    for (const auto& s : samples) values.push_back(s.*field);
    const double mean = std::accumulate(values.begin(), values.end(), 0.0) / values.size();
    double var = 0;
    for (double v : values) var += (v - mean) * (v - mean);
    const double cv = std::sqrt(var / values.size()) / mean;
    std::cout << label << " median=" << quantile(values, 0.5)
              << " p10=" << quantile(values, 0.1)
              << " p90=" << quantile(values, 0.9)
              << " cv=" << cv * 100.0 << "%\n";
}

int positive_arg(const char* text) {
    const int value = std::stoi(text);
    if (value <= 0) throw std::invalid_argument("benchmark arguments must be positive");
    return value;
}

} // namespace

int main(int argc, char** argv) {
    try {
        int layers = 48;
        int experts = 10;
        int samples_per_repeat = 20;
        int repeats = 3;
        bool nested = false;
        for (int i = 1; i < argc; i += 2) {
            if (i + 1 >= argc) throw std::invalid_argument("expected value after option");
            const std::string key(argv[i]);
            if (key == "--nested") {
                const int value = std::stoi(argv[i + 1]);
                if (value != 0 && value != 1) throw std::invalid_argument("--nested must be 0 or 1");
                nested = (value == 1);
                continue;
            }
            if (key == "--test-host-dma") {
                const int host_gb = positive_arg(argv[i + 1]);
                Context ctx(nullptr, true);
                std::cout << "[Host Pool Test] Testing " << host_gb << " GiB pinned USM host allocation in 1 GiB chunks..." << std::endl;
                std::vector<void*> chunks;
                const size_t chunk_bytes = 1024ULL * 1024ULL * 1024ULL;
                auto t0 = Clock::now();
                for (int g = 0; g < host_gb; ++g) {
                    try {
                        void* p = ctx.alloc_host(chunk_bytes);
                        chunks.push_back(p);
                    } catch (const std::exception& e) {
                        std::cout << "Failed at chunk " << g << " (" << g << " GiB allocated): " << e.what() << std::endl;
                        break;
                    }
                }
                auto t1 = Clock::now();
                std::cout << "[Host Pool Test] Successfully allocated " << chunks.size() << " GiB pinned host memory in "
                          << elapsed_ms(t0, t1) << " ms" << std::endl;

                if (!chunks.empty()) {
                    constexpr size_t test_xfer_bytes = 2534400ULL * 100;
                    void* dev_buf = ctx.alloc_device(test_xfer_bytes);
                    std::cout << "[Host Pool Test] Testing DMA transfer of " << test_xfer_bytes / (1024.0 * 1024.0) << " MB from pinned host..." << std::endl;
                    std::vector<double> dma_bw;
                    for (int iter = 0; iter < 10; ++iter) {
                        auto t_start = Clock::now();
                        auto ev = ctx.queue().memcpy(dev_buf, chunks[0], test_xfer_bytes);
                        ev.wait();
                        auto t_end = Clock::now();
                        double ms = elapsed_ms(t_start, t_end);
                        double gbps = (test_xfer_bytes / 1e9) / (ms / 1e3);
                        dma_bw.push_back(gbps);
                    }
                    std::sort(dma_bw.begin(), dma_bw.end());
                    std::cout << "[Host Pool Test] Direct Pinned H2D DMA Bandwidth: median=" << dma_bw[5]
                              << " GB/s (p10=" << dma_bw[1] << ", p90=" << dma_bw[8] << ")" << std::endl;
                    ctx.free_device(dev_buf);
                }
                for (void* p : chunks) ctx.free_host(p);
                std::cout << "[Host Pool Test] Cleanup complete." << std::endl;
                return 0;
            }
            const int value = positive_arg(argv[i + 1]);
            if (key == "--layers") layers = value;
            else if (key == "--experts") experts = value;
            else if (key == "--samples") samples_per_repeat = value;
            else if (key == "--repeats") repeats = value;
            else throw std::invalid_argument("unknown option: " + key);
        }
        if (experts > 16) throw std::invalid_argument("the MoE kernel supports at most 16 experts");

        Context ctx(nullptr, true);
        auto& q = ctx.queue();
        const size_t slot_count = static_cast<size_t>(layers) * experts;
        const size_t slot_bytes = nested ? kCompactSlotBytes : kSlotBytes;
        const size_t pool_bytes = slot_count * slot_bytes;
        std::cout << std::fixed << std::setprecision(3)
                  << "device=" << q.get_device().get_info<sycl::info::device::name>() << "\n"
                  << "sycl_driver=" << q.get_device().get_info<sycl::info::device::driver_version>() << "\n"
                  << "layers=" << layers << " experts=" << experts
                  << " nested=" << nested << " slot_bytes=" << slot_bytes << " pool_bytes=" << pool_bytes
                  << " samples=" << samples_per_repeat << " repeats=" << repeats << "\n";

        // Each slot has the production packed + f32 or nested-code geometry.
        auto* pool = static_cast<uint8_t*>(ctx.alloc_device(pool_bytes));
        float* metadata = nested ? static_cast<float*>(ctx.alloc_device(
            static_cast<size_t>(layers) * kLayerNestedFloats * sizeof(float))) : nullptr;
        std::vector<ops::MoeExpertSlotDev> slots(slot_count);
        for (size_t s = 0; s < slot_count; ++s) {
            uint8_t* base = pool + s * slot_bytes;
            const size_t gu_scale_bytes = kGateUpScales * (nested ? 1 : sizeof(float));
            const size_t dn_offset = kGateUpPacked + gu_scale_bytes;
            const size_t dn_scale_offset = dn_offset + kDownPacked;
            auto& slot = slots[s];
            slot.gate_up_packed = base;
            slot.down_packed = base + dn_offset;
            if (nested) {
                const size_t layer = s / static_cast<size_t>(experts);
                const size_t expert = s % static_cast<size_t>(experts);
                float* layer_meta = metadata + layer * kLayerNestedFloats;
                float* gu_map = layer_meta + kCheckpointExperts * kGuNestedPerExpert;
                float* dn_scales = gu_map + 256;
                float* dn_map = dn_scales + kCheckpointExperts * kDnNestedPerExpert;
                slot.gate_up_nested = {base + kGateUpPacked,
                    layer_meta + expert * kGuNestedPerExpert, gu_map, 0.0f, kNestedBlock};
                slot.down_nested = {base + dn_scale_offset,
                    dn_scales + expert * kDnNestedPerExpert, dn_map, 0.0f, kNestedBlock};
            } else {
                slot.gate_up_absmax = reinterpret_cast<float*>(base + kGateUpPacked);
                slot.down_absmax = reinterpret_cast<float*>(base + dn_scale_offset);
            }
        }
        q.memset(pool, 0x56, pool_bytes);
        constexpr size_t scales_per_slot = kGateUpScales + kDownScales;
        if (nested) {
            q.fill(metadata, 0.02f, static_cast<size_t>(layers) * kLayerNestedFloats);
            q.parallel_for(sycl::range<1>(static_cast<size_t>(layers) * 512), [=](sycl::id<1> id) {
                const size_t layer = id[0] / 512;
                const size_t map_index = id[0] % 512;
                float* layer_meta = metadata + layer * kLayerNestedFloats;
                float* gu_map = layer_meta + kCheckpointExperts * kGuNestedPerExpert;
                float* dn_map = gu_map + 256 + kCheckpointExperts * kDnNestedPerExpert;
                (map_index < 256 ? gu_map : dn_map)[map_index % 256] = 1.0f;
            });
        } else {
            q.parallel_for(sycl::range<1>(slot_count * scales_per_slot), [=](sycl::id<1> id) {
                const size_t index = id[0];
                const size_t slot = index / scales_per_slot;
                const size_t scale = index % scales_per_slot;
                uint8_t* base = pool + slot * slot_bytes;
                if (scale < kGateUpScales) {
                    reinterpret_cast<float*>(base + kGateUpPacked)[scale] = 0.02f;
                } else {
                    reinterpret_cast<float*>(base + kGateUpPacked +
                                             kGateUpScales * sizeof(float) + kDownPacked)
                        [scale - kGateUpScales] = 0.02f;
                }
            });
        }

        auto* x = static_cast<bf16*>(ctx.alloc_device(kIn * sizeof(bf16)));
        auto* h = static_cast<bf16*>(ctx.alloc_device(static_cast<size_t>(experts) *
                                                     kIntermediate * sizeof(bf16)));
        auto* out = static_cast<bf16*>(ctx.alloc_device(kIn * sizeof(bf16)));
        const std::vector<bf16> x_host(kIn, bf16(0.03f));
        const std::vector<float> weights(experts, 1.0f / experts);
        ctx.memcpy_h2d(x, x_host.data(), x_host.size() * sizeof(bf16));

        auto gate = [&](int layer) {
            return ops::moe_decode_gate_up_swiglu_grouped(
                ctx, slots.data() + static_cast<size_t>(layer) * experts,
                nullptr, x, h, experts, kIn, kIntermediate, kBlock);
        };
        auto down = [&](int layer) {
            return ops::moe_decode_down_weighted_accumulate(
                ctx, slots.data() + static_cast<size_t>(layer) * experts,
                nullptr, weights.data(), h, nullptr, out, experts,
                kIntermediate, kIn, kBlock);
        };

        // CPU reference for the uniform input and real slot geometry. The
        // randomized indexing test remains AilaQwen4ExpKernelTests.
        gate(0);
        down(0);
        ctx.synchronize();
        std::vector<bf16> h_host(static_cast<size_t>(experts) * kIntermediate);
        std::vector<bf16> out_host(kIn);
        ctx.memcpy_d2h(h_host.data(), h, h_host.size() * sizeof(bf16));
        ctx.memcpy_d2h(out_host.data(), out, out_host.size() * sizeof(bf16));
        constexpr float q5 = -0.18477343022823334f;
        constexpr float q6 = -0.09105003625154495f;
        const float input = static_cast<float>(x_host[0]);
        const float gu_sum = static_cast<float>(kIn / 2) * (q5 + q6) * 0.02f * input;
        const float h_ref = static_cast<float>(bf16(
            gu_sum / (1.0f + std::exp(-gu_sum)) * gu_sum));
        const float down_sum = static_cast<float>(kIntermediate / 2) *
                               (q5 + q6) * 0.02f * h_ref;
        const float out_ref = static_cast<float>(bf16(down_sum));
        double h_error = 0;
        double out_error = 0;
        for (bf16 v : h_host) {
            const float actual = static_cast<float>(v);
            if (!std::isfinite(actual) || !std::isfinite(h_ref)) {
                throw std::runtime_error("non-finite gate/up output or CPU reference");
            }
            h_error = std::max(h_error, static_cast<double>(std::abs(actual - h_ref)));
        }
        for (bf16 v : out_host) {
            const float actual = static_cast<float>(v);
            if (!std::isfinite(actual) || !std::isfinite(out_ref)) {
                throw std::runtime_error("non-finite down output or CPU reference");
            }
            out_error = std::max(out_error, static_cast<double>(std::abs(actual - out_ref)));
        }
        std::cout << "cpu_reference h_max_abs_error=" << h_error
                  << " out_max_abs_error=" << out_error << "\n";
        if (h_error > 0.001 || out_error > 0.001) {
            throw std::runtime_error("MoE kernel output disagrees with CPU reference");
        }

        // Randomize the full pool after the simple CPU check. Uniform bytes can
        // make a streaming control unrealistically fast on compressed memory.
        q.parallel_for(sycl::range<1>(pool_bytes / sizeof(uint32_t)), [=](sycl::id<1> id) {
            uint32_t v = static_cast<uint32_t>(id[0]);
            v ^= v >> 16;
            v *= 0x7feb352dU;
            v ^= v >> 15;
            v *= 0x846ca68bU;
            v ^= v >> 16;
            reinterpret_cast<uint32_t*>(pool)[id[0]] = v;
        });
        if (nested) {
            // Real checkpoint-sized per-layer metadata: 512 expert scale slices
            // and two 256-entry maps. Only ten slices are touched per layer.
            q.parallel_for(sycl::range<1>(static_cast<size_t>(layers) * kLayerNestedFloats),
                [=](sycl::id<1> id) {
                    const size_t layer = id[0] / kLayerNestedFloats;
                    const size_t index = id[0] % kLayerNestedFloats;
                    const size_t gu_map_begin = kCheckpointExperts * kGuNestedPerExpert;
                    const size_t dn_scale_begin = gu_map_begin + 256;
                    const size_t dn_map_begin = dn_scale_begin + kCheckpointExperts * kDnNestedPerExpert;
                    if (index < gu_map_begin || (index >= dn_scale_begin && index < dn_map_begin)) {
                        metadata[layer * kLayerNestedFloats + index] =
                            0.015f + static_cast<float>(index % 17) * 0.0003f;
                    }
                });
        } else {
            q.parallel_for(sycl::range<1>(slot_count * scales_per_slot), [=](sycl::id<1> id) {
                const size_t index = id[0];
                const size_t slot = index / scales_per_slot;
                const size_t scale = index % scales_per_slot;
                const float value = 0.015f + static_cast<float>(index % 17) * 0.0003f;
                uint8_t* base = pool + slot * slot_bytes;
                if (scale < kGateUpScales) {
                    reinterpret_cast<float*>(base + kGateUpPacked)[scale] = value;
                } else {
                    reinterpret_cast<float*>(base + kGateUpPacked +
                                             kGateUpScales * sizeof(float) + kDownPacked)
                        [scale - kGateUpScales] = value;
                }
            });
        }
        ctx.synchronize();

        // The first forward JITs both kernels; these untimed full passes settle
        // GPU clocks and exercise all 48 distinct layer weight sets.
        for (int warmup = 0; warmup < 3; ++warmup) {
            for (int layer = 0; layer < layers; ++layer) {
                gate(layer);
                down(layer);
            }
            ctx.synchronize();
        }

        std::vector<Sample> measurements;
        measurements.reserve(static_cast<size_t>(samples_per_repeat) * repeats);
        std::vector<sycl::event> events;
        events.reserve(static_cast<size_t>(layers) * 2);
        for (int repeat = 0; repeat < repeats; ++repeat) {
            for (int sample = 0; sample < samples_per_repeat; ++sample) {
                events.clear();
                const auto wall_start = Clock::now();
                for (int layer = 0; layer < layers; ++layer) {
                    events.push_back(gate(layer));
                    events.push_back(down(layer));
                }
                const auto submitted = Clock::now();
                ctx.synchronize();
                const auto complete = Clock::now();
                Sample m;
                m.wall_ms = elapsed_ms(wall_start, complete);
                m.submit_ms = elapsed_ms(wall_start, submitted);
                m.device_span_ms = static_cast<double>(end_ns(events.back()) -
                                                       start_ns(events.front())) / 1e6;
                for (size_t i = 0; i < events.size(); i += 2) {
                    m.gate_device_ms += static_cast<double>(end_ns(events[i]) -
                                                            start_ns(events[i])) / 1e6;
                    m.down_device_ms += static_cast<double>(end_ns(events[i + 1]) -
                                                            start_ns(events[i + 1])) / 1e6;
                }
                measurements.push_back(m);
            }
        }
        std::cout << "moe_batch_ms (all times per " << layers << " layers):\n";
        report("wall", measurements, &Sample::wall_ms);
        report("enqueue_loop", measurements, &Sample::submit_ms);
        report("device_span", measurements, &Sample::device_span_ms);
        report("gate_device_sum", measurements, &Sample::gate_device_ms);
        report("down_device_sum", measurements, &Sample::down_device_ms);

        // Smallest no-op kernel batch with the same number of in-order
        // submissions. Enqueue time includes runtime work and backpressure.
        auto empty_batch = [&]() {
            std::vector<sycl::event> empty_events;
            empty_events.reserve(static_cast<size_t>(layers) * 2);
            const auto begin = Clock::now();
            for (int i = 0; i < layers * 2; ++i) {
                empty_events.push_back(q.single_task([] {}));
            }
            const auto submitted = Clock::now();
            ctx.synchronize();
            Sample m;
            m.wall_ms = elapsed_ms(begin, Clock::now());
            m.submit_ms = elapsed_ms(begin, submitted);
            m.device_span_ms = static_cast<double>(end_ns(empty_events.back()) -
                                                   start_ns(empty_events.front())) / 1e6;
            for (const auto& event : empty_events) {
                m.gate_device_ms += static_cast<double>(end_ns(event) - start_ns(event)) / 1e6;
            }
            return m;
        };
        empty_batch();
        std::vector<Sample> empty_samples;
        for (int i = 0; i < samples_per_repeat * repeats; ++i) {
            empty_samples.push_back(empty_batch());
        }
        std::cout << "empty_batch_ms (" << layers * 2 << " launches):\n";
        report("wall", empty_samples, &Sample::wall_ms);
        report("enqueue_loop", empty_samples, &Sample::submit_ms);
        report("device_span", empty_samples, &Sample::device_span_ms);
        report("kernel_device_sum", empty_samples, &Sample::gate_device_ms);

        // A coalesced streaming-read control over the same bytes. It is an
        // achievable bound, not a substitute for the MoE access pattern.
        const size_t stream_chunks = pool_bytes / 256;
        auto* sums = static_cast<uint32_t*>(ctx.alloc_device(stream_chunks * sizeof(uint32_t)));
        auto stream_read = [&]() {
            const auto begin = Clock::now();
            auto event = q.parallel_for(sycl::range<1>(stream_chunks), [=](sycl::id<1> id) {
                const auto* words = reinterpret_cast<const uint32_t*>(pool) + id[0] * 64;
                uint32_t sum = 0;
                for (int block = 0; block < 4; ++block) {
                    const auto v = *reinterpret_cast<const sycl::vec<uint32_t, 16>*>(words + block * 16);
                    for (int lane = 0; lane < 16; ++lane) sum += v[lane];
                }
                sums[id[0]] = sum;
            });
            const auto submitted = Clock::now();
            ctx.synchronize();
            const auto complete = Clock::now();
            return Sample{elapsed_ms(begin, complete), elapsed_ms(begin, submitted),
                          static_cast<double>(end_ns(event) - start_ns(event)) / 1e6};
        };
        stream_read();
        std::vector<Sample> stream_samples;
        for (int i = 0; i < samples_per_repeat * repeats; ++i) {
            stream_samples.push_back(stream_read());
        }
        std::cout << "streaming_read_ms (" << pool_bytes << " input bytes):\n";
        report("wall", stream_samples, &Sample::wall_ms);
        report("device", stream_samples, &Sample::gate_device_ms);
        const double stream_device_ms = quantile([&] {
            std::vector<double> values;
            for (const auto& s : stream_samples) values.push_back(s.gate_device_ms);
            return values;
        }(), 0.5);
        std::cout << "streaming_read_GBps=" << pool_bytes / (stream_device_ms * 1e6) << "\n";

        ctx.free_device(sums);
        ctx.free_device(out);
        ctx.free_device(h);
        ctx.free_device(x);
        if (metadata) ctx.free_device(metadata);
        ctx.free_device(pool);
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "q4e_moe_bench failed: " << e.what() << "\n";
        return 1;
    }
}
