// Qwen4Exp host-USM streaming-read probe (role P5, 2026-10-16).
//
// Question: what is the effective kernel-read rate R from host USM (pinned
// sycl::malloc_host) on this A770 box? Candidate Z (delete the H2D pump and
// point the MoE consumer descriptors at host memory) is only feasible if the
// kernel can stream host bytes at a rate comparable to the 9.89 GB/s H2D
// memcpy control measured by the supply bench. Z has two variants:
//   Z1 - keep the host fill, read the existing staging bank directly.
//   Z2 - read the large-page arena slice directly (delete fill + H2D).
// See docs/plans/qwen38-r1-prefill-overlap-design-2026-10-16.md section 8.
//
// Arms per size (median of --reps, warmup discarded):
//   h2d       queue memcpy host->device, the bandwidth control.
//   dev-flat  streaming read of device USM, uint32 per lane, coalesced.
//   host-flat same pattern, host USM pointer.
//   dev-flat16 / host-flat16: 16B vector loads (vec<uint32_t,4> per lane).
//   dev-moe / host-moe: production expert-major gate/up shape at the real
//   geometry (src/ops/MoeExpertOps.cpp:1584): one sub-group per packed row,
//   16 lanes x 20 uint32 steps of 64B, two distant streams per step (gate
//   and up row), plus leader-only absmax float gathers (40 floats per row,
//   blocksize 64). Each device arm mirrors the host arm byte for byte so
//   R(device) is a roofline reference for the same access pattern.
//
// Guard: every arm reads every 32-bit word of the effective prefix exactly
// once. The kernel accumulates two order-independent sums (sum, sum-of-xor)
// into a device sink; the host compares both against a CPU reference. Any
// mismatch fails the arm closed (guard_ok=0), so a partially executed walk
// cannot report an inflated GB/s.
//
// ASCII only. No allocation inside the timed reps. GPU execution is the
// coordinator's; this file only needs a syntax check here.

#include "core/Context.hpp"

#include <sycl/sycl.hpp>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

constexpr uint32_t kXorMask = 0xA5A5A5A5u;
constexpr size_t kMoeCellBytes = 1440;   // one packed row: 1280B weights + 160B absmax
constexpr size_t kMoePairBytes = 2 * kMoeCellBytes;
constexpr size_t kMoeCodesBytes = 160;
constexpr int kMoeSteps = 20;            // 1280B row / (16 lanes * 4B per step)
constexpr int kWg = 256;
constexpr int kSg = 16;
constexpr int kSgsPerWg = kWg / kSg;

struct Sink {
    uint64_t sum;
    uint64_t sum2;
};

struct Refs {
    uint64_t sum = 0;
    uint64_t sum2 = 0;
};

uint32_t mix(uint32_t x) {
    x ^= x >> 16;
    x *= 0x7feb352dU;
    x ^= x >> 15;
    x *= 0x846ca68bU;
    return x ^ (x >> 16);
}

double median(std::vector<double> values) {
    if (values.empty()) return 0.0;
    std::sort(values.begin(), values.end());
    return values[values.size() / 2];
}

void check(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}

// Keep the total thread count bounded while still covering every unit exactly
// once: thread g reads units g, g+threads, g+2*threads, ...
size_t pick_threads(size_t units, size_t cap) {
    size_t threads = std::min(units, cap);
    threads = std::max<size_t>(threads, kWg);
    return ((threads + kWg - 1) / kWg) * kWg;
}

// Flat streaming read. Vec == 1 loads uint32 per lane; Vec == 4 loads a
// 16B sycl::vec<uint32_t,4> per lane.
template <int Vec>
sycl::event launch_flat(Context& ctx, const uint32_t* src, size_t units,
                        size_t threads, Sink* sink) {
    return ctx.queue().submit([&](sycl::handler& cgh) {
        sycl::local_accessor<uint64_t, 1> red(sycl::range<1>(2 * kWg), cgh);
        cgh.parallel_for(sycl::nd_range<1>(threads, kWg), [=](sycl::nd_item<1> item) {
            const size_t gid = item.get_global_id(0);
            uint64_t a1 = 0;
            uint64_t a2 = 0;
            for (size_t u = gid; u < units; u += threads) {
                if constexpr (Vec == 4) {
                    const sycl::vec<uint32_t, 4> v =
                        *reinterpret_cast<const sycl::vec<uint32_t, 4>*>(src + u * 4);
#pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        const uint32_t w = v[j];
                        a1 += static_cast<uint64_t>(w);
                        a2 += static_cast<uint64_t>(w ^ kXorMask);
                    }
                } else {
                    const uint32_t w = src[u];
                    a1 += static_cast<uint64_t>(w);
                    a2 += static_cast<uint64_t>(w ^ kXorMask);
                }
            }
            const int lid = static_cast<int>(item.get_local_id(0));
            red[lid] = a1;
            red[kWg + lid] = a2;
            item.barrier(sycl::access::fence_space::local_space);
            if (lid == 0) {
                uint64_t s1 = 0;
                uint64_t s2 = 0;
                for (int i = 0; i < kWg; ++i) {
                    s1 += red[i];
                    s2 += red[kWg + i];
                }
                sycl::atomic_ref<uint64_t, sycl::memory_order::relaxed,
                                 sycl::memory_scope::device,
                                 sycl::access::address_space::global_space>(sink->sum)
                    .fetch_add(s1);
                sycl::atomic_ref<uint64_t, sycl::memory_order::relaxed,
                                 sycl::memory_scope::device,
                                 sycl::access::address_space::global_space>(sink->sum2)
                    .fetch_add(s2);
            }
        });
    });
}

// Production expert-major gate/up access shape. The buffer is a sequence of
// 1440B cells (1280B packed row + 40 absmax floats). One sub-group owns one
// pair of cells: cell0 = pair, cell1 = pair + pairs, i.e. two streams half
// the buffer apart, like the gate and up halves of a real expert slot.
sycl::event launch_moe(Context& ctx, const uint8_t* base, size_t pairs, Sink* sink) {
    const size_t groups = (pairs + kSgsPerWg - 1) / kSgsPerWg;
    const size_t threads = groups * kWg;
    return ctx.queue().submit([&](sycl::handler& cgh) {
        sycl::local_accessor<uint64_t, 1> red(sycl::range<1>(2 * kWg), cgh);
        cgh.parallel_for(sycl::nd_range<1>(threads, kWg),
                         [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
            const int lid = static_cast<int>(item.get_local_id(0));
            const int sg_id = lid / kSg;
            const int lane = lid % kSg;
            const size_t pair = static_cast<size_t>(item.get_group(0)) * kSgsPerWg +
                                static_cast<size_t>(sg_id);
            uint64_t a1 = 0;
            uint64_t a2 = 0;
            if (pair < pairs) {
                const uint8_t* cell0 = base + pair * kMoePairBytes;
                const uint8_t* cell1 = base + (pair + pairs) * kMoeCellBytes;
                const auto* codes0 =
                    reinterpret_cast<const uint32_t*>(cell0 + kMoeCellBytes - kMoeCodesBytes);
                const auto* codes1 =
                    reinterpret_cast<const uint32_t*>(cell1 + kMoeCellBytes - kMoeCodesBytes);
                for (int k = 0; k < kMoeSteps; ++k) {
                    const size_t off =
                        static_cast<size_t>(lane) * 4 + static_cast<size_t>(k) * 64;
                    const uint32_t gw = *reinterpret_cast<const uint32_t*>(cell0 + off);
                    const uint32_t uw = *reinterpret_cast<const uint32_t*>(cell1 + off);
                    a1 += static_cast<uint64_t>(gw);
                    a2 += static_cast<uint64_t>(gw ^ kXorMask);
                    a1 += static_cast<uint64_t>(uw);
                    a2 += static_cast<uint64_t>(uw ^ kXorMask);
                    if (lane == 0) {
                        const uint32_t c0 = codes0[2 * k];
                        const uint32_t c1 = codes0[2 * k + 1];
                        a1 += static_cast<uint64_t>(c0) + static_cast<uint64_t>(c1);
                        a2 += static_cast<uint64_t>(c0 ^ kXorMask) +
                              static_cast<uint64_t>(c1 ^ kXorMask);
                    } else if (lane == 8) {
                        const uint32_t c0 = codes1[2 * k];
                        const uint32_t c1 = codes1[2 * k + 1];
                        a1 += static_cast<uint64_t>(c0) + static_cast<uint64_t>(c1);
                        a2 += static_cast<uint64_t>(c0 ^ kXorMask) +
                              static_cast<uint64_t>(c1 ^ kXorMask);
                    }
                }
            }
            red[lid] = a1;
            red[kWg + lid] = a2;
            item.barrier(sycl::access::fence_space::local_space);
            if (lid == 0) {
                uint64_t s1 = 0;
                uint64_t s2 = 0;
                for (int i = 0; i < kWg; ++i) {
                    s1 += red[i];
                    s2 += red[kWg + i];
                }
                sycl::atomic_ref<uint64_t, sycl::memory_order::relaxed,
                                 sycl::memory_scope::device,
                                 sycl::access::address_space::global_space>(sink->sum)
                    .fetch_add(s1);
                sycl::atomic_ref<uint64_t, sycl::memory_order::relaxed,
                                 sycl::memory_scope::device,
                                 sycl::access::address_space::global_space>(sink->sum2)
                    .fetch_add(s2);
            }
        });
    });
}

struct Params {
    std::vector<size_t> sizes_mib{512, 1024};
    int reps = 5;
    int warmup = 1;
    size_t threads_cap = 1u << 20;
    std::vector<std::string> arms;  // empty selects every arm
};

struct ArmOutcome {
    bool ran = false;
    bool guard_ok = false;
    size_t bytes = 0;
    double wall_ms = 0.0;
    double kernel_ms = -1.0;
    double gbps = 0.0;
};

size_t parse_size(std::string text) {
    size_t value = static_cast<size_t>(std::stoull(text));
    if (value == 0) throw std::invalid_argument("size must be positive");
    return value;
}

Params parse_params(int argc, char** argv) {
    Params p;
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        const auto value_at = [&](const char* name) {
            if (i + 1 >= argc) throw std::invalid_argument(std::string(name) + " needs a value");
            return std::string(argv[++i]);
        };
        if (key == "--sizes") {
            p.sizes_mib.clear();
            std::string list = value_at("--sizes");
            size_t start = 0;
            while (start <= list.size()) {
                const size_t comma = list.find(',', start);
                const std::string part =
                    list.substr(start, comma == std::string::npos ? std::string::npos : comma - start);
                if (!part.empty()) p.sizes_mib.push_back(parse_size(part));
                if (comma == std::string::npos) break;
                start = comma + 1;
            }
            if (p.sizes_mib.empty()) throw std::invalid_argument("--sizes list is empty");
        } else if (key == "--reps") {
            p.reps = std::stoi(value_at("--reps"));
            if (p.reps < 1) throw std::invalid_argument("--reps must be >= 1");
        } else if (key == "--warmup") {
            p.warmup = std::stoi(value_at("--warmup"));
            if (p.warmup < 0) throw std::invalid_argument("--warmup must be >= 0");
        } else if (key == "--threads-cap") {
            p.threads_cap = parse_size(value_at("--threads-cap"));
        } else if (key == "--arms") {
            std::string list = value_at("--arms");
            size_t start = 0;
            while (start <= list.size()) {
                const size_t comma = list.find(',', start);
                const std::string part =
                    list.substr(start, comma == std::string::npos ? std::string::npos : comma - start);
                if (!part.empty()) p.arms.push_back(part);
                if (comma == std::string::npos) break;
                start = comma + 1;
            }
        } else if (key == "--smoke") {
            p.sizes_mib = {256};
            p.reps = 2;
        } else if (key == "--help" || key == "-h") {
            std::cout << "usage: AilaQwen4ExpHostReadBench [--sizes 512,1024] [--reps 5]"
                      << " [--warmup 1] [--threads-cap 1048576] [--smoke]"
                      << " [--arms h2d,dev-flat,host-flat,dev-flat16,host-flat16,dev-moe,host-moe]\n";
            std::exit(0);
        } else {
            throw std::invalid_argument("unknown argument: " + key);
        }
    }
    return p;
}

void run_size(const Params& p, size_t requested_bytes, Context& ctx, sycl::queue& q) {
    // Every arm walks whole 2880B pairs, so the effective prefix is rounded
    // down; the same prefix is used for the memcpy control and the reference.
    const size_t bytes = (requested_bytes / kMoePairBytes) * kMoePairBytes;
    check(bytes >= 10 * kMoePairBytes, "size too small for the moe shape");
    const size_t words = bytes / 4;
    const size_t vec_units = words / 4;
    const size_t pairs = bytes / kMoePairBytes;
    check(words % 4 == 0, "effective bytes must be divisible by 16");

    auto* host_bytes = static_cast<uint8_t*>(ctx.alloc_host(bytes));
    auto* dev_bytes = static_cast<uint8_t*>(ctx.alloc_device(bytes));
    auto* sink = static_cast<Sink*>(ctx.alloc_device(sizeof(Sink)));
    struct Drain {
        Context& ctx;
        void* host;
        void* dev;
        void* sink;
        ~Drain() {
            try {
                ctx.synchronize();
            } catch (...) {
            }
            try {
                ctx.free_device(sink);
            } catch (...) {
            }
            try {
                ctx.free_device(dev);
            } catch (...) {
            }
            try {
                ctx.free_host(host);
            } catch (...) {
            }
        }
    } drain{ctx, host_bytes, dev_bytes, sink};

    auto* host_words = reinterpret_cast<uint32_t*>(host_bytes);
    Refs ref;
    const auto ref_start = Clock::now();
    for (size_t i = 0; i < words; ++i) {
        const uint32_t v = mix(static_cast<uint32_t>(i) + 37);
        host_words[i] = v;
        ref.sum += static_cast<uint64_t>(v);
        ref.sum2 += static_cast<uint64_t>(v ^ kXorMask);
    }
    const double ref_ms = std::chrono::duration<double, std::milli>(Clock::now() - ref_start).count();

    // Fill the device buffer once so the device arms read identical bytes.
    ctx.memcpy_h2d(dev_bytes, host_bytes, bytes);
    check(sycl::get_pointer_type(host_bytes, q.get_context()) == sycl::usm::alloc::host,
          "host buffer is not host USM");
    check(sycl::get_pointer_type(dev_bytes, q.get_context()) == sycl::usm::alloc::device,
          "device buffer is not device USM");

    const bool vec_aligned =
        (reinterpret_cast<uintptr_t>(host_bytes) % 16 == 0) &&
        (reinterpret_cast<uintptr_t>(dev_bytes) % 16 == 0);

    std::cout << "[HR CONFIG] {\"requested_bytes\":" << requested_bytes
              << ",\"effective_bytes\":" << bytes << ",\"words\":" << words
              << ",\"pairs\":" << pairs << ",\"reps\":" << p.reps
              << ",\"warmup\":" << p.warmup
              << ",\"threads_cap\":" << p.threads_cap
              << ",\"dev_ptr\":" << (const void*)dev_bytes
              << ",\"host_ptr\":" << (const void*)host_bytes
              << ",\"vec16_aligned\":" << (vec_aligned ? 1 : 0) << "}\n";
    std::cout << "[HR REF] {\"bytes\":" << bytes << ",\"sum\":" << ref.sum
              << ",\"sum2\":" << ref.sum2 << ",\"ref_ms\":" << std::fixed
              << std::setprecision(1) << ref_ms << "}\n";
    std::cout.unsetf(std::ios::fixed);

    const auto selected = [&](const char* name) {
        if (p.arms.empty()) return true;
        return std::find(p.arms.begin(), p.arms.end(), std::string(name)) != p.arms.end();
    };

    // submit() returns the timed event. The sink is zeroed and read back
    // outside the timed window; the guard compares against the CPU reference.
    const auto measure = [&](const char* arm, size_t arm_bytes, auto&& submit) -> ArmOutcome {
        ArmOutcome outcome;
        outcome.ran = true;
        outcome.bytes = arm_bytes;
        outcome.guard_ok = true;
        std::vector<double> wall_ms;
        std::vector<double> kernel_ms;
        for (int r = 0; r < p.reps + p.warmup; ++r) {
            q.memset(sink, 0, sizeof(Sink));
            q.wait();
            const auto t0 = Clock::now();
            sycl::event event = submit();
            event.wait();
            const double wall = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
            double kernel = -1.0;
            try {
                kernel = static_cast<double>(
                             event.get_profiling_info<sycl::info::event_profiling::command_end>() -
                             event.get_profiling_info<sycl::info::event_profiling::command_start>()) /
                         1e6;
            } catch (...) {
                kernel = -1.0;
            }
            Sink host_sink{};
            q.memcpy(&host_sink, sink, sizeof(Sink));
            q.wait();
            outcome.guard_ok = outcome.guard_ok && host_sink.sum == ref.sum &&
                               host_sink.sum2 == ref.sum2;
            if (r >= p.warmup) {
                wall_ms.push_back(wall);
                if (kernel > 0.0) kernel_ms.push_back(kernel);
            }
        }
        outcome.wall_ms = median(wall_ms);
        outcome.kernel_ms = kernel_ms.empty() ? -1.0 : median(kernel_ms);
        outcome.gbps = outcome.wall_ms > 0.0
                           ? static_cast<double>(arm_bytes) / (outcome.wall_ms / 1000.0) / 1e9
                           : 0.0;
        std::cout << "[HR BENCH] {\"arm\":\"" << arm << "\",\"bytes\":" << arm_bytes
                  << ",\"median_ms\":" << std::fixed << std::setprecision(3) << outcome.wall_ms
                  << ",\"gbps\":" << std::setprecision(3) << outcome.gbps
                  << ",\"kernel_ms\":" << std::setprecision(3) << outcome.kernel_ms
                  << ",\"guard_ok\":" << (outcome.guard_ok ? 1 : 0) << "}\n";
        std::cout.unsetf(std::ios::fixed);
        return outcome;
    };

    const size_t flat_threads = pick_threads(words, p.threads_cap);
    const size_t flat16_threads = pick_threads(vec_units, p.threads_cap);

    ArmOutcome h2d;
    ArmOutcome dev_flat;
    ArmOutcome host_flat;
    ArmOutcome dev_flat16;
    ArmOutcome host_flat16;
    ArmOutcome dev_moe;
    ArmOutcome host_moe;

    if (selected("h2d")) {
        h2d = measure("h2d", bytes, [&]() { return q.memcpy(dev_bytes, host_bytes, bytes); });
        // The memcpy itself carries no checksum; prove the device copy by
        // reading it back through the flat kernel once.
        q.memset(sink, 0, sizeof(Sink));
        q.wait();
        launch_flat<1>(ctx, reinterpret_cast<const uint32_t*>(dev_bytes), words, flat_threads, sink)
            .wait();
        Sink host_sink{};
        q.memcpy(&host_sink, sink, sizeof(Sink));
        q.wait();
        h2d.guard_ok = h2d.guard_ok && host_sink.sum == ref.sum && host_sink.sum2 == ref.sum2;
        std::cout << "[HR GUARD] {\"arm\":\"h2d\",\"device_content_matches_ref\":"
                  << (host_sink.sum == ref.sum && host_sink.sum2 == ref.sum2 ? 1 : 0) << "}\n";
    }
    if (selected("dev-flat")) {
        dev_flat = measure("dev-flat", bytes, [&]() {
            return launch_flat<1>(ctx, reinterpret_cast<const uint32_t*>(dev_bytes), words,
                                  flat_threads, sink);
        });
    }
    if (selected("host-flat")) {
        host_flat = measure("host-flat", bytes, [&]() {
            return launch_flat<1>(ctx, reinterpret_cast<const uint32_t*>(host_bytes), words,
                                  flat_threads, sink);
        });
    }
    if (selected("dev-flat16") && vec_aligned) {
        dev_flat16 = measure("dev-flat16", bytes, [&]() {
            return launch_flat<4>(ctx, reinterpret_cast<const uint32_t*>(dev_bytes), vec_units,
                                  flat16_threads, sink);
        });
    }
    if (!vec_aligned && (selected("dev-flat16") || selected("host-flat16"))) {
        std::cout << "[HR GUARD] {\"arm\":\"flat16\",\"skipped\":\"base_ptr_not_16B_aligned\"}\n";
    }
    if (selected("host-flat16") && vec_aligned) {
        host_flat16 = measure("host-flat16", bytes, [&]() {
            return launch_flat<4>(ctx, reinterpret_cast<const uint32_t*>(host_bytes), vec_units,
                                  flat16_threads, sink);
        });
    }
    if (selected("dev-moe")) {
        dev_moe = measure("dev-moe", bytes,
                          [&]() { return launch_moe(ctx, dev_bytes, pairs, sink); });
    }
    if (selected("host-moe")) {
        host_moe = measure("host-moe", bytes,
                           [&]() { return launch_moe(ctx, host_bytes, pairs, sink); });
    }

    const auto gbps_or = [](const ArmOutcome& o) {
        return o.ran ? o.gbps : -1.0;
    };
    const bool guard_all = (!h2d.ran || h2d.guard_ok) && (!dev_flat.ran || dev_flat.guard_ok) &&
                           (!host_flat.ran || host_flat.guard_ok) &&
                           (!dev_flat16.ran || dev_flat16.guard_ok) &&
                           (!host_flat16.ran || host_flat16.guard_ok) &&
                           (!dev_moe.ran || dev_moe.guard_ok) &&
                           (!host_moe.ran || host_moe.guard_ok);
    std::cout << "[HR SUMMARY] {\"size_mib\":" << (requested_bytes / (1024 * 1024))
              << ",\"bytes\":" << bytes
              << ",\"h2d_gbps\":" << std::fixed << std::setprecision(3) << gbps_or(h2d)
              << ",\"dev_flat_gbps\":" << gbps_or(dev_flat)
              << ",\"host_flat_gbps\":" << gbps_or(host_flat)
              << ",\"dev_flat16_gbps\":" << gbps_or(dev_flat16)
              << ",\"host_flat16_gbps\":" << gbps_or(host_flat16)
              << ",\"dev_moe_gbps\":" << gbps_or(dev_moe)
              << ",\"host_moe_gbps\":" << gbps_or(host_moe)
              << ",\"R_flat_gbps\":" << gbps_or(host_flat)
              << ",\"R_moe_gbps\":" << gbps_or(host_moe)
              << ",\"guard_all_ok\":" << (guard_all ? 1 : 0) << "}\n";
    std::cout.unsetf(std::ios::fixed);
}

void print_capabilities(Context& ctx) {
    const sycl::device dev = ctx.queue().get_device();
    const auto aspect = [&](sycl::aspect a) { return dev.has(a) ? 1 : 0; };
    const std::string name = dev.get_info<sycl::info::device::name>();
    const std::string vendor = dev.get_info<sycl::info::device::vendor>();
    const std::string driver = dev.get_info<sycl::info::device::driver_version>();
    size_t local_mem = 0;
    size_t global_mem = 0;
    try {
        local_mem = dev.get_info<sycl::info::device::local_mem_size>();
        global_mem = dev.get_info<sycl::info::device::global_mem_size>();
    } catch (...) {
    }
    std::cout << "[HR CAP] {\"name\":\"" << name << "\",\"vendor\":\"" << vendor
              << "\",\"driver\":\"" << driver << "\",\"usm_host_allocations\":"
              << aspect(sycl::aspect::usm_host_allocations)
              << ",\"usm_device_allocations\":" << aspect(sycl::aspect::usm_device_allocations)
              << ",\"usm_shared_allocations\":" << aspect(sycl::aspect::usm_shared_allocations)
              << ",\"usm_system_allocations\":" << aspect(sycl::aspect::usm_system_allocations)
              << ",\"is_gpu\":" << (dev.is_gpu() ? 1 : 0)
              << ",\"local_mem_size\":" << local_mem << ",\"global_mem_size\":" << global_mem
              << "}\n";
    // A discrete GPU with usm_host_allocations means kernel reads of
    // malloc_host pointers cross the PCIe link; SYCL exposes no finer peer
    // query, so R itself is the proof of the path (see the doc).
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Params params = parse_params(argc, argv);
        Context ctx(nullptr, true);
        print_capabilities(ctx);
        if (!ctx.queue().get_device().has(sycl::aspect::usm_host_allocations)) {
            std::cout << "[HR ERROR] device lacks usm_host_allocations\n";
            return 2;
        }
        for (const size_t mib : params.sizes_mib) {
            run_size(params, mib * 1024 * 1024, ctx, ctx.queue());
        }
        return 0;
    } catch (const std::exception& error) {
        std::cout << "[HR ERROR] " << error.what() << "\n";
        return 1;
    }
}
