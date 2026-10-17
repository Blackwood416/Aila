// P0 probe for the proposed Qwen4Exp whole-token graph:
//   route publish (D->H) -> CPU expert work -> completion flag (H->D)
//   -> copy-from-mapped kernel -> bitwise verify -> completion notify (D->H)
//
// The probe deliberately uses Aila's Context and Limited SYCL Graph path.
// It does not use queue.memcpy inside the graph: the mapped host result is
// transferred by a kernel, matching the Strata WDDM-safe pattern.
//
// No main-project source is modified. The GPU run is opt-in through run.ps1.

#include "core/Context.hpp"

#include <sycl/sycl.hpp>
#include <sycl/bit_cast.hpp>
#include <sycl/ext/intel/experimental/cache_control_properties.hpp>
#include <sycl/ext/oneapi/experimental/annotated_ptr/annotated_ptr.hpp>
#include <sycl/ext/oneapi/experimental/graph.hpp>

#include <immintrin.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace ex = sycl::ext::oneapi::experimental;
namespace iex = sycl::ext::intel::experimental;
namespace oex = sycl::ext::oneapi::experimental;
using Clock = std::chrono::steady_clock;

namespace {

constexpr int kTopK = 10;
constexpr int kResultWidth = 8;
constexpr int kResultCount = kTopK * kResultWidth;
constexpr uint32_t kDefaultSpinMax = 20000;
constexpr uint32_t kPollFlushEvery = 1024;

// Intel cache-control hints. These are the load/store half of the Strata
// doorbell fix: system-scope atomics alone can remain in the L1/L3 cache on Arc.
using UncachedRead = decltype(oex::properties(
    iex::read_hint<iex::cache_control<
        iex::cache_mode::uncached,
        oex::cache_level::L1,
        oex::cache_level::L3>>));
using UncachedWrite = decltype(oex::properties(
    iex::write_hint<iex::cache_control<
        iex::cache_mode::uncached,
        oex::cache_level::L1,
        oex::cache_level::L3>>));

template <class T>
inline T uncached_load(const T* p) {
    oex::annotated_ptr<T, UncachedRead> a(const_cast<T*>(p));
    return a[0];
}

template <class T>
inline void uncached_store(T* p, T v) {
    oex::annotated_ptr<T, UncachedWrite> a(p);
    a[0] = v;
}

struct alignas(64) ControlRing {
    uint32_t route_seq;
    uint32_t cpu_flag;
    uint32_t result_seq;
    uint32_t done_seq;
    uint32_t timeout_count;
    uint32_t stale_count;
    uint32_t mismatch_count;
    uint32_t host_route_error_count;
    uint64_t timeout_spins;
    uint64_t reserved[3];
};

struct alignas(64) RouteRing {
    int32_t ids[kTopK];
    float weights[kTopK];
    uint32_t reserved[12];
};

struct alignas(64) ResultRing {
    float values[kResultCount];
    uint32_t reserved[16];
};

struct alignas(64) HostRing {
    ControlRing ctrl;
    RouteRing route;
    ResultRing result;
};

struct alignas(64) DeviceEpoch {
    uint32_t seq;
    uint32_t reserved[15];
};

static_assert(sizeof(ControlRing) == 64);
static_assert(sizeof(RouteRing) == 128);
static_assert(sizeof(ResultRing) == 384);
static_assert(sizeof(HostRing) == 576);
static_assert(sizeof(DeviceEpoch) == 64);

inline uint32_t expected_id(uint32_t seq, uint32_t k) {
    return (seq * 37u + k * 53u) & 511u;
}

inline uint32_t expected_weight_bits(uint32_t seq, uint32_t k) {
    return (seq * 13u + k * 7u) & 255u;
}

inline float expected_weight(uint32_t seq, uint32_t k) {
    return static_cast<float>(expected_weight_bits(seq, k));
}

inline float expected_result(uint32_t seq, uint32_t k, uint32_t j) {
    const uint32_t id = expected_id(seq, k);
    const uint32_t base = (seq & 1023u) * 4096u;
    return static_cast<float>(base + id * kResultWidth + j);
}

inline uint32_t bit_u32(float v) {
    return sycl::bit_cast<uint32_t>(v);
}

inline volatile uint32_t* volatile_u32(uint32_t* p) {
    return reinterpret_cast<volatile uint32_t*>(p);
}

inline volatile float* volatile_float(float* p) {
    return reinterpret_cast<volatile float*>(p);
}

void submit_route_publish(sycl::queue& q, HostRing* ring, DeviceEpoch* epoch) {
    q.submit([&](sycl::handler& h) {
        h.single_task([=]() {
            const uint32_t seq = epoch->seq + 1u;
            epoch->seq = seq;
            for (uint32_t k = 0; k < kTopK; ++k) {
                uncached_store(&ring->route.ids[k],
                               static_cast<int32_t>(expected_id(seq, k)));
                uncached_store(&ring->route.weights[k],
                               expected_weight(seq, k));
            }
            // Payload is uncached and ordered before the sequence publication.
            sycl::atomic_fence(sycl::memory_order::release,
                               sycl::memory_scope::system);
            uncached_store(&ring->ctrl.route_seq, seq);
        });
    });
}

void submit_device_wait(sycl::queue& q, HostRing* ring, DeviceEpoch* epoch,
                        uint32_t spin_max) {
    q.submit([&](sycl::handler& h) {
        h.single_task([=]() {
            const uint32_t seq = epoch->seq;
            uint32_t flag = 0;
            uint32_t spins = 0;
            for (;;) {
                // Keep the acquire fence inside the retry loop. Besides ordering,
                // it prevents the compiler from hoisting a non-volatile load out
                // of the polling loop.
                sycl::atomic_fence(sycl::memory_order::acquire,
                                   sycl::memory_scope::system);
                flag = uncached_load(&ring->ctrl.cpu_flag);
                if (flag == seq) break;
                if (++spins >= spin_max) break;
            }
            if (flag != seq) {
                uncached_store(&ring->ctrl.timeout_count,
                               uncached_load(&ring->ctrl.timeout_count) + 1u);
                uncached_store(&ring->ctrl.timeout_spins,
                               static_cast<uint64_t>(spins));
                return;
            }
            // The matching flag establishes publication of result_seq/result.
            sycl::atomic_fence(sycl::memory_order::acquire,
                               sycl::memory_scope::system);

            bool stale = uncached_load(&ring->ctrl.result_seq) != seq;
            if (!stale) {
                for (uint32_t k = 0; k < kTopK; ++k) {
                    for (uint32_t j = 0; j < kResultWidth; ++j) {
                        const uint32_t idx = k * kResultWidth + j;
                        const float got =
                            uncached_load(&ring->result.values[idx]);
                        const float expected = expected_result(seq, k, j);
                        if (bit_u32(got) != bit_u32(expected)) {
                            stale = true;
                            break;
                        }
                    }
                    if (stale) break;
                }
            }
            if (stale) {
                uncached_store(&ring->ctrl.stale_count,
                               uncached_load(&ring->ctrl.stale_count) + 1u);
            }
        });
    });
}

// The graph must not contain queue.memcpy: a copy-engine node can split the
// WDDM submission. This kernel reads mapped pinned host USM directly.
void submit_copy_from_mapped(sycl::queue& q, HostRing* ring, float* device_result) {
    q.parallel_for(sycl::range<1>(kResultCount), [=](sycl::id<1> id) {
        const size_t i = id[0];
        device_result[i] = uncached_load(&ring->result.values[i]);
    });
}

void submit_bitwise_verify(sycl::queue& q, HostRing* ring, DeviceEpoch* epoch,
                           const float* device_result) {
    q.submit([&](sycl::handler& h) {
        h.single_task([=]() {
            const uint32_t seq = epoch->seq;
            bool mismatch = uncached_load(&ring->ctrl.result_seq) != seq;
            if (!mismatch) {
                for (uint32_t k = 0; k < kTopK; ++k) {
                    for (uint32_t j = 0; j < kResultWidth; ++j) {
                        const uint32_t idx = k * kResultWidth + j;
                        const float got = device_result[idx];
                        const float expected = expected_result(seq, k, j);
                        if (bit_u32(got) != bit_u32(expected)) {
                            mismatch = true;
                            break;
                        }
                    }
                    if (mismatch) break;
                }
            }
            if (mismatch) {
                uncached_store(&ring->ctrl.mismatch_count,
                               uncached_load(&ring->ctrl.mismatch_count) + 1u);
            }
            sycl::atomic_fence(sycl::memory_order::release,
                               sycl::memory_scope::system);
            uncached_store(&ring->ctrl.done_seq, seq);
        });
    });
}

double us_between(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double, std::micro>(b - a).count();
}

double percentile(std::vector<double> v, double p) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    const double pos = p * static_cast<double>(v.size() - 1);
    const size_t i = static_cast<size_t>(pos + 0.5);
    return v[std::min(i, v.size() - 1)];
}

void burn_cpu_us(int cpu_us) {
    if (cpu_us <= 0) return;
    const auto deadline = Clock::now() + std::chrono::microseconds(cpu_us);
    while (Clock::now() < deadline) {
        for (int i = 0; i < 64; ++i) _mm_pause();
    }
}

void release_graph_submission(const sycl::event& graph_event) {
    // A driver entry is required to discharge the L0 dynamic batch. Querying
    // the graph event is the release primitive already measured in Aila's
    // submit probes; it neither waits nor changes graph semantics.
    (void)graph_event
        .get_info<sycl::info::event::command_execution_status>();
}

enum class PollResult { Reached, Timeout, QueueEnded, Error };

PollResult poll_host_word(sycl::queue& q, volatile uint32_t* word,
                          uint32_t want, int timeout_ms, std::string& error) {
    const auto deadline = Clock::now() + std::chrono::milliseconds(timeout_ms);
    uint32_t polls = 0;
    while (*word != want) {
        if (*word > want) {
            error = "ring sequence advanced past expected value";
            return PollResult::Error;
        }
        if ((++polls & (kPollFlushEvery - 1u)) == 0u) {
            try {
                if (q.ext_oneapi_empty() && *word != want) {
                    error = "queue became empty before expected ring value";
                    return PollResult::QueueEnded;
                }
            } catch (const sycl::exception& e) {
                error = std::string("queue query failed: ") + e.what();
                return PollResult::Error;
            }
        }
        if (Clock::now() >= deadline) {
            std::ostringstream os;
            os << "host timeout waiting for ring value " << want
               << " (last=" << *word << ", polls=" << polls << ')';
            error = os.str();
            return PollResult::Timeout;
        }
        for (int i = 0; i < 64; ++i) _mm_pause();
    }
    std::atomic_thread_fence(std::memory_order_acquire);
    return PollResult::Reached;
}

bool host_route_matches(HostRing* ring, uint32_t seq, std::string& error) {
    for (uint32_t k = 0; k < kTopK; ++k) {
        const int32_t got_id = ring->route.ids[k];
        const float got_weight = ring->route.weights[k];
        if (got_id != static_cast<int32_t>(expected_id(seq, k)) ||
            bit_u32(got_weight) != bit_u32(expected_weight(seq, k))) {
            std::ostringstream os;
            os << "route payload stale at k=" << k << " seq=" << seq;
            error = os.str();
            return false;
        }
    }
    return true;
}

void write_host_result(HostRing* ring, uint32_t value_seq, uint32_t publish_seq) {
    volatile float* values = volatile_float(ring->result.values);
    for (uint32_t k = 0; k < kTopK; ++k) {
        for (uint32_t j = 0; j < kResultWidth; ++j) {
            values[k * kResultWidth + j] =
                expected_result(value_seq, k, j);
        }
    }
    std::atomic_thread_fence(std::memory_order_seq_cst);
    *volatile_u32(&ring->ctrl.result_seq) = publish_seq;
    std::atomic_thread_fence(std::memory_order_seq_cst);
    _mm_sfence();
    *volatile_u32(&ring->ctrl.cpu_flag) = publish_seq;
}

struct Handoff {
    uint32_t seq = 0;
    double launch_to_route_us = 0.0;
    double route_to_flag_us = 0.0;
    double route_to_done_us = 0.0;
    double replay_wall_us = 0.0;
    uint32_t timeout_delta = 0;
    uint32_t stale_delta = 0;
    uint32_t mismatch_delta = 0;
};

Handoff service_replay(Context& ctx,
                       ex::command_graph<ex::graph_state::executable>& graph,
                       HostRing* ring,
                       uint32_t expected_seq,
                       int cpu_us,
                       int host_timeout_ms,
                       bool stale_injection) {
    sycl::queue& q = ctx.queue();
    auto& ctrl = ring->ctrl;
    const uint32_t timeout_before = *volatile_u32(&ctrl.timeout_count);
    const uint32_t stale_before = *volatile_u32(&ctrl.stale_count);
    const uint32_t mismatch_before = *volatile_u32(&ctrl.mismatch_count);

    const auto t_launch = Clock::now();
    auto graph_event = q.ext_oneapi_graph(graph);
    release_graph_submission(graph_event);

    std::string error;
    const PollResult route_poll =
        poll_host_word(q, volatile_u32(&ctrl.route_seq), expected_seq,
                       host_timeout_ms, error);
    if (route_poll != PollResult::Reached) {
        try { graph_event.wait_and_throw(); } catch (...) {}
        throw std::runtime_error("route " + error);
    }
    const auto t_route = Clock::now();
    if (!host_route_matches(ring, expected_seq, error)) {
        *volatile_u32(&ctrl.host_route_error_count) =
            *volatile_u32(&ctrl.host_route_error_count) + 1u;
        try { graph_event.wait_and_throw(); } catch (...) {}
        throw std::runtime_error(error);
    }

    const uint32_t value_seq =
        stale_injection ? expected_seq - 1u : expected_seq;
    burn_cpu_us(cpu_us);
    write_host_result(ring, value_seq, expected_seq);
    const auto t_flag = Clock::now();

    const PollResult done_poll =
        poll_host_word(q, volatile_u32(&ctrl.done_seq), expected_seq,
                       host_timeout_ms, error);
    if (done_poll != PollResult::Reached) {
        try { graph_event.wait_and_throw(); } catch (...) {}
        throw std::runtime_error("done " + error);
    }
    const auto t_done = Clock::now();
    graph_event.wait_and_throw();
    const auto t_end = Clock::now();

    Handoff out;
    out.seq = expected_seq;
    out.launch_to_route_us = us_between(t_launch, t_route);
    out.route_to_flag_us = us_between(t_route, t_flag);
    out.route_to_done_us = us_between(t_route, t_done);
    out.replay_wall_us = us_between(t_launch, t_end);
    out.timeout_delta =
        *volatile_u32(&ctrl.timeout_count) - timeout_before;
    out.stale_delta = *volatile_u32(&ctrl.stale_count) - stale_before;
    out.mismatch_delta =
        *volatile_u32(&ctrl.mismatch_count) - mismatch_before;
    return out;
}

struct ArmStats {
    int cpu_us = 0;
    size_t measured = 0;
    bool fresh_ok = true;
    bool stale_detected = false;
    std::vector<double> launch_to_route;
    std::vector<double> route_to_flag;
    std::vector<double> route_to_done;
    std::vector<double> wall;
};

void print_arm(const ArmStats& s) {
    const bool ok = s.fresh_ok && s.stale_detected && s.measured >= 1000;
    std::cout << "[TOKEN GRAPH ARM] cpu_us=" << s.cpu_us
              << " replays=" << s.measured
              << " ok=" << (ok ? 1 : 0)
              << " fresh_bitexact=" << (s.fresh_ok ? 1 : 0)
              << " stale_detected=" << (s.stale_detected ? 1 : 0)
              << " launch_to_route_us_median="
              << std::fixed << std::setprecision(3)
              << percentile(s.launch_to_route, 0.50)
              << " launch_to_route_us_p99="
              << percentile(s.launch_to_route, 0.99)
              << " route_to_flag_us_median="
              << percentile(s.route_to_flag, 0.50)
              << " route_to_flag_us_p99="
              << percentile(s.route_to_flag, 0.99)
              << " handoff_us_median="
              << percentile(s.route_to_done, 0.50)
              << " handoff_us_p99="
              << percentile(s.route_to_done, 0.99)
              << " replay_wall_us_median="
              << percentile(s.wall, 0.50)
              << " replay_wall_us_p99="
              << percentile(s.wall, 0.99)
              << '\n';
}

std::vector<int> parse_cpu_sweep(const std::string& text) {
    std::vector<int> out;
    std::stringstream ss(text);
    std::string item;
    while (std::getline(ss, item, ',')) {
        if (item.empty()) continue;
        const int value = std::stoi(item);
        if (value < 0) throw std::runtime_error("cpu-us values must be >= 0");
        out.push_back(value);
    }
    if (out.empty()) throw std::runtime_error("cpu-us sweep is empty");
    return out;
}

struct Options {
    uint32_t replays = 1000;
    int warmup = 10;
    uint32_t spin_max = kDefaultSpinMax;
    int host_timeout_ms = 2000;
    std::vector<int> cpu_sweep{0, 50, 200, 500};
};

Options parse_options(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto next = [&](const char* name) -> std::string {
            if (i + 1 >= argc)
                throw std::runtime_error(std::string("missing value for ") + name);
            return argv[++i];
        };
        if (arg == "--replays") {
            o.replays = static_cast<uint32_t>(std::stoul(next("--replays")));
        } else if (arg == "--warmup") {
            o.warmup = std::stoi(next("--warmup"));
        } else if (arg == "--spin-max") {
            o.spin_max = static_cast<uint32_t>(std::stoul(next("--spin-max")));
        } else if (arg == "--host-timeout-ms") {
            o.host_timeout_ms = std::stoi(next("--host-timeout-ms"));
        } else if (arg == "--cpu-us") {
            o.cpu_sweep = parse_cpu_sweep(next("--cpu-us"));
        } else if (arg == "--help" || arg == "-h") {
            std::cout
                << "usage: probe_token_graph.exe [--replays N] [--warmup N]\n"
                   "       [--cpu-us 0,50,200,500] [--spin-max N]\n"
                   "       [--host-timeout-ms N]\n";
            std::exit(0);
        } else {
            throw std::runtime_error("unknown argument: " + arg);
        }
    }
    if (o.replays < 1000)
        throw std::runtime_error("--replays must be >= 1000 for P0");
    if (o.warmup < 0) throw std::runtime_error("--warmup must be >= 0");
    if (!o.spin_max) throw std::runtime_error("--spin-max must be > 0");
    if (o.host_timeout_ms <= 0)
        throw std::runtime_error("--host-timeout-ms must be > 0");
    return o;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_options(argc, argv);
        Context ctx;
        sycl::queue& q = ctx.queue();
        const sycl::device dev = q.get_device();
        const bool limited =
            dev.has(sycl::aspect::ext_oneapi_limited_graph);
        const bool full = dev.has(sycl::aspect::ext_oneapi_graph);

        std::cout << "[TOKEN GRAPH CAPS] limited=" << (limited ? 1 : 0)
                  << " full=" << (full ? 1 : 0)
                  << " device=" << dev.get_info<sycl::info::device::name>()
                  << '\n';
        if (!limited && !full) {
            std::cout << "[TOKEN GRAPH SKIP] static graph unsupported\n";
            return 2;
        }

        HostRing* ring = static_cast<HostRing*>(ctx.alloc_host(sizeof(HostRing)));
        if (reinterpret_cast<uintptr_t>(ring) % alignof(HostRing) != 0) {
            throw std::runtime_error("host ring is not 64-byte aligned");
        }
        DeviceEpoch* epoch =
            static_cast<DeviceEpoch*>(ctx.alloc_device(sizeof(DeviceEpoch)));
        float* device_result =
            static_cast<float*>(ctx.alloc_device(sizeof(float) * kResultCount));
        std::memset(ring, 0, sizeof(HostRing));
        q.memset(epoch, 0, sizeof(DeviceEpoch)).wait_and_throw();
        q.memset(device_result, 0, sizeof(float) * kResultCount)
            .wait_and_throw();

        std::cout << "[TOKEN GRAPH CONFIG] replays=" << options.replays
                  << " warmup=" << options.warmup
                  << " spin_max=" << options.spin_max
                  << " host_timeout_ms=" << options.host_timeout_ms
                  << " cpu_sweep=";
        for (size_t i = 0; i < options.cpu_sweep.size(); ++i) {
            if (i) std::cout << ',';
            std::cout << options.cpu_sweep[i];
        }
        std::cout << '\n';
        std::cout
            << "[TOKEN GRAPH RING] allocation=aila_context_alloc_host"
               " ring_bytes=" << sizeof(HostRing)
            << " device_access=uncached_l1_l3"
               " copy=copy_from_mapped_kernel"
               " bounded_device_spin=1"
               " release=event_status_then_queue_empty_fallback\n";

        ctx.synchronize();
        uint32_t expected_seq = 0;
        bool all_ok = true;
        {
            ex::command_graph<ex::graph_state::modifiable> recording(
                q.get_context(), q.get_device());
            recording.begin_recording(q);
            try {
                submit_route_publish(q, ring, epoch);
                submit_device_wait(q, ring, epoch, options.spin_max);
                submit_copy_from_mapped(q, ring, device_result);
                submit_bitwise_verify(q, ring, epoch, device_result);
                recording.end_recording(q);
            } catch (...) {
                recording.end_recording(q);
                throw;
            }
            auto executable = recording.finalize();
            std::cout << "[TOKEN GRAPH CAPTURE] status=ok nodes=4\n";

            for (int cpu_us : options.cpu_sweep) {
                ArmStats stats;
                stats.cpu_us = cpu_us;

                for (int w = 0; w < options.warmup; ++w) {
                    ++expected_seq;
                    const Handoff h = service_replay(
                        ctx, executable, ring, expected_seq, cpu_us,
                        options.host_timeout_ms, false);
                    if (h.timeout_delta || h.stale_delta || h.mismatch_delta) {
                        throw std::runtime_error(
                            "warmup exposed timeout/stale/mismatch");
                    }
                }

                for (uint32_t r = 0; r < options.replays; ++r) {
                    ++expected_seq;
                    const Handoff h = service_replay(
                        ctx, executable, ring, expected_seq, cpu_us,
                        options.host_timeout_ms, false);
                    stats.launch_to_route.push_back(h.launch_to_route_us);
                    stats.route_to_flag.push_back(h.route_to_flag_us);
                    stats.route_to_done.push_back(h.route_to_done_us);
                    stats.wall.push_back(h.replay_wall_us);
                    if (h.timeout_delta || h.stale_delta || h.mismatch_delta) {
                        stats.fresh_ok = false;
                        break;
                    }
                }
                stats.measured = stats.route_to_done.size();

                // One deliberate stale-result replay proves the detector is live.
                ++expected_seq;
                const Handoff stale = service_replay(
                    ctx, executable, ring, expected_seq, cpu_us,
                    options.host_timeout_ms, true);
                stats.stale_detected =
                    stale.timeout_delta == 0 && stale.stale_delta >= 1 &&
                    stale.mismatch_delta >= 1;

                print_arm(stats);
                if (!stats.fresh_ok || !stats.stale_detected ||
                    stats.measured < 1000) {
                    all_ok = false;
                }
            }

            std::cout << "[TOKEN GRAPH STATUS] timeouts="
                      << *volatile_u32(&ring->ctrl.timeout_count)
                      << " stale="
                      << *volatile_u32(&ring->ctrl.stale_count)
                      << " mismatches="
                      << *volatile_u32(&ring->ctrl.mismatch_count)
                      << " host_route_errors="
                      << *volatile_u32(&ring->ctrl.host_route_error_count)
                      << '\n';
            std::cout << "[TOKEN GRAPH VERDICT] all_ok=" << (all_ok ? 1 : 0)
                      << " bit_exact=" << (all_ok ? 1 : 0)
                      << " stale_detector=" << (all_ok ? 1 : 0) << '\n';
        }

        ctx.synchronize();
        ctx.free_device(device_result);
        ctx.free_device(epoch);
        ctx.free_host(ring);
        return all_ok ? 0 : 1;
    } catch (const std::exception& e) {
        std::cerr << "[TOKEN GRAPH ERROR] " << e.what() << '\n';
        return 1;
    }
}
