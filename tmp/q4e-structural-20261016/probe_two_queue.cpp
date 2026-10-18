// Can a second in-order compute queue create an independent cold-expert lane
// on the production Windows/L0 path?
//
// Arms:
//   compute_only: dependency-chain FP32 work, small footprint (attention-like)
//   memory_only:  large streaming read (cold-expert-weight-like)
//   serial:       both kernels in order on q0
//   overlap:      same kernels on q0/q1 after a common start event
//
// The overlap arm uses profiling events.  If the two kernels co-execute,
// overlap_us > 0 and overlap_fraction should be materially above zero.
#include <sycl/sycl.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

using Clock = std::chrono::steady_clock;

namespace {

constexpr int kComputeGroups = 128;
constexpr int kComputeThreads = 256;
constexpr int kComputeSteps = 65536;
constexpr int kMemoryGroups = 512;
constexpr int kMemoryThreads = 256;

float* g_compute = nullptr;
float* g_compute_out = nullptr;
float* g_stream = nullptr;
float* g_stream_out = nullptr;
size_t g_stream_count = 0;
float* g_copy_host = nullptr;
float* g_copy_dev = nullptr;
size_t g_copy_count = 0;

sycl::event submit_compute(sycl::queue& q, const sycl::event* dep) {
    float* in = g_compute;
    float* out = g_compute_out;
    return q.submit([&](sycl::handler& h) {
        if (dep) h.depends_on(*dep);
        h.parallel_for(
            sycl::nd_range<1>(sycl::range<1>(kComputeGroups * kComputeThreads),
                              sycl::range<1>(kComputeThreads)),
            [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
                const int tid = static_cast<int>(it.get_global_linear_id());
                float v = in[(tid * 7) % 1024] + 1.0f;
                const float a = in[(tid * 13 + 3) % 1024] * 1.0e-7f;
                const float b = in[(tid * 17 + 5) % 1024] * 1.0e-9f;
#pragma unroll 4
                for (int s = 0; s < kComputeSteps; ++s) {
                    v = sycl::fma(v, 0.99999994f, a);
                    v = sycl::fma(v, 0.99999982f, b);
                }
                if ((tid & 31) == 0) out[tid >> 5] = v;
            });
    });
}

sycl::event submit_stream(sycl::queue& q, const sycl::event* dep) {
    const float* src = g_stream;
    float* out = g_stream_out;
    const size_t n = g_stream_count;
    const int threads = kMemoryGroups * kMemoryThreads;
    return q.submit([&](sycl::handler& h) {
        if (dep) h.depends_on(*dep);
        h.parallel_for(
            sycl::nd_range<1>(sycl::range<1>(threads),
                              sycl::range<1>(kMemoryThreads)),
            [=](sycl::nd_item<1> it) {
                const size_t tid = it.get_global_linear_id();
                const size_t stride = static_cast<size_t>(threads);
                float acc = 0.0f;
                for (size_t i = tid; i < n; i += stride)
                    acc += src[i];
                if (tid < 32) out[tid] = acc;
            });
    });
}

sycl::event submit_host_copy(sycl::queue& q, const sycl::event* dep) {
    if (dep) {
        return q.memcpy(g_copy_dev, g_copy_host, g_copy_count * sizeof(float),
                        std::vector<sycl::event>{*dep});
    }
    return q.memcpy(g_copy_dev, g_copy_host, g_copy_count * sizeof(float));
}

double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v.empty() ? 0.0 : v[v.size() / 2];
}

double us_between(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double, std::micro>(b - a).count();
}

uint64_t event_start_us(const sycl::event& e) {
    return e.get_profiling_info<sycl::info::event_profiling::command_start>();
}

uint64_t event_end_us(const sycl::event& e) {
    return e.get_profiling_info<sycl::info::event_profiling::command_end>();
}

} // namespace

int main(int argc, char** argv) {
    int iters = 20;
    int warmup = 3;
    int mem_mb = 1024;
    int copy_mb = 32;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--iters" && i + 1 < argc) iters = std::stoi(argv[++i]);
        else if (a == "--warmup" && i + 1 < argc) warmup = std::stoi(argv[++i]);
        else if (a == "--mem-mb" && i + 1 < argc) mem_mb = std::stoi(argv[++i]);
        else if (a == "--copy-mb" && i + 1 < argc) copy_mb = std::stoi(argv[++i]);
    }

    sycl::queue q0{sycl::gpu_selector_v,
                   sycl::property_list{sycl::property::queue::in_order(),
                                       sycl::property::queue::enable_profiling()}};
    sycl::queue q1{q0.get_context(), q0.get_device(),
                   sycl::property_list{sycl::property::queue::in_order(),
                                       sycl::property::queue::enable_profiling()}};

    g_compute = sycl::malloc_device<float>(1024, q0);
    g_compute_out = sycl::malloc_device<float>(kComputeGroups * kComputeThreads / 32, q0);
    g_stream_count = static_cast<size_t>(mem_mb) * 1024 * 1024 / sizeof(float);
    g_stream = sycl::malloc_device<float>(g_stream_count, q0);
    g_stream_out = sycl::malloc_device<float>(32, q0);
    g_copy_count = static_cast<size_t>(copy_mb) * 1024 * 1024 / sizeof(float);
    g_copy_host = sycl::malloc_host<float>(g_copy_count, q0);
    g_copy_dev = sycl::malloc_device<float>(g_copy_count, q0);
    if (!g_compute || !g_compute_out || !g_stream || !g_stream_out ||
        !g_copy_host || !g_copy_dev)
        throw std::runtime_error("device allocation failed");

    q0.fill(g_compute, 0.5f, 1024);
    q0.fill(g_compute_out, 0.0f, kComputeGroups * kComputeThreads / 32);
    q0.fill(g_stream, 0.25f, g_stream_count);
    q0.fill(g_stream_out, 0.0f, 32);
    std::fill(g_copy_host, g_copy_host + g_copy_count, 0.5f);
    q0.fill(g_copy_dev, 0.0f, g_copy_count);
    q0.wait_and_throw();

    std::cout << std::fixed << std::setprecision(3) << std::unitbuf;
    std::cout << "[2Q CONFIG] device="
              << q0.get_device().get_info<sycl::info::device::name>()
              << " compute(groups=" << kComputeGroups
              << ",steps=" << kComputeSteps
              << ") stream_mb=" << mem_mb
              << " copy_mb=" << copy_mb
              << " iters=" << iters << '\n';

    for (int i = 0; i < warmup; ++i) {
        auto a = submit_compute(q0, nullptr);
        auto b = submit_stream(q0, nullptr);
        b.wait_and_throw();
        sycl::event s = q0.single_task([] {});
        auto c = submit_compute(q0, nullptr);
        auto d = submit_stream(q1, &s);
        auto x = submit_host_copy(q1, &s);
        c.wait_and_throw();
        d.wait_and_throw();
        x.wait_and_throw();
    }

    std::vector<double> compute_only, memory_only, serial, overlap;
    std::vector<double> copy_only, serial_copy, overlap_copy;
    std::vector<double> overlap_us_values, overlap_fraction, speedup_vs_serial;
    std::vector<double> copy_overlap_us, copy_overlap_fraction;
    for (int i = 0; i < iters; ++i) {
        auto t0 = Clock::now();
        auto ce = submit_compute(q0, nullptr);
        ce.wait_and_throw();
        compute_only.push_back(us_between(t0, Clock::now()));

        t0 = Clock::now();
        auto me = submit_stream(q0, nullptr);
        me.wait_and_throw();
        memory_only.push_back(us_between(t0, Clock::now()));

        t0 = Clock::now();
        auto serial_c = submit_compute(q0, nullptr);
        auto serial_m = submit_stream(q0, nullptr);
        serial_m.wait_and_throw();
        serial.push_back(us_between(t0, Clock::now()));

        t0 = Clock::now();
        auto start = q0.single_task([=] {});
        auto q0c = submit_compute(q0, &start);
        auto q1m = submit_stream(q1, &start);
        q0c.wait_and_throw();
        q1m.wait_and_throw();
        overlap.push_back(us_between(t0, Clock::now()));

        const uint64_t c0 = event_start_us(q0c), c1 = event_end_us(q0c);
        const uint64_t m0 = event_start_us(q1m), m1 = event_end_us(q1m);
        const uint64_t inter0 = std::max(c0, m0), inter1 = std::min(c1, m1);
        const uint64_t span0 = std::min(c0, m0), span1 = std::max(c1, m1);
        const double ov = inter1 > inter0 ? double(inter1 - inter0) : 0.0;
        overlap_us_values.push_back(ov);
        overlap_fraction.push_back(span1 > span0 ? ov / double(span1 - span0) : 0.0);
        speedup_vs_serial.push_back(median(serial) / std::max(1.0, overlap.back()));

        t0 = Clock::now();
        auto copy_e = submit_host_copy(q1, nullptr);
        copy_e.wait_and_throw();
        copy_only.push_back(us_between(t0, Clock::now()));

        t0 = Clock::now();
        auto serial_c2 = submit_compute(q0, nullptr);
        auto serial_copy_e = submit_host_copy(q0, nullptr);
        serial_copy_e.wait_and_throw();
        serial_copy.push_back(us_between(t0, Clock::now()));

        t0 = Clock::now();
        auto start_copy = q0.single_task([=] {});
        auto compute_copy = submit_compute(q0, &start_copy);
        auto copy_overlap = submit_host_copy(q1, &start_copy);
        compute_copy.wait_and_throw();
        copy_overlap.wait_and_throw();
        overlap_copy.push_back(us_between(t0, Clock::now()));

        const uint64_t cc0 = event_start_us(compute_copy), cc1 = event_end_us(compute_copy);
        const uint64_t mc0 = event_start_us(copy_overlap), mc1 = event_end_us(copy_overlap);
        const uint64_t co0 = std::max(cc0, mc0), co1 = std::min(cc1, mc1);
        const uint64_t cs0 = std::min(cc0, mc0), cs1 = std::max(cc1, mc1);
        const double cov = co1 > co0 ? double(co1 - co0) : 0.0;
        copy_overlap_us.push_back(cov);
        copy_overlap_fraction.push_back(cs1 > cs0 ? cov / double(cs1 - cs0) : 0.0);
    }

    const double c_only = median(compute_only);
    const double m_only = median(memory_only);
    const double ser = median(serial);
    const double ov = median(overlap);
    std::cout << "[2Q ARM] compute_only_us=" << c_only << '\n';
    std::cout << "[2Q ARM] memory_only_us=" << m_only << '\n';
    std::cout << "[2Q ARM] serial_us=" << ser << '\n';
    std::cout << "[2Q ARM] overlap_us=" << ov
              << " saved_vs_serial_us=" << (ser - ov)
              << " speedup=" << (ser / std::max(1.0, ov)) << '\n';
    std::cout << "[2Q ARM] copy_only_us=" << median(copy_only) << '\n';
    std::cout << "[2Q ARM] serial_compute_copy_us=" << median(serial_copy) << '\n';
    std::cout << "[2Q ARM] overlap_compute_copy_us=" << median(overlap_copy)
              << " saved_vs_serial_us=" << (median(serial_copy) - median(overlap_copy))
              << " speedup=" << (median(serial_copy) / std::max(1.0, median(overlap_copy)))
              << '\n';
    std::cout << "[2Q OVERLAP] interval_us_median=" << median(overlap_us_values)
              << " fraction_median=" << median(overlap_fraction)
              << " fraction_min=" << *std::min_element(overlap_fraction.begin(), overlap_fraction.end())
              << " fraction_max=" << *std::max_element(overlap_fraction.begin(), overlap_fraction.end())
              << '\n';
    std::cout << "[2Q COPY OVERLAP] interval_us_median=" << median(copy_overlap_us)
              << " fraction_median=" << median(copy_overlap_fraction)
              << " fraction_min=" << *std::min_element(copy_overlap_fraction.begin(), copy_overlap_fraction.end())
              << " fraction_max=" << *std::max_element(copy_overlap_fraction.begin(), copy_overlap_fraction.end())
              << '\n';

    sycl::free(g_copy_dev, q0);
    sycl::free(g_copy_host, q0);
    sycl::free(g_stream_out, q0);
    sycl::free(g_stream, q0);
    sycl::free(g_compute_out, q0);
    sycl::free(g_compute, q0);
    return 0;
}
