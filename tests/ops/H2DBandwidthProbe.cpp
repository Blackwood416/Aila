// Measures achievable host-to-device copy bandwidth on this machine.
//
// The prefill path stages ~42.6 GB of expert weights per 2K-token pass and an
// observer-free trace shows it takes 19 s, i.e. 2.24 GB/s. This probe answers
// whether that number is a hardware ceiling (the platform's PCIe link) or an
// implementation artefact, by copying the same total volume at several chunk
// sizes with several copies in flight.
//
// Usage: AilaH2DBandwidthProbe [total_mb] [chunk_kb]
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <chrono>
#include <vector>

#include "src/core/Context.hpp"

using Clock = std::chrono::steady_clock;

static double ms_since(Clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

int main(int argc, char** argv) {
    const size_t total_mb = (argc > 1) ? static_cast<size_t>(std::max(16, std::atoi(argv[1]))) : 512;
    const size_t chunk_kb = (argc > 2) ? static_cast<size_t>(std::max(4, std::atoi(argv[2]))) : 2400;
    const size_t total_bytes = total_mb * 1024ULL * 1024ULL;
    const size_t chunk_bytes = chunk_kb * 1024ULL;
    const size_t chunks = (total_bytes + chunk_bytes - 1) / chunk_bytes;

    try {
        Context ctx(nullptr, /*profiling=*/true);

        void* host = ctx.alloc_host(chunk_bytes);
        void* dev = ctx.alloc_device(chunk_bytes);
        if (!host || !dev) {
            std::printf("allocation failed host=%p dev=%p\n", host, dev);
            return 1;
        }
        std::memset(host, 0x5a, chunk_bytes);

        // Warm up: first copies pay page-fault and context setup costs.
        for (int i = 0; i < 8; ++i) ctx.memcpy_h2d(dev, host, chunk_bytes);
        ctx.synchronize();

        const int samples = 5;
        std::vector<double> rates;
        for (int s = 0; s < samples; ++s) {
            const auto t0 = Clock::now();
            for (size_t i = 0; i < chunks; ++i) ctx.memcpy_h2d(dev, host, chunk_bytes);
            ctx.synchronize();
            const double ms = ms_since(t0);
            // The loop reuses one device buffer, so a copy of the same size is
            // the realistic unit of expert staging (one expert is ~2.4 MB).
            rates.push_back(static_cast<double>(chunks) * static_cast<double>(chunk_bytes) / (ms * 1e6));
            std::printf("sample %d: %zu chunks of %zu KB in %.1f ms -> %.2f GB/s\n",
                        s, chunks, chunk_kb, ms, rates.back());
        }
        std::sort(rates.begin(), rates.end());
        std::printf("median %.2f GB/s  (total %.0f MB per sample, chunk %zu KB)\n",
                    rates[rates.size() / 2], static_cast<double>(total_mb), chunk_kb);
        std::printf("per-copy cost %.1f us\n",
                    1e3 * (total_bytes / static_cast<double>(chunks)) / (rates[rates.size() / 2] * 1e9) * 1e6);

        // Larger single transfer with the same total volume, for comparison:
        // if this is much faster, the per-copy overhead is what costs us.
        // Rotating the destination across a large region, as the real expert
        // pool does, is the case that matters: the fixed per-copy cost there may
        // include page/TLB setup that a single reused buffer never pays.
        void* dev_pool = ctx.alloc_device(1024ULL * 1024ULL * 1024ULL);
        if (host && dev_pool) {
            const size_t pool = 1024ULL * 1024ULL * 1024ULL;
            const size_t rotating_chunks = (total_bytes + chunk_bytes - 1) / chunk_bytes;
            std::vector<double> rot;
            for (int s = 0; s < samples; ++s) {
                const auto t0 = Clock::now();
                size_t off = 0;
                for (size_t i = 0; i < rotating_chunks; ++i) {
                    ctx.memcpy_h2d(static_cast<unsigned char*>(dev_pool) + off, host, chunk_bytes);
                    off = (off + chunk_bytes) % (pool - chunk_bytes);
                }
                ctx.synchronize();
                const double ms = ms_since(t0);
                rot.push_back(static_cast<double>(rotating_chunks) * static_cast<double>(chunk_bytes) / (ms * 1e6));
            }
            std::sort(rot.begin(), rot.end());
            std::printf("rotating destination over 1 GB: median %.2f GB/s\n", rot[rot.size() / 2]);
            ctx.free_device(dev_pool);
        }

        void* host_big = ctx.alloc_host(total_bytes);
        void* dev_big = ctx.alloc_device(total_bytes);
        if (host_big && dev_big) {
            std::memset(host_big, 0x3c, total_bytes);
            ctx.memcpy_h2d(dev_big, host_big, total_bytes);
            ctx.synchronize();
            const auto t0 = Clock::now();
            ctx.memcpy_h2d(dev_big, host_big, total_bytes);
            ctx.synchronize();
            const double ms = ms_since(t0);
            std::printf("single %.0f MB copy: %.1f ms -> %.2f GB/s\n",
                        static_cast<double>(total_mb), ms,
                        static_cast<double>(total_bytes) / (ms * 1e6));
        }
        std::printf("PROBE_OK\n");
        return 0;
    } catch (const std::exception& e) {
        std::printf("PROBE_FAILED: %s\n", e.what());
        return 1;
    }
}
