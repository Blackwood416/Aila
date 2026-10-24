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

        // The engine keeps resident experts in a VirtualAlloc(MEM_LARGE_PAGES) arena and then
        // copies each expert into a pinned staging buffer before DMA-ing it, which costs 7.1 s per
        // prefill pass. The question that decides whether that hop can be deleted: will
        // queue::memcpy accept a large-page pointer (not a SYCL USM allocation) as the source, and
        // if it does, does it DMA at link speed or bounce through a runtime copy?
        try {
            void* large = ctx.alloc_large_host(256ULL * 1024ULL * 1024ULL);
            if (large && dev) {
                std::memset(large, 0xa5, 256ULL * 1024ULL * 1024ULL);
                const size_t large_chunks = (256ULL * 1024ULL * 1024ULL) / chunk_bytes;
                ctx.memcpy_h2d(dev, large, chunk_bytes);
                ctx.synchronize();
                const auto t0 = Clock::now();
                for (size_t i = 0; i < large_chunks; ++i) ctx.memcpy_h2d(dev, large, chunk_bytes);
                ctx.synchronize();
                const double ms = ms_since(t0);
                std::printf("large-page source: %zu chunks of %zu KB in %.1f ms -> %.2f GB/s\n",
                            large_chunks, chunk_kb, ms,
                            static_cast<double>(large_chunks) * static_cast<double>(chunk_bytes) / (ms * 1e6));
                ctx.free_host(large);
            }
        } catch (const std::exception& e) {
            std::printf("large-page source: REJECTED by the runtime (%s)\n", e.what());
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

        // Does a second queue actually overlap with compute on this driver? The
        // engine runs everything on one in-order queue, so a copy and a kernel can
        // never overlap there. This block measures the premise for fixing that: a
        // busy integer kernel on the main queue against the same 42.6 GB of
        // expert-sized copies on a second queue.
        {
            const size_t spin_iters = 40000000;  // tuned below by measurement
            auto burn = [&](sycl::queue& q) {
                int* out = static_cast<int*>(ctx.alloc_device(sizeof(int)));
                const sycl::event ev = q.submit([&](sycl::handler& h) {
                    int* o = out;
                    h.single_task([=]() {
                        int acc = 0;
                        for (size_t i = 0; i < spin_iters; ++i) acc += static_cast<int>(i & 0x3ff);
                        o[0] = acc;
                    });
                });
                return std::make_pair(ev, out);
            };

            sycl::queue copy_q(ctx.queue().get_device(),
                               sycl::property_list{sycl::property::queue::in_order()});

            // Sequential: kernel, then the copies, all on the main queue.
            auto seq_burn = burn(ctx.queue());
            const auto t0 = Clock::now();
            seq_burn.first.wait();
            const double kernel_ms = ms_since(t0);
            const auto t1 = Clock::now();
            for (size_t i = 0; i < chunks; ++i) ctx.memcpy_h2d(dev, host, chunk_bytes);
            ctx.synchronize();
            const double seq_copy_ms = ms_since(t1);

            // Concurrent: kernel on the main queue, copies on the second queue.
            const auto t2 = Clock::now();
            auto con_burn = burn(ctx.queue());
            for (size_t i = 0; i < chunks; ++i) copy_q.memcpy(dev, host, chunk_bytes).wait();
            copy_q.wait();
            con_burn.first.wait();
            const double con_ms = ms_since(t2);

            std::printf("overlap test: kernel %.1f ms, copies %.1f ms, sum %.1f ms, "
                        "concurrent %.1f ms -> %s\n",
                        kernel_ms, seq_copy_ms, kernel_ms + seq_copy_ms, con_ms,
                        (con_ms < 0.85 * (kernel_ms + seq_copy_ms)) ? "OVERLAPS" : "SERIALISED");
            ctx.free_device(seq_burn.second);
            ctx.free_device(con_burn.second);
        }
        std::printf("PROBE_OK\n");
        return 0;
    } catch (const std::exception& e) {
        std::printf("PROBE_FAILED: %s\n", e.what());
        return 1;
    }
}
