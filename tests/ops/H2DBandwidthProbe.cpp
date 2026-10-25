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

#include <level_zero/ze_api.h>

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
                // Same source, but through sycl::queue::memcpy - the entry point the engine's
                // direct_arena staging uses (Qwen4ExpAdaptiveCache.cpp:677 records 9.05-9.23 GB/s
                // for it) - rather than Context::memcpy_h2d, which measured far lower above.
                void* large2 = ctx.alloc_large_host(256ULL * 1024ULL * 1024ULL);
                if (large2) {
                    std::memset(large2, 0x5a, 256ULL * 1024ULL * 1024ULL);
                    sycl::queue& q = ctx.queue();
                    q.memcpy(dev, large2, chunk_bytes);
                    q.wait_and_throw();
                    const auto t1 = Clock::now();
                    sycl::event last{};
                    for (size_t i = 0; i < large_chunks; ++i)
                        last = q.memcpy(dev, large2, chunk_bytes);
                    const double enqueue_ms = ms_since(t1);
                    last.wait_and_throw();
                    const double ms2 = ms_since(t1);
                    std::printf("large-page via queue.memcpy: %zu chunks of %zu KB in %.1f ms -> %.2f GB/s "
                                "(enqueue alone %.1f ms, wait %.1f ms) -> %s\n",
                                large_chunks, chunk_kb, ms2,
                                static_cast<double>(large_chunks) * static_cast<double>(chunk_bytes) / (ms2 * 1e6),
                                enqueue_ms, ms2 - enqueue_ms,
                                (enqueue_ms > 0.7 * ms2) ? "BLOCKING" : "ASYNC");
                    ctx.free_host(large2);
                }
            }
        } catch (const std::exception& e) {
            std::printf("large-page source: REJECTED by the runtime (%s)\n", e.what());
        }

        // Argon's rank-2 claim: a VirtualAlloc(MEM_LARGE_PAGES) pointer is locked by the OS but unknown to
        // Level Zero, so every queue::memcpy makes NEO do per-call host-pointer registration, which is why
        // the enqueue costs ~111 us. Importing the arena once should make those calls cheap. This tests the
        // premise only - nothing here changes the engine.
        {
            void* arena = ctx.alloc_large_host(256ULL * 1024ULL * 1024ULL);
            void* dev2 = ctx.alloc_device(chunk_bytes);
            if (arena && dev2) {
                std::memset(arena, 0x11, 256ULL * 1024ULL * 1024ULL);
                const size_t n = (256ULL * 1024ULL * 1024ULL) / chunk_bytes;
                sycl::queue& q = ctx.queue();

                auto time_copies = [&](const char* tag) {
                    q.memcpy(dev2, arena, chunk_bytes);
                    q.wait_and_throw();
                    const auto t0 = Clock::now();
                    sycl::event last{};
                    for (size_t i = 0; i < n; ++i) last = q.memcpy(dev2, arena, chunk_bytes);
                    const double enq = ms_since(t0);
                    last.wait_and_throw();
                    const double tot = ms_since(t0);
                    std::printf("%s: %zu copies, enqueue %.1f ms (%.1f us/copy), total %.1f ms\n",
                                tag, n, enq, enq * 1000.0 / static_cast<double>(n), tot);
                };

                time_copies("arena, before import");

                // Import through the Level Zero driver extension, if the driver offers it.
                bool imported = false;
                try {
                    auto ze_context = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(q.get_context());
                    ze_driver_handle_t driver = nullptr;
                    uint32_t driver_count = 0;
                    if (zeDriverGet(&driver_count, nullptr) == ZE_RESULT_SUCCESS && driver_count > 0) {
                        std::vector<ze_driver_handle_t> drivers(driver_count);
                        zeDriverGet(&driver_count, drivers.data());
                        driver = drivers[0];
                    }
                    using import_fn = ze_result_t (*)(ze_driver_handle_t, void*, size_t);
                    void* raw = nullptr;
                    auto fn = (driver && zeDriverGetExtensionFunctionAddress(
                                   driver, "zexDriverImportExternalPointer", &raw) == ZE_RESULT_SUCCESS)
                        ? reinterpret_cast<import_fn>(raw)
                        : nullptr;
                    if (fn != nullptr) {
                        const ze_result_t r = fn(driver, arena, 256ULL * 1024ULL * 1024ULL);
                        imported = (r == ZE_RESULT_SUCCESS);
                        std::printf("zexDriverImportExternalPointer: %s (0x%x)\n",
                                    imported ? "OK" : "failed", static_cast<unsigned>(r));
                    } else {
                        std::printf("zexDriverImportExternalPointer: extension not available\n");
                    }
                    (void)ze_context;
                } catch (const std::exception& e) {
                    std::printf("zexDriverImportExternalPointer: threw (%s)\n", e.what());
                }

                time_copies(imported ? "arena, after import" : "arena, import unavailable");
                ctx.free_device(dev2);
                ctx.free_host(arena);
            }
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
