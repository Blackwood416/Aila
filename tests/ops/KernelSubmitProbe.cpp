// Micro-probe for the cost of a queue submission itself.
//
// Motivation: two fusions on the same stack, both removing submissions, paid
// very differently. Fusing three 256-work-group tail kernels into one removed 96
// submissions and 3.2 ms of host time (33 us each); folding a single-work-group
// classifier into the top-k kernel removed 48 submissions and only 0.57 ms
// (12 us each). So "submission = 53 us" is an average, not a law, and the
// question is what a submission actually costs as a function of its shape.
//
// This probe answers that directly: it enqueues K trivial kernels back to back,
// times only the enqueue loop (the queue is in-order and async, so that loop is
// the submission path), then drains and reports both halves. Four shapes:
// few arguments vs many, small grid vs large grid.

#include "src/ops/AttentionOps.cpp"

// Same stubs the other probes provide: these globals live in the engine
// translation unit, not in the ops implementation.
namespace aila::env {
int g_q35_prefill_step_override = -1;
bool g_kv_quant_override = false;
}

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

using bf16 = sycl::ext::oneapi::bfloat16;

// A kernel that touches nothing but its arguments, so the only thing being
// measured is the submission of its argument block plus the launch.
template <int NArgs>
float submit_shape(sycl::queue &q, int work_groups, int threads,
                   int iterations, double &submit_us) {
    // Dummy USM targets so the arguments are real pointers, not constants.
    int *sink = sycl::malloc_device<int>(1, q);
    q.memset(sink, 0, sizeof(int));
    q.wait();

    double enqueue_us = 0.0;
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < iterations; ++i) {
        q.parallel_for(sycl::nd_range<1>(sycl::range<1>(
                                            static_cast<size_t>(work_groups) * threads),
                                        sycl::range<1>(threads)),
                       [=](sycl::nd_item<1> item) {
                           // Consume every argument so it cannot be folded away.
                           float acc = 0.0f;
#pragma unroll
                           for (int a = 0; a < NArgs; ++a) {
                               acc += static_cast<float>(a) * 1e-7f;
                           }
                           if (acc < -1.0f) {
                               sink[0] = 1;
                           }
                           (void)item;
                       });
    }
    const auto t1 = std::chrono::steady_clock::now();
    q.wait();
    const auto t2 = std::chrono::steady_clock::now();

    enqueue_us = std::chrono::duration<double, std::micro>(t1 - t0).count();
    submit_us = enqueue_us / iterations;
    sycl::free(sink, q);
    return static_cast<float>(
        std::chrono::duration<double, std::micro>(t2 - t1).count() / iterations);
}

// Same as submit_shape but with NLocal dynamic shared-memory accessors, which
// the production kernels use heavily. A local_accessor is constructed on the
// host for every submission and registered with the handler, so if submissions
// of the real kernels cost far more than a bare kernel does, this is the most
// likely reason.
template <int NLocal>
float submit_shape_local(sycl::queue &q, int work_groups, int threads,
                         int iterations, double &submit_us) {
    int *sink = sycl::malloc_device<int>(1, q);
    q.memset(sink, 0, sizeof(int));
    q.wait();

    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < iterations; ++i) {
        q.submit([&](sycl::handler &h) {
            sycl::local_accessor<float, 1> a0(sycl::range<1>(64), h);
            sycl::local_accessor<float, 1> a1(sycl::range<1>(128), h);
            sycl::local_accessor<float, 1> a2(sycl::range<1>(2), h);
            sycl::local_accessor<int, 1> a3(sycl::range<1>(128), h);
            h.parallel_for(sycl::nd_range<1>(
                               sycl::range<1>(static_cast<size_t>(work_groups) * threads),
                               sycl::range<1>(threads)),
                           [=](sycl::nd_item<1> item) {
                               const int lid = static_cast<int>(item.get_local_id(0));
                               if constexpr (NLocal >= 1) a0[lid & 63] = 1.0f;
                               if constexpr (NLocal >= 2) a1[lid & 127] = 2.0f;
                               if constexpr (NLocal >= 3) a2[0] = 3.0f;
                               if constexpr (NLocal >= 4) a3[lid & 127] = lid;
                               sycl::group_barrier(item.get_group());
                               float s = 0.0f;
                               if constexpr (NLocal >= 1) s += a0[0];
                               if constexpr (NLocal >= 2) s += a1[0];
                               if constexpr (NLocal >= 3) s += a2[0];
                               if constexpr (NLocal >= 4) s += static_cast<float>(a3[0]);
                               if (s < -1.0f) sink[0] = 1;
                           });
        });
    }
    const auto t1 = std::chrono::steady_clock::now();
    q.wait();
    const auto t2 = std::chrono::steady_clock::now();
    submit_us = std::chrono::duration<double, std::micro>(t1 - t0).count() / iterations;
    sycl::free(sink, q);
    return static_cast<float>(
        std::chrono::duration<double, std::micro>(t2 - t1).count() / iterations);
}

}  // namespace

int main(int argc, char **argv) {
    int iterations = 200;
    if (argc > 1) {
        iterations = std::atoi(argv[1]);
        if (iterations < 20) iterations = 20;
    }

    Context ctx(nullptr, /*profiling=*/false);
    auto &q = ctx.queue();

    std::printf("iterations=%d\n", iterations);
    std::printf("%-28s %14s %14s\n", "shape", "submit_us/kernel", "drain_us/kernel");

    double submit_us = 0.0;
    double drain_us = 0.0;
    drain_us = submit_shape<2>(q, 1, 32, iterations, submit_us);
    std::printf("%-28s %14.2f %14.2f\n", "args=2   grid=1x32", submit_us, drain_us);
    drain_us = submit_shape<2>(q, 256, 256, iterations, submit_us);
    std::printf("%-28s %14.2f %14.2f\n", "args=2   grid=256x256", submit_us, drain_us);
    drain_us = submit_shape<200>(q, 1, 32, iterations, submit_us);
    std::printf("%-28s %14.2f %14.2f\n", "args=200 grid=1x32", submit_us, drain_us);
    drain_us = submit_shape<200>(q, 256, 256, iterations, submit_us);
    std::printf("%-28s %14.2f %14.2f\n", "args=200 grid=256x256", submit_us, drain_us);
    drain_us = submit_shape_local<1>(q, 256, 256, iterations, submit_us);
    std::printf("%-28s %14.2f %14.2f\n", "1 local_accessor  grid=256", submit_us, drain_us);
    drain_us = submit_shape_local<4>(q, 256, 256, iterations, submit_us);
    std::printf("%-28s %14.2f %14.2f\n", "4 local_accessors grid=256", submit_us, drain_us);

    std::printf("%s\n", "SUBMIT_PROBE_OK");
    return 0;
}
