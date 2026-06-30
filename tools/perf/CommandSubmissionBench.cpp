#include <level_zero/ze_api.h>
#include <sycl/ext/oneapi/experimental/graph.hpp>
#include <sycl/sycl.hpp>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using Clock = std::chrono::high_resolution_clock;

struct Options {
    int warmup = 1000;
    int iters = 10000;
    size_t bytes = 4096;
    int ops_per_graph = 16;
};

struct TimingStats {
    double avg_us = 0.0;
    double min_us = 0.0;
    double p50_us = 0.0;
    double p95_us = 0.0;
    double max_us = 0.0;
};

struct BenchResult {
    std::string backend;
    std::string mode;
    std::string device;
    int warmup = 0;
    int iters = 0;
    size_t bytes = 0;
    int ops_per_iter = 1;
    TimingStats host;
    TimingStats total;
};

double elapsed_us(Clock::time_point begin, Clock::time_point end) {
    return std::chrono::duration<double, std::micro>(end - begin).count();
}

TimingStats summarize(std::vector<double> values) {
    if (values.empty()) {
        return {};
    }
    std::sort(values.begin(), values.end());
    const double sum = std::accumulate(values.begin(), values.end(), 0.0);
    auto percentile = [&](double p) {
        const double pos = p * static_cast<double>(values.size() - 1);
        const size_t lo = static_cast<size_t>(pos);
        const size_t hi = std::min(values.size() - 1, lo + 1);
        const double frac = pos - static_cast<double>(lo);
        return values[lo] * (1.0 - frac) + values[hi] * frac;
    };
    TimingStats stats;
    stats.avg_us = sum / static_cast<double>(values.size());
    stats.min_us = values.front();
    stats.p50_us = percentile(0.50);
    stats.p95_us = percentile(0.95);
    stats.max_us = values.back();
    return stats;
}

Options parse_args(int argc, char** argv) {
    Options opts;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto read_value = [&](const char* name) -> std::string {
            if (i + 1 >= argc) {
                throw std::runtime_error(std::string("missing value for ") + name);
            }
            return argv[++i];
        };
        if (arg == "--warmup") {
            opts.warmup = std::max(0, std::stoi(read_value("--warmup")));
        } else if (arg == "--iters") {
            opts.iters = std::max(1, std::stoi(read_value("--iters")));
        } else if (arg == "--bytes") {
            opts.bytes = std::max<size_t>(1, static_cast<size_t>(
                std::stoull(read_value("--bytes"))));
        } else if (arg == "--ops-per-graph") {
            opts.ops_per_graph = std::max(1, std::stoi(read_value("--ops-per-graph")));
        } else if (arg == "--help" || arg == "-h") {
            std::cout
                << "Usage: AilaCommandSubmissionBench [--warmup N] [--iters N] "
                   "[--bytes N] [--ops-per-graph N]\n";
            std::exit(0);
        } else {
            throw std::runtime_error("unknown argument: " + arg);
        }
    }
    return opts;
}

std::string env_value(const char* name) {
#ifdef _WIN32
    char* raw = nullptr;
    size_t len = 0;
    if (_dupenv_s(&raw, &len, name) != 0 || raw == nullptr) {
        return "";
    }
    std::string value(raw);
    std::free(raw);
    return value;
#else
    const char* raw = std::getenv(name);
    return raw ? std::string(raw) : std::string();
#endif
}

void print_result(const BenchResult& r) {
    std::cout << r.backend << ","
              << r.mode << ","
              << '"' << r.device << '"' << ","
              << r.warmup << ","
              << r.iters << ","
              << r.bytes << ","
              << r.ops_per_iter << ","
              << std::fixed << std::setprecision(3)
              << r.host.avg_us << ","
              << r.host.min_us << ","
              << r.host.p50_us << ","
              << r.host.p95_us << ","
              << r.host.max_us << ","
              << r.total.avg_us << ","
              << r.total.min_us << ","
              << r.total.p50_us << ","
              << r.total.p95_us << ","
              << r.total.max_us << "\n";
}

template <typename Fn>
void print_optional_result(const char* mode, Fn&& fn) {
    try {
        print_result(fn());
    } catch (const std::exception& e) {
        std::cout << "skip," << mode << "," << e.what() << "\n";
    }
}

BenchResult run_sycl_memset(const Options& opts) {
    sycl::queue q{sycl::gpu_selector_v, sycl::property::queue::in_order()};
    void* ptr = sycl::malloc_device(opts.bytes, q);
    if (!ptr) {
        throw std::runtime_error("sycl::malloc_device failed");
    }

    for (int i = 0; i < opts.warmup; ++i) {
        q.memset(ptr, 0, opts.bytes).wait();
    }

    std::vector<double> host_us;
    std::vector<double> total_us;
    host_us.reserve(static_cast<size_t>(opts.iters));
    total_us.reserve(static_cast<size_t>(opts.iters));
    for (int i = 0; i < opts.iters; ++i) {
        const auto begin = Clock::now();
        sycl::event ev = q.memset(ptr, i & 0xff, opts.bytes);
        const auto submitted = Clock::now();
        ev.wait();
        const auto done = Clock::now();
        host_us.push_back(elapsed_us(begin, submitted));
        total_us.push_back(elapsed_us(begin, done));
    }

    BenchResult result;
    result.backend = "sycl";
    result.mode = "queue_memset";
    result.device = q.get_device().get_info<sycl::info::device::name>();
    result.warmup = opts.warmup;
    result.iters = opts.iters;
    result.bytes = opts.bytes;
    result.ops_per_iter = 1;
    result.host = summarize(std::move(host_us));
    result.total = summarize(std::move(total_us));
    sycl::free(ptr, q);
    return result;
}

BenchResult run_sycl_single_task(const Options& opts) {
    sycl::queue q{sycl::gpu_selector_v, sycl::property::queue::in_order()};
    auto* ptr = static_cast<uint32_t*>(sycl::malloc_shared(sizeof(uint32_t), q));
    if (!ptr) {
        throw std::runtime_error("sycl::malloc_shared failed");
    }
    *ptr = 0;

    for (int i = 0; i < opts.warmup; ++i) {
        q.single_task([=]() { ptr[0] += 1; }).wait();
    }

    std::vector<double> host_us;
    std::vector<double> total_us;
    host_us.reserve(static_cast<size_t>(opts.iters));
    total_us.reserve(static_cast<size_t>(opts.iters));
    for (int i = 0; i < opts.iters; ++i) {
        const auto begin = Clock::now();
        sycl::event ev = q.single_task([=]() { ptr[0] += 1; });
        const auto submitted = Clock::now();
        ev.wait();
        const auto done = Clock::now();
        host_us.push_back(elapsed_us(begin, submitted));
        total_us.push_back(elapsed_us(begin, done));
    }

    BenchResult result;
    result.backend = "sycl";
    result.mode = "queue_single_task";
    result.device = q.get_device().get_info<sycl::info::device::name>();
    result.warmup = opts.warmup;
    result.iters = opts.iters;
    result.bytes = sizeof(uint32_t);
    result.ops_per_iter = 1;
    result.host = summarize(std::move(host_us));
    result.total = summarize(std::move(total_us));
    sycl::free(ptr, q);
    return result;
}

BenchResult run_sycl_memset_graph(const Options& opts) {
    sycl::queue q{sycl::gpu_selector_v, sycl::property::queue::in_order()};
    const size_t total_bytes = opts.bytes * static_cast<size_t>(opts.ops_per_graph);
    auto* ptr = static_cast<uint8_t*>(sycl::malloc_device(total_bytes, q));
    if (!ptr) {
        throw std::runtime_error("sycl::malloc_device failed");
    }

    auto submit_graph = [&](int seed) {
        sycl::event last;
        for (int op = 0; op < opts.ops_per_graph; ++op) {
            last = q.memset(ptr + static_cast<size_t>(op) * opts.bytes,
                            (seed + op) & 0xff,
                            opts.bytes);
        }
        last.wait();
    };

    for (int i = 0; i < opts.warmup; ++i) {
        submit_graph(i);
    }

    std::vector<double> host_us;
    std::vector<double> total_us;
    host_us.reserve(static_cast<size_t>(opts.iters));
    total_us.reserve(static_cast<size_t>(opts.iters));
    for (int i = 0; i < opts.iters; ++i) {
        const auto begin = Clock::now();
        sycl::event last;
        for (int op = 0; op < opts.ops_per_graph; ++op) {
            last = q.memset(ptr + static_cast<size_t>(op) * opts.bytes,
                            (i + op) & 0xff,
                            opts.bytes);
        }
        const auto submitted = Clock::now();
        last.wait();
        const auto done = Clock::now();
        host_us.push_back(elapsed_us(begin, submitted));
        total_us.push_back(elapsed_us(begin, done));
    }

    BenchResult result;
    result.backend = "sycl";
    result.mode = "queue_memset_graph";
    result.device = q.get_device().get_info<sycl::info::device::name>();
    result.warmup = opts.warmup;
    result.iters = opts.iters;
    result.bytes = opts.bytes;
    result.ops_per_iter = opts.ops_per_graph;
    result.host = summarize(std::move(host_us));
    result.total = summarize(std::move(total_us));
    sycl::free(ptr, q);
    return result;
}

BenchResult run_sycl_kernel_chain_batch(const Options& opts) {
    sycl::queue q{sycl::gpu_selector_v, sycl::property::queue::in_order()};
    const size_t elems =
        std::max<size_t>(1, (opts.bytes + sizeof(uint32_t) - 1) / sizeof(uint32_t));
    auto* data = static_cast<uint32_t*>(
        sycl::malloc_device(elems * sizeof(uint32_t), q));
    auto* param = static_cast<uint32_t*>(
        sycl::malloc_shared(sizeof(uint32_t), q));
    if (!data || !param) {
        throw std::runtime_error("sycl USM allocation failed");
    }

    q.memset(data, 0, elems * sizeof(uint32_t)).wait();
    *param = 0;

    auto submit_chain = [&](uint32_t seed) {
        *param = seed;
        sycl::event last;
        for (int op = 0; op < opts.ops_per_graph; ++op) {
            last = q.parallel_for(sycl::range<1>(elems), [=](sycl::id<1> idx) {
                const uint32_t i = static_cast<uint32_t>(idx[0]);
                data[i] = data[i] * 1664525u + 1013904223u + param[0] +
                          static_cast<uint32_t>(op) + i;
            });
        }
        last.wait();
    };

    for (int i = 0; i < opts.warmup; ++i) {
        submit_chain(static_cast<uint32_t>(i));
    }

    std::vector<double> host_us;
    std::vector<double> total_us;
    host_us.reserve(static_cast<size_t>(opts.iters));
    total_us.reserve(static_cast<size_t>(opts.iters));
    for (int i = 0; i < opts.iters; ++i) {
        const auto begin = Clock::now();
        *param = static_cast<uint32_t>(i);
        sycl::event last;
        for (int op = 0; op < opts.ops_per_graph; ++op) {
            last = q.parallel_for(sycl::range<1>(elems), [=](sycl::id<1> idx) {
                const uint32_t j = static_cast<uint32_t>(idx[0]);
                data[j] = data[j] * 1664525u + 1013904223u + param[0] +
                          static_cast<uint32_t>(op) + j;
            });
        }
        const auto submitted = Clock::now();
        last.wait();
        const auto done = Clock::now();
        host_us.push_back(elapsed_us(begin, submitted));
        total_us.push_back(elapsed_us(begin, done));
    }

    BenchResult result;
    result.backend = "sycl";
    result.mode = "queue_kernel_chain";
    result.device = q.get_device().get_info<sycl::info::device::name>();
    result.warmup = opts.warmup;
    result.iters = opts.iters;
    result.bytes = opts.bytes;
    result.ops_per_iter = opts.ops_per_graph;
    result.host = summarize(std::move(host_us));
    result.total = summarize(std::move(total_us));
    sycl::free(data, q);
    sycl::free(param, q);
    return result;
}

BenchResult run_sycl_kernel_chain_command_graph(const Options& opts) {
    namespace exp = sycl::ext::oneapi::experimental;

    sycl::queue q{sycl::gpu_selector_v, sycl::property::queue::in_order()};
    if (!q.get_device().has(sycl::aspect::ext_oneapi_graph)) {
        throw std::runtime_error("device does not report ext_oneapi_graph support");
    }

    const size_t elems =
        std::max<size_t>(1, (opts.bytes + sizeof(uint32_t) - 1) / sizeof(uint32_t));
    auto* data = static_cast<uint32_t*>(
        sycl::malloc_device(elems * sizeof(uint32_t), q));
    auto* param = static_cast<uint32_t*>(
        sycl::malloc_shared(sizeof(uint32_t), q));
    if (!data || !param) {
        throw std::runtime_error("sycl USM allocation failed");
    }

    q.memset(data, 0, elems * sizeof(uint32_t)).wait();
    *param = 0;

    exp::command_graph graph{q};
    graph.begin_recording(q);
    for (int op = 0; op < opts.ops_per_graph; ++op) {
        q.parallel_for(sycl::range<1>(elems), [=](sycl::id<1> idx) {
            const uint32_t i = static_cast<uint32_t>(idx[0]);
            data[i] = data[i] * 1664525u + 1013904223u + param[0] +
                      static_cast<uint32_t>(op) + i;
        });
    }
    graph.end_recording(q);
    auto executable_graph = graph.finalize();

    for (int i = 0; i < opts.warmup; ++i) {
        *param = static_cast<uint32_t>(i);
        q.ext_oneapi_graph(executable_graph).wait();
    }

    std::vector<double> host_us;
    std::vector<double> total_us;
    host_us.reserve(static_cast<size_t>(opts.iters));
    total_us.reserve(static_cast<size_t>(opts.iters));
    for (int i = 0; i < opts.iters; ++i) {
        const auto begin = Clock::now();
        *param = static_cast<uint32_t>(i);
        sycl::event ev = q.ext_oneapi_graph(executable_graph);
        const auto submitted = Clock::now();
        ev.wait();
        const auto done = Clock::now();
        host_us.push_back(elapsed_us(begin, submitted));
        total_us.push_back(elapsed_us(begin, done));
    }

    BenchResult result;
    result.backend = "sycl";
    result.mode = "command_graph_kernel_chain";
    result.device = q.get_device().get_info<sycl::info::device::name>();
    result.warmup = opts.warmup;
    result.iters = opts.iters;
    result.bytes = opts.bytes;
    result.ops_per_iter = opts.ops_per_graph;
    result.host = summarize(std::move(host_us));
    result.total = summarize(std::move(total_us));
    sycl::free(data, q);
    sycl::free(param, q);
    return result;
}

void check_ze(ze_result_t result, const char* expr) {
    if (result != ZE_RESULT_SUCCESS) {
        throw std::runtime_error(std::string(expr) + " failed with ze_result_t=" +
                                 std::to_string(static_cast<int>(result)));
    }
}

#define CHECK_ZE(expr) check_ze((expr), #expr)

struct LevelZeroState {
    ze_driver_handle_t driver = nullptr;
    ze_device_handle_t device = nullptr;
    ze_context_handle_t context = nullptr;
    uint32_t queue_ordinal = 0;
    std::string device_name;

    explicit LevelZeroState() {
        CHECK_ZE(zeInit(0));
        uint32_t driver_count = 0;
        CHECK_ZE(zeDriverGet(&driver_count, nullptr));
        if (driver_count == 0) {
            throw std::runtime_error("no Level Zero drivers found");
        }
        std::vector<ze_driver_handle_t> drivers(driver_count);
        CHECK_ZE(zeDriverGet(&driver_count, drivers.data()));

        for (ze_driver_handle_t candidate_driver : drivers) {
            uint32_t device_count = 0;
            CHECK_ZE(zeDeviceGet(candidate_driver, &device_count, nullptr));
            std::vector<ze_device_handle_t> devices(device_count);
            CHECK_ZE(zeDeviceGet(candidate_driver, &device_count, devices.data()));
            for (ze_device_handle_t candidate_device : devices) {
                ze_device_properties_t props{};
                props.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;
                CHECK_ZE(zeDeviceGetProperties(candidate_device, &props));
                if (!device || props.type == ZE_DEVICE_TYPE_GPU) {
                    driver = candidate_driver;
                    device = candidate_device;
                    device_name = props.name;
                    if (props.type == ZE_DEVICE_TYPE_GPU) {
                        break;
                    }
                }
            }
            if (device) {
                break;
            }
        }
        if (!device) {
            throw std::runtime_error("no Level Zero devices found");
        }

        ze_context_desc_t context_desc{};
        context_desc.stype = ZE_STRUCTURE_TYPE_CONTEXT_DESC;
        CHECK_ZE(zeContextCreate(driver, &context_desc, &context));

        uint32_t group_count = 0;
        CHECK_ZE(zeDeviceGetCommandQueueGroupProperties(device, &group_count, nullptr));
        std::vector<ze_command_queue_group_properties_t> groups(group_count);
        for (auto& group : groups) {
            group.stype = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_GROUP_PROPERTIES;
        }
        CHECK_ZE(zeDeviceGetCommandQueueGroupProperties(device, &group_count, groups.data()));
        bool found_queue = false;
        for (uint32_t i = 0; i < group_count; ++i) {
            if ((groups[i].flags & ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COMPUTE) != 0) {
                queue_ordinal = i;
                found_queue = true;
                break;
            }
        }
        if (!found_queue) {
            throw std::runtime_error("no Level Zero compute command queue group found");
        }
    }

    LevelZeroState(const LevelZeroState&) = delete;
    LevelZeroState& operator=(const LevelZeroState&) = delete;

    ~LevelZeroState() {
        if (context) {
            zeContextDestroy(context);
        }
    }
};

void* alloc_l0_device(LevelZeroState& l0, size_t bytes) {
    ze_device_mem_alloc_desc_t alloc_desc{};
    alloc_desc.stype = ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC;
    void* ptr = nullptr;
    CHECK_ZE(zeMemAllocDevice(l0.context, &alloc_desc, bytes, 64, l0.device, &ptr));
    return ptr;
}

ze_command_queue_desc_t command_queue_desc(const LevelZeroState& l0) {
    ze_command_queue_desc_t desc{};
    desc.stype = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC;
    desc.ordinal = l0.queue_ordinal;
    desc.index = 0;
    desc.mode = ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS;
    desc.priority = ZE_COMMAND_QUEUE_PRIORITY_NORMAL;
    return desc;
}

ze_command_list_handle_t create_regular_list(LevelZeroState& l0) {
    ze_command_list_desc_t desc{};
    desc.stype = ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC;
    desc.commandQueueGroupOrdinal = l0.queue_ordinal;
    ze_command_list_handle_t list = nullptr;
    CHECK_ZE(zeCommandListCreate(l0.context, l0.device, &desc, &list));
    return list;
}

ze_command_queue_handle_t create_queue(LevelZeroState& l0) {
    ze_command_queue_desc_t desc = command_queue_desc(l0);
    ze_command_queue_handle_t queue = nullptr;
    CHECK_ZE(zeCommandQueueCreate(l0.context, l0.device, &desc, &queue));
    return queue;
}

void append_fill(ze_command_list_handle_t list, void* ptr, size_t bytes, uint32_t pattern) {
    CHECK_ZE(zeCommandListAppendMemoryFill(
        list, ptr, &pattern, sizeof(pattern), bytes, nullptr, 0, nullptr));
}

uint8_t* byte_ptr(void* ptr, size_t offset) {
    return static_cast<uint8_t*>(ptr) + offset;
}

BenchResult run_l0_regular_record_each(LevelZeroState& l0, const Options& opts) {
    void* ptr = alloc_l0_device(l0, opts.bytes);
    ze_command_queue_handle_t queue = create_queue(l0);
    ze_command_list_handle_t list = create_regular_list(l0);

    auto run_once = [&](uint32_t pattern) {
        CHECK_ZE(zeCommandListReset(list));
        append_fill(list, ptr, opts.bytes, pattern);
        CHECK_ZE(zeCommandListClose(list));
        CHECK_ZE(zeCommandQueueExecuteCommandLists(queue, 1, &list, nullptr));
        CHECK_ZE(zeCommandQueueSynchronize(queue, UINT64_MAX));
    };
    for (int i = 0; i < opts.warmup; ++i) {
        run_once(static_cast<uint32_t>(i));
    }

    std::vector<double> host_us;
    std::vector<double> total_us;
    host_us.reserve(static_cast<size_t>(opts.iters));
    total_us.reserve(static_cast<size_t>(opts.iters));
    for (int i = 0; i < opts.iters; ++i) {
        const auto begin = Clock::now();
        CHECK_ZE(zeCommandListReset(list));
        append_fill(list, ptr, opts.bytes, static_cast<uint32_t>(i));
        CHECK_ZE(zeCommandListClose(list));
        CHECK_ZE(zeCommandQueueExecuteCommandLists(queue, 1, &list, nullptr));
        const auto submitted = Clock::now();
        CHECK_ZE(zeCommandQueueSynchronize(queue, UINT64_MAX));
        const auto done = Clock::now();
        host_us.push_back(elapsed_us(begin, submitted));
        total_us.push_back(elapsed_us(begin, done));
    }

    CHECK_ZE(zeCommandListDestroy(list));
    CHECK_ZE(zeCommandQueueDestroy(queue));
    CHECK_ZE(zeMemFree(l0.context, ptr));

    BenchResult result;
    result.backend = "level_zero";
    result.mode = "regular_record_each_fill";
    result.device = l0.device_name;
    result.warmup = opts.warmup;
    result.iters = opts.iters;
    result.bytes = opts.bytes;
    result.ops_per_iter = 1;
    result.host = summarize(std::move(host_us));
    result.total = summarize(std::move(total_us));
    return result;
}

BenchResult run_l0_regular_replay(LevelZeroState& l0, const Options& opts) {
    void* ptr = alloc_l0_device(l0, opts.bytes);
    ze_command_queue_handle_t queue = create_queue(l0);
    ze_command_list_handle_t list = create_regular_list(l0);

    append_fill(list, ptr, opts.bytes, 0x01020304u);
    CHECK_ZE(zeCommandListClose(list));
    for (int i = 0; i < opts.warmup; ++i) {
        CHECK_ZE(zeCommandQueueExecuteCommandLists(queue, 1, &list, nullptr));
        CHECK_ZE(zeCommandQueueSynchronize(queue, UINT64_MAX));
    }

    std::vector<double> host_us;
    std::vector<double> total_us;
    host_us.reserve(static_cast<size_t>(opts.iters));
    total_us.reserve(static_cast<size_t>(opts.iters));
    for (int i = 0; i < opts.iters; ++i) {
        const auto begin = Clock::now();
        CHECK_ZE(zeCommandQueueExecuteCommandLists(queue, 1, &list, nullptr));
        const auto submitted = Clock::now();
        CHECK_ZE(zeCommandQueueSynchronize(queue, UINT64_MAX));
        const auto done = Clock::now();
        host_us.push_back(elapsed_us(begin, submitted));
        total_us.push_back(elapsed_us(begin, done));
    }

    CHECK_ZE(zeCommandListDestroy(list));
    CHECK_ZE(zeCommandQueueDestroy(queue));
    CHECK_ZE(zeMemFree(l0.context, ptr));

    BenchResult result;
    result.backend = "level_zero";
    result.mode = "regular_replay_fill";
    result.device = l0.device_name;
    result.warmup = opts.warmup;
    result.iters = opts.iters;
    result.bytes = opts.bytes;
    result.ops_per_iter = 1;
    result.host = summarize(std::move(host_us));
    result.total = summarize(std::move(total_us));
    return result;
}

BenchResult run_l0_regular_replay_graph(LevelZeroState& l0, const Options& opts) {
    const size_t total_bytes = opts.bytes * static_cast<size_t>(opts.ops_per_graph);
    void* ptr = alloc_l0_device(l0, total_bytes);
    ze_command_queue_handle_t queue = create_queue(l0);
    ze_command_list_handle_t list = create_regular_list(l0);

    for (int op = 0; op < opts.ops_per_graph; ++op) {
        append_fill(list,
                    byte_ptr(ptr, static_cast<size_t>(op) * opts.bytes),
                    opts.bytes,
                    static_cast<uint32_t>(0x01020304u + op));
    }
    CHECK_ZE(zeCommandListClose(list));
    for (int i = 0; i < opts.warmup; ++i) {
        CHECK_ZE(zeCommandQueueExecuteCommandLists(queue, 1, &list, nullptr));
        CHECK_ZE(zeCommandQueueSynchronize(queue, UINT64_MAX));
    }

    std::vector<double> host_us;
    std::vector<double> total_us;
    host_us.reserve(static_cast<size_t>(opts.iters));
    total_us.reserve(static_cast<size_t>(opts.iters));
    for (int i = 0; i < opts.iters; ++i) {
        const auto begin = Clock::now();
        CHECK_ZE(zeCommandQueueExecuteCommandLists(queue, 1, &list, nullptr));
        const auto submitted = Clock::now();
        CHECK_ZE(zeCommandQueueSynchronize(queue, UINT64_MAX));
        const auto done = Clock::now();
        host_us.push_back(elapsed_us(begin, submitted));
        total_us.push_back(elapsed_us(begin, done));
    }

    CHECK_ZE(zeCommandListDestroy(list));
    CHECK_ZE(zeCommandQueueDestroy(queue));
    CHECK_ZE(zeMemFree(l0.context, ptr));

    BenchResult result;
    result.backend = "level_zero";
    result.mode = "regular_replay_fill_graph";
    result.device = l0.device_name;
    result.warmup = opts.warmup;
    result.iters = opts.iters;
    result.bytes = opts.bytes;
    result.ops_per_iter = opts.ops_per_graph;
    result.host = summarize(std::move(host_us));
    result.total = summarize(std::move(total_us));
    return result;
}

BenchResult run_l0_immediate(LevelZeroState& l0, const Options& opts) {
    void* ptr = alloc_l0_device(l0, opts.bytes);
    ze_command_queue_desc_t desc = command_queue_desc(l0);
    ze_command_list_handle_t list = nullptr;
    CHECK_ZE(zeCommandListCreateImmediate(l0.context, l0.device, &desc, &list));

    auto run_once = [&](uint32_t pattern) {
        append_fill(list, ptr, opts.bytes, pattern);
        CHECK_ZE(zeCommandListHostSynchronize(list, UINT64_MAX));
    };
    for (int i = 0; i < opts.warmup; ++i) {
        run_once(static_cast<uint32_t>(i));
    }

    std::vector<double> host_us;
    std::vector<double> total_us;
    host_us.reserve(static_cast<size_t>(opts.iters));
    total_us.reserve(static_cast<size_t>(opts.iters));
    for (int i = 0; i < opts.iters; ++i) {
        const auto begin = Clock::now();
        append_fill(list, ptr, opts.bytes, static_cast<uint32_t>(i));
        const auto submitted = Clock::now();
        CHECK_ZE(zeCommandListHostSynchronize(list, UINT64_MAX));
        const auto done = Clock::now();
        host_us.push_back(elapsed_us(begin, submitted));
        total_us.push_back(elapsed_us(begin, done));
    }

    CHECK_ZE(zeCommandListDestroy(list));
    CHECK_ZE(zeMemFree(l0.context, ptr));

    BenchResult result;
    result.backend = "level_zero";
    result.mode = "immediate_fill";
    result.device = l0.device_name;
    result.warmup = opts.warmup;
    result.iters = opts.iters;
    result.bytes = opts.bytes;
    result.ops_per_iter = 1;
    result.host = summarize(std::move(host_us));
    result.total = summarize(std::move(total_us));
    return result;
}

BenchResult run_l0_immediate_graph(LevelZeroState& l0, const Options& opts) {
    const size_t total_bytes = opts.bytes * static_cast<size_t>(opts.ops_per_graph);
    void* ptr = alloc_l0_device(l0, total_bytes);
    ze_command_queue_desc_t desc = command_queue_desc(l0);
    ze_command_list_handle_t list = nullptr;
    CHECK_ZE(zeCommandListCreateImmediate(l0.context, l0.device, &desc, &list));

    auto run_once = [&](int seed) {
        for (int op = 0; op < opts.ops_per_graph; ++op) {
            append_fill(list,
                        byte_ptr(ptr, static_cast<size_t>(op) * opts.bytes),
                        opts.bytes,
                        static_cast<uint32_t>(seed + op));
        }
        CHECK_ZE(zeCommandListHostSynchronize(list, UINT64_MAX));
    };
    for (int i = 0; i < opts.warmup; ++i) {
        run_once(i);
    }

    std::vector<double> host_us;
    std::vector<double> total_us;
    host_us.reserve(static_cast<size_t>(opts.iters));
    total_us.reserve(static_cast<size_t>(opts.iters));
    for (int i = 0; i < opts.iters; ++i) {
        const auto begin = Clock::now();
        for (int op = 0; op < opts.ops_per_graph; ++op) {
            append_fill(list,
                        byte_ptr(ptr, static_cast<size_t>(op) * opts.bytes),
                        opts.bytes,
                        static_cast<uint32_t>(i + op));
        }
        const auto submitted = Clock::now();
        CHECK_ZE(zeCommandListHostSynchronize(list, UINT64_MAX));
        const auto done = Clock::now();
        host_us.push_back(elapsed_us(begin, submitted));
        total_us.push_back(elapsed_us(begin, done));
    }

    CHECK_ZE(zeCommandListDestroy(list));
    CHECK_ZE(zeMemFree(l0.context, ptr));

    BenchResult result;
    result.backend = "level_zero";
    result.mode = "immediate_fill_graph";
    result.device = l0.device_name;
    result.warmup = opts.warmup;
    result.iters = opts.iters;
    result.bytes = opts.bytes;
    result.ops_per_iter = opts.ops_per_graph;
    result.host = summarize(std::move(host_us));
    result.total = summarize(std::move(total_us));
    return result;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options opts = parse_args(argc, argv);
        std::cout << "ur_l0_use_immediate_commandlists="
                  << env_value("UR_L0_USE_IMMEDIATE_COMMANDLISTS") << "\n";
        std::cout << "backend,mode,device,warmup,iters,bytes,"
                  << "ops_per_iter,"
                  << "host_avg_us,host_min_us,host_p50_us,host_p95_us,host_max_us,"
                  << "total_avg_us,total_min_us,total_p50_us,total_p95_us,total_max_us\n";

        print_result(run_sycl_memset(opts));
        print_result(run_sycl_single_task(opts));
        print_result(run_sycl_memset_graph(opts));
        print_result(run_sycl_kernel_chain_batch(opts));
        print_optional_result("command_graph_kernel_chain",
                              [&]() { return run_sycl_kernel_chain_command_graph(opts); });

        LevelZeroState l0;
        print_result(run_l0_regular_record_each(l0, opts));
        print_result(run_l0_regular_replay(l0, opts));
        print_result(run_l0_regular_replay_graph(l0, opts));
        print_result(run_l0_immediate(l0, opts));
        print_result(run_l0_immediate_graph(l0, opts));
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "AilaCommandSubmissionBench error: " << e.what() << "\n";
        return 1;
    }
}
