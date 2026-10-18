// Enumerate Level Zero command queue groups without a SYCL queue.
// This answers whether the adapter exposes more than one physical compute
// group that could back an independent cold-expert lane.
#include <level_zero/ze_api.h>

#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace {

const char* result_name(ze_result_t r) {
    switch (r) {
    case ZE_RESULT_SUCCESS: return "SUCCESS";
    case ZE_RESULT_ERROR_UNINITIALIZED: return "UNINITIALIZED";
    case ZE_RESULT_ERROR_DEVICE_LOST: return "DEVICE_LOST";
    case ZE_RESULT_ERROR_INVALID_ARGUMENT: return "INVALID_ARGUMENT";
    case ZE_RESULT_ERROR_UNSUPPORTED_FEATURE: return "UNSUPPORTED_FEATURE";
    case ZE_RESULT_ERROR_DEPENDENCY_UNAVAILABLE: return "DEPENDENCY_UNAVAILABLE";
    case ZE_RESULT_ERROR_INSUFFICIENT_PERMISSIONS: return "INSUFFICIENT_PERMISSIONS";
    case ZE_RESULT_ERROR_NOT_AVAILABLE: return "NOT_AVAILABLE";
    default: return "OTHER";
    }
}

std::string group_flags(ze_command_queue_group_property_flags_t flags) {
    std::string s;
    if (flags & ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COMPUTE) s += "COMPUTE|";
    if (flags & ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COPY) s += "COPY|";
    if (flags & ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COOPERATIVE_KERNELS) s += "COOPERATIVE|";
    if (flags & ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_METRICS) s += "METRICS|";
    if (!s.empty()) s.pop_back();
    return s.empty() ? "NONE" : s;
}

void check(ze_result_t r, const char* what) {
    if (r != ZE_RESULT_SUCCESS)
        throw std::runtime_error(std::string(what) + ": " + result_name(r));
}

} // namespace

int main() {
    try {
        check(zeInit(ZE_INIT_FLAG_GPU_ONLY), "zeInit");

        uint32_t driver_count = 0;
        check(zeDriverGet(&driver_count, nullptr), "zeDriverGet count");
        std::vector<ze_driver_handle_t> drivers(driver_count);
        check(zeDriverGet(&driver_count, drivers.data()), "zeDriverGet");
        std::cout << "[L0] drivers=" << driver_count << '\n';

        for (uint32_t di = 0; di < driver_count; ++di) {
            uint32_t device_count = 0;
            check(zeDeviceGet(drivers[di], &device_count, nullptr), "zeDeviceGet count");
            std::vector<ze_device_handle_t> devices(device_count);
            check(zeDeviceGet(drivers[di], &device_count, devices.data()), "zeDeviceGet");
            std::cout << "[L0] driver=" << di << " devices=" << device_count << '\n';

            for (uint32_t ui = 0; ui < device_count; ++ui) {
                ze_device_properties_t dp{};
                dp.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;
                check(zeDeviceGetProperties(devices[ui], &dp), "zeDeviceGetProperties");
                std::cout << std::hex << std::showbase
                          << "[L0] device=" << ui
                          << " name=\"" << dp.name << "\""
                          << " type=" << dp.type
                          << " vendor=0x" << dp.vendorId
                          << " deviceId=0x" << dp.deviceId
                          << std::dec << std::noshowbase
                          << " slices=" << dp.numSlices
                          << " eusPerSubslice=" << dp.numEUsPerSubslice
                          << '\n';

                uint32_t subdevice_count = 0;
                const auto sub_result = zeDeviceGetSubDevices(
                    devices[ui], &subdevice_count, nullptr);
                std::cout << "[L0] subdevices=";
                if (sub_result == ZE_RESULT_SUCCESS)
                    std::cout << subdevice_count;
                else
                    std::cout << "UNSUPPORTED(" << result_name(sub_result) << ")";
                std::cout << '\n';

                uint32_t group_count = 0;
                auto r = zeDeviceGetCommandQueueGroupProperties(
                    devices[ui], &group_count, nullptr);
                if (r != ZE_RESULT_SUCCESS) {
                    std::cout << "[L0] groups=UNSUPPORTED(" << result_name(r) << ")\n";
                    continue;
                }
                std::vector<ze_command_queue_group_properties_t> groups(group_count);
                for (auto& g : groups) {
                    g.stype = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_GROUP_PROPERTIES;
                }
                check(zeDeviceGetCommandQueueGroupProperties(
                          devices[ui], &group_count, groups.data()),
                      "zeDeviceGetCommandQueueGroupProperties");

                int compute_groups = 0;
                int copy_groups = 0;
                std::cout << "[L0] groups=" << group_count << '\n';
                for (uint32_t gi = 0; gi < group_count; ++gi) {
                    const auto& g = groups[gi];
                    const bool compute =
                        (g.flags & ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COMPUTE) != 0;
                    const bool copy =
                        (g.flags & ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COPY) != 0;
                    compute_groups += compute ? 1 : 0;
                    copy_groups += copy ? 1 : 0;
                    std::cout << "[L0 GROUP] ordinal=" << gi
                              << " flags=" << group_flags(g.flags)
                              << " numQueues=" << g.numQueues
                              << " maxFillPattern=" << g.maxMemoryFillPatternSize
                              << '\n';
                }
                std::cout << "[L0 SUMMARY] compute_groups=" << compute_groups
                          << " copy_groups=" << copy_groups
                          << " independent_compute_lane="
                          << (compute_groups > 1 ? "POSSIBLE" : "NO")
                          << '\n';
            }
        }
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "[L0 ERROR] " << e.what() << '\n';
        return 1;
    }
}
