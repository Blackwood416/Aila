#pragma once

#include <cstddef>
#include <stdexcept>
#include <string>
#include <sycl/sycl.hpp>

#if defined(AILA_HAS_LEVEL_ZERO)
#include <level_zero/ze_api.h>
#include <sycl/ext/oneapi/backend/level_zero.hpp>
#endif

namespace aila::memory {

struct UncachedHostAllocation {
    void* ptr = nullptr;
    void* native_context = nullptr;
};

// Level Zero host USM that device code may poll while the host updates it.
// The opaque context is retained so release cannot accidentally use sycl::free.
inline UncachedHostAllocation AllocateUncachedHost(
    const sycl::context& context, size_t bytes) {
    if (!bytes) {
        throw std::invalid_argument(
            "Uncached host allocation must have positive size");
    }
#if defined(AILA_HAS_LEVEL_ZERO)
    if (context.get_backend() != sycl::backend::ext_oneapi_level_zero) {
        throw std::runtime_error(
            "Uncached host allocation requires the Level Zero SYCL backend");
    }
    const ze_context_handle_t ze_context =
        sycl::get_native<sycl::backend::ext_oneapi_level_zero>(context);
    if (!ze_context) {
        throw std::runtime_error(
            "Level Zero returned a null context for uncached host allocation");
    }

    ze_host_mem_alloc_desc_t descriptor{
        ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC, nullptr,
        ZE_HOST_MEM_ALLOC_FLAG_BIAS_UNCACHED};
    void* ptr = nullptr;
    const ze_result_t result =
        zeMemAllocHost(ze_context, &descriptor, bytes, 64, &ptr);
    if (result != ZE_RESULT_SUCCESS || !ptr) {
        if (ptr) {
            (void)zeMemFree(ze_context, ptr);
        }
        throw std::runtime_error(
            "Level Zero uncached host allocation failed (ze_result_t=" +
            std::to_string(static_cast<int>(result)) + ")");
    }
    return {ptr, reinterpret_cast<void*>(ze_context)};
#else
    (void)context;
    throw std::runtime_error(
        "Uncached host allocation requires a build with Level Zero SDK support");
#endif
}

inline void FreeUncachedHost(void* native_context, void* ptr) noexcept {
    if (!ptr) {
        return;
    }
#if defined(AILA_HAS_LEVEL_ZERO)
    if (native_context) {
        (void)zeMemFree(
            reinterpret_cast<ze_context_handle_t>(native_context), ptr);
    }
#else
    (void)native_context;
#endif
}

} // namespace aila::memory
