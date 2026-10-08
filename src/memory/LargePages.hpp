#pragma once

#include <cstddef>

namespace aila::memory {

// Windows physical residency, not SYCL USM. CPU access and explicit copies only.
// Allocation failure is reported; it must never silently become pageable RAM.
size_t LargePageAllocationSize(size_t bytes);
void* AllocateLargePages(size_t rounded_bytes);
void FreeLargePages(void* ptr) noexcept;

} // namespace aila::memory
