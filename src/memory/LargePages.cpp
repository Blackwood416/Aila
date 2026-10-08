#include "memory/LargePages.hpp"

#include <windows.h>
#include <limits>
#include <stdexcept>
#include <string>

#pragma comment(lib, "advapi32.lib")

namespace aila::memory {
namespace {

void enable_lock_pages() {
    static const bool enabled = [] {
        HANDLE token = nullptr;
        if (!OpenProcessToken(GetCurrentProcess(), TOKEN_ADJUST_PRIVILEGES | TOKEN_QUERY, &token))
            throw std::runtime_error("OpenProcessToken for large pages failed: " + std::to_string(GetLastError()));
        TOKEN_PRIVILEGES privileges{};
        privileges.PrivilegeCount = 1;
        DWORD error = ERROR_SUCCESS;
        if (!LookupPrivilegeValueW(nullptr, L"SeLockMemoryPrivilege", &privileges.Privileges[0].Luid)) {
            error = GetLastError();
        } else {
            privileges.Privileges[0].Attributes = SE_PRIVILEGE_ENABLED;
            SetLastError(ERROR_SUCCESS);
            const BOOL adjusted = AdjustTokenPrivileges(token, FALSE, &privileges, 0, nullptr, nullptr);
            error = GetLastError(); // TRUE can still mean ERROR_NOT_ALL_ASSIGNED.
            if (!adjusted && error == ERROR_SUCCESS) error = ERROR_PRIVILEGE_NOT_HELD;
        }
        CloseHandle(token);
        if (error != ERROR_SUCCESS)
            throw std::runtime_error("Enabling SeLockMemoryPrivilege failed: " + std::to_string(error));
        return true;
    }();
    (void)enabled;
}

} // namespace

size_t LargePageAllocationSize(size_t bytes) {
    const size_t page = GetLargePageMinimum();
    if (!page || !bytes || bytes > std::numeric_limits<size_t>::max() - (page - 1))
        throw std::invalid_argument("Invalid Windows large-page allocation size");
    return (bytes + page - 1) / page * page;
}

void* AllocateLargePages(size_t rounded_bytes) {
    if (LargePageAllocationSize(rounded_bytes) != rounded_bytes)
        throw std::invalid_argument("Large-page allocation is not page aligned");
    enable_lock_pages();
    void* ptr = VirtualAlloc(nullptr, rounded_bytes, MEM_RESERVE | MEM_COMMIT | MEM_LARGE_PAGES, PAGE_READWRITE);
    if (!ptr) {
        const DWORD error = GetLastError();
        throw std::runtime_error("MEM_LARGE_PAGES allocation failed: bytes=" + std::to_string(rounded_bytes) +
                                 " win32_error=" + std::to_string(error));
    }
    return ptr;
}

void FreeLargePages(void* ptr) noexcept {
    if (ptr) VirtualFree(ptr, 0, MEM_RELEASE);
}

} // namespace aila::memory
