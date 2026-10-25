#include "memory/L0HostImport.hpp"

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace aila::memory {
namespace {

// Minimal Level Zero surface, resolved at run time. Only these three entry
// points are needed, so the engine keeps zero link-time dependency on the
// loader and works unchanged on machines without the extension.
using ZeResult = int32_t;
using ZeDriver = void*;
constexpr ZeResult kZeResultSuccess = 0;

using DriverGetFn = ZeResult (*)(uint32_t*, ZeDriver*);
using ExtensionAddressFn = ZeResult (*)(ZeDriver, const char*, void**);
using ImportFn = ZeResult (*)(ZeDriver, void*, size_t);
using ReleaseFn = ZeResult (*)(ZeDriver, void*);

struct Api {
    DriverGetFn driver_get = nullptr;
    ExtensionAddressFn extension_address = nullptr;
    ImportFn import = nullptr;
    ReleaseFn release = nullptr;
    ZeDriver driver = nullptr;
};

void* open_loader() {
#ifdef _WIN32
    // The SYCL runtime already loaded the loader in this process; reuse it so a
    // second copy cannot be initialised next to it.
    HMODULE module = GetModuleHandleW(L"ze_loader.dll");
    if (!module) module = LoadLibraryW(L"ze_loader.dll");
    return module;
#else
    return dlopen("libze_loader.so.1", RTLD_NOW | RTLD_LOCAL);
#endif
}

void* resolve(void* module, const char* name) {
    if (!module) return nullptr;
#ifdef _WIN32
    return reinterpret_cast<void*>(GetProcAddress(static_cast<HMODULE>(module), name));
#else
    return dlsym(module, name);
#endif
}

const Api& api() {
    static const Api resolved = [] {
        Api out{};
        void* module = open_loader();
        if (!module) {
            std::fprintf(stderr, "[L0HostImport] ze_loader not present; large-page imports disabled\n");
            return out;
        }
        out.driver_get = reinterpret_cast<DriverGetFn>(resolve(module, "zeDriverGet"));
        out.extension_address =
            reinterpret_cast<ExtensionAddressFn>(resolve(module, "zeDriverGetExtensionFunctionAddress"));
        if (!out.driver_get || !out.extension_address) {
            std::fprintf(stderr, "[L0HostImport] zeDriverGet/zeDriverGetExtensionFunctionAddress missing\n");
            return out;
        }
        uint32_t count = 0;
        if (out.driver_get(&count, nullptr) != kZeResultSuccess || count == 0) {
            std::fprintf(stderr, "[L0HostImport] no Level Zero driver enumerated\n");
            return out;
        }
        // A single-GPU engine: the first driver owns the Arc the queue runs on.
        out.driver = nullptr;
        if (out.driver_get(&count, &out.driver) != kZeResultSuccess) {
            std::fprintf(stderr, "[L0HostImport] zeDriverGet failed\n");
            return out;
        }
        void* raw = nullptr;
        if (out.extension_address(out.driver, "zexDriverImportExternalPointer", &raw) == kZeResultSuccess) {
            out.import = reinterpret_cast<ImportFn>(raw);
        }
        raw = nullptr;
        if (out.extension_address(out.driver, "zexDriverReleaseImportedPointer", &raw) == kZeResultSuccess) {
            out.release = reinterpret_cast<ReleaseFn>(raw);
        }
        if (!out.import || !out.release) {
            std::fprintf(stderr, "[L0HostImport] zexDriverImportExternalPointer unavailable\n");
            out.import = nullptr;
            out.release = nullptr;
        }
        return out;
    }();
    return resolved;
}

std::atomic<uint64_t> g_import_bytes{0};
std::atomic<uint64_t> g_import_failures{0};
std::atomic<uint64_t> g_import_calls{0};

// Opt-in while the engine-side behaviour is still being validated. Read with a
// plain cached getenv so this translation unit keeps no util-library dependency
// (the probe targets link it without the env module).
bool import_enabled() {
    static const bool enabled = [] {
        const char* raw = std::getenv("AILA_Q4E_L0_IMPORT_HOST");
        if (!raw || !*raw) return false;
        return !(raw[0] == '0' && raw[1] == '\0');
    }();
    return enabled;
}

} // namespace

bool LevelZeroHostImportAvailable() { return api().import != nullptr; }

bool ImportHostRangeForLevelZero(void* ptr, size_t bytes) {
    if (!ptr || !bytes || !import_enabled()) return false;
    const Api& a = api();
    if (!a.import) return false;
    const ZeResult status = a.import(a.driver, ptr, bytes);
    if (status != kZeResultSuccess) {
        const uint64_t failures = g_import_failures.fetch_add(1, std::memory_order_relaxed) + 1;
        if (failures <= 3)
            std::fprintf(stderr, "[L0HostImport] import of %zu bytes rejected (0x%x)\n", bytes,
                         static_cast<unsigned>(status));
        return false;
    }
    g_import_bytes.fetch_add(bytes, std::memory_order_relaxed);
    const uint64_t calls = g_import_calls.fetch_add(1, std::memory_order_relaxed) + 1;
    // One line proves the path is live without flooding startup: the host arena
    // allocates one range per 256 MB (127 ranges for the 32 GiB default).
    if (calls == 1)
        std::fprintf(stderr, "[L0HostImport] first range imported (%zu bytes); later ranges log only on failure\n",
                     bytes);
    return true;
}

void ReleaseHostRangeFromLevelZero(void* ptr) noexcept {
    if (!ptr) return;
    const Api& a = api();
    if (!a.release) return;
    a.release(a.driver, ptr);
}

} // namespace aila::memory
