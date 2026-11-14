#pragma once

#include <cstdint>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <Windows.h>
#include <dxgi1_4.h>
#endif

namespace aila {

struct DxgiLocalVramInfo {
    bool valid = false;
    uint64_t dedicatedVideoMemoryBytes = 0;
    uint64_t budgetBytes = 0;
    uint64_t currentUsageBytes = 0;
    uint64_t availableBytes = 0;
};

// Queries Windows DXGI local VRAM budget & current process-wide usage for the
// primary Intel/discrete hardware GPU adapter without adding a static dxgi.lib
// dependency. Returns {false, ...} on non-Windows or if DXGI is unavailable.
inline DxgiLocalVramInfo QueryDxgiLocalVram() noexcept {
    DxgiLocalVramInfo result;
#ifdef _WIN32
    using CreateDXGIFactory1Fn = HRESULT(WINAPI*)(REFIID, void**);
    HMODULE dxgiMod = LoadLibraryA("dxgi.dll");
    if (!dxgiMod) return result;

    auto createFactory = reinterpret_cast<CreateDXGIFactory1Fn>(
        GetProcAddress(dxgiMod, "CreateDXGIFactory1"));
    if (!createFactory) {
        FreeLibrary(dxgiMod);
        return result;
    }

    IDXGIFactory4* factory = nullptr;
    if (FAILED(createFactory(__uuidof(IDXGIFactory4), reinterpret_cast<void**>(&factory))) || !factory) {
        FreeLibrary(dxgiMod);
        return result;
    }

    IDXGIAdapter3* bestAdapter = nullptr;
    SIZE_T bestDedicated = 0;
    bool bestIsIntel = false;

    for (UINT idx = 0;; ++idx) {
        IDXGIAdapter1* adapter1 = nullptr;
        if (factory->EnumAdapters1(idx, &adapter1) == DXGI_ERROR_NOT_FOUND) break;
        if (!adapter1) continue;

        DXGI_ADAPTER_DESC1 desc{};
        if (SUCCEEDED(adapter1->GetDesc1(&desc)) &&
            (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) == 0 &&
            desc.DedicatedVideoMemory > 0) {
            IDXGIAdapter3* adapter3 = nullptr;
            if (SUCCEEDED(adapter1->QueryInterface(__uuidof(IDXGIAdapter3), reinterpret_cast<void**>(&adapter3))) &&
                adapter3) {
                const bool isIntel = (desc.VendorId == 0x8086);
                const bool better =
                    !bestAdapter ||
                    (isIntel && !bestIsIntel) ||
                    (isIntel == bestIsIntel && desc.DedicatedVideoMemory > bestDedicated);
                if (better) {
                    if (bestAdapter) bestAdapter->Release();
                    bestAdapter = adapter3;
                    bestDedicated = desc.DedicatedVideoMemory;
                    bestIsIntel = isIntel;
                } else {
                    adapter3->Release();
                }
            }
        }
        adapter1->Release();
    }

    if (bestAdapter) {
        DXGI_QUERY_VIDEO_MEMORY_INFO localInfo{};
        if (SUCCEEDED(bestAdapter->QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &localInfo))) {
            result.valid = true;
            result.dedicatedVideoMemoryBytes = static_cast<uint64_t>(bestDedicated);
            result.budgetBytes = static_cast<uint64_t>(localInfo.Budget);
            result.currentUsageBytes = static_cast<uint64_t>(localInfo.CurrentUsage);
            result.availableBytes = result.budgetBytes > result.currentUsageBytes
                ? (result.budgetBytes - result.currentUsageBytes)
                : 0;
        }
        bestAdapter->Release();
    }

    factory->Release();
    FreeLibrary(dxgiMod);
#endif
    return result;
}

} // namespace aila
