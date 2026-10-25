#pragma once
#include <cstddef>

namespace aila::memory {

// Import a VirtualAlloc(MEM_LARGE_PAGES) range into the Level Zero driver.
//
// The Windows large-page arena is locked by the OS, but Level Zero does not know
// the range is host-resident, so every queue::memcpy out of it makes NEO run its
// per-call host-pointer registration path. Measured with
// build-refactor/AilaH2DBandwidthProbe.exe: 23.7 us per 2.4 MB enqueue before the
// import and 4.0 us after (6x). Prefill issues ~24.6k such expert copies per
// pass, so the enqueue cost is worth removing.
//
// Returns true when the driver accepted the range. A false return is not fatal:
// the range simply keeps the un-imported cost. The lookup is done through the
// Level Zero loader at run time (LoadLibrary + GetProcAddress), so callers do not
// need a ze_loader link dependency.
bool ImportHostRangeForLevelZero(void* ptr, size_t bytes);

// Release a range previously accepted by ImportHostRangeForLevelZero.
void ReleaseHostRangeFromLevelZero(void* ptr) noexcept;

// True when the driver exposes zexDriverImportExternalPointer and a driver was
// found. Diagnostic only.
bool LevelZeroHostImportAvailable();

} // namespace aila::memory
