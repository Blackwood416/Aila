#pragma once

#include "ipc/IpcProtocol.hpp"

#include <cstddef>
#include <string>

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#define AILA_IPC_UNDEFINE_WIN32_LEAN_AND_MEAN
#endif

#ifndef NOMINMAX
#define NOMINMAX
#define AILA_IPC_UNDEFINE_NOMINMAX
#endif

#include <Windows.h>

#ifdef AILA_IPC_UNDEFINE_NOMINMAX
#undef NOMINMAX
#undef AILA_IPC_UNDEFINE_NOMINMAX
#endif

#ifdef AILA_IPC_UNDEFINE_WIN32_LEAN_AND_MEAN
#undef WIN32_LEAN_AND_MEAN
#undef AILA_IPC_UNDEFINE_WIN32_LEAN_AND_MEAN
#endif

namespace aila::ipc {

bool write_frame(HANDLE handle, const Frame& frame, std::string& error);
bool read_frame(HANDLE handle, Frame& frame, std::string& error);
bool write_all(HANDLE handle, const void* data, size_t size, std::string& error);
bool read_exact(HANDLE handle, void* data, size_t size, std::string& error);

} // namespace aila::ipc
