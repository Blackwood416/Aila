#include "ipc/Win32Pipe.hpp"

#include <algorithm>
#include <array>
#include <cstring>
#include <exception>
#include <limits>
#include <new>
#include <utility>
#include <vector>

namespace aila::ipc {
namespace {

constexpr size_t kFramePrefixBytes = 8;

bool is_end_of_pipe_error(DWORD error_code) {
    return error_code == ERROR_BROKEN_PIPE || error_code == ERROR_HANDLE_EOF;
}

std::string format_system_message(DWORD error_code) {
    wchar_t* wide_message = nullptr;
    const DWORD wide_length = FormatMessageW(
        FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM |
            FORMAT_MESSAGE_IGNORE_INSERTS,
        nullptr,
        error_code,
        0,
        reinterpret_cast<wchar_t*>(&wide_message),
        0,
        nullptr);
    if (wide_length == 0 || wide_message == nullptr) {
        return {};
    }

    size_t trimmed_length = wide_length;
    while (trimmed_length > 0) {
        const wchar_t last = wide_message[trimmed_length - 1];
        if (last != L'\r' && last != L'\n' && last != L' ' && last != L'\t') {
            break;
        }
        --trimmed_length;
    }

    std::string message;
    if (trimmed_length <= static_cast<size_t>((std::numeric_limits<int>::max)())) {
        const int utf8_length = WideCharToMultiByte(
            CP_UTF8,
            0,
            wide_message,
            static_cast<int>(trimmed_length),
            nullptr,
            0,
            nullptr,
            nullptr);
        if (utf8_length > 0) {
            message.resize(static_cast<size_t>(utf8_length));
            WideCharToMultiByte(
                CP_UTF8,
                0,
                wide_message,
                static_cast<int>(trimmed_length),
                message.data(),
                utf8_length,
                nullptr,
                nullptr);
        }
    }

    LocalFree(wide_message);
    return message;
}

std::string format_win32_error(const char* operation, DWORD error_code) {
    std::string error = operation;
    error += is_end_of_pipe_error(error_code)
        ? " failed: unexpected end of pipe (Win32 error "
        : " failed with Win32 error ";
    error += std::to_string(error_code);
    if (is_end_of_pipe_error(error_code)) {
        error += ')';
    }

    const std::string system_message = format_system_message(error_code);
    if (!system_message.empty()) {
        error += ": ";
        error += system_message;
    }
    return error;
}

bool validate_lengths(
    uint32_t header_bytes,
    uint32_t attachment_bytes,
    size_t& frame_bytes,
    std::string& error) {
    if (header_bytes > kMaxHeaderBytes) {
        error = "declared header length exceeds kMaxHeaderBytes";
        return false;
    }
    if (attachment_bytes > kMaxAttachmentBytes) {
        error = "declared attachment length exceeds kMaxAttachmentBytes";
        return false;
    }

    frame_bytes = kFramePrefixBytes;
    if (header_bytes > (std::numeric_limits<size_t>::max)() - frame_bytes) {
        error = "declared frame length overflows addressable size";
        return false;
    }
    frame_bytes += header_bytes;
    if (attachment_bytes > (std::numeric_limits<size_t>::max)() - frame_bytes) {
        error = "declared frame length overflows addressable size";
        return false;
    }
    frame_bytes += attachment_bytes;
    return true;
}

} // namespace

bool write_all(HANDLE handle, const void* data, size_t size, std::string& error) {
    error.clear();
    if (size != 0 && data == nullptr) {
        error = "write_all received a null buffer";
        return false;
    }

    const auto* bytes = static_cast<const std::byte*>(data);
    size_t offset = 0;
    while (offset < size) {
        const size_t remaining = size - offset;
        const DWORD chunk_bytes = static_cast<DWORD>((std::min)(
            remaining,
            static_cast<size_t>((std::numeric_limits<DWORD>::max)())));
        DWORD written = 0;
        if (WriteFile(handle, bytes + offset, chunk_bytes, &written, nullptr) == FALSE) {
            const DWORD error_code = GetLastError();
            error = format_win32_error("WriteFile", error_code);
            return false;
        }
        if (written == 0) {
            error = "WriteFile succeeded without writing data";
            return false;
        }
        if (written > chunk_bytes) {
            error = "WriteFile reported more bytes than requested";
            return false;
        }
        offset += written;
    }
    return true;
}

bool read_exact(HANDLE handle, void* data, size_t size, std::string& error) {
    error.clear();
    if (size != 0 && data == nullptr) {
        error = "read_exact received a null buffer";
        return false;
    }

    auto* bytes = static_cast<std::byte*>(data);
    size_t offset = 0;
    while (offset < size) {
        const size_t remaining = size - offset;
        const DWORD chunk_bytes = static_cast<DWORD>((std::min)(
            remaining,
            static_cast<size_t>((std::numeric_limits<DWORD>::max)())));
        DWORD read = 0;
        if (ReadFile(handle, bytes + offset, chunk_bytes, &read, nullptr) == FALSE) {
            const DWORD error_code = GetLastError();
            error = format_win32_error("ReadFile", error_code);
            return false;
        }
        if (read == 0) {
            error = "ReadFile succeeded without reading data: unexpected end of pipe";
            return false;
        }
        if (read > chunk_bytes) {
            error = "ReadFile reported more bytes than requested";
            return false;
        }
        offset += read;
    }
    return true;
}

bool write_frame(HANDLE handle, const Frame& frame, std::string& error) {
    error.clear();
    const std::vector<std::byte> encoded = encode_frame(frame);
    if (encoded.empty()) {
        error = "failed to encode frame for pipe write";
        return false;
    }
    return write_all(handle, encoded.data(), encoded.size(), error);
}

bool read_frame(HANDLE handle, Frame& frame, std::string& error) {
    error.clear();

    std::array<std::byte, kFramePrefixBytes> prefix{};
    if (!read_exact(handle, prefix.data(), prefix.size(), error)) {
        return false;
    }

    const uint32_t header_bytes = read_u32_le(prefix.data());
    const uint32_t attachment_bytes = read_u32_le(prefix.data() + 4);
    size_t frame_bytes = 0;
    if (!validate_lengths(header_bytes, attachment_bytes, frame_bytes, error)) {
        return false;
    }

    try {
        std::vector<std::byte> encoded(frame_bytes);
        std::memcpy(encoded.data(), prefix.data(), prefix.size());
        if (frame_bytes > prefix.size() &&
            !read_exact(
                handle,
                encoded.data() + prefix.size(),
                frame_bytes - prefix.size(),
                error)) {
            return false;
        }

        Frame decoded;
        if (!decode_frame(encoded, decoded, error)) {
            return false;
        }
        frame = std::move(decoded);
        return true;
    } catch (const std::bad_alloc&) {
        error = "failed to allocate bounded frame buffer";
        return false;
    } catch (const std::exception& exception) {
        error = std::string("failed to read frame: ") + exception.what();
        return false;
    } catch (...) {
        error = "failed to read frame: unknown error";
        return false;
    }
}

} // namespace aila::ipc
