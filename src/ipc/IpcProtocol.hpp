#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace aila::ipc {

inline constexpr uint32_t kProtocolVersion = 1;
inline constexpr uint32_t kPublicAbiVersion = 1;
inline constexpr uint32_t kMaxHeaderBytes = 1024 * 1024;
inline constexpr uint32_t kMaxAttachmentBytes = 512 * 1024 * 1024;
inline constexpr uint64_t kMaxStreamEventCount = 1'000'000;

struct FrameHeader {
    uint32_t protocol = kProtocolVersion;
    uint32_t abi = kPublicAbiVersion;
    uint64_t request_id = 0;
    std::string kind;
    std::string method;
    std::string payload_json = "{}";
};

struct Frame {
    FrameHeader header;
    std::vector<std::byte> attachment;
};

void write_u32_le(std::byte* destination, uint32_t value);
uint32_t read_u32_le(const std::byte* source);

bool encoded_header_size(const FrameHeader& header, size_t& size);
std::vector<std::byte> encode_frame(const Frame& frame);
bool decode_frame(const std::vector<std::byte>& bytes, Frame& frame, std::string& error);

} // namespace aila::ipc
