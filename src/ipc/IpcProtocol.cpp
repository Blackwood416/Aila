#include "ipc/IpcProtocol.hpp"

#include "simdjson.h"

#include <exception>
#include <limits>
#include <string_view>
#include <utility>

namespace aila::ipc {
namespace {

constexpr size_t kFramePrefixBytes = 8;

bool append_bounded(std::string& output, std::string_view value) {
    if (value.size() > kMaxHeaderBytes - output.size()) {
        return false;
    }
    output.append(value);
    return true;
}

bool append_json_string(std::string& output, std::string_view value) {
    if (!append_bounded(output, "\"")) {
        return false;
    }

    constexpr char hex[] = "0123456789abcdef";
    for (const unsigned char character : value) {
        switch (character) {
            case '\"':
                if (!append_bounded(output, "\\\"")) return false;
                break;
            case '\\':
                if (!append_bounded(output, "\\\\")) return false;
                break;
            case '\b':
                if (!append_bounded(output, "\\b")) return false;
                break;
            case '\f':
                if (!append_bounded(output, "\\f")) return false;
                break;
            case '\n':
                if (!append_bounded(output, "\\n")) return false;
                break;
            case '\r':
                if (!append_bounded(output, "\\r")) return false;
                break;
            case '\t':
                if (!append_bounded(output, "\\t")) return false;
                break;
            default:
                if (character < 0x20) {
                    char escaped[] = {'\\', 'u', '0', '0', hex[character >> 4], hex[character & 0x0f]};
                    if (!append_bounded(output, std::string_view(escaped, sizeof(escaped)))) return false;
                } else {
                    const char byte = static_cast<char>(character);
                    if (!append_bounded(output, std::string_view(&byte, 1))) return false;
                }
                break;
        }
    }

    return append_bounded(output, "\"");
}

bool read_uint64_field(
    simdjson::dom::object object,
    const char* name,
    uint64_t& output,
    std::string& error) {
    simdjson::dom::element element;
    if (object.at_key(name).get(element) != simdjson::SUCCESS ||
        element.get_uint64().get(output) != simdjson::SUCCESS) {
        error = std::string("header field '") + name + "' must be an unsigned integer";
        return false;
    }
    return true;
}

bool read_string_field(
    simdjson::dom::object object,
    const char* name,
    std::string& output,
    std::string& error) {
    simdjson::dom::element element;
    std::string_view value;
    if (object.at_key(name).get(element) != simdjson::SUCCESS ||
        element.get_string().get(value) != simdjson::SUCCESS) {
        error = std::string("header field '") + name + "' must be a string";
        return false;
    }
    output.assign(value);
    return true;
}

bool serialize_header_json(const FrameHeader& header, std::string& header_json) {
    header_json.clear();
    if (header.payload_json.size() > kMaxHeaderBytes ||
        header.kind.size() > kMaxHeaderBytes ||
        header.method.size() > kMaxHeaderBytes) {
        return false;
    }

    simdjson::dom::parser payload_parser;
    simdjson::dom::element payload;
    if (payload_parser.parse(header.payload_json).get(payload) != simdjson::SUCCESS) {
        return false;
    }
    const std::string normalized_payload = simdjson::minify(payload);

    if (!(append_bounded(header_json, "{\"protocol\":") &&
        append_bounded(header_json, std::to_string(header.protocol)) &&
        append_bounded(header_json, ",\"abi\":") &&
        append_bounded(header_json, std::to_string(header.abi)) &&
        append_bounded(header_json, ",\"requestId\":") &&
        append_bounded(header_json, std::to_string(header.request_id)) &&
        append_bounded(header_json, ",\"kind\":") &&
        append_json_string(header_json, header.kind) &&
        append_bounded(header_json, ",\"method\":") &&
        append_json_string(header_json, header.method) &&
        append_bounded(header_json, ",\"payload\":") &&
        append_bounded(header_json, normalized_payload) &&
        append_bounded(header_json, "}"))) {
        return false;
    }
    simdjson::dom::parser header_parser;
    simdjson::dom::element validated_header;
    return header_parser.parse(header_json).get(validated_header) == simdjson::SUCCESS;
}

} // namespace

void write_u32_le(std::byte* destination, uint32_t value) {
    destination[0] = static_cast<std::byte>(value & 0xffu);
    destination[1] = static_cast<std::byte>((value >> 8) & 0xffu);
    destination[2] = static_cast<std::byte>((value >> 16) & 0xffu);
    destination[3] = static_cast<std::byte>((value >> 24) & 0xffu);
}

uint32_t read_u32_le(const std::byte* source) {
    return static_cast<uint32_t>(source[0]) |
        (static_cast<uint32_t>(source[1]) << 8) |
        (static_cast<uint32_t>(source[2]) << 16) |
        (static_cast<uint32_t>(source[3]) << 24);
}

bool encoded_header_size(const FrameHeader& header, size_t& size) {
    size = 0;
    try {
        std::string header_json;
        if (!serialize_header_json(header, header_json)) return false;
        size = header_json.size();
        return true;
    } catch (...) {
        size = 0;
        return false;
    }
}

std::vector<std::byte> encode_frame(const Frame& frame) {
    try {
        if (frame.attachment.size() > kMaxAttachmentBytes) {
            return {};
        }

        std::string header_json;
        if (!serialize_header_json(frame.header, header_json)) {
            return {};
        }

        if (header_json.size() > std::numeric_limits<uint32_t>::max() ||
            frame.attachment.size() > std::numeric_limits<uint32_t>::max()) {
            return {};
        }
        if (header_json.size() > std::numeric_limits<size_t>::max() - kFramePrefixBytes ||
            frame.attachment.size() >
                std::numeric_limits<size_t>::max() - kFramePrefixBytes - header_json.size()) {
            return {};
        }

        std::vector<std::byte> encoded(
            kFramePrefixBytes + header_json.size() + frame.attachment.size());
        write_u32_le(encoded.data(), static_cast<uint32_t>(header_json.size()));
        write_u32_le(encoded.data() + 4, static_cast<uint32_t>(frame.attachment.size()));

        size_t offset = kFramePrefixBytes;
        for (const char character : header_json) {
            encoded[offset++] = static_cast<std::byte>(character);
        }
        for (const std::byte byte : frame.attachment) {
            encoded[offset++] = byte;
        }
        return encoded;
    } catch (...) {
        return {};
    }
}

bool decode_frame(const std::vector<std::byte>& bytes, Frame& frame, std::string& error) {
    try {
        error.clear();
        if (bytes.size() < kFramePrefixBytes) {
            error = "frame is shorter than the 8-byte length prefix";
            return false;
        }

        const uint32_t header_bytes = read_u32_le(bytes.data());
        const uint32_t attachment_bytes = read_u32_le(bytes.data() + 4);
        if (header_bytes > kMaxHeaderBytes) {
            error = "declared header length exceeds kMaxHeaderBytes";
            return false;
        }
        if (attachment_bytes > kMaxAttachmentBytes) {
            error = "declared attachment length exceeds kMaxAttachmentBytes";
            return false;
        }

        size_t expected_size = kFramePrefixBytes;
        if (header_bytes > std::numeric_limits<size_t>::max() - expected_size) {
            error = "declared frame length overflows addressable size";
            return false;
        }
        expected_size += header_bytes;
        if (attachment_bytes > std::numeric_limits<size_t>::max() - expected_size) {
            error = "declared frame length overflows addressable size";
            return false;
        }
        expected_size += attachment_bytes;
        if (bytes.size() != expected_size) {
            error = bytes.size() < expected_size
                ? "frame is truncated relative to its declared lengths"
                : "frame contains trailing bytes beyond its declared lengths";
            return false;
        }

        const char* header_data = reinterpret_cast<const char*>(bytes.data() + kFramePrefixBytes);
        const std::string header_json(header_data, header_bytes);
        simdjson::dom::parser parser;
        simdjson::dom::element root;
        const simdjson::error_code parse_error = parser.parse(header_json).get(root);
        if (parse_error != simdjson::SUCCESS) {
            error = std::string("invalid header JSON: ") + simdjson::error_message(parse_error);
            return false;
        }

        simdjson::dom::object object;
        if (root.get_object().get(object) != simdjson::SUCCESS) {
            error = "header JSON must be an object";
            return false;
        }

        Frame decoded;
        uint64_t protocol = 0;
        uint64_t abi = 0;
        if (!read_uint64_field(object, "protocol", protocol, error) ||
            protocol > std::numeric_limits<uint32_t>::max()) {
            if (error.empty()) error = "header field 'protocol' is out of range";
            return false;
        }
        if (!read_uint64_field(object, "abi", abi, error) ||
            abi > std::numeric_limits<uint32_t>::max()) {
            if (error.empty()) error = "header field 'abi' is out of range";
            return false;
        }
        decoded.header.protocol = static_cast<uint32_t>(protocol);
        decoded.header.abi = static_cast<uint32_t>(abi);
        if (!read_uint64_field(object, "requestId", decoded.header.request_id, error) ||
            !read_string_field(object, "kind", decoded.header.kind, error) ||
            !read_string_field(object, "method", decoded.header.method, error)) {
            return false;
        }

        simdjson::dom::element payload;
        if (object.at_key("payload").get(payload) != simdjson::SUCCESS) {
            error = "header field 'payload' is required";
            return false;
        }
        decoded.header.payload_json = simdjson::minify(payload);

        const auto attachment_begin = bytes.begin() +
            static_cast<std::vector<std::byte>::difference_type>(kFramePrefixBytes + header_bytes);
        decoded.attachment.assign(attachment_begin, bytes.end());
        frame = std::move(decoded);
        return true;
    } catch (const std::exception& exception) {
        error = std::string("failed to decode frame: ") + exception.what();
        return false;
    } catch (...) {
        error = "failed to decode frame: unknown error";
        return false;
    }
}

} // namespace aila::ipc
