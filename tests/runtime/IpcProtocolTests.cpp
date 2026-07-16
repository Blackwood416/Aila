#include "ipc/IpcProtocol.hpp"

#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

namespace {

using aila::ipc::Frame;

[[noreturn]] void fail(const char* test_name, const std::string& message) {
    std::cerr << "FAILED: " << test_name << ": " << message << '\n';
    std::exit(1);
}

void expect(bool condition, const char* test_name, const std::string& message) {
    if (!condition) {
        fail(test_name, message);
    }
}

std::vector<std::byte> make_raw_frame(
    const std::string& header_json,
    uint32_t declared_attachment_bytes = 0) {
    std::vector<std::byte> bytes(8 + header_json.size());
    aila::ipc::write_u32_le(bytes.data(), static_cast<uint32_t>(header_json.size()));
    aila::ipc::write_u32_le(bytes.data() + 4, declared_attachment_bytes);
    for (size_t index = 0; index < header_json.size(); ++index) {
        bytes[8 + index] = static_cast<std::byte>(header_json[index]);
    }
    return bytes;
}

void test_little_endian_helpers() {
    constexpr const char* name = "little-endian helpers";
    std::byte bytes[4]{};
    aila::ipc::write_u32_le(bytes, 0x78563412u);

    expect(bytes[0] == std::byte{0x12}, name, "least-significant byte was not written first");
    expect(bytes[3] == std::byte{0x78}, name, "most-significant byte was not written last");
    expect(aila::ipc::read_u32_le(bytes) == 0x78563412u, name, "read did not invert write");
}

void test_round_trip() {
    constexpr const char* name = "frame round trip";
    Frame input;
    input.header.protocol = aila::ipc::kProtocolVersion;
    input.header.abi = aila::ipc::kPublicAbiVersion;
    input.header.request_id = 42;
    input.header.kind = "request";
    input.header.method = "engine.init";
    input.header.payload_json = R"({"model":"C:\\models\\qwen"})";
    input.attachment = {std::byte{1}, std::byte{2}, std::byte{3}};

    const std::vector<std::byte> encoded = aila::ipc::encode_frame(input);
    expect(!encoded.empty(), name, "encoding returned an empty frame");

    Frame output;
    std::string error;
    expect(aila::ipc::decode_frame(encoded, output, error), name, error);
    expect(output.header.request_id == 42, name, "request id changed");
    expect(output.header.method == "engine.init", name, "method changed");
    expect(output.header.payload_json == input.header.payload_json, name, "payload JSON changed");
    expect(output.attachment == input.attachment, name, "attachment changed");
}

void test_truncated_frame_is_rejected() {
    constexpr const char* name = "truncated frame";
    Frame input;
    input.header.kind = "request";
    input.header.method = "engine.init";
    input.attachment = {std::byte{1}, std::byte{2}, std::byte{3}};
    std::vector<std::byte> encoded = aila::ipc::encode_frame(input);
    expect(!encoded.empty(), name, "fixture encoding failed");
    encoded.pop_back();

    Frame output;
    std::string error;
    expect(!aila::ipc::decode_frame(encoded, output, error), name, "truncated bytes were accepted");
    expect(!error.empty(), name, "rejection did not explain the error");
}

void test_oversized_header_is_rejected_from_prefix() {
    constexpr const char* name = "oversized header prefix";
    std::vector<std::byte> bytes(8);
    aila::ipc::write_u32_le(bytes.data(), aila::ipc::kMaxHeaderBytes + 1);
    aila::ipc::write_u32_le(bytes.data() + 4, 0);

    Frame output;
    std::string error;
    expect(!aila::ipc::decode_frame(bytes, output, error), name, "oversized header was accepted");
    expect(error.find("header") != std::string::npos, name, "error did not identify the header bound");
}

void test_malformed_json_is_rejected() {
    constexpr const char* name = "malformed JSON";
    const std::vector<std::byte> bytes = make_raw_frame("{]");

    Frame output;
    std::string error;
    expect(!aila::ipc::decode_frame(bytes, output, error), name, "malformed JSON was accepted");
    expect(error.find("JSON") != std::string::npos, name, "error did not identify malformed JSON");
}

void test_oversized_attachment_is_rejected_from_prefix() {
    constexpr const char* name = "oversized attachment prefix";
    const std::vector<std::byte> bytes =
        make_raw_frame("{}", aila::ipc::kMaxAttachmentBytes + 1);

    Frame output;
    std::string error;
    expect(!aila::ipc::decode_frame(bytes, output, error), name, "oversized attachment was accepted");
    expect(error.find("attachment") != std::string::npos, name, "error did not identify attachment bound");
}

} // namespace

int main() {
    test_little_endian_helpers();
    test_round_trip();
    test_truncated_frame_is_rejected();
    test_oversized_header_is_rejected_from_prefix();
    test_malformed_json_is_rejected();
    test_oversized_attachment_is_rejected_from_prefix();
    std::cout << "AilaIpcProtocolTests passed\n";
    return 0;
}
