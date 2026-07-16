#include "ipc/IpcProtocol.hpp"
#include "ipc/Win32Pipe.hpp"

#include <algorithm>
#include <array>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace {

using aila::ipc::Frame;

[[noreturn]] void fail(const char* test_name, const std::string& message) {
    throw std::runtime_error(std::string("FAILED: ") + test_name + ": " + message);
}

void expect(bool condition, const char* test_name, const std::string& message) {
    if (!condition) {
        fail(test_name, message);
    }
}

class UniqueHandle {
public:
    UniqueHandle() = default;
    explicit UniqueHandle(HANDLE handle) : handle_(handle) {}

    UniqueHandle(const UniqueHandle&) = delete;
    UniqueHandle& operator=(const UniqueHandle&) = delete;

    UniqueHandle(UniqueHandle&& other) noexcept : handle_(other.handle_) {
        other.handle_ = nullptr;
    }

    UniqueHandle& operator=(UniqueHandle&& other) noexcept {
        if (this != &other) {
            reset();
            handle_ = other.handle_;
            other.handle_ = nullptr;
        }
        return *this;
    }

    ~UniqueHandle() { reset(); }

    HANDLE get() const { return handle_; }

private:
    void reset() {
        if (handle_ != nullptr && handle_ != INVALID_HANDLE_VALUE) {
            CloseHandle(handle_);
        }
        handle_ = nullptr;
    }

    HANDLE handle_ = nullptr;
};

class JoiningThread {
public:
    explicit JoiningThread(std::thread thread) noexcept : thread_(std::move(thread)) {}
    JoiningThread(const JoiningThread&) = delete;
    JoiningThread& operator=(const JoiningThread&) = delete;
    ~JoiningThread() {
        if (thread_.joinable()) {
            thread_.join();
        }
    }

    void join() {
        if (thread_.joinable()) {
            thread_.join();
        }
    }

private:
    std::thread thread_;
};

std::pair<UniqueHandle, UniqueHandle> create_pipe(const char* test_name) {
    HANDLE read_handle = nullptr;
    HANDLE write_handle = nullptr;
    const BOOL created = CreatePipe(&read_handle, &write_handle, nullptr, 0);
    UniqueHandle reader(read_handle);
    UniqueHandle writer(write_handle);
    expect(created != FALSE, test_name, "CreatePipe failed");
    return {std::move(reader), std::move(writer)};
}

Frame make_pipe_frame(
    uint64_t request_id,
    std::string method,
    std::string payload_json,
    std::vector<std::byte> attachment) {
    Frame frame;
    frame.header.request_id = request_id;
    frame.header.kind = "request";
    frame.header.method = std::move(method);
    frame.header.payload_json = std::move(payload_json);
    frame.attachment = std::move(attachment);
    return frame;
}

struct WriterResult {
    bool success = false;
    std::string error;
};

bool write_fragmented(
    HANDLE handle,
    const std::vector<std::byte>& bytes,
    std::string& error) {
    constexpr std::array<size_t, 3> fragment_sizes{1, 3, 7};
    size_t offset = 0;
    size_t fragment_index = 0;
    while (offset < bytes.size()) {
        const size_t bytes_to_write =
            (std::min)(fragment_sizes[fragment_index % fragment_sizes.size()], bytes.size() - offset);
        if (!aila::ipc::write_all(handle, bytes.data() + offset, bytes_to_write, error)) {
            return false;
        }
        offset += bytes_to_write;
        ++fragment_index;
    }
    return true;
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

void test_pipe_reads_consecutive_fragmented_frames() {
    constexpr const char* name = "consecutive fragmented pipe frames";
    auto [reader, writer] = create_pipe(name);

    const Frame first = make_pipe_frame(
        101,
        "engine.init",
        R"({"model":"qwen","device":0})",
        {std::byte{0x01}, std::byte{0x02}, std::byte{0x03}});
    const Frame second = make_pipe_frame(
        102,
        "engine.generate",
        R"({"prompt":"hello","tokens":4})",
        {std::byte{0xf0}, std::byte{0x0d}});
    const std::vector<std::byte> first_encoded = aila::ipc::encode_frame(first);
    const std::vector<std::byte> second_encoded = aila::ipc::encode_frame(second);
    expect(!first_encoded.empty() && !second_encoded.empty(), name, "fixture encoding failed");

    WriterResult writer_result;
    JoiningThread writer_thread(std::thread(
        [writer = std::move(writer), &first_encoded, &second_encoded, &writer_result]() {
            writer_result.success =
                write_fragmented(writer.get(), first_encoded, writer_result.error) &&
                write_fragmented(writer.get(), second_encoded, writer_result.error);
        }));

    Frame first_output;
    Frame second_output;
    std::string first_error;
    std::string second_error;
    const bool first_read = aila::ipc::read_frame(reader.get(), first_output, first_error);
    const bool second_read = aila::ipc::read_frame(reader.get(), second_output, second_error);
    writer_thread.join();

    expect(writer_result.success, name, "writer failed: " + writer_result.error);
    expect(first_read, name, "first read failed: " + first_error);
    expect(second_read, name, "second read failed: " + second_error);
    expect(first_output.header.request_id == first.header.request_id, name, "first request id changed");
    expect(first_output.header.method == first.header.method, name, "first method changed");
    expect(first_output.header.payload_json == first.header.payload_json, name, "first payload changed");
    expect(first_output.attachment == first.attachment, name, "first attachment changed");
    expect(second_output.header.request_id == second.header.request_id, name, "second request id changed");
    expect(second_output.header.method == second.header.method, name, "second method changed");
    expect(second_output.header.payload_json == second.header.payload_json, name, "second payload changed");
    expect(second_output.attachment == second.attachment, name, "second attachment changed");
}

void test_pipe_write_frame_round_trip() {
    constexpr const char* name = "pipe write_frame round trip";
    auto [reader, writer] = create_pipe(name);
    const Frame input = make_pipe_frame(
        103,
        "engine.shutdown",
        R"({"reason":"test"})",
        {std::byte{0xaa}, std::byte{0x55}});

    WriterResult writer_result;
    JoiningThread writer_thread(std::thread(
        [writer = std::move(writer), &input, &writer_result]() {
            writer_result.success =
                aila::ipc::write_frame(writer.get(), input, writer_result.error);
        }));

    Frame output;
    std::string read_error;
    const bool read_success = aila::ipc::read_frame(reader.get(), output, read_error);
    writer_thread.join();

    expect(writer_result.success, name, "writer failed: " + writer_result.error);
    expect(read_success, name, "reader failed: " + read_error);
    expect(output.header.request_id == input.header.request_id, name, "request id changed");
    expect(output.header.method == input.header.method, name, "method changed");
    expect(output.header.payload_json == input.header.payload_json, name, "payload changed");
    expect(output.attachment == input.attachment, name, "attachment changed");
}

void test_pipe_rejects_truncated_frame_at_eof() {
    constexpr const char* name = "truncated pipe frame";
    auto [reader, writer] = create_pipe(name);
    const Frame input = make_pipe_frame(
        104,
        "engine.generate",
        R"({"prompt":"partial"})",
        {std::byte{0x10}, std::byte{0x20}, std::byte{0x30}, std::byte{0x40}});
    const std::vector<std::byte> encoded = aila::ipc::encode_frame(input);
    expect(!encoded.empty(), name, "fixture encoding failed");

    std::string write_error;
    const bool write_success =
        aila::ipc::write_all(writer.get(), encoded.data(), encoded.size() / 2, write_error);
    writer = UniqueHandle{};

    Frame sentinel = make_pipe_frame(999, "unchanged", R"({"sentinel":true})", {std::byte{0xee}});
    const Frame original_sentinel = sentinel;
    std::string read_error;
    const bool read_success = aila::ipc::read_frame(reader.get(), sentinel, read_error);

    expect(write_success, name, "partial fixture write failed: " + write_error);
    expect(!read_success, name, "truncated frame was accepted");
    expect(
        read_error.find("unexpected end of pipe") != std::string::npos,
        name,
        "EOF error was not reported: " + read_error);
    expect(sentinel.header.request_id == original_sentinel.header.request_id, name, "failed read changed frame");
    expect(sentinel.header.method == original_sentinel.header.method, name, "failed read changed method");
    expect(sentinel.attachment == original_sentinel.attachment, name, "failed read changed attachment");
}

} // namespace

int main() {
    try {
        test_little_endian_helpers();
        test_round_trip();
        test_truncated_frame_is_rejected();
        test_oversized_header_is_rejected_from_prefix();
        test_malformed_json_is_rejected();
        test_oversized_attachment_is_rejected_from_prefix();
        test_pipe_reads_consecutive_fragmented_frames();
        test_pipe_write_frame_round_trip();
        test_pipe_rejects_truncated_frame_at_eof();
        std::cout << "AilaIpcProtocolTests passed\n";
        return 0;
    } catch (const std::exception& exception) {
        std::cerr << exception.what() << '\n';
        return 1;
    }
}
