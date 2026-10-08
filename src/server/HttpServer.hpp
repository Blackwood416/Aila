#pragma once

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>

namespace aila::server {

struct ServerOptions {
    std::filesystem::path workerExecutable;
    std::filesystem::path modelDirectory;
    std::filesystem::path shutdownFile;
    std::filesystem::path reportFile;
    std::string expectedBuildId;
    std::string modelId = "qwen3";
    std::string backend = "qwen3_nf4";
    uint16_t port = 8080;
    uint32_t maxSessions = 4;
    uint32_t maxDecodeBatchSize = 1;
    uint32_t contextTokens = 4096;
    uint32_t prefillChunkSize = 512;
    bool reuseSessions = false;
    uint32_t maxNewTokens = 1024;
    size_t maxBodyBytes = 65536;
    size_t maxOutputBytes = 262144;
    size_t streamBufferBytes = 65536;
    uint64_t maxDeviceBytes = 0;
    uint64_t maxPinnedHostBytes = 0;
    uint64_t maxSessionHostBytes = 0;
    std::chrono::milliseconds streamStallTimeout{5000};
    std::chrono::milliseconds requestTimeout{120000};
    std::chrono::milliseconds drainTimeout{2000};
    // Opt-in CPU ledger observation. Zero leaves the original report/traffic unchanged.
    std::chrono::milliseconds memorySampleInterval{0};
};

// A local HTTP/1.1 text service. Device code is confined to the independently
// launched Worker; HTTP loops only touch ordinary C++ host state and sockets.
int RunHttpServer(const ServerOptions& options, std::atomic<bool>& stopRequested);

} // namespace aila::server
