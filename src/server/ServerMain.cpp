#include "server/HttpServer.hpp"

#include <Windows.h>
#include <charconv>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string_view>

#ifndef AILA_WORKER_BUILD_ID
#error "The ordinary C++ server requires the expected Worker build identity"
#endif

namespace {

std::atomic<bool> stopRequested{false};

BOOL WINAPI ConsoleControl(DWORD event) {
    if (event == CTRL_C_EVENT || event == CTRL_BREAK_EVENT || event == CTRL_CLOSE_EVENT ||
        event == CTRL_SHUTDOWN_EVENT) {
        stopRequested.store(true);
        return TRUE;
    }
    return FALSE;
}

uint64_t Number(std::string_view text, uint64_t minimum, uint64_t maximum, const std::string& name) {
    uint64_t value = 0;
    const auto parsed = std::from_chars(text.data(), text.data() + text.size(), value);
    if (parsed.ec != std::errc{} || parsed.ptr != text.data() + text.size() || value < minimum || value > maximum)
        throw std::invalid_argument("Invalid " + name);
    return value;
}

void Usage() {
    std::cout << "AilaServer --worker <AilaWorker.exe> --model <model directory> [options]\n"
                 "  --backend NAME              qwen3_nf4 (default), qwen35_nf4 or qwen35_bf16\n"
                 "  --port N                    Local 127.0.0.1 port (8080)\n"
                 "  --model-id NAME             HTTP model name (qwen3 / qwen3.5)\n"
                 "  --context N                 Session capacity, 32..4096 (4096)\n"
                 "  --sessions N                Concurrent requests, 1..4 (4)\n"
                 "  --decode-batch-size N       Compatible decode rows, 1..4 and <= sessions (1)\n"
                 "  --max-new-tokens N          Per-request generation limit, 1..1024 (1024)\n"
                 "  --max-output-bytes N        Per-request serialized output limit (262144)\n"
                 "  --max-device-bytes N         Worker explicit USM device ceiling (0=unlimited)\n"
                 "  --max-pinned-host-bytes N    Worker pinned host ceiling (0=unlimited)\n"
                 "  --max-session-host-bytes N   Total Session host mirror payload (0=unlimited)\n"
                 "  --memory-sample-ms N         Runtime USM samples, 0=off or 100..60000\n"
                 "  --stream-buffer-bytes N     Trantor pending output gate (65536)\n"
                 "  --stream-stall-ms N          Slow-client timeout (5000)\n"
                 "  --request-timeout-ms N       Total request deadline (120000)\n"
                 "  --shutdown-file PATH        Shut down when this file appears\n"
                 "  --report PATH               Write bounded final metrics as JSON\n"
                 "Ctrl+C stops admission, cancels requests, and shuts down Worker.\n";
}

} // namespace

int main(int argc, char** argv) {
    try {
        aila::server::ServerOptions options;
        options.expectedBuildId = AILA_WORKER_BUILD_ID;
        bool modelIdProvided = false;
        for (int index = 1; index < argc; ++index) {
            const std::string name(argv[index]);
            if (name == "--help" || name == "-h") { Usage(); return 0; }
            if (index + 1 == argc) throw std::invalid_argument("Missing value for " + name);
            const std::string value(argv[++index]);
            if (name == "--worker") options.workerExecutable = std::filesystem::u8path(value);
            else if (name == "--model") options.modelDirectory = std::filesystem::u8path(value);
            else if (name == "--backend") options.backend = value;
            else if (name == "--model-id") { options.modelId = value; modelIdProvided = true; }
            else if (name == "--port") options.port = static_cast<uint16_t>(Number(value, 1, 65535, name));
            else if (name == "--context") options.contextTokens = static_cast<uint32_t>(Number(value, 32, 4096, name));
            else if (name == "--sessions") options.maxSessions = static_cast<uint32_t>(Number(value, 1, 4, name));
            else if (name == "--decode-batch-size") options.maxDecodeBatchSize = static_cast<uint32_t>(Number(value, 1, 4, name));
            else if (name == "--max-new-tokens") options.maxNewTokens = static_cast<uint32_t>(Number(value, 1, 1024, name));
            else if (name == "--max-output-bytes") options.maxOutputBytes = static_cast<size_t>(Number(value, 4096, 4 * 1024 * 1024, name));
            else if (name == "--max-device-bytes") options.maxDeviceBytes = Number(value, 0, (std::numeric_limits<uint64_t>::max)(), name);
            else if (name == "--max-pinned-host-bytes") options.maxPinnedHostBytes = Number(value, 0, (std::numeric_limits<uint64_t>::max)(), name);
            else if (name == "--max-session-host-bytes") options.maxSessionHostBytes = Number(value, 0, (std::numeric_limits<uint64_t>::max)(), name);
            else if (name == "--memory-sample-ms") options.memorySampleInterval = std::chrono::milliseconds(Number(value, 0, 60000, name));
            else if (name == "--stream-buffer-bytes") options.streamBufferBytes = static_cast<size_t>(Number(value, 32768, 1024 * 1024, name));
            else if (name == "--stream-stall-ms") options.streamStallTimeout = std::chrono::milliseconds(Number(value, 100, 60000, name));
            else if (name == "--request-timeout-ms") options.requestTimeout = std::chrono::milliseconds(Number(value, 1000, 600000, name));
            else if (name == "--shutdown-file") options.shutdownFile = std::filesystem::u8path(value);
            else if (name == "--report") options.reportFile = std::filesystem::u8path(value);
            else throw std::invalid_argument("Unknown option " + name);
        }
        if (options.workerExecutable.empty() || options.modelDirectory.empty()) {
            Usage();
            throw std::invalid_argument("--worker and --model are required");
        }
        if (options.backend != "qwen3_nf4" && options.backend != "qwen35_nf4" && options.backend != "qwen35_bf16")
            throw std::invalid_argument("--backend must be qwen3_nf4, qwen35_nf4 or qwen35_bf16");
        if (!modelIdProvided) options.modelId = options.backend == "qwen3_nf4" ? "qwen3" : "qwen3.5";
        if (options.modelId.empty() || options.modelId.size() > 128 ||
            options.modelId.find_first_not_of("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._/-") != std::string::npos)
            throw std::invalid_argument("--model-id must contain 1..128 ASCII model-name characters");
        if (options.maxNewTokens >= options.contextTokens)
            options.maxNewTokens = options.contextTokens - 1;
        options.workerExecutable = std::filesystem::absolute(options.workerExecutable).lexically_normal();
        options.modelDirectory = std::filesystem::absolute(options.modelDirectory).lexically_normal();
        SetConsoleCtrlHandler(ConsoleControl, TRUE);
        const int result = aila::server::RunHttpServer(options, stopRequested);
        SetConsoleCtrlHandler(ConsoleControl, FALSE);
        return result;
    } catch (const std::exception& error) {
        std::cerr << "AilaServer: " << error.what() << '\n';
        return 1;
    }
}
