#include "server/HttpServer.hpp"

#include <Windows.h>
#include <charconv>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string_view>
#include <vector>

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

std::filesystem::path DefaultWorkerExecutable() {
    std::vector<wchar_t> buffer(MAX_PATH);
    for (;;) {
        const DWORD copied = GetModuleFileNameW(nullptr, buffer.data(), static_cast<DWORD>(buffer.size()));
        if (copied == 0) {
            return {};
        }
        if (copied < buffer.size()) {
            const auto exeDir = std::filesystem::path(std::wstring(buffer.data(), copied)).parent_path();
            const auto splitWorker = (exeDir / L"aila_runtime" / L"AilaWorker.exe").lexically_normal();
            std::error_code ec;
            if (std::filesystem::is_regular_file(splitWorker, ec)) {
                return splitWorker;
            }
            const auto siblingWorker = (exeDir / L"AilaWorker.exe").lexically_normal();
            if (std::filesystem::is_regular_file(siblingWorker, ec)) {
                return siblingWorker;
            }
            return {};
        }
        if (buffer.size() > (std::numeric_limits<DWORD>::max)() / 2) {
            return {};
        }
        buffer.resize(buffer.size() * 2);
    }
}

void Usage() {
    std::cout << "AilaServer --model <model directory> [--worker <AilaWorker.exe>] [options]\n"
                 "  --worker PATH               Path to AilaWorker.exe (default: aila_runtime\\AilaWorker.exe beside AilaServer.exe)\n"
                 "  --backend NAME              qwen3_nf4 (default), qwen35_nf4, qwen35_bf16 or qwen4exp_nf4\n"
                 "  --kv-type TYPE              auto (default), bf16, or int8 (qwen4exp_nf4 only)\n"
                 "  --port N                    Local 127.0.0.1 port (8080)\n"
                 "  --model-id NAME             HTTP model name (qwen3 / qwen3.5)\n"
                 "  --context N                 Session capacity, 32..262144 (4096)\n"
                 "  --prefill-chunk-size N       Prefill work per scheduler step, 1..8192 (8192 for qwen4exp_nf4, 512 otherwise)\n"
                 "  --mtp / --no-mtp            Enable or disable speculative MTP decoding (qwen4exp_nf4)\n"
                 "  --mtp-max-context N         Maximum context tracked by MTP (32768)\n"
                 "  --reuse-sessions N           Reuse exact completed token prefixes, 0/1 (0)\n"
                 "  --sessions N                Concurrent requests, 1..4 (4)\n"
                 "  --decode-batch-size N       Compatible decode rows, 1..4 and <= sessions (1)\n"
                 "  --max-new-tokens N          Per-request generation limit, 1..32768 (1024)\n"
                 "  --default-temperature N       Default temperature when omitted, 0..2 (0.6)\n"
                 "  --max-body-bytes N           Request body and rendered prompt limit (65536)\n"
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
        options.kvType = "auto";
        bool modelIdProvided = false;
        bool kvTypeProvided = false;
        bool prefillChunkProvided = false;
        int mtpMode = -1;
        uint32_t mtpMaxContext = 0;
        for (int index = 1; index < argc; ++index) {
            const std::string name(argv[index]);
            if (name == "--help" || name == "-h") { Usage(); return 0; }
            if (name == "--mtp") { mtpMode = 1; continue; }
            if (name == "--no-mtp") { mtpMode = 0; continue; }
            if (index + 1 == argc) throw std::invalid_argument("Missing value for " + name);
            const std::string value(argv[++index]);
            if (name == "--worker") options.workerExecutable = std::filesystem::u8path(value);
            else if (name == "--model") options.modelDirectory = std::filesystem::u8path(value);
            else if (name == "--backend") options.backend = value;
            else if (name == "--kv-type") { options.kvType = value; kvTypeProvided = true; }
            else if (name == "--model-id") { options.modelId = value; modelIdProvided = true; }
            else if (name == "--port") options.port = static_cast<uint16_t>(Number(value, 1, 65535, name));
            else if (name == "--context") options.contextTokens = static_cast<uint32_t>(Number(value, 32, 262144, name));
            else if (name == "--prefill-chunk-size") { options.prefillChunkSize = static_cast<uint32_t>(Number(value, 1, 8192, name)); prefillChunkProvided = true; }
            else if (name == "--mtp-max-context") mtpMaxContext = static_cast<uint32_t>(Number(value, 2, 262144, name));
            else if (name == "--reuse-sessions") options.reuseSessions = Number(value, 0, 1, name) != 0;
            else if (name == "--sessions") options.maxSessions = static_cast<uint32_t>(Number(value, 1, 4, name));
            else if (name == "--decode-batch-size") options.maxDecodeBatchSize = static_cast<uint32_t>(Number(value, 1, 4, name));
            else if (name == "--max-new-tokens") options.maxNewTokens = static_cast<uint32_t>(Number(value, 1, 32768, name));
            else if (name == "--default-temperature") {
                char* end = nullptr;
                const double parsed = std::strtod(value.c_str(), &end);
                if (value.empty() || !end || *end != '\0' || !std::isfinite(parsed) || parsed < 0.0 || parsed > 2.0 || (parsed > 0.0 && static_cast<float>(parsed) <= 0.0f))
                    throw std::invalid_argument("Invalid --default-temperature");
                options.defaultTemperature = parsed;
            }
            else if (name == "--max-body-bytes") options.maxBodyBytes = static_cast<size_t>(Number(value, 65536, 4 * 1024 * 1024, name));
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
        if (options.workerExecutable.empty()) {
            options.workerExecutable = DefaultWorkerExecutable();
        }
        if (options.modelDirectory.empty()) {
            Usage();
            throw std::invalid_argument("--model is required");
        }
        if (options.workerExecutable.empty()) {
            Usage();
            throw std::invalid_argument(
                "--worker is required (auto-lookup checked aila_runtime\\AilaWorker.exe and AilaWorker.exe beside AilaServer.exe)");
        }
        if (options.backend != "qwen3_nf4" && options.backend != "qwen35_nf4" &&
            options.backend != "qwen35_bf16" && options.backend != "qwen4exp_nf4")
            throw std::invalid_argument("--backend must be qwen3_nf4, qwen35_nf4, qwen35_bf16 or qwen4exp_nf4");
        if (!prefillChunkProvided && options.backend == "qwen4exp_nf4")
            options.prefillChunkSize = 8192;
        if (!kvTypeProvided) {
            if (const char* value = std::getenv("AILA_SERVER_KV_TYPE")) options.kvType = value;
        }
        if (options.kvType == "auto") {
            const char* envQuant = std::getenv("AILA_KV_QUANT");
            const char* envFmt = std::getenv("AILA_KV_QUANT_FORMAT");
            const bool envInt8 = envQuant && std::atoi(envQuant) != 0 &&
                envFmt && (std::string(envFmt) == "int8" || std::string(envFmt) == "INT8");
            if (options.backend == "qwen4exp_nf4" && (options.contextTokens > 65536 || envInt8)) {
                options.kvType = "int8";
            } else {
                options.kvType = "bf16";
            }
        }
        if (options.kvType != "bf16" && options.kvType != "int8")
            throw std::invalid_argument("--kv-type must be auto, bf16 or int8");
        if (options.kvType == "int8" && options.backend != "qwen4exp_nf4")
            throw std::invalid_argument("--kv-type int8 is only supported by qwen4exp_nf4");
        if (mtpMode == 1) {
            _putenv_s("AILA_Q4E_SERVER_MTP", "1");
        } else if (mtpMode == 0) {
            _putenv_s("AILA_Q4E_SERVER_MTP", "0");
            _putenv_s("AILA_Q4E_MTP_DRAFT", "0");
        }
        if (mtpMaxContext >= 2) {
            _putenv_s("AILA_Q4E_MTP_MAX_CONTEXT", std::to_string(mtpMaxContext).c_str());
        }
        if (!modelIdProvided) options.modelId = options.backend == "qwen3_nf4" ? "qwen3"
            : options.backend == "qwen4exp_nf4" ? "qwen4exp" : "qwen3.5";
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
