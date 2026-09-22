#include "runtime/InferenceClient.hpp"
#include "profile/Profiling.hpp"
#include "utils/Tokenizer.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <charconv>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <locale>
#include <map>
#include <memory>
#include <mutex>
#include <numeric>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#ifndef AILA_IPC_WORKER_PATH
#error "IPC benchmark requires the actual Worker target path"
#endif
#ifndef AILA_BUILD_ID
#error "IPC benchmark requires the Worker build identity"
#endif

// This executable owns one isolated Worker and loads one model exactly once.
// It measures client-observed delivery of actual IPC token events. Read() can
// consume already buffered events, including several from one pipe read; these
// intervals are not GPU kernel/decode latency. No SYCL Context is created here.
namespace {
namespace host = aila::runtime;
namespace wire = aila::ipc::v2;
namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;
using Tokens = std::vector<int32_t>;
using namespace std::chrono_literals;
constexpr std::array<size_t, 3> Widths{1, 2, 4};
constexpr uint32_t PrefillChunk = 512;
constexpr auto RequestTimeout = 300s;
constexpr auto ReclaimTimeout = 30s;
constexpr auto ShutdownTimeout = 30s;

void Expect(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

std::string FailureText(std::exception_ptr failure) {
    try { if (failure) std::rethrow_exception(failure); }
    catch (const std::exception& error) { return error.what(); }
    catch (...) { return "Unknown exception"; }
    return {};
}

template <typename T>
std::string Number(T value) {
    std::ostringstream text;
    text.imbue(std::locale::classic());
    text << std::setprecision(17) << value;
    return text.str();
}

std::string Json(std::string_view value) {
    constexpr char digits[] = "0123456789abcdef";
    std::string result = "\"";
    for (unsigned char c : value) {
        if (c == '"' || c == '\\') { result += '\\'; result += static_cast<char>(c); }
        else if (c < 0x20) {
            result += "\\u00";
            result += digits[c >> 4]; result += digits[c & 15];
        } else result += static_cast<char>(c);
    }
    result += '"';
    return result;
}

void Csv(std::ostream& output, const std::vector<std::string>& fields) {
    bool first = true;
    for (const auto& field : fields) {
        if (!first) output << ',';
        first = false;
        output << '"';
        for (const char c : field) {
            if (c == '"') output << '"';
            output << c;
        }
        output << '"';
    }
    output << '\n';
}

std::ofstream Open(const fs::path& path) {
    std::ofstream output;
    output.exceptions(std::ios::failbit | std::ios::badbit);
    output.imbue(std::locale::classic());
    output.open(path, std::ios::out | std::ios::binary | std::ios::trunc);
    output << std::setprecision(17);
    return output;
}

struct Arguments {
    std::string backend = "qwen3_nf4";
    fs::path model, output;
    uint32_t context = 2048, newTokens = 32;
    size_t warmup = 2, rounds = 10;
};

size_t Count(const std::string& value, size_t minimum, size_t maximum, const std::string& option) {
    size_t count = 0;
    const auto parsed = std::from_chars(value.data(), value.data() + value.size(), count);
    Expect(parsed.ec == std::errc{} && parsed.ptr == value.data() + value.size() &&
               count >= minimum && count <= maximum,
           option + " must be an integer in " + Number(minimum) + ".." + Number(maximum));
    return count;
}

Arguments Parse(int argc, char** argv) {
    Arguments result;
    std::set<std::string> seen;
    for (int i = 1; i < argc; ++i) {
        const std::string option = argv[i];
        Expect(seen.insert(option).second, "Duplicate option: " + option);
        Expect(i + 1 < argc, "Missing value for " + option);
        const std::string value = argv[++i];
        if (option == "--backend") {
            Expect(value == "qwen3_nf4" || value == "qwen35_nf4", "Unsupported --backend");
            result.backend = value;
        } else if (option == "--model" || option == "--output") {
            Expect(!value.empty(), "Empty " + option);
            auto path = fs::absolute(fs::u8path(value)).lexically_normal();
            if (option == "--model") result.model = std::move(path);
            else result.output = std::move(path);
        } else if (option == "--context") {
            result.context = static_cast<uint32_t>(Count(value, 2048, 4096, option));
            Expect(result.context == 2048 || result.context == 4096, "--context must be 2048 or 4096");
        } else if (option == "--new-tokens") {
            result.newTokens = static_cast<uint32_t>(Count(value, 32, 64, option));
            Expect(result.newTokens == 32 || result.newTokens == 64, "--new-tokens must be 32 or 64");
        } else if (option == "--warmup") result.warmup = Count(value, 2, 10000, option);
        else if (option == "--rounds") result.rounds = Count(value, 1, 10000, option);
        else throw std::invalid_argument("Unknown option: " + option);
    }
    Expect(!result.model.empty() && fs::is_directory(result.model), "--model must name an existing model directory");
    Expect(!result.output.empty(), "--output must name a new, unique directory");
    Expect(!fs::exists(result.output), "Output directory already exists; select a new --output directory");
    return result;
}

struct Distribution {
    size_t count = 0;
    double minimum = 0, mean = 0, p50 = 0, p95 = 0, p99 = 0, maximum = 0;
};

Distribution Describe(std::vector<double> values) {
    Distribution result;
    if (values.empty()) return result;
    for (const double value : values) Expect(std::isfinite(value) && value >= 0, "Invalid timing sample");
    std::sort(values.begin(), values.end());
    const auto percentile = [&](double p) {
        const double at = p * static_cast<double>(values.size() - 1);
        const auto lower = static_cast<size_t>(at);
        const auto upper = std::min(lower + 1, values.size() - 1);
        return values[lower] + (values[upper] - values[lower]) * (at - static_cast<double>(lower));
    };
    result.count = values.size();
    result.minimum = values.front(); result.maximum = values.back();
    result.mean = std::accumulate(values.begin(), values.end(), 0.0) / static_cast<double>(values.size());
    result.p50 = percentile(0.50); result.p95 = percentile(0.95); result.p99 = percentile(0.99);
    return result;
}

void AppendDistribution(std::vector<std::string>& row, const Distribution& value) {
    row.push_back(Number(value.count));
    for (const double item : {value.minimum, value.mean, value.p50, value.p95, value.p99, value.maximum})
        row.push_back(value.count ? Number(item) : "");
}

void AppendDistributionHeader(std::vector<std::string>& row, const std::string& prefix) {
    row.push_back(prefix + "_count");
    for (const char* name : {"minimum_ms", "mean_ms", "p50_ms", "p95_ms", "p99_ms", "maximum_ms"})
        row.push_back(prefix + "_" + name);
}

class StartGate {
public:
    explicit StartGate(size_t expected) : expected_(expected) {}
    bool ArriveAndWait() {
        std::unique_lock<std::mutex> lock(mutex_);
        ++arrived_;
        changed_.notify_all();
        changed_.wait(lock, [&] { return open_ || aborted_; });
        return !aborted_;
    }
    void OpenWhenReady() {
        std::unique_lock<std::mutex> lock(mutex_);
        Expect(changed_.wait_for(lock, 15s, [&] { return arrived_ == expected_ || aborted_; }) && !aborted_,
               "Request reader threads did not reach the start barrier");
        open_ = true;
        changed_.notify_all();
    }
    void Abort() {
        std::lock_guard<std::mutex> lock(mutex_);
        aborted_ = true;
        changed_.notify_all();
    }
private:
    std::mutex mutex_;
    std::condition_variable changed_;
    size_t expected_, arrived_ = 0;
    bool open_ = false, aborted_ = false;
};

struct Fixture {
    std::string key, kind, source;
    uint64_t seed = 0;
    Tokens prompt;
};
struct Reference { Tokens tokens; std::string label; bool ready = false; };
struct TokenPoint {
    uint64_t sequence;
    int32_t id;
    int64_t readNs;
    uint64_t promptTokens, generatedTokens, deliveredTokens;
};
struct RequestRecord {
    size_t fixture = 0, lane = 0;
    std::string label, referenceLabel, error;
    uint64_t sessionId = 0, requestId = 0;
    int64_t startNs = -1, acceptedNs = -1, terminalNs = -1;
    size_t terminalCount = 0;
    bool passed = false;
    std::vector<TokenPoint> tokens;
    host::InferenceReadEvent terminal;
    host::InferenceRequestSnapshot snapshot;
    std::shared_ptr<host::InferenceRequest> handle;
    std::exception_ptr failure;
};
struct Batch {
    size_t id, width, round;
    std::string phase, profile;
};
struct Group {
    size_t batches = 0, requests = 0;
    uint64_t completionTokens = 0;
    double spanSeconds = 0;
    std::vector<double> ttftMs, durationMs, deliveryGapMs;
};

std::vector<double> Gaps(const RequestRecord& row) {
    std::vector<double> result;
    if (row.tokens.size() > 1) result.reserve(row.tokens.size() - 1);
    // Never subtract timestamps belonging to different requests.
    for (size_t i = 1; i < row.tokens.size(); ++i)
        result.push_back(static_cast<double>(row.tokens[i].readNs - row.tokens[i - 1].readNs) / 1e6);
    return result;
}

void WriteLocal(std::ostream& out, const host::InferenceClientSnapshot& value) {
    out << "{\"status\":" << Json(wire::to_string(value.status)) << ",\"epoch\":" << Json(value.epoch)
        << ",\"outstanding_requests\":" << value.outstandingRequests
        << ",\"retained_request_handles\":" << value.retainedRequestHandles
        << ",\"pending_commands\":" << value.pendingCommands << ",\"queued_commands\":" << value.queuedCommands
        << ",\"reserved_tokens\":" << value.reservedTokens << ",\"ingress_bytes\":" << value.ingressBytes
        << ",\"transport_bytes\":" << value.transportBytes << ",\"peak_ingress_bytes\":" << value.peakIngressBytes
        << ",\"peak_transport_bytes\":" << value.peakTransportBytes
        << ",\"received_event_ordinal\":" << value.receivedEventOrdinal
        << ",\"dropped_logs\":" << value.droppedLogs << ",\"error\":" << Json(value.error) << '}';
}

void WriteRemote(std::ostream& out, const wire::WorkerInfo& value) {
    out << "{\"status\":" << Json(wire::to_string(value.domainStatus))
        << ",\"accepting\":" << (value.accepting ? "true" : "false")
        << ",\"stopping\":" << (value.stopping ? "true" : "false")
        << ",\"faulted\":" << (value.faulted ? "true" : "false")
        << ",\"outstanding_requests\":" << value.outstandingRequests
        << ",\"active_requests\":" << value.activeRequests << ",\"queued_requests\":" << value.queuedRequests
        << ",\"ipc_outstanding_requests\":" << value.ipcOutstandingRequests
        << ",\"retained_request_handles\":" << value.retainedRequestHandles
        << ",\"reserved_tokens\":" << value.reservedTokens
        << ",\"reserved_session_cache_bytes\":" << value.reservedSessionCacheBytes
        << ",\"pending_commands_including_query\":" << value.pendingCommands
        << ",\"ingress_bytes_including_query\":" << value.ingressBytes
        << ",\"transport_bytes_including_query\":" << value.transportBytes
        << ",\"dropped_logs\":" << value.droppedLogs << '}';
}

bool LocalEmpty(const host::InferenceClientSnapshot& value) {
    return value.outstandingRequests == 0 && value.retainedRequestHandles == 0 && value.pendingCommands == 0 &&
        value.queuedCommands == 0 && value.reservedTokens == 0 && value.ingressBytes == 0 && value.transportBytes == 0;
}

class Runner {
public:
    explicit Runner(Arguments args)
        : args_(std::move(args)), executable_(fs::absolute(fs::u8path(AILA_IPC_WORKER_PATH)).lexically_normal()) {
        Expect(fs::is_regular_file(executable_), "Compiled Worker path does not exist");
        fs::create_directories(args_.output.parent_path());
        Expect(fs::create_directory(args_.output), "Output directory was already created by another run");
        tokens_ = Open(args_.output / "tokens.csv");
        requests_ = Open(args_.output / "requests.csv");
        batches_ = Open(args_.output / "batches.csv");
        reclamation_ = Open(args_.output / "reclamation.jsonl");
        Csv(tokens_, {"batch", "request", "phase", "profile", "concurrency", "round", "lane", "fixture", "prompt_kind",
                      "prompt_tokens", "seed", "reference_request", "inference_id", "session_id", "sequence", "token_id",
                      "read_return_ns", "since_submit_start_ns", "delivery_gap_ns", "prompt_tokens_processed",
                      "generated_tokens", "delivered_tokens"});
        std::vector<std::string> header{
            "batch", "request", "phase", "profile", "concurrency", "round", "lane", "fixture", "prompt_kind",
            "prompt_tokens", "seed", "reference_request", "inference_id", "session_id", "start_ns", "submit_return_ns",
            "first_token_ns", "last_token_ns", "terminal_ns", "terminal_count", "token_count", "ttft_ms", "request_duration_ms",
            "terminal_sequence", "terminal_status", "end_reason", "terminal_prompt_tokens", "terminal_generated_tokens",
            "terminal_event_count", "terminal_error_code", "terminal_message", "local_failure", "delivery_abandoned",
            "discarded_published_tokens", "accepted", "final_received", "terminal_received", "completion_ready", "terminal_consumed"};
        AppendDistributionHeader(header, "delivery_gap");
        header.insert(header.end(), {"passed", "error"});
        Csv(requests_, header);
        Csv(batches_, {"batch", "phase", "profile", "concurrency", "round", "passed", "requests", "completion_tokens",
                       "start_ns", "last_terminal_ns", "span_seconds", "offered_overlap", "completion_tokens_per_second", "error"});
    }

    void Run() {
        std::string failure, shutdownFailure;
        bool workloadPassed = false, cleanShutdown = false;
        try {
            MakeFixtures();
            WriteMetadata();
            StartWorker();
            for (size_t fixture = 0; fixture < fixtures_.size(); ++fixture)
                RunBatch("reference", fixtures_[fixture].kind, 0, {fixture});
            for (const std::string profile : {"short", "mixed"}) {
                for (const size_t width : Widths) {
                    for (size_t round = 0; round < args_.warmup; ++round)
                        RunBatch("warmup", profile, round, Select(profile, width, round));
                    for (size_t round = 0; round < args_.rounds; ++round)
                        RunBatch("measured", profile, round, Select(profile, width, round));
                }
            }
            workloadPassed = true;
        } catch (...) { failure = FailureText(std::current_exception()); }

        // All reader threads are joined and child handles released by RunBatch,
        // including its failure path. Always attempt the owned Worker's normal
        // shutdown; InferenceClient verifies its close barrier and exit code.
        if (client_) {
            if (model_) { model_->Release(); model_.reset(); }
            try { WaitReclaimed("model-released"); }
            catch (...) {
                const auto message = FailureText(std::current_exception());
                if (failure.empty()) failure = message;
                else failure += "; final reclamation: " + message;
            }
            try {
                client_->Shutdown(ShutdownTimeout);
                finalLocal_ = client_->Snapshot();
                Expect(finalLocal_.status == wire::DomainStatus::Closed && LocalEmpty(finalLocal_) && finalLocal_.error.empty(),
                       "Shutdown did not close a fully reclaimed client domain");
                cleanShutdown = true;
            } catch (...) {
                shutdownFailure = FailureText(std::current_exception());
                finalLocal_ = client_->Snapshot();
            }
        }
        bool passed = workloadPassed && cleanShutdown && failure.empty() && shutdownFailure.empty();
        if (passed) {
            try { WriteSummary(); }
            catch (...) { failure = FailureText(std::current_exception()); passed = false; }
        }
        auto result = Open(args_.output / "result.json");
        result << "{\n  \"schema\":\"aila.ipc_benchmark.v1\",\n  \"passed\":" << (passed ? "true" : "false")
               << ",\n  \"workload_passed\":" << (workloadPassed ? "true" : "false")
               << ",\n  \"clean_shutdown_and_zero_exit\":" << (cleanShutdown ? "true" : "false")
               << ",\n  \"model_loads\":" << modelLoads_ << ",\n  \"unique_session_ids\":" << sessionIds_.size()
               << ",\n  \"unique_request_ids\":" << requestIds_.size() << ",\n  \"batches\":" << nextBatch_
               << ",\n  \"error\":" << Json(failure) << ",\n  \"shutdown_error\":" << Json(shutdownFailure)
               << ",\n  \"final_client\":";
        WriteLocal(result, finalLocal_);
        result << "\n}\n";
        result.close();
        Expect(passed, "IPC benchmark failed: " + failure + (shutdownFailure.empty() ? "" : "; shutdown: " + shutdownFailure));
        std::cout << "[PASS] IPC benchmark: one model load, independent token references, zero retained request handles, "
                     "normal Worker exit. Output=" << args_.output.u8string() << '\n';
    }

private:
    int64_t NowNs() const {
        return std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - origin_).count();
    }

    host::InferenceRequestOptions RequestOptions(const Fixture& fixture) const {
        host::InferenceRequestOptions options;
        options.prefillChunkSize = PrefillChunk;
        options.generation.maxNewTokens = args_.newTokens;
        options.generation.doSample = true;
        options.generation.temperature = 0.6;
        options.generation.topK = 20;
        options.generation.topP = 0.95;
        options.generation.useFixedSeed = true;
        options.generation.samplingSeed = fixture.seed;
        // An empty stop set makes the exact requested token count observable,
        // even if a generated token happens to be EOS in a chat tokenizer.
        return options;
    }

    void MakeFixtures() {
        Tokenizer tokenizer;
        Expect(tokenizer.load(args_.model.u8string()), "Could not load benchmark tokenizer");
        const std::array<std::string, 4> sources{
            "Aster records the temperature each morning and compares consecutive measurements. ",
            "Beryl keeps a journal of train arrivals, distances, and elapsed minutes. ",
            "Coral describes a C++ vector algorithm and several distinct boundary conditions. ",
            "Dahlia lists observations about books, gardens, colours, and simple arithmetic. "};
        auto prompts = Open(args_.output / "prompts.csv");
        Csv(prompts, {"fixture", "prompt_kind", "seed", "prompt_tokens", "index", "token_id"});
        for (size_t lane = 0; lane < sources.size(); ++lane) {
            const auto source = tokenizer.encode(sources[lane]);
            Expect(!source.empty(), "Tokenizer returned an empty source fixture");
            for (size_t kind = 0; kind < 2; ++kind) {
                Fixture fixture;
                fixture.kind = kind ? "long" : "short";
                fixture.key = fixture.kind + "-lane-" + Number(lane);
                fixture.source = sources[lane]; fixture.seed = 42 + lane;
                const size_t count = kind ? 1024 : 32;
                fixture.prompt.reserve(count);
                for (size_t i = 0; i < count; ++i) {
                    fixture.prompt.push_back(source[i % source.size()]);
                    Csv(prompts, {fixture.key, fixture.kind, Number(fixture.seed), Number(count), Number(i), Number(fixture.prompt.back())});
                }
                fixtures_.push_back(std::move(fixture));
            }
        }
        references_.resize(fixtures_.size());
    }

    void WriteMetadata() {
        auto output = Open(args_.output / "metadata.json");
        const auto wallNs = std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
        output << "{\n  \"schema\":\"aila.ipc_benchmark.v1\",\n  \"backend\":" << Json(args_.backend)
               << ",\n  \"model_directory\":" << Json(args_.model.u8string())
               << ",\n  \"output_directory\":" << Json(args_.output.u8string())
               << ",\n  \"worker_executable\":" << Json(executable_.u8string())
               << ",\n  \"worker_runtime_directory\":" << Json(executable_.parent_path().u8string())
               << ",\n  \"build_id\":" << Json(AILA_BUILD_ID)
               << ",\n  \"metadata_unix_ns\":" << wallNs << ",\n  \"context\":" << args_.context
               << ",\n  \"warmup_rounds_per_configuration\":" << args_.warmup
               << ",\n  \"measured_rounds_per_configuration\":" << args_.rounds
               << ",\n  \"new_tokens\":" << args_.newTokens << ",\n  \"prefill_chunk_size\":" << PrefillChunk
               << ",\n  \"concurrency\":[1,2,4],\n  \"profiles\":[\"short\",\"mixed\"],"
                  "\n  \"sampling\":{\"do_sample\":true,\"temperature\":0.6,\"top_k\":20,\"top_p\":0.95,"
                  "\"use_fixed_seed\":true,\"repetition_penalty\":1,\"presence_penalty\":0,\"frequency_penalty\":0,\"stop_token_ids\":[]},"
                  "\n  \"clock\":\"Nanoseconds since this process run's steady_clock origin. Wall-clock metadata is separate.\","
                  "\n  \"request_timing\":\"One independent reader thread per request; a start barrier precedes Submit. Start is immediately before Submit; token and terminal timestamps are recorded immediately after Read returns. Session creation, post-completion queries, releases, file writes and between-batch probes are excluded.\","
                  "\n  \"delivery_gap\":\"Difference between successive actual token Read-return timestamps within the same request. First-token delay is TTFT, not a token gap. Zero and near-zero gaps are retained. The client/pipe may batch several token deliveries; these are not GPU computation latencies.\","
                  "\n  \"percentiles\":\"Linear interpolation at p*(count-1); counts accompany each distribution. Group gaps pool already calculated within-request gaps, never cross-request timestamp differences. Small-sample p99 is descriptive.\","
                  "\n  \"throughput\":\"Completed tokens divided by the sum of measured batch spans, each from earliest Submit start to latest terminal Read return. This excludes idle setup/cleanup and is not sustained open-loop throughput.\","
                  "\n  \"references\":\"Every exact prompt/seed runs alone in a fresh Session before concurrent warmup. All warmup/measured token IDs and completed summaries must match it. Reference and warmup samples never enter summary.csv.\","
                  "\n  \"mixed\":\"Long=1024 actual tokens when (lane+round)%2==0, otherwise short=32. Width one alternates long/short across rounds. Fresh Session per request; no chat template is applied.\","
                  "\n  \"identity\":\"InferenceClient::Start uses the compiled real Worker path and build ID, verifies the isolated executable/runtime/current-directory/PATH handshake, and owns its lifetime. Content hashes are supplied separately by the orchestrating run.\","
                  "\n  \"reclamation\":\"Request handles and all public IPC/Scheduler request, token and cache reservations must be zero after every wave. Worker pending/ingress/transport counters include the observing worker.query itself. Final Shutdown validates the close barrier and zero process exit.\","
                  "\n  \"result_gate\":\"Use measurements only if result.json has passed=true. Raw CSV files also retain failed and excluded runs.\","
                  "\n  \"fixtures\":[";
        for (size_t i = 0; i < fixtures_.size(); ++i) {
            const auto& fixture = fixtures_[i];
            if (i) output << ',';
            output << "\n    {\"key\":" << Json(fixture.key) << ",\"kind\":" << Json(fixture.kind)
                   << ",\"seed\":" << fixture.seed << ",\"prompt_tokens\":" << fixture.prompt.size()
                   << ",\"source_text\":" << Json(fixture.source) << '}';
        }
        output << "\n  ]\n}\n";
    }

    void StartWorker() {
        host::InferenceClientOptions options;
        options.handshakeTimeout = 10s; options.commandTimeout = RequestTimeout; options.shutdownTimeout = ShutdownTimeout;
        if (args_.backend == "qwen35_nf4") {
            auto& caps = options.requiredCapabilities;
            caps.erase(std::remove(caps.begin(), caps.end(), "qwen3_nf4_bf16_kv"), caps.end());
            caps.emplace_back("qwen35_nf4_bf16_kv");
        }
        client_ = host::InferenceClient::Start(executable_.parent_path(), executable_, AILA_BUILD_ID, options);
        epoch_ = client_->Snapshot().epoch;
        Expect(wire::valid_epoch(epoch_), "Worker did not publish a valid epoch");
        host::InferenceModelOptions modelOptions;
        modelOptions.maxSequenceLength = args_.context; modelOptions.maxSessions = 4;
        model_ = args_.backend == "qwen35_nf4" ? client_->LoadQwen35(args_.model.u8string(), modelOptions)
                                             : client_->LoadQwen3(args_.model.u8string(), modelOptions);
        ++modelLoads_;
        const auto& info = model_->Info();
        Expect(info.modelId && info.vocabSize && info.maxSequenceLength == args_.context &&
                   info.sessionReservationBytes && model_->Epoch() == epoch_, "Loaded model metadata/epoch mismatch");
        const std::string capability(wire::model_capability(args_.backend));
        Expect(std::find(info.capabilities.begin(), info.capabilities.end(), capability) != info.capabilities.end(),
               "Loaded model does not advertise its requested backend");
        for (const auto& fixture : fixtures_) {
            Expect(fixture.prompt.size() + args_.newTokens <= info.maxSequenceLength, "Token fixture exceeds context");
            for (const auto token : fixture.prompt)
                Expect(token >= 0 && static_cast<uint32_t>(token) < info.vocabSize, "Tokenizer emitted an out-of-vocabulary fixture token");
        }
        auto identity = Open(args_.output / "worker.json");
        identity << "{\"handshake_validated\":true,\"epoch\":" << Json(epoch_)
                 << ",\"build_id\":" << Json(AILA_BUILD_ID) << ",\"model_id\":" << info.modelId
                 << ",\"vocab_size\":" << info.vocabSize << ",\"context\":" << info.maxSequenceLength
                 << ",\"session_cache_bytes\":" << info.sessionReservationBytes << ",\"model_loads\":1}\n";
        WaitReclaimed("model-loaded");
    }

    std::vector<size_t> Select(const std::string& profile, size_t width, size_t round) const {
        std::vector<size_t> selected;
        for (size_t lane = 0; lane < width; ++lane)
            selected.push_back(lane * 2 + (profile == "mixed" && (lane + round) % 2 == 0 ? 1 : 0));
        return selected;
    }

    void WaitReclaimed(const std::string& stage) {
        const auto deadline = Clock::now() + ReclaimTimeout;
        host::InferenceClientSnapshot local;
        wire::WorkerInfo remote;
        bool passed = false;
        do {
            remote = client_->Query(2s);
            local = client_->Snapshot();
            Expect(remote.domainStatus == wire::DomainStatus::Ready && remote.accepting && !remote.stopping &&
                       !remote.faulted && local.status == wire::DomainStatus::Ready && local.error.empty() && local.epoch == epoch_,
                   stage + ": Worker stopped or faulted during reclamation");
            // The remote query counts itself as one pending command. Its
            // frame/response reservations are not live inference payload.
            passed = LocalEmpty(local) && remote.pendingCommands <= 1 && remote.ipcOutstandingRequests == 0 &&
                remote.retainedRequestHandles == 0 && remote.outstandingRequests == 0 && remote.activeRequests == 0 &&
                remote.queuedRequests == 0 && remote.reservedTokens == 0 && remote.reservedSessionCacheBytes == 0;
            if (passed) break;
            std::this_thread::sleep_for(2ms);
        } while (Clock::now() < deadline);
        reclamation_ << "{\"stage\":" << Json(stage) << ",\"observed_ns\":" << NowNs()
                     << ",\"passed\":" << (passed ? "true" : "false") << ",\"client\":";
        WriteLocal(reclamation_, local);
        reclamation_ << ",\"worker\":";
        WriteRemote(reclamation_, remote);
        reclamation_ << "}\n";
        reclamation_.flush();
        Expect(passed, stage + ": public request handles or IPC/Scheduler reservations were not reclaimed");
    }

    std::shared_ptr<host::InferenceSession> CreateSession() {
        const auto deadline = Clock::now() + ReclaimTimeout;
        for (;;) {
            try {
                auto session = model_->CreateSession(10s);
                const auto state = session->Query(2s);
                Expect(session->Epoch() == epoch_ && state.modelId == model_->Info().modelId && state.sessionId &&
                           state.capacity == args_.context && state.cacheBytes == model_->Info().sessionReservationBytes &&
                           !state.busy && state.contextLength == 0, "New Session has incorrect identity, cache or context");
                Expect(sessionIds_.insert(state.sessionId).second, "Worker reused a Session ID in one epoch");
                return session;
            } catch (const host::InferenceClientError& error) {
                if (error.RemoteCode() != wire::ErrorCode::ResourceLimit || Clock::now() >= deadline) throw;
                std::this_thread::sleep_for(2ms);
            }
        }
    }

    void ReadRequest(RequestRecord& row, const std::shared_ptr<host::InferenceSession>& session,
                     StartGate& gate, std::atomic<bool>& abort) {
        const auto& fixture = fixtures_[row.fixture];
        Tokens prompt = fixture.prompt;
        auto options = RequestOptions(fixture);
        Expect(gate.ArriveAndWait() && !abort.load(), "Concurrent wave aborted before Submit");
        row.startNs = NowNs();
        const auto deadline = Clock::now() + RequestTimeout;
        row.handle = session->Submit(std::move(prompt), std::move(options), RequestTimeout);
        row.acceptedNs = NowNs();
        row.requestId = row.handle->Id();
        Expect(row.requestId && row.handle->Epoch() == epoch_, "Accepted request has the wrong identity/epoch");
        for (;;) {
            Expect(!abort.load(), "Concurrent peer failed");
            Expect(Clock::now() < deadline, "Request exceeded its wall-clock timeout");
            const auto read = row.handle->Read(100ms);
            const int64_t returned = NowNs(); // Timestamp before inspecting or validating the returned event.
            if (read.status == host::InferenceReadStatus::Timeout) continue;
            Expect(read.status == host::InferenceReadStatus::Event && read.event, "Request closed without its unique terminal");
            const auto& event = *read.event;
            if (event.terminal) {
                ++row.terminalCount;
                row.terminalNs = returned;
                row.terminal = event;
                const auto next = row.handle->Read(0ms);
                Expect(next.status == host::InferenceReadStatus::Closed && !next.event, "A second terminal/token was exposed after terminal");
                row.snapshot = row.handle->Snapshot();
                return;
            }
            row.tokens.push_back({event.sequence, event.tokenId, returned, event.summary.promptTokensProcessed,
                                  event.summary.generatedTokens, event.deliveredTokens});
            Expect(row.tokens.size() <= args_.newTokens, "Worker delivered more tokens than requested");
            Expect(!event.localFailure && !event.deliveryAbandoned && event.discardedPublishedTokens == 0,
                   "Token delivery was abandoned or locally fabricated");
        }
    }

    void Validate(RequestRecord& row, const std::shared_ptr<host::InferenceSession>& session) const {
        const auto& fixture = fixtures_[row.fixture];
        const auto& reference = references_[row.fixture];
        Expect(row.terminalCount == 1 && row.tokens.size() == args_.newTokens, "Wrong terminal or actual token count");
        const auto& terminal = row.terminal;
        const auto& summary = terminal.summary;
        Expect(!terminal.localFailure && !terminal.deliveryAbandoned && terminal.discardedPublishedTokens == 0 &&
                   terminal.deliveredTokens == args_.newTokens && terminal.sequence == uint64_t(args_.newTokens) + 1 &&
                   summary.status == wire::RequestStatus::Completed && summary.endReason == wire::EndReason::Length &&
                   summary.promptTokensProcessed == fixture.prompt.size() && summary.generatedTokens == args_.newTokens &&
                   summary.eventCount == uint64_t(args_.newTokens) + 1 && !summary.errorCode && summary.message.empty(),
               "Terminal does not describe the exact completed token stream");
        const auto& snapshot = row.snapshot;
        Expect(snapshot.inferenceId == row.requestId && snapshot.epoch == epoch_ && snapshot.accepted && snapshot.finalReceived &&
                   snapshot.terminalReceived && snapshot.completionReady && snapshot.terminalConsumed && !snapshot.released &&
                   !snapshot.deliveryAbandoned && !snapshot.localFailure && snapshot.deliveredTokens == args_.newTokens &&
                   snapshot.discardedPublishedTokens == 0 && snapshot.bufferedTokens == 0 &&
                   snapshot.consumedSequence == uint64_t(args_.newTokens) + 1,
               "Client did not consume matching terminal/final events and every preceding token");
        Expect(row.startNs >= 0 && row.acceptedNs >= row.startNs && row.tokens.front().readNs >= row.acceptedNs &&
                   row.terminalNs >= row.tokens.back().readNs && row.terminalNs > row.startNs, "Request timestamps are not monotonic");
        for (size_t i = 0; i < row.tokens.size(); ++i) {
            const auto& point = row.tokens[i];
            Expect(point.sequence == i + 1 && point.generatedTokens == i + 1 && point.deliveredTokens == i + 1 &&
                       point.promptTokens == fixture.prompt.size() && point.id >= 0 &&
                       static_cast<uint32_t>(point.id) < model_->Info().vocabSize &&
                       (i == 0 || point.readNs >= row.tokens[i - 1].readNs), "Token sequence/count/vocabulary/timestamp mismatch");
            if (!row.referenceLabel.empty()) {
                Expect(reference.ready && reference.tokens.size() == args_.newTokens, "Missing independent single-request reference");
                Expect(point.id == reference.tokens[i], "Independent reference divergence at token " + Number(i) +
                       ": actual=" + Number(point.id) + " reference=" + Number(reference.tokens[i]));
            }
        }
        const auto state = session->Query(2s);
        Expect(!state.busy && state.contextLength == fixture.prompt.size() + row.tokens.size() &&
                   state.sessionId == row.sessionId && state.modelId == model_->Info().modelId,
               "Completed Session did not consume the last generated token into its own cache");
    }

    void RunBatch(const std::string& phase, const std::string& profile, size_t round, const std::vector<size_t>& selected) {
        const Batch batch{++nextBatch_, selected.size(), round, phase, profile};
        std::vector<RequestRecord> rows(selected.size());
        std::vector<std::shared_ptr<host::InferenceSession>> sessions;
        std::vector<std::thread> readers;
        StartGate gate(selected.size());
        std::atomic<bool> abort{false};
        std::string error;
        try {
            Expect(!selected.empty() && selected.size() <= 4, "Invalid batch width");
            for (size_t lane = 0; lane < selected.size(); ++lane) {
                auto& row = rows[lane];
                row.fixture = selected[lane]; row.lane = lane;
                row.label = Number(batch.id) + "-" + phase + "-lane-" + Number(lane);
                row.tokens.reserve(args_.newTokens + 1);
                if (phase != "reference") {
                    Expect(references_[row.fixture].ready, "Concurrent run cannot install its own reference");
                    row.referenceLabel = references_[row.fixture].label;
                }
                sessions.push_back(CreateSession());
                row.sessionId = sessions.back()->Info().sessionId;
            }
            readers.reserve(selected.size());
            for (size_t lane = 0; lane < selected.size(); ++lane) {
                readers.emplace_back([&, lane] {
                    try { ReadRequest(rows[lane], sessions[lane], gate, abort); }
                    catch (...) {
                        rows[lane].failure = std::current_exception();
                        abort.store(true);
                        gate.Abort();
                    }
                });
            }
            gate.OpenWhenReady();
        } catch (...) {
            error = FailureText(std::current_exception());
            abort.store(true); gate.Abort();
        }
        // Even a partially created thread group must be joined before any
        // Session/Request handle or the gate can be destroyed.
        for (auto& reader : readers) if (reader.joinable()) reader.join();
        for (size_t lane = 0; lane < rows.size(); ++lane) {
            auto& row = rows[lane];
            try {
                if (row.failure) std::rethrow_exception(row.failure);
                Expect(row.handle && lane < sessions.size(), "Request was never submitted");
                Expect(requestIds_.insert(row.requestId).second, "Worker reused an inference ID in one epoch");
                Validate(row, sessions[lane]);
                row.passed = true;
            } catch (...) {
                row.error = FailureText(std::current_exception());
                if (error.empty()) error = row.label + ": " + row.error;
            }
        }
        int64_t firstStart = -1, lastStart = -1, firstTerminal = -1, lastTerminal = -1;
        uint64_t tokenCount = 0;
        bool timed = true;
        for (const auto& row : rows) {
            timed = timed && row.startNs >= 0 && row.terminalNs >= row.startNs;
            firstStart = firstStart < 0 ? row.startNs : std::min(firstStart, row.startNs);
            lastStart = std::max(lastStart, row.startNs);
            firstTerminal = firstTerminal < 0 ? row.terminalNs : std::min(firstTerminal, row.terminalNs);
            lastTerminal = std::max(lastTerminal, row.terminalNs);
            tokenCount += row.tokens.size();
        }
        const bool overlap = timed && lastStart < firstTerminal;
        if (error.empty() && !overlap) error = "Offered requests did not overlap; cannot report this concurrency";
        for (auto& row : rows) if (row.handle) { row.handle->Release(); row.handle.reset(); }
        for (auto& session : sessions) { session->Release(); session.reset(); }
        sessions.clear();
        try { WaitReclaimed("batch-" + Number(batch.id)); }
        catch (...) {
            const auto message = FailureText(std::current_exception());
            if (error.empty()) error = message;
            else error += "; reclamation: " + message;
        }
        const double span = timed ? static_cast<double>(lastTerminal - firstStart) / 1e9 : 0;
        WriteBatch(batch, rows);
        Csv(batches_, {Number(batch.id), phase, profile, Number(batch.width), Number(round), Number(error.empty()),
                       Number(rows.size()), Number(tokenCount), timed ? Number(firstStart) : "", timed ? Number(lastTerminal) : "",
                       timed ? Number(span) : "", Number(overlap), span > 0 ? Number(static_cast<double>(tokenCount) / span) : "", error});
        tokens_.flush(); requests_.flush(); batches_.flush();
        Expect(error.empty(), "Batch " + Number(batch.id) + ": " + error);
        if (phase == "reference") {
            Expect(rows.size() == 1 && !references_[rows[0].fixture].ready, "Independent reference must run alone exactly once");
            auto& reference = references_[rows[0].fixture];
            for (const auto& point : rows[0].tokens) reference.tokens.push_back(point.id);
            reference.label = rows[0].label; reference.ready = true;
        } else if (phase == "measured") {
            auto& group = groups_[{profile, batch.width}];
            ++group.batches; group.requests += rows.size(); group.completionTokens += tokenCount; group.spanSeconds += span;
            for (const auto& row : rows) {
                group.ttftMs.push_back(static_cast<double>(row.tokens.front().readNs - row.startNs) / 1e6);
                group.durationMs.push_back(static_cast<double>(row.terminalNs - row.startNs) / 1e6);
                const auto gaps = Gaps(row);
                group.deliveryGapMs.insert(group.deliveryGapMs.end(), gaps.begin(), gaps.end());
            }
        }
        std::cout << "[PASS] phase=" << phase << " profile=" << profile << " concurrency=" << batch.width
                  << " round=" << round << " tokens=" << tokenCount << " span_seconds=" << span << '\n';
    }

    std::vector<std::string> Prefix(const Batch& batch, const RequestRecord& row) const {
        const auto& fixture = fixtures_[row.fixture];
        return {Number(batch.id), row.label, batch.phase, batch.profile, Number(batch.width), Number(batch.round), Number(row.lane),
                fixture.key, fixture.kind, Number(fixture.prompt.size()), Number(fixture.seed), row.referenceLabel,
                Number(row.requestId), Number(row.sessionId)};
    }

    void WriteBatch(const Batch& batch, const std::vector<RequestRecord>& rows) {
        for (const auto& row : rows) {
            const auto prefix = Prefix(batch, row);
            for (size_t i = 0; i < row.tokens.size(); ++i) {
                const auto& point = row.tokens[i];
                auto fields = prefix;
                fields.insert(fields.end(), {Number(point.sequence), Number(point.id), Number(point.readNs),
                    Number(point.readNs - row.startNs), i ? Number(point.readNs - row.tokens[i - 1].readNs) : "",
                    Number(point.promptTokens), Number(point.generatedTokens), Number(point.deliveredTokens)});
                Csv(tokens_, fields);
            }
            auto fields = prefix;
            const auto& terminal = row.terminal;
            const auto& summary = terminal.summary;
            fields.insert(fields.end(), {row.startNs >= 0 ? Number(row.startNs) : "", row.acceptedNs >= 0 ? Number(row.acceptedNs) : "",
                row.tokens.empty() ? "" : Number(row.tokens.front().readNs), row.tokens.empty() ? "" : Number(row.tokens.back().readNs),
                row.terminalNs >= 0 ? Number(row.terminalNs) : "", Number(row.terminalCount), Number(row.tokens.size()),
                row.tokens.empty() ? "" : Number(static_cast<double>(row.tokens.front().readNs - row.startNs) / 1e6),
                row.terminalNs >= 0 ? Number(static_cast<double>(row.terminalNs - row.startNs) / 1e6) : "",
                Number(terminal.sequence), std::string(wire::to_string(summary.status)), std::string(wire::to_string(summary.endReason)),
                Number(summary.promptTokensProcessed), Number(summary.generatedTokens), Number(summary.eventCount),
                summary.errorCode ? std::string(wire::to_string(*summary.errorCode)) : "", summary.message,
                Number(terminal.localFailure), Number(terminal.deliveryAbandoned), Number(terminal.discardedPublishedTokens),
                Number(row.snapshot.accepted), Number(row.snapshot.finalReceived), Number(row.snapshot.terminalReceived),
                Number(row.snapshot.completionReady), Number(row.snapshot.terminalConsumed)});
            AppendDistribution(fields, Describe(Gaps(row)));
            fields.insert(fields.end(), {Number(row.passed), row.error});
            Csv(requests_, fields);
        }
    }

    void WriteSummary() {
        Expect(groups_.size() == 6 && modelLoads_ == 1, "Not all six measured configurations used the same loaded model");
        auto output = Open(args_.output / "summary.csv");
        std::vector<std::string> header{"profile", "concurrency", "measured_batches", "measured_requests", "completion_tokens",
                                        "batch_span_seconds_total", "completion_tokens_per_second_over_batch_spans"};
        for (const char* prefix : {"ttft", "request_duration", "delivery_gap"}) AppendDistributionHeader(header, prefix);
        Csv(output, header);
        Group total;
        const auto write = [&](const std::string& profile, const std::string& width, const Group& group) {
            std::vector<std::string> row{profile, width, Number(group.batches), Number(group.requests), Number(group.completionTokens),
                Number(group.spanSeconds), Number(static_cast<double>(group.completionTokens) / group.spanSeconds)};
            AppendDistribution(row, Describe(group.ttftMs));
            AppendDistribution(row, Describe(group.durationMs));
            AppendDistribution(row, Describe(group.deliveryGapMs));
            Csv(output, row);
        };
        for (const auto& [key, group] : groups_) {
            Expect(group.batches == args_.rounds && group.requests == args_.rounds * key.second &&
                       group.completionTokens == group.requests * args_.newTokens &&
                       group.deliveryGapMs.size() == group.requests * (args_.newTokens - 1) && group.spanSeconds > 0,
                   "Measured statistics include missing, extra or excluded samples");
            write(key.first, Number(key.second), group);
            total.batches += group.batches; total.requests += group.requests;
            total.completionTokens += group.completionTokens; total.spanSeconds += group.spanSeconds;
            total.ttftMs.insert(total.ttftMs.end(), group.ttftMs.begin(), group.ttftMs.end());
            total.durationMs.insert(total.durationMs.end(), group.durationMs.begin(), group.durationMs.end());
            total.deliveryGapMs.insert(total.deliveryGapMs.end(), group.deliveryGapMs.begin(), group.deliveryGapMs.end());
        }
        write("all_configurations", "", total);
    }

    Arguments args_;
    fs::path executable_;
    Clock::time_point origin_ = Clock::now();
    std::ofstream tokens_, requests_, batches_, reclamation_;
    std::vector<Fixture> fixtures_;
    std::vector<Reference> references_;
    std::shared_ptr<host::InferenceClient> client_;
    std::shared_ptr<host::InferenceModel> model_;
    std::string epoch_;
    host::InferenceClientSnapshot finalLocal_;
    size_t modelLoads_ = 0, nextBatch_ = 0;
    std::set<uint64_t> sessionIds_, requestIds_;
    std::map<std::pair<std::string, size_t>, Group> groups_;
};
} // namespace

int main(int argc, char** argv) {
    try {
        if (argc == 2 && std::string(argv[1]) == "--help") {
            std::cout << "Usage: IpcBenchmark --model <NF4 directory> --output <new unique directory>\n"
                         "                    [--backend qwen3_nf4|qwen35_nf4] [--context 2048|4096]\n"
                         "                    [--warmup 2..10000] [--rounds 1..10000] [--new-tokens 32|64]\n"
                         "Defaults: qwen3_nf4, context 2048, warmup 2, rounds 10, new tokens 32.\n"
                         "Runs one real Worker/model, eight independent width-one prompt/seed references, then\n"
                         "short (32 tokens) / mixed (32 and 1024 tokens) at concurrency 1, 2 and 4.\n"
                         "Every completed stream is compared token by token. Reference/warmup samples are excluded.\n"
                         "CSV timestamps measure client IPC token delivery, including buffered/batched reads;\n"
                         "they are not GPU kernel latency. Run serially with all other GPU work.\n";
            return 0;
        }
        aila::set_log_level(aila::LogLevel::Warning);
        Runner runner(Parse(argc, argv));
        runner.Run();
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "[FAIL] " << error.what() << '\n';
        return 1;
    }
}
