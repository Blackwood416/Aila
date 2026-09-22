#include <winsock2.h>

#include "server/HttpServer.hpp"
#include "server/ChatCodec.hpp"
#include "profile/Profiling.hpp"
#include "runtime/InferenceClient.hpp"
#include "utils/Tokenizer.hpp"

#include <drogon/drogon.h>
#include <trantor/net/TcpConnection.h>
#include <algorithm>
#include <condition_variable>
#include <deque>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <thread>
#include <utility>
#include <vector>

namespace aila::server {
namespace {

namespace host = runtime;
namespace wire = ipc::v2;
using Clock = std::chrono::steady_clock;
using namespace std::chrono_literals;
using Reply = std::function<void(const drogon::HttpResponsePtr&)>;
constexpr size_t MaxFrameBytes = 16 * 1024;
constexpr size_t ClosingChunkBytes = 5;
constexpr size_t MaxConnections = 128;
constexpr size_t RecentRecords = 64;
constexpr int SocketSendBuffer = 16 * 1024;

std::string JsonText(const Json::Value& value) {
    Json::StreamWriterBuilder writer;
    writer["indentation"] = "";
    writer["emitUTF8"] = true;
    return Json::writeString(writer, value);
}

size_t ChunkBytes(size_t dataBytes) {
    size_t digits = 1;
    for (size_t count = dataBytes; count >= 16; count /= 16) ++digits;
    return dataBytes + digits + 4;
}

drogon::HttpResponsePtr JsonResponse(std::string body, int status = 200) {
    auto response = drogon::HttpResponse::newHttpResponse();
    response->setStatusCode(static_cast<drogon::HttpStatusCode>(status));
    response->setContentTypeString("application/json; charset=utf-8");
    response->addHeader("Cache-Control", "no-store");
    response->setBody(std::move(body));
    return response;
}

ApiError ClientError(const host::InferenceClientError& error) {
    if (error.Code() == host::ClientErrorCode::Timeout)
        return {504, "inference_timeout", "Inference command timed out"};
    const auto code = error.RemoteCode();
    if (error.Code() == host::ClientErrorCode::ResourceLimit || code == wire::ErrorCode::ResourceLimit || code == wire::ErrorCode::Busy)
        return {429, "server_busy", "Inference capacity is currently occupied"};
    if (code == wire::ErrorCode::ContextOverflow)
        return {400, "context_length_exceeded", "The request exceeds the Session capacity"};
    if (code == wire::ErrorCode::InvalidArgument || code == wire::ErrorCode::UnsupportedModel)
        return {400, "invalid_request", std::string(error.what()).substr(0, 1024)};
    return {503, "worker_unavailable", std::string(error.what()).substr(0, 1024)};
}

class JobPool final {
public:
    explicit JobPool(size_t count) : limit(count) {
        try {
            for (size_t index = 0; index < count; ++index) threads.emplace_back([this] { Run(); });
        } catch (...) { Stop(); throw; }
    }
    ~JobPool() { Stop(); }
    bool Post(std::function<void()> job) {
        {
            std::lock_guard<std::mutex> lock(mutex);
            if (stopping || jobs.size() >= limit) return false;
            jobs.push_back(std::move(job));
        }
        changed.notify_one();
        return true;
    }
    void Stop() noexcept {
        { std::lock_guard<std::mutex> lock(mutex); stopping = true; }
        changed.notify_all();
        for (auto& thread : threads) if (thread.joinable()) thread.join();
        threads.clear();
    }
private:
    void Run() noexcept {
        for (;;) {
            std::function<void()> job;
            {
                std::unique_lock<std::mutex> lock(mutex);
                changed.wait(lock, [&] { return stopping || !jobs.empty(); });
                if (jobs.empty()) return;
                job = std::move(jobs.front());
                jobs.pop_front();
            }
            try { job(); } catch (...) { /* Each admitted job owns its cleanup. */ }
        }
    }
    const size_t limit;
    std::mutex mutex;
    std::condition_variable changed;
    std::deque<std::function<void()>> jobs;
    std::vector<std::thread> threads;
    bool stopping = false;
};

class Service;

class RequestRun final : public std::enable_shared_from_this<RequestRun> {
public:
    RequestRun(std::shared_ptr<Service> owner, const trantor::TcpConnectionPtr& connection,
               Reply reply, std::string body, ChatIdentity identity);
    void Execute() noexcept;
    void ArmDeadline();
    void Cancel(const std::string& reason);
    void Disconnected();
    void JobFinished();
    const std::string& Id() const { return identity.id; }
    const trantor::TcpConnection* Key() const { return key; }
    bool Reclaimable() const { return jobDone.load() && ioDone.load(); }

private:
    bool Alive();
    bool Offer(std::string frame);
    bool OfferAll(std::vector<std::string> frames);
    void StartStream(drogon::ResponseStreamPtr value);
    void SendHeaders();
    void SendJson(std::string body, int status);
    void SendError(const ApiError& error, ChatOutput* output);
    void Pump();
    void SchedulePump();
    void EndStream();
    void DrainOnLoop();
    void CloseOnLoop(bool force);
    void MarkIoDone();
    std::string CancellationReason() const;

    std::weak_ptr<Service> owner;
    std::weak_ptr<trantor::TcpConnection> connection;
    const trantor::TcpConnection* key;
    trantor::EventLoop* loop;
    Reply reply;
    std::string body;
    ChatIdentity identity;
    Clock::time_point deadline;
    mutable std::mutex mutex;
    std::condition_variable changed;
    std::string stagedFrame;
    std::string cancellationReason;
    bool framePending = false;
    std::atomic<bool> cancelled{false}, jobDone{false}, ioDone{false}, ioFailed{false}, deliveryComplete{false};
    bool responseStarted = false; // Request job only.
    bool streaming = false;       // Request job only.
    // These fields are accessed only on the connection's I/O loop.
    drogon::ResponseStreamPtr stream;
    trantor::TimerId timer = trantor::InvalidTimerId;
    trantor::TimerId deadlineTimer = trantor::InvalidTimerId;
    Clock::time_point stalledAt{}, drainDeadline{};
    bool draining = false;
};

struct Record {
    std::string id;
    std::string status = "running";
    std::optional<uint64_t> generated;
    uint64_t peakPending = 0, pauses = 0;
    Json::Value ToJson() const {
        Json::Value value;
        value["id"] = id;
        value["status"] = status;
        value["generated_tokens"] = generated ? Json::Value(Json::UInt64(*generated)) : Json::Value();
        value["peak_pending_bytes"] = Json::UInt64(peakPending);
        value["pause_ticks"] = Json::UInt64(pauses);
        return value;
    }
};

class Service final : public std::enable_shared_from_this<Service> {
public:
    Service(ServerOptions options, std::shared_ptr<host::InferenceClient> client,
            std::shared_ptr<host::InferenceModel> model, std::shared_ptr<const Tokenizer> tokenizer)
        : options(std::move(options)), client(std::move(client)), model(std::move(model)),
          tokenizer(std::move(tokenizer)), pool(this->options.maxSessions) {
        chat.modelId = this->options.modelId;
        chat.maxContextTokens = this->options.contextTokens;
        chat.maxPromptTokens = std::min<uint32_t>(4096, this->options.contextTokens);
        chat.maxNewTokens = this->options.maxNewTokens;
        chat.maxOutputBytes = this->options.maxOutputBytes;
        chat.family = this->options.backend == "qwen3_nf4" ? ModelFamily::Qwen3Dense : ModelFamily::Qwen35Hybrid;
    }
    ~Service() { StopMemorySampling(); }

    void StartMemorySampling() {
        if (options.memorySampleInterval == 0ms) return;
        memorySamplingStopped = false; // Started before the shutdown thread/HTTP loop.
        memoryThread = std::thread([this] {
            std::unique_lock<std::mutex> lock(mutex);
            while (!memoryStopRequested) {
                lock.unlock();
                std::optional<wire::RuntimeMemoryInfo> sample;
                const char* error = nullptr;
                try {
                    sample = client->QueryMemory(1s);
                    if (sample->usage && (sample->usage->deviceLimitBytes != options.maxDeviceBytes ||
                        sample->usage->pinnedHostLimitBytes != options.maxPinnedHostBytes)) {
                        sample.reset();
                        error = "limit_mismatch";
                    }
                }
                catch (const host::InferenceClientError& failure) {
                    error = failure.Code() == host::ClientErrorCode::Timeout ? "timeout" : "query_failed";
                } catch (...) { error = "query_failed"; }
                lock.lock();
                memoryObservedAt = Clock::now();
                memoryInfo = std::move(sample);
                memoryError = error;
                ++memorySamples;
                if (error) {
                    ++memoryFailures;
                    // A timed-out RPC can still be pending. Never accumulate
                    // more diagnostic calls or silently retry a failed sample.
                    break;
                }
                if (memoryInfo->state == wire::RuntimeMemoryState::Busy) ++memoryBusySamples;
                if (memoryChanged.wait_for(lock, options.memorySampleInterval, [&] { return memoryStopRequested; })) break;
            }
            memorySamplingStopped = true;
        });
    }

    void StopMemorySampling() {
        { std::lock_guard<std::mutex> lock(mutex); memoryStopRequested = true; }
        memoryChanged.notify_all();
        if (memoryThread.joinable()) memoryThread.join();
    }

    void Connected(const trantor::TcpConnectionPtr& connection) {
        std::shared_ptr<RequestRun> activeRun;
        {
            std::lock_guard<std::mutex> lock(mutex);
            if (connection->connected()) {
                if (connections.emplace(connection.get(), Connection{connection, {}, 0}).second) ++connectionsAccepted;
            } else {
                const auto found = connections.find(connection.get());
                if (found != connections.end()) {
                    ++connectionsClosed;
                    activeRun = found->second.run.lock();
                    connections.erase(found);
                }
            }
        }
        idle.notify_all();
        if (activeRun) activeRun->Disconnected();
        if (stopping.load() && connection->connected()) connection->forceClose();
    }

    bool ClaimConnection(const trantor::TcpConnectionPtr& connection) {
        std::shared_ptr<RequestRun> activeRun;
        bool first = false;
        {
            std::lock_guard<std::mutex> lock(mutex);
            const auto found = connections.find(connection.get());
            if (found != connections.end()) {
                first = ++found->second.requests == 1;
                if (!first) { ++reusedRejected; activeRun = found->second.run.lock(); }
            }
        }
        if (!first) {
            if (activeRun) activeRun->Cancel("connection_reused");
            connection->forceClose();
        }
        return first;
    }

    void SocketOptions(int descriptor) {
        const int requested = SocketSendBuffer;
        int actual = 0, size = sizeof(actual);
        const auto socket = static_cast<SOCKET>(descriptor);
        const bool success = setsockopt(socket, SOL_SOCKET, SO_SNDBUF,
            reinterpret_cast<const char*>(&requested), sizeof(requested)) == 0 &&
            getsockopt(socket, SOL_SOCKET, SO_SNDBUF, reinterpret_cast<char*>(&actual), &size) == 0;
        std::lock_guard<std::mutex> lock(mutex);
        ++socketOptionCallbacks;
        if (!success) ++socketOptionFailures;
        else { minSendBuffer = minSendBuffer == 0 ? actual : std::min(minSendBuffer, actual); maxSendBuffer = std::max(maxSendBuffer, actual); }
    }

    void Handle(const drogon::HttpRequestPtr& request, Reply callback) {
        if (stopping.load() || client->Snapshot().status != wire::DomainStatus::Ready)
            throw ApiError(503, "worker_unavailable", "The inference service is not ready");
        auto connection = request->getConnectionPtr().lock();
        if (!connection || !connection->connected()) return;
        const auto& contentType = request->getHeader("content-type");
        const auto delimiter = contentType.find(';');
        if (contentType.substr(0, delimiter) != "application/json")
            throw ApiError(415, "unsupported_media_type", "Use application/json");
        if (!request->getHeader("content-encoding").empty() && request->getHeader("content-encoding") != "identity")
            throw ApiError(415, "unsupported_content_encoding", "Compressed request bodies are not supported");
        if (request->body().size() > options.maxBodyBytes)
            throw ApiError(413, "request_too_large", "The JSON body exceeds the configured limit");
        ChatIdentity identity;
        identity.id = "chatcmpl-" + client->Snapshot().epoch.substr(0, 12) + "-" + std::to_string(++nextId);
        identity.model = options.modelId;
        identity.createdUnixSeconds = std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
        auto run = std::make_shared<RequestRun>(shared_from_this(), connection, std::move(callback),
                                               std::string(request->body()), std::move(identity));
        {
            std::lock_guard<std::mutex> lock(mutex);
            if (stopping.load() || active.size() >= options.maxSessions) {
                ++rejected;
                throw ApiError(429, "server_busy", "All request slots are occupied; retry after a request finishes");
            }
            const auto found = connections.find(connection.get());
            if (found == connections.end()) return;
            active.emplace(run.get(), Active{run, Record{run->Id()}});
            found->second.run = run;
            ++admitted;
        }
        try {
            run->ArmDeadline();
            if (!pool.Post([run] { run->Execute(); })) {
                Outcome(run.get(), "shutdown", {});
                run->Cancel("shutdown");
                run->JobFinished();
            }
        } catch (...) {
            Outcome(run.get(), "failed", {});
            run->Cancel("dispatch_failed");
            run->JobFinished();
            // The connection is explicitly closed. Do not attempt a second
            // HTTP callback after synchronous forceClose has already won.
        }
    }

    void Observe(const RequestRun* run, size_t pending, size_t handoff, bool paused) {
        std::lock_guard<std::mutex> lock(mutex);
        if (pending != std::numeric_limits<size_t>::max()) peakPending = std::max<uint64_t>(peakPending, pending);
        peakHandoff = std::max<uint64_t>(peakHandoff, handoff);
        if (paused) ++pauseTicks;
        const auto found = active.find(run);
        if (found != active.end()) {
            if (pending != std::numeric_limits<size_t>::max()) found->second.record.peakPending = std::max<uint64_t>(found->second.record.peakPending, pending);
            if (paused) ++found->second.record.pauses;
        }
    }

    void Outcome(const RequestRun* run, std::string status, std::optional<uint64_t> generated) {
        std::lock_guard<std::mutex> lock(mutex);
        const auto found = active.find(run);
        if (found == active.end() || found->second.record.status != "running") return;
        found->second.record.status = std::move(status);
        found->second.record.generated = generated;
        const auto& result = found->second.record.status;
        if (result == "completed") ++completed;
        else if (result == "failed") ++failed;
        else {
            ++cancelled;
            if (result == "disconnect") ++disconnectCancellations;
            if (result == "slow_client") ++slowCancellations;
        }
    }

    void Retire(const RequestRun* run) {
        {
            std::lock_guard<std::mutex> lock(mutex);
            const auto found = active.find(run);
            if (found == active.end() || !found->second.run->Reclaimable()) return;
            if (recent.size() == RecentRecords) recent.pop_front();
            recent.push_back(found->second.record);
            active.erase(found);
        }
        idle.notify_all();
    }

    void BeginShutdown() {
        stopping.store(true);
        std::vector<std::shared_ptr<RequestRun>> runs;
        std::vector<trantor::TcpConnectionPtr> sockets;
        {
            std::lock_guard<std::mutex> lock(mutex);
            for (const auto& item : active) runs.push_back(item.second.run);
            for (const auto& item : connections) if (auto socket = item.second.connection.lock()) sockets.push_back(std::move(socket));
        }
        for (const auto& run : runs) run->Cancel("shutdown");
        for (const auto& socket : sockets) socket->forceClose();
    }

    bool FinishShutdown() {
        StopMemorySampling();
        bool graceful = true;
        try { client->Shutdown(30s); }
        catch (const std::exception& error) { graceful = false; std::cerr << "Worker shutdown: " << error.what() << '\n'; }
        pool.Stop();
        std::unique_lock<std::mutex> lock(mutex);
        const bool idleNow = idle.wait_for(lock, options.drainTimeout, [&] { return active.empty() && connections.empty(); });
        shutdownGraceful = graceful && idleNow;
        shutdownIdle = idleNow;
        return shutdownGraceful;
    }

    Json::Value Report() const {
        const auto snapshot = client->Snapshot(); // No RPC or device lock on an HTTP loop.
        std::lock_guard<std::mutex> lock(mutex);
        Json::Value value;
        value["status"] = snapshot.status == wire::DomainStatus::Ready && !stopping.load() ? "ok" :
            stopping.load() && snapshot.status == wire::DomainStatus::Ready ? "stopping" : std::string(wire::to_string(snapshot.status));
        value["model"] = options.modelId;
        value["backend"] = options.backend;
        value["max_decode_batch_size"] = options.maxDecodeBatchSize;
        if (model->Info().memory) {
            const auto& memory = *model->Info().memory;
            auto& budget = value["memory"];
            budget["max_device_bytes"] = Json::UInt64(memory.limits.maxDeviceBytes);
            budget["max_pinned_host_bytes"] = Json::UInt64(memory.limits.maxPinnedHostBytes);
            budget["max_session_host_bytes"] = Json::UInt64(memory.limits.maxSessionHostBytes);
            budget["model_resident_device_bytes_at_load"] = Json::UInt64(memory.residentDeviceBytes);
            budget["session_host_reservation_bytes"] = Json::UInt64(memory.sessionHostReservationBytes);
            budget["session_device_reservation_bytes"] = Json::UInt64(model->Info().sessionReservationBytes);
        }
        if (options.memorySampleInterval != 0ms) {
            auto& sample = value["runtime_memory"];
            sample["interval_ms"] = Json::Int64(options.memorySampleInterval.count());
            sample["epoch"] = snapshot.epoch;
            sample["samples"] = Json::UInt64(memorySamples);
            sample["busy_samples"] = Json::UInt64(memoryBusySamples);
            sample["failures"] = Json::UInt64(memoryFailures);
            sample["sampling_stopped"] = memorySamplingStopped;
            sample["state"] = memoryError ? "error" : memoryInfo ? std::string(wire::to_string(memoryInfo->state)) : "pending";
            sample["error"] = memoryError ? Json::Value(memoryError) : Json::Value();
            sample["age_ms"] = memorySamples ? Json::Value(Json::Int64(
                std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() - memoryObservedAt).count())) : Json::Value();
            sample["memory"] = Json::Value();
            if (memoryInfo && memoryInfo->usage) {
                const auto& usage = *memoryInfo->usage;
                auto& data = sample["memory"];
                data["device_bytes"] = Json::UInt64(usage.deviceBytes);
                data["peak_device_bytes"] = Json::UInt64(usage.peakDeviceBytes);
                data["pinned_host_bytes"] = Json::UInt64(usage.pinnedHostBytes);
                data["peak_pinned_host_bytes"] = Json::UInt64(usage.peakPinnedHostBytes);
                data["device_allocations"] = Json::UInt64(usage.deviceAllocations);
                data["pinned_host_allocations"] = Json::UInt64(usage.pinnedHostAllocations);
                data["device_limit_bytes"] = Json::UInt64(usage.deviceLimitBytes);
                data["pinned_host_limit_bytes"] = Json::UInt64(usage.pinnedHostLimitBytes);
            }
        }
        value["active_requests"] = Json::UInt64(active.size());
        auto& counters = value["counters"];
        counters["admitted"] = Json::UInt64(admitted); counters["rejected"] = Json::UInt64(rejected);
        counters["completed"] = Json::UInt64(completed); counters["failed"] = Json::UInt64(failed);
        counters["cancelled"] = Json::UInt64(cancelled);
        counters["disconnect_cancellations"] = Json::UInt64(disconnectCancellations);
        counters["slow_client_cancellations"] = Json::UInt64(slowCancellations);
        auto& transport = value["transport"];
        transport["stream_limit_bytes"] = Json::UInt64(options.streamBufferBytes);
        transport["frame_limit_bytes"] = Json::UInt64(MaxFrameBytes);
        transport["peak_pending_bytes"] = Json::UInt64(peakPending);
        transport["peak_handoff_bytes"] = Json::UInt64(peakHandoff);
        transport["pause_ticks"] = Json::UInt64(pauseTicks);
        auto& ipc = value["ipc"];
        ipc["status"] = std::string(wire::to_string(snapshot.status)); ipc["epoch"] = snapshot.epoch;
        ipc["outstanding_requests"] = Json::UInt64(snapshot.outstandingRequests);
        ipc["pending_commands"] = Json::UInt64(snapshot.pendingCommands);
        auto& sockets = value["connections"];
        sockets["active"] = Json::UInt64(connections.size()); sockets["accepted"] = Json::UInt64(connectionsAccepted);
        sockets["closed"] = Json::UInt64(connectionsClosed); sockets["reused_rejected"] = Json::UInt64(reusedRejected);
        sockets["socket_option_callbacks"] = Json::UInt64(socketOptionCallbacks);
        sockets["socket_option_failures"] = Json::UInt64(socketOptionFailures);
        sockets["min_send_buffer"] = minSendBuffer; sockets["max_send_buffer"] = maxSendBuffer;
        value["recent_requests"] = Json::Value(Json::arrayValue);
        for (const auto& record : recent) value["recent_requests"].append(record.ToJson());
        value["shutdown"]["graceful"] = shutdownGraceful;
        value["shutdown"]["idle"] = shutdownIdle;
        return value;
    }

    ServerOptions options;
    ChatSettings chat;
    std::shared_ptr<host::InferenceClient> client;
    std::shared_ptr<host::InferenceModel> model;
    std::shared_ptr<const Tokenizer> tokenizer;
    std::atomic<bool> stopping{false};

private:
    struct Connection { std::weak_ptr<trantor::TcpConnection> connection; std::weak_ptr<RequestRun> run; uint64_t requests; };
    struct Active { std::shared_ptr<RequestRun> run; Record record; };
    JobPool pool;
    mutable std::mutex mutex;
    std::condition_variable idle;
    std::map<const trantor::TcpConnection*, Connection> connections;
    std::map<const RequestRun*, Active> active;
    std::deque<Record> recent;
    std::atomic<uint64_t> nextId{0};
    uint64_t admitted = 0, rejected = 0, completed = 0, failed = 0, cancelled = 0;
    uint64_t disconnectCancellations = 0, slowCancellations = 0;
    uint64_t peakPending = 0, peakHandoff = 0, pauseTicks = 0;
    uint64_t connectionsAccepted = 0, connectionsClosed = 0, reusedRejected = 0;
    uint64_t socketOptionCallbacks = 0, socketOptionFailures = 0;
    int minSendBuffer = 0, maxSendBuffer = 0;
    bool shutdownGraceful = false, shutdownIdle = false;
    // The HTTP loop holds mutex only while copying CPU state, never during RPC.
    std::thread memoryThread;
    std::condition_variable memoryChanged;
    bool memoryStopRequested = false, memorySamplingStopped = true;
    std::optional<wire::RuntimeMemoryInfo> memoryInfo;
    Clock::time_point memoryObservedAt{};
    const char* memoryError = nullptr;
    uint64_t memorySamples = 0, memoryBusySamples = 0, memoryFailures = 0;
};

RequestRun::RequestRun(std::shared_ptr<Service> service, const trantor::TcpConnectionPtr& connection,
                       Reply reply, std::string body, ChatIdentity identity)
    : owner(service), connection(connection), key(connection.get()), loop(connection->getLoop()),
      reply(std::move(reply)), body(std::move(body)), identity(std::move(identity)),
      deadline(Clock::now() + service->options.requestTimeout) {}

bool RequestRun::Alive() {
    if (!cancelled.load() && Clock::now() >= deadline) Cancel("request_timeout");
    return !cancelled.load() && !ioFailed.load();
}

void RequestRun::ArmDeadline() {
    loop->runInLoop([self = shared_from_this()] {
        const double remaining = std::max(0.001, std::chrono::duration<double>(self->deadline - Clock::now()).count());
        self->deadlineTimer = self->loop->runAfter(remaining, [weak = self->weak_from_this()] {
            if (auto run = weak.lock()) {
                run->deadlineTimer = trantor::InvalidTimerId;
                run->Cancel("request_timeout");
            }
        });
    });
}

std::string RequestRun::CancellationReason() const {
    std::lock_guard<std::mutex> lock(mutex);
    return cancellationReason.empty() ? "cancelled" : cancellationReason;
}

void RequestRun::Cancel(const std::string& reason) {
    {
        std::lock_guard<std::mutex> lock(mutex);
        if (!cancelled.load() && !deliveryComplete.load()) {
            cancellationReason = reason;
            cancelled.store(true);
        }
    }
    changed.notify_all();
    loop->runInLoop([self = shared_from_this()] { self->CloseOnLoop(true); });
}

void RequestRun::Disconnected() { Cancel("disconnect"); }

void RequestRun::JobFinished() {
    jobDone.store(true);
    if (auto service = owner.lock()) service->Retire(this);
}

bool RequestRun::Offer(std::string frame) {
    if (!Alive()) return false;
    if (frame.size() > MaxFrameBytes) throw ApiError(500, "stream_frame_too_large", "A formatted stream event exceeds the frame limit");
    const auto bytes = frame.size();
    {
        std::lock_guard<std::mutex> lock(mutex);
        if (cancelled.load()) return false;
        if (framePending) throw std::logic_error("Only one HTTP handoff frame is permitted");
        stagedFrame = std::move(frame);
        framePending = true;
    }
    if (auto service = owner.lock()) service->Observe(this, 0, bytes, false);
    // One job waits for acceptance before queuing another callback or reading
    // another IPC token. This also bounds the cross-thread loop handoff.
    loop->queueInLoop([self = shared_from_this()] { self->Pump(); });
    std::unique_lock<std::mutex> lock(mutex);
    while (framePending && !cancelled.load() && !ioFailed.load()) {
        changed.wait_for(lock, 10ms);
        if (Clock::now() >= deadline) {
            lock.unlock(); Cancel("request_timeout"); lock.lock();
        }
    }
    return !cancelled.load() && !ioFailed.load();
}

bool RequestRun::OfferAll(std::vector<std::string> frames) {
    for (auto& frame : frames) if (!Offer(std::move(frame))) return false;
    return true;
}

void RequestRun::StartStream(drogon::ResponseStreamPtr value) {
    loop->assertInLoopThread();
    stream = std::move(value);
    if (cancelled.load()) { CloseOnLoop(true); return; }
    Pump();
}

void RequestRun::SendHeaders() {
    auto response = drogon::HttpResponse::newAsyncStreamResponse(
        [weak = weak_from_this()](drogon::ResponseStreamPtr stream) {
            if (auto self = weak.lock()) self->StartStream(std::move(stream));
        }, true);
    response->setContentTypeString("text/event-stream; charset=utf-8");
    response->addHeader("Cache-Control", "no-cache, no-transform");
    response->addHeader("X-Accel-Buffering", "no");
    response->addHeader("X-Request-Id", Id());
    responseStarted = true;
    reply(response);
}

void RequestRun::SendJson(std::string text, int status) {
    auto response = JsonResponse(std::move(text), status);
    response->addHeader("X-Request-Id", Id());
    {
        std::lock_guard<std::mutex> lock(mutex);
        if (cancelled.load() || ioFailed.load()) return;
        responseStarted = true;
        deliveryComplete.store(true);
    }
    // Terminal admission and cancellation share the lock; the callback itself
    // stays outside it because socket close callbacks may reenter this object.
    try { reply(response); }
    catch (...) { ioFailed.store(true); throw; }
    // A finite non-streaming body is separately bounded by maxOutputBytes.
    // The connection deadline also releases a peer that never drains it.
    const auto timeout = owner.lock()->options.drainTimeout;
    loop->queueInLoop([self = shared_from_this(), timeout] {
        self->MarkIoDone();
        const auto weak = self->connection;
        self->loop->runAfter(std::chrono::duration<double>(timeout).count(), [weak] {
            if (auto conn = weak.lock()) if (!conn->disconnected()) conn->forceClose();
        });
    });
}

void RequestRun::SendError(const ApiError& error, ChatOutput* output) {
    if (cancelled.load()) return;
    if (!responseStarted) { SendJson(error.ToJson(), error.httpStatus); return; }
    if (streaming && output) {
        auto result = output->Fail(error);
        if (OfferAll(std::move(result.sseFrames))) EndStream();
    } else Cancel("response_failed");
}

void RequestRun::SchedulePump() {
    if (timer != trantor::InvalidTimerId) return;
    timer = loop->runAfter(0.005, [weak = weak_from_this()] {
        if (auto self = weak.lock()) { self->timer = trantor::InvalidTimerId; self->Pump(); }
    });
}

void RequestRun::Pump() {
    loop->assertInLoopThread();
    if (cancelled.load() || ioFailed.load()) { CloseOnLoop(true); return; }
    if (draining) { DrainOnLoop(); return; }
    if (!stream) return;
    auto conn = connection.lock();
    auto service = owner.lock();
    if (!conn || !conn->connected() || !service) { Cancel("disconnect"); return; }
    std::string frame;
    { std::lock_guard<std::mutex> lock(mutex); if (!framePending) return; frame = stagedFrame; }
    const size_t pending = conn->pendingWriteBytes();
    const size_t limit = service->options.streamBufferBytes;
    service->Observe(this, pending, frame.size(), false);
    // Unsupported TLS/file/synchronous-stream nodes fail closed. The getter
    // counts actual ordinary/async buffer bytes, including earlier HTTP writes.
    if (pending == std::numeric_limits<size_t>::max() || pending > limit) { Cancel("transport_limit"); return; }
    const size_t required = ChunkBytes(frame.size()) + ClosingChunkBytes;
    if (required > limit - pending) {
        service->Observe(this, pending, frame.size(), true);
        if (stalledAt == Clock::time_point{}) stalledAt = Clock::now();
        if (Clock::now() - stalledAt >= service->options.streamStallTimeout) { Cancel("slow_client"); return; }
        SchedulePump();
        return;
    }
    stalledAt = Clock::time_point{};
    {
        std::lock_guard<std::mutex> lock(mutex);
        if (cancelled.load() || ioFailed.load()) return;
        // This is the publication boundary for the already-validated frame.
        // Cancellation after a terminal reservation may close its transport,
        // but cannot replace that committed application terminal.
        if (frame == "data: [DONE]\n\n") deliveryComplete.store(true);
    }
    if (!stream->send(frame)) {
        ioFailed.store(true);
        CloseOnLoop(true);
        return;
    }
    service->Observe(this, conn->pendingWriteBytes(), 0, false);
    {
        std::lock_guard<std::mutex> lock(mutex);
        stagedFrame.clear();
        framePending = false;
    }
    changed.notify_all();
}

void RequestRun::EndStream() {
    loop->queueInLoop([self = shared_from_this()] {
        self->loop->assertInLoopThread();
        if (self->cancelled.load()) { self->CloseOnLoop(true); return; }
        self->draining = true;
        if (self->stream) { self->stream->close(); self->stream.reset(); }
        auto service = self->owner.lock();
        if (!service) { self->CloseOnLoop(true); return; }
        self->drainDeadline = Clock::now() + service->options.drainTimeout;
        if (auto conn = self->connection.lock()) conn->shutdown();
        self->DrainOnLoop();
    });
}

void RequestRun::DrainOnLoop() {
    auto conn = connection.lock();
    if (!conn || conn->disconnected()) { CloseOnLoop(false); return; }
    if (Clock::now() >= drainDeadline) { CloseOnLoop(true); return; }
    // closeWrite is a half-close, not peer EOF. Keep the connection and slot
    // until its close callback or the bounded drain deadline.
    SchedulePump();
}

void RequestRun::MarkIoDone() {
    loop->assertInLoopThread();
    if (deadlineTimer != trantor::InvalidTimerId) loop->invalidateTimer(deadlineTimer);
    deadlineTimer = trantor::InvalidTimerId;
    ioDone.store(true);
    changed.notify_all();
    if (auto service = owner.lock()) service->Retire(this);
}

void RequestRun::CloseOnLoop(bool force) {
    loop->assertInLoopThread();
    if (timer != trantor::InvalidTimerId) loop->invalidateTimer(timer);
    timer = trantor::InvalidTimerId;
    if (stream) { stream->close(); stream.reset(); }
    { std::lock_guard<std::mutex> lock(mutex); stagedFrame.clear(); framePending = false; }
    if (force) if (auto conn = connection.lock()) if (!conn->disconnected()) conn->forceClose();
    MarkIoDone();
}

void RequestRun::Execute() noexcept {
    auto service = owner.lock();
    std::shared_ptr<host::InferenceSession> session;
    std::shared_ptr<host::InferenceRequest> request;
    std::unique_ptr<ChatOutput> output;
    std::optional<uint64_t> generated;
    std::string outcome = "failed";
    try {
        if (!service || !Alive()) throw ApiError(503, "server_stopping", "Request cancelled before preparation");
        auto prepared = PrepareChat(body, *service->tokenizer, service->chat);
        body.clear(); body.shrink_to_fit();
        streaming = prepared.stream;
        output = std::make_unique<ChatOutput>(identity, prepared, service->options.maxOutputBytes);
        // Session release is asynchronous. A just-finished predecessor can
        // still own its final lifecycle task; permit a short bounded retry.
        const auto createDeadline = std::min(deadline, Clock::now() + 2s);
        while (Alive()) {
            try {
                const auto budget = std::max(1ms, std::chrono::duration_cast<std::chrono::milliseconds>(deadline - Clock::now()));
                session = service->model->CreateSession(budget);
                break;
            }
            catch (const host::InferenceClientError& error) {
                if (error.RemoteCode() != wire::ErrorCode::ResourceLimit || Clock::now() >= createDeadline) throw;
                std::this_thread::sleep_for(2ms);
            }
        }
        if (!session || !Alive()) throw ApiError(503, "request_cancelled", "Request cancelled before submit");
        const auto submitBudget = std::max(1ms, std::chrono::duration_cast<std::chrono::milliseconds>(deadline - Clock::now()));
        request = session->Submit(std::move(prepared.promptIds), prepared.options, submitBudget);
        if (Alive() && streaming) { SendHeaders(); OfferAll(output->Begin()); }
        while (Alive()) {
            const auto next = request->Read(10ms);
            if (next.status == host::InferenceReadStatus::Timeout) continue;
            if (next.status == host::InferenceReadStatus::Closed || !next.event)
                throw ApiError(502, "incomplete_inference", "Worker closed the request without a terminal event");
            if (next.event->terminal) {
                generated = next.event->summary.generatedTokens;
                auto completion = output->Finish(next.event->summary, next.event->localFailure || next.event->deliveryAbandoned);
                outcome = completion.failed ? "failed" : "completed";
                if (streaming) { if (OfferAll(std::move(completion.sseFrames))) EndStream(); }
                else SendJson(std::move(completion.json), completion.httpStatus);
                break;
            }
            auto frames = output->PushText(DecodeChatToken(*service->tokenizer, prepared, next.event->tokenId));
            if (streaming && !OfferAll(std::move(frames))) break;
        }
    } catch (const ApiError& error) {
        try { SendError(error, output.get()); } catch (...) { Cancel("response_failed"); }
    } catch (const host::InferenceClientError& error) {
        try { SendError(ClientError(error), output.get()); } catch (...) { Cancel("response_failed"); }
    } catch (const std::exception& error) {
        try { SendError(ApiError(500, "inference_error", std::string(error.what()).substr(0, 1024)), output.get()); }
        catch (...) { Cancel("response_failed"); }
    } catch (...) {
        try { SendError(ApiError(500, "inference_error", "Unknown request failure"), output.get()); }
        catch (...) { Cancel("response_failed"); }
    }
    // Release is nonblocking: the connection privately cancels/drains/ACKs.
    // No GPU resource is destroyed on an HTTP loop or socket callback.
    if (request) { request->Release(); request.reset(); }
    if (session) { session->Release(); session.reset(); }
    if (cancelled.load()) outcome = CancellationReason();
    else if (ioFailed.load()) outcome = "failed";
    if (service) service->Outcome(this, std::move(outcome), generated);
    JobFinished();
}

} // namespace

int RunHttpServer(const ServerOptions& options, std::atomic<bool>& stopRequested) {
    if (options.memorySampleInterval < 0ms || options.memorySampleInterval > 60000ms ||
        (options.memorySampleInterval != 0ms && options.memorySampleInterval < 100ms))
        throw std::invalid_argument("--memory-sample-ms must be 0 or 100..60000");
    const auto capability = wire::model_capability(options.backend);
    if (capability.empty()) throw std::invalid_argument("Unsupported server backend: " + options.backend);
    wire::Limits serverLimits;
    serverLimits.maxActiveRequests = options.maxSessions;
    wire::ValidationError batchingError;
    if (!wire::validate_decode_batch_size(options.maxDecodeBatchSize, serverLimits, batchingError))
        throw std::invalid_argument("--decode-batch-size must be 1..4 and not exceed --sessions");
    aila::set_log_level(aila::LogLevel::Warning);
    auto tokenizer = std::make_shared<Tokenizer>();
    if (!tokenizer->load(options.modelDirectory.u8string())) throw std::runtime_error("Could not load the model tokenizer");
    host::InferenceClientOptions clientOptions;
    clientOptions.limits.maxActiveRequests = options.maxSessions;
    clientOptions.maxDecodeBatchSize = options.maxDecodeBatchSize;
    clientOptions.handshakeTimeout = 10s;
    clientOptions.commandTimeout = 120s;
    clientOptions.shutdownTimeout = 30s;
    if (std::find(clientOptions.requiredCapabilities.begin(), clientOptions.requiredCapabilities.end(), capability) ==
        clientOptions.requiredCapabilities.end()) clientOptions.requiredCapabilities.emplace_back(capability);
    clientOptions.requiredCapabilities.emplace_back("allocation_limits_v1");
    if (options.memorySampleInterval != 0ms) clientOptions.requiredCapabilities.emplace_back("runtime_memory_v1");
    auto client = host::InferenceClient::Start(options.workerExecutable.parent_path(), options.workerExecutable,
                                              options.expectedBuildId, clientOptions);
    host::InferenceModelOptions modelOptions;
    modelOptions.maxSequenceLength = options.contextTokens;
    modelOptions.maxSessions = options.maxSessions;
    modelOptions.allocationLimits = wire::AllocationLimits{
        options.maxDeviceBytes, options.maxPinnedHostBytes, options.maxSessionHostBytes};
    std::shared_ptr<host::InferenceModel> model;
    if (options.backend == "qwen3_nf4") model = client->LoadQwen3(options.modelDirectory.u8string(), modelOptions);
    else if (options.backend == "qwen35_nf4") model = client->LoadQwen35(options.modelDirectory.u8string(), modelOptions);
    else model = client->LoadQwen35Bf16(options.modelDirectory.u8string(), modelOptions);
    if (tokenizer->vocab_size() <= 0 || static_cast<uint32_t>(tokenizer->vocab_size()) > model->Info().vocabSize)
        throw std::runtime_error("Tokenizer vocabulary does not fit the loaded model");
    auto service = std::make_shared<Service>(options, client, model, tokenizer);
    auto& app = drogon::app();
    app.setLogLevel(trantor::Logger::kWarn).setThreadNum(2);
    app.disableSigtermHandling();
    app.enableGzip(false).enableBrotli(false).enableRequestStream(false);
    app.setClientMaxBodySize(options.maxBodyBytes).setClientMaxMemoryBodySize(options.maxBodyBytes);
    app.setMaxConnectionNum(MaxConnections).setMaxConnectionNumPerIP(MaxConnections);
    app.setIdleConnectionTimeout(static_cast<size_t>(options.requestTimeout.count() / 1000 + 10));
    app.setAfterAcceptSockOptCallback([service](int descriptor) { service->SocketOptions(descriptor); });
    app.setConnectionCallback([service](const trantor::TcpConnectionPtr& connection) { service->Connected(connection); });
    app.registerPreRoutingAdvice([service](const drogon::HttpRequestPtr& request,
        drogon::AdviceCallback&& reply, drogon::AdviceChainCallback&& next) {
        auto conn = request->getConnectionPtr().lock();
        if (!conn || !service->ClaimConnection(conn)) return;
        if (service->stopping.load()) { reply(JsonResponse(ApiError(503, "server_stopping", "Server is stopping").ToJson(), 503)); return; }
        if (request->version() != drogon::Version::kHttp11 || !request->getHeader("upgrade").empty()) {
            reply(JsonResponse(ApiError(400, "unsupported_protocol", "Use plaintext HTTP/1.1 without Upgrade").ToJson(), 400)); return;
        }
        const auto& path = request->path();
        if (path != "/health" && path != "/v1/models" && path != "/v1/chat/completions") {
            reply(JsonResponse(ApiError(404, "not_found", "Unknown endpoint").ToJson(), 404)); return;
        }
        next();
    });
    app.registerPreSendingAdvice([](const drogon::HttpRequestPtr&, const drogon::HttpResponsePtr& response) {
        // Drogon 1.9.11 skips asyncStreamCallback when its internal close flag
        // is true. Own the final close in RequestRun, including callers that
        // explicitly send Connection: close; advertise that ownership on wire.
        response->setCloseConnection(!response->asyncStreamCallback());
        response->addHeader("Connection", "close");
    });
    app.registerHandler("/health", [service](const drogon::HttpRequestPtr&, Reply&& reply) {
        auto report = service->Report();
        const int status = report["status"].asString() == "ok" ? 200 : 503;
        reply(JsonResponse(JsonText(report), status));
    }, {drogon::Get});
    app.registerHandler("/v1/models", [service](const drogon::HttpRequestPtr&, Reply&& reply) {
        Json::Value value;
        value["object"] = "list";
        Json::Value item;
        item["id"] = service->options.modelId; item["object"] = "model";
        item["created"] = Json::Int64(0); item["owned_by"] = "aila";
        value["data"] = Json::Value(Json::arrayValue); value["data"].append(item);
        reply(JsonResponse(JsonText(value)));
    }, {drogon::Get});
    app.registerHandler("/v1/chat/completions", [service](const drogon::HttpRequestPtr& request, Reply&& reply) {
        try { service->Handle(request, reply); }
        catch (const ApiError& error) { reply(JsonResponse(error.ToJson(), error.httpStatus)); }
        catch (const std::exception&) { reply(JsonResponse(ApiError(500, "request_dispatch_failed", "Could not dispatch the request").ToJson(), 500)); }
    }, {drogon::Post});
    app.registerBeginningAdvice([service] {
        Json::Value ready;
        ready["event"] = "ready"; ready["address"] = "127.0.0.1";
        ready["port"] = service->options.port; ready["model"] = service->options.modelId;
        ready["backend"] = service->options.backend;
        ready["max_decode_batch_size"] = service->options.maxDecodeBatchSize;
        ready["worker_build_id"] = service->options.expectedBuildId;
        ready["epoch"] = service->client->Snapshot().epoch;
        std::cout << JsonText(ready) << std::endl;
    });
    app.addListener("127.0.0.1", options.port);
    std::atomic<bool> appEnded{false}, shutdownSucceeded{false};
    std::thread shutdownThread;
    try {
        service->StartMemorySampling();
        shutdownThread = std::thread([&] {
        while (!stopRequested.load() && !appEnded.load()) {
            if (!options.shutdownFile.empty()) {
                std::error_code error;
                if (std::filesystem::exists(options.shutdownFile, error)) stopRequested.store(true);
            }
            std::this_thread::sleep_for(25ms);
        }
        service->BeginShutdown();
        shutdownSucceeded.store(service->FinishShutdown());
        if (!appEnded.load()) app.getLoop()->queueInLoop([&app] { app.quit(); });
        });
    } catch (...) { service->StopMemorySampling(); throw; }
    try { app.run(); }
    catch (...) { appEnded.store(true); stopRequested.store(true); shutdownThread.join(); throw; }
    appEnded.store(true);
    stopRequested.store(true);
    shutdownThread.join();
    const auto report = service->Report();
    if (!options.reportFile.empty()) {
        if (!options.reportFile.parent_path().empty()) std::filesystem::create_directories(options.reportFile.parent_path());
        std::ofstream file(options.reportFile, std::ios::binary);
        file << JsonText(report) << '\n';
        file.close();
        if (!file) throw std::runtime_error("Could not write the server shutdown report");
    }
    std::cout << "AilaServer shutdown " << (shutdownSucceeded.load() ? "complete" : "failed") << std::endl;
    return shutdownSucceeded.load() ? 0 : 2;
}

} // namespace aila::server
