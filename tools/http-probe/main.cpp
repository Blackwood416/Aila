#ifdef _WIN32
#include <winsock2.h>
#include <process.h>
#else
#include <sys/socket.h>
#include <unistd.h>
#endif
#include <drogon/drogon.h>
#include <trantor/net/TcpConnection.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <fstream>
#include <iostream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace {
using Clock = std::chrono::steady_clock;
constexpr size_t QueueLimit = 64 * 1024;
constexpr size_t HeaderAllowance = 4096;
constexpr size_t FinalChunkBytes = 5;
constexpr int SocketSendBuffer = 16 * 1024;
constexpr size_t MaxStreams = 8;
constexpr size_t MaxRecords = 256;

class Stream;

struct Record {
    std::string status = "active";
    size_t tokens = 0;
    size_t doneEvents = 0;
    size_t terminalTransitions = 0;
    size_t submittedWireBytes = 0;
    size_t socketBytesSent = 0;
    size_t queuedUpperBound = 0;
    size_t peakQueuedUpperBound = 0;
    size_t pauseTicks = 0;
    bool timerReleased = false;
    bool producerDestroyed = false;
    bool disconnectCallbackSeen = false;

    Json::Value ToJson() const {
        Json::Value value;
        value["status"] = status;
        value["tokens"] = Json::UInt64(tokens);
        value["done_events"] = Json::UInt64(doneEvents);
        value["terminal_transitions"] = Json::UInt64(terminalTransitions);
        value["submitted_wire_bytes"] = Json::UInt64(submittedWireBytes);
        value["socket_bytes_sent"] = Json::UInt64(socketBytesSent);
        value["queued_upper_bound"] = Json::UInt64(queuedUpperBound);
        value["peak_queued_upper_bound"] = Json::UInt64(peakQueuedUpperBound);
        value["pause_ticks"] = Json::UInt64(pauseTicks);
        value["timer_released"] = timerReleased;
        value["producer_destroyed"] = producerDestroyed;
        value["disconnect_callback_seen"] = disconnectCallbackSeen;
        return value;
    }
};

struct Registry {
    mutable std::mutex mutex;
    std::unordered_map<std::string, Record> records;
    std::unordered_map<const trantor::TcpConnection*, std::shared_ptr<Stream>> active;
    std::atomic<bool> stopping{false};
    bool forcedShutdown = false;
    int minSendBuffer = 0;
    int maxSendBuffer = 0;
    size_t socketOptionFailures = 0;
    size_t socketOptionCallbacks = 0;
    size_t connectionCloseCallbacks = 0;

    template<class Function>
    void Update(const std::string& id, Function&& update) {
        std::lock_guard<std::mutex> lock(mutex);
        update(records.at(id));
    }

    std::shared_ptr<Stream> Find(const trantor::TcpConnection* connection) {
        std::lock_guard<std::mutex> lock(mutex);
        const auto found = active.find(connection);
        return found == active.end() ? nullptr : found->second;
    }

    std::vector<std::shared_ptr<Stream>> Active() {
        std::lock_guard<std::mutex> lock(mutex);
        std::vector<std::shared_ptr<Stream>> result;
        for (const auto& item : active) result.push_back(item.second);
        return result;
    }

    Json::Value ToJson() const {
        std::lock_guard<std::mutex> lock(mutex);
        Json::Value result;
        result["ok"] = true;
        result["active_producers"] = Json::UInt64(active.size());
        result["queue_limit"] = Json::UInt64(QueueLimit);
        result["header_allowance"] = Json::UInt64(HeaderAllowance);
        result["min_socket_send_buffer"] = minSendBuffer;
        result["max_socket_send_buffer"] = maxSendBuffer;
        result["socket_option_failures"] = Json::UInt64(socketOptionFailures);
        result["socket_option_callbacks"] = Json::UInt64(socketOptionCallbacks);
        result["connection_close_callbacks"] = Json::UInt64(connectionCloseCallbacks);
        result["forced_shutdown"] = forcedShutdown;
        result["stopping"] = stopping.load();
        result["records"] = Json::Value(Json::objectValue);
        for (const auto& item : records) result["records"][item.first] = item.second.ToJson();
        return result;
    }
};

std::string EncodeJson(const Json::Value& value) {
    Json::StreamWriterBuilder writer;
    writer["indentation"] = "";
    writer["emitUTF8"] = true;
    return Json::writeString(writer, value);
}

size_t ChunkBytes(size_t dataBytes) {
    size_t digits = 1;
    for (size_t remaining = dataBytes; remaining >= 16; remaining /= 16) ++digits;
    return dataBytes + digits + 4; // hexadecimal length, CRLF, data, CRLF
}

// One SSE owns one plaintext HTTP/1.1 connection. All stream calls and
// bytesSent() reads occur on that connection's IO loop. bytesSent() counts
// bytes accepted by the socket, so the accounting bounds Trantor's pending
// stream node as well as this producer's submissions, not just a local queue.
// The fixed response has no cookies, sessions, compression or other writers;
// its headers fit HeaderAllowance. Kernel buffers are separately constrained
// by SO_SNDBUF and are not counted as process heap by this metric.
class Stream final : public std::enable_shared_from_this<Stream> {
public:
    Stream(std::shared_ptr<Registry> registry, const trantor::TcpConnectionPtr& connection,
           std::string id, size_t count, size_t padding, double interval)
        : registry(registry), connection(connection), key(connection.get()),
          loop(connection->getLoop()), id(std::move(id)), count(count),
          padding(padding, 'x'), interval(interval), sentAtStart(connection->bytesSent()) {}

    ~Stream() {
        if (auto owner = registry.lock()) {
            owner->Update(id, [](Record& record) { record.producerDestroyed = true; });
        }
    }

    void Start(drogon::ResponseStreamPtr value) {
        loop->assertInLoopThread();
        stream = std::move(value);
        if (terminal) { stream.reset(); return; }
        Schedule(0.1); // Makes headers-before-first-event observable on the wire.
    }

    void Cancel(const std::string& reason) {
        loop->assertInLoopThread();
        Finish(reason, false);
    }

    void Disconnected() {
        if (auto owner = registry.lock()) {
            owner->Update(id, [](Record& record) { record.disconnectCallbackSeen = true; });
        }
        Cancel("disconnected");
    }

    void CancelOnLoop(std::string reason) {
        loop->queueInLoop([self = shared_from_this(), reason = std::move(reason)] {
            self->Cancel(reason);
        });
    }

private:
    void Schedule(double delay) {
        timer = loop->runAfter(delay, [weak = weak_from_this()] {
            if (auto self = weak.lock()) self->Tick();
        });
    }

    size_t Observe(const trantor::TcpConnectionPtr& conn) {
        const auto sent = conn->bytesSent() - sentAtStart;
        const auto pending = HeaderAllowance + submitted;
        const auto upper = sent < pending ? pending - sent : 0;
        if (auto owner = registry.lock()) {
            owner->Update(id, [&](Record& record) {
                record.submittedWireBytes = submitted;
                record.socketBytesSent = sent;
                record.queuedUpperBound = upper;
                record.peakQueuedUpperBound = std::max(record.peakQueuedUpperBound, upper);
            });
        }
        return upper;
    }

    void Tick() {
        loop->assertInLoopThread();
        timer = trantor::InvalidTimerId;
        if (terminal) return;
        auto conn = connection.lock();
        if (!conn || !conn->connected()) { Finish("disconnected", false); return; }
        const bool last = emitted == count;
        if (nextFrame.empty()) {
            Json::Value data;
            data["id"] = id;
            if (last) {
                data["count"] = Json::UInt64(emitted);
                nextFrame = "event: done\ndata: " + EncodeJson(data) + "\n\n";
            } else {
                data["index"] = Json::UInt64(emitted);
                data["text"] = u8"汉字🙂";
                data["padding"] = padding;
                nextFrame = "event: token\nid: " + id + ":" + std::to_string(emitted) +
                            "\ndata: " + EncodeJson(data) + "\n\n";
            }
        }
        const size_t frameBytes = ChunkBytes(nextFrame.size());
        if (Observe(conn) + frameBytes + FinalChunkBytes > QueueLimit) {
            if (auto owner = registry.lock()) owner->Update(id, [](Record& record) { ++record.pauseTicks; });
            Schedule(interval);
            return;
        }
        if (!stream->send(nextFrame)) { Finish("disconnected", false); return; }
        submitted += frameBytes;
        nextFrame.clear();
        Observe(conn);
        if (last) {
            if (auto owner = registry.lock()) owner->Update(id, [](Record& record) { ++record.doneEvents; });
            Finish("completed", true);
            return;
        }
        ++emitted;
        if (auto owner = registry.lock()) owner->Update(id, [](Record& record) { ++record.tokens; });
        Schedule(interval);
    }

    void Finish(const std::string& reason, bool graceful) {
        if (terminal) return;
        terminal = true; // Before any close callback can reenter.
        if (timer != trantor::InvalidTimerId) loop->invalidateTimer(timer);
        timer = trantor::InvalidTimerId;
        auto conn = connection.lock();
        if (stream) {
            submitted += FinalChunkBytes;
            stream->close();
            stream.reset();
        }
        if (conn) Observe(conn);
        if (auto owner = registry.lock()) {
            std::lock_guard<std::mutex> lock(owner->mutex);
            auto& record = owner->records.at(id);
            record.status = reason;
            ++record.terminalTransitions;
            record.timerReleased = true;
            owner->active.erase(key);
        }
        // Never call transport code under the registry mutex.
        if (conn && conn->connected()) {
            if (graceful) conn->shutdown();
            else conn->forceClose();
        }
    }

    std::weak_ptr<Registry> registry;
    std::weak_ptr<trantor::TcpConnection> connection;
    const trantor::TcpConnection* key;
    trantor::EventLoop* loop;
    std::string id;
    size_t count;
    std::string padding;
    double interval;
    size_t sentAtStart;
    size_t submitted = 0;
    size_t emitted = 0;
    bool terminal = false;
    std::string nextFrame;
    drogon::ResponseStreamPtr stream;
    trantor::TimerId timer = trantor::InvalidTimerId;
};

int Parameter(const drogon::HttpRequestPtr& request, const char* name,
              int fallback, int minimum, int maximum) {
    const auto& text = request->getParameter(name);
    if (text.empty()) return fallback;
    size_t used = 0;
    const int value = std::stoi(text, &used);
    if (used != text.size() || value < minimum || value > maximum) {
        throw std::invalid_argument(std::string("Invalid parameter: ") + name);
    }
    return value;
}

void CheckShutdown(const std::shared_ptr<Registry>& registry, Clock::time_point deadline) {
    if (registry->Active().empty()) { drogon::app().quit(); return; }
    if (Clock::now() >= deadline) {
        { std::lock_guard<std::mutex> lock(registry->mutex); registry->forcedShutdown = true; }
        drogon::app().quit();
        return;
    }
    drogon::app().getLoop()->runAfter(0.01, [registry, deadline] { CheckShutdown(registry, deadline); });
}

} // namespace

int main(int argc, char** argv) {
    try {
        int port = 18781;
        std::string reportPath;
        for (int i = 1; i < argc; ++i) {
            const std::string argument = argv[i];
            if (argument == "--port" && i + 1 < argc) port = std::stoi(argv[++i]);
            else if (argument == "--report" && i + 1 < argc) reportPath = argv[++i];
            else throw std::invalid_argument("Usage: AilaHttpProbe --port N --report PATH");
        }
        if (port < 1 || port > 65535 || reportPath.empty()) throw std::invalid_argument("Invalid port/report");
        const auto registry = std::make_shared<Registry>();
        auto& application = drogon::app();
        application.setLogLevel(trantor::Logger::kWarn).setThreadNum(2);
        application.disableSigtermHandling();
        application.setIdleConnectionTimeout(5);
        application.setClientMaxBodySize(1024);
        application.enableGzip(false);
        application.enableBrotli(false);
        application.setAfterAcceptSockOptCallback([registry](int descriptor) {
            const int requested = SocketSendBuffer;
            int actual = 0;
            int size = sizeof(actual);
#ifdef _WIN32
            const auto socket = static_cast<SOCKET>(descriptor);
#else
            const auto socket = descriptor;
#endif
            const bool ok = setsockopt(socket, SOL_SOCKET, SO_SNDBUF,
                reinterpret_cast<const char*>(&requested), sizeof(requested)) == 0 &&
                getsockopt(socket, SOL_SOCKET, SO_SNDBUF, reinterpret_cast<char*>(&actual), &size) == 0;
            std::lock_guard<std::mutex> lock(registry->mutex);
            ++registry->socketOptionCallbacks;
            if (!ok) ++registry->socketOptionFailures;
            else {
                if (registry->minSendBuffer == 0) registry->minSendBuffer = actual;
                registry->minSendBuffer = std::min(registry->minSendBuffer, actual);
                registry->maxSendBuffer = std::max(registry->maxSendBuffer, actual);
            }
        });
        application.setConnectionCallback([registry](const trantor::TcpConnectionPtr& connection) {
            if (!connection->connected()) {
                { std::lock_guard<std::mutex> lock(registry->mutex); ++registry->connectionCloseCallbacks; }
                if (auto stream = registry->Find(connection.get())) stream->Disconnected();
            }
        });
        application.registerHandler("/health", [registry](const drogon::HttpRequestPtr&,
            std::function<void(const drogon::HttpResponsePtr&)>&& callback) {
            callback(drogon::HttpResponse::newHttpJsonResponse(registry->ToJson()));
        }, {drogon::Get});
        application.registerHandler("/stats", [registry](const drogon::HttpRequestPtr&,
            std::function<void(const drogon::HttpResponsePtr&)>&& callback) {
            callback(drogon::HttpResponse::newHttpJsonResponse(registry->ToJson()));
        }, {drogon::Get});
        application.registerHandler("/sse", [registry](const drogon::HttpRequestPtr& request,
            std::function<void(const drogon::HttpResponsePtr&)>&& callback) {
            try {
                auto connection = request->getConnectionPtr().lock();
                if (!connection) return;
                const std::string id = request->getParameter("id");
                if (id.empty() || id.size() > 48 || id.find_first_not_of("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-") != std::string::npos) {
                    throw std::invalid_argument("An ASCII request id of at most 48 characters is required");
                }
                const auto count = Parameter(request, "count", 8, 1, 100000);
                const auto padding = Parameter(request, "padding", 0, 0, 16000);
                const auto interval = Parameter(request, "interval_ms", 40, 1, 1000) / 1000.0;
                std::shared_ptr<Stream> state;
                {
                    std::lock_guard<std::mutex> lock(registry->mutex);
                    if (registry->stopping || registry->active.size() >= MaxStreams ||
                        registry->records.size() >= MaxRecords || registry->records.count(id) ||
                        registry->active.count(connection.get())) {
                        throw std::invalid_argument("Probe is stopping, full, or id/connection is already owned");
                    }
                    state = std::make_shared<Stream>(registry, connection, id, count, padding, interval);
                    registry->records.emplace(id, Record{});
                    registry->active.emplace(connection.get(), state);
                }
                auto response = drogon::HttpResponse::newAsyncStreamResponse(
                    [weak = std::weak_ptr<Stream>(state)](drogon::ResponseStreamPtr stream) {
                        if (auto owner = weak.lock()) owner->Start(std::move(stream));
                    }, true);
                response->setContentTypeString("text/event-stream; charset=utf-8");
                response->addHeader("Cache-Control", "no-cache, no-transform");
                response->addHeader("X-Accel-Buffering", "no");
                callback(response);
            } catch (const std::exception& error) {
                Json::Value body;
                body["error"] = error.what();
                auto response = drogon::HttpResponse::newHttpJsonResponse(body);
                response->setStatusCode(drogon::k400BadRequest);
                callback(response);
            }
        }, {drogon::Get});
        application.registerHandler("/shutdown", [registry](const drogon::HttpRequestPtr&,
            std::function<void(const drogon::HttpResponsePtr&)>&& callback) {
            registry->stopping = true;
            for (const auto& stream : registry->Active()) stream->CancelOnLoop("shutdown");
            Json::Value body;
            body["stopping"] = true;
            callback(drogon::HttpResponse::newHttpJsonResponse(body));
            drogon::app().getLoop()->runAfter(0.1, [registry] {
                CheckShutdown(registry, Clock::now() + std::chrono::seconds(1));
            });
        }, {drogon::Post});
        application.addListener("127.0.0.1", static_cast<uint16_t>(port));
        std::cout << "Listening on 127.0.0.1:" << port << std::endl;
        application.run();
        const auto finalReport = registry->ToJson();
        std::ofstream report(reportPath, std::ios::binary);
        report << EncodeJson(finalReport) << '\n';
        report.close();
        if (!report || finalReport["active_producers"].asUInt64() != 0 || finalReport["forced_shutdown"].asBool()) return 2;
        std::cout << "Probe shutdown complete" << std::endl;
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << std::endl;
        return 1;
    }
}
