#include "core/ExecutionTrace.hpp"
#include "utils/EnvUtils.hpp"

#include <atomic>
#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <mutex>
#include <stdexcept>
#include <system_error>
#include <thread>
#include <utility>

#ifdef _WIN32
#include <windows.h>
#else
#include <cerrno>
#include <fcntl.h>
#include <unistd.h>
#endif

namespace aila::diagnostics {
namespace {

std::atomic<bool> processFailed{false};

uint64_t process_id() noexcept {
#ifdef _WIN32
    return GetCurrentProcessId();
#else
    return static_cast<uint64_t>(getpid());
#endif
}

uint64_t thread_id() noexcept {
#ifdef _WIN32
    return GetCurrentThreadId();
#else
    // A monotonic process-local ID avoids hash collisions and zero values.
    static std::atomic<uint64_t> next{1};
    thread_local const uint64_t value = next.fetch_add(1);
    return value;
#endif
}

int64_t utc_ns() noexcept {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
}

const char* stage_name(ExecutionStage stage) noexcept {
    switch (stage) {
    case ExecutionStage::ContextInitialize: return "context_initialize";
    case ExecutionStage::ContextRelease: return "context_release";
    case ExecutionStage::DeviceShutdown: return "device_shutdown";
    case ExecutionStage::ModelLoad: return "model_load";
    case ExecutionStage::ModelDestroy: return "model_destroy";
    case ExecutionStage::SessionCreate: return "session_create";
    case ExecutionStage::SessionDestroy: return "session_destroy";
    case ExecutionStage::SessionReset: return "session_reset";
    case ExecutionStage::SessionTruncate: return "session_truncate";
    case ExecutionStage::Forward: return "forward";
    case ExecutionStage::BackendForward: return "backend_forward";
    case ExecutionStage::Sampling: return "sampling";
    case ExecutionStage::LogitsRead: return "logits_read";
    case ExecutionStage::QueueWait: return "queue_wait";
    case ExecutionStage::CopyH2DWait: return "copy_h2d_wait";
    case ExecutionStage::CopyD2HWait: return "copy_d2h_wait";
    case ExecutionStage::WorkspaceRelease: return "workspace_release";
    case ExecutionStage::UsmFree: return "usm_free";
    case ExecutionStage::DecodeBatch: return "decode_batch";
    }
    return "invalid";
}

class FileSink final : public ExecutionTraceSink {
public:
    explicit FileSink(const std::filesystem::path& path) {
#ifdef _WIN32
        handle_ = CreateFileW(path.c_str(), GENERIC_WRITE, FILE_SHARE_READ, nullptr,
                              CREATE_NEW, FILE_ATTRIBUTE_NORMAL | FILE_FLAG_WRITE_THROUGH, nullptr);
        if (handle_ == INVALID_HANDLE_VALUE)
            throw std::system_error(static_cast<int>(GetLastError()), std::system_category(),
                                    "Cannot create execution trace");
#else
        handle_ = open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0600);
        if (handle_ < 0) throw std::system_error(errno, std::generic_category(), "Cannot create execution trace");
#endif
    }
    ~FileSink() override {
#ifdef _WIN32
        if (handle_ != INVALID_HANDLE_VALUE) CloseHandle(handle_);
#else
        if (handle_ >= 0) close(handle_);
#endif
    }
    void WriteAndFlush(std::string_view record) override {
#ifdef _WIN32
        DWORD written = 0;
        if (!WriteFile(handle_, record.data(), static_cast<DWORD>(record.size()), &written, nullptr) ||
            written != record.size() || !FlushFileBuffers(handle_))
            throw std::runtime_error("Execution trace write/flush failed");
#else
        // Any short/interrupted write is a failed trace, not a retried record.
        if (write(handle_, record.data(), record.size()) != static_cast<ssize_t>(record.size()) ||
            fsync(handle_) != 0)
            throw std::runtime_error("Execution trace write/flush failed");
#endif
    }
private:
#ifdef _WIN32
    HANDLE handle_ = INVALID_HANDLE_VALUE;
#else
    int handle_ = -1;
#endif
};

} // namespace

struct ExecutionTrace::Impl {
    explicit Impl(size_t limit) : maxBytes(limit) {
        static std::atomic<uint64_t> next{1};
        id = "execution-" + std::to_string(process_id()) + "-" +
             std::to_string(utc_ns()) + "-" + std::to_string(next.fetch_add(1));
    }
    std::unique_ptr<ExecutionTraceSink> sink;
    std::filesystem::path path;
    std::string id;
    std::mutex mutex;
    std::atomic<bool> failed{false};
    std::atomic<uint64_t> nextObject{1}, nextScope{1};
    uint64_t sequence = 0;
    size_t bytes = 0;
    size_t maxBytes;
    const std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
};

ExecutionTrace::ExecutionTrace(const std::filesystem::path& directory)
    : impl_(std::make_unique<Impl>(MaxBytes)) {
    if (directory.empty()) throw std::invalid_argument("Execution trace directory is empty");
    std::filesystem::create_directories(directory);
    impl_->path = std::filesystem::absolute(directory) / (impl_->id + ".jsonl");
    impl_->sink = std::make_unique<FileSink>(impl_->path);
    Record("trace_open", 0, 0, "trace", {}, "ok");
    Check(); // Fail before DeviceState constructs a SYCL Context.
}

ExecutionTrace::ExecutionTrace(std::unique_ptr<ExecutionTraceSink> sink, size_t maxBytes)
    : impl_(std::make_unique<Impl>(maxBytes)) {
    if (!sink || maxBytes == 0 || maxBytes > MaxBytes) throw std::invalid_argument("Invalid execution trace sink/limit");
    impl_->sink = std::move(sink);
    Record("trace_open", 0, 0, "trace", {}, "ok");
    Check();
}

ExecutionTrace::~ExecutionTrace() {
    Record("trace_close", 0, 0, "trace", {}, "ok");
}

std::unique_ptr<ExecutionTrace> ExecutionTrace::FromEnvironment() {
    static const std::string directory = aila::env::read_string("AILA_EXECUTION_TRACE_DIR");
    if (directory.empty()) return {};
    return std::make_unique<ExecutionTrace>(std::filesystem::u8path(directory));
}

bool ExecutionTrace::ProcessFailed() noexcept { return processFailed.load(); }
uint64_t ExecutionTrace::NextObjectId() noexcept { return impl_->nextObject.fetch_add(1); }
uint64_t ExecutionTrace::NextScopeId() noexcept { return impl_->nextScope.fetch_add(1); }
bool ExecutionTrace::Failed() const noexcept { return impl_->failed.load(); }
const std::filesystem::path& ExecutionTrace::Path() const noexcept { return impl_->path; }

void ExecutionTrace::Check() const {
    if (Failed()) throw std::runtime_error("Execution trace write/flush failed or exceeded 64 MiB; no new device work is allowed");
}

void ExecutionTrace::Record(const char* event, uint64_t scope, uint64_t parent,
                            const char* operation, const ExecutionFields& fields,
                            const char* status) noexcept {
    if (Failed()) return;
    // Keep the lock alive through the failure latch: another writer must not
    // append into a partially written record while this call is unwinding.
    std::unique_lock<std::mutex> lock(impl_->mutex, std::defer_lock);
    try {
        lock.lock();
        if (Failed()) return;
        const char* phase = fields.phase == ExecutionPhase::Prefill ? "prefill" :
                            fields.phase == ExecutionPhase::Decode ? "decode" : "none";
        const char* memory = fields.memory == TraceMemory::Device ? "device" :
                             fields.memory == TraceMemory::PinnedHost ? "pinned_host" : "none";
        const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - impl_->start).count();
        char record[1536];
        const int length = std::snprintf(record, sizeof(record),
            "{\"schema_version\":1,\"trace_id\":\"%s\",\"seq\":%" PRIu64
            ",\"pid\":%" PRIu64 ",\"tid\":%" PRIu64 ",\"utc_ns\":%" PRId64
            ",\"steady_ns\":%" PRId64 ",\"event\":\"%s\",\"scope_id\":%" PRIu64
            ",\"parent_scope_id\":%" PRIu64 ",\"operation\":\"%s\",\"status\":\"%s\""
            ",\"model_id\":%" PRIu64 ",\"session_id\":%" PRIu64 ",\"phase\":\"%s\""
            ",\"token_count\":%" PRId64 ",\"position\":%" PRId64 ",\"target_position\":%" PRId64
            ",\"bytes\":%" PRIu64 ",\"memory\":\"%s\"}\n",
            impl_->id.c_str(), ++impl_->sequence, process_id(), thread_id(), utc_ns(),
            static_cast<int64_t>(elapsed), event, scope, parent, operation, status,
            fields.modelId, fields.sessionId, phase, fields.tokenCount, fields.position,
            fields.targetPosition, fields.bytes, memory);
        if (length <= 0 || static_cast<size_t>(length) >= sizeof(record) ||
            static_cast<size_t>(length) > impl_->maxBytes - impl_->bytes)
            throw std::runtime_error("Execution trace byte limit reached");
        impl_->sink->WriteAndFlush({record, static_cast<size_t>(length)});
        impl_->bytes += static_cast<size_t>(length);
    } catch (...) {
        impl_->failed.store(true);
        processFailed.store(true);
    }
}

thread_local ExecutionScope* ExecutionScope::current_ = nullptr;

ExecutionScope::ExecutionScope(ExecutionTrace* trace, ExecutionStage stage,
                               ExecutionFields fields) noexcept
    : trace_(trace), operation_(stage_name(stage)), fields_(fields) {
    if (!trace_) return;
    previous_ = current_;
    for (auto parent = previous_; parent; parent = parent->previous_) {
        if (parent->trace_ != trace_ || parent->completed_) continue;
        parent_ = parent->id_;
        const auto& inherited = parent->fields_;
        if (!fields_.modelId) fields_.modelId = inherited.modelId;
        if (!fields_.sessionId) fields_.sessionId = inherited.sessionId;
        if (fields_.phase == ExecutionPhase::None) fields_.phase = inherited.phase;
        if (fields_.tokenCount == -1) fields_.tokenCount = inherited.tokenCount;
        if (fields_.position == -1) fields_.position = inherited.position;
        if (fields_.targetPosition == -1) fields_.targetPosition = inherited.targetPosition;
        // Allocation size/kind never inherit from another operation.
        break;
    }
    id_ = trace_->NextScopeId();
    trace_->Record("begin", id_, parent_, operation_, fields_, "ok");
    current_ = this;
}

ExecutionScope::~ExecutionScope() {
    if (!trace_) return;
    if (!completed_) trace_->Record("end", id_, parent_, operation_, fields_, "error");
    current_ = previous_;
}

void ExecutionScope::Complete() noexcept {
    if (!trace_ || completed_) return;
    trace_->Record("end", id_, parent_, operation_, fields_, "ok");
    completed_ = true;
}

} // namespace aila::diagnostics
