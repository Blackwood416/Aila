#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <string_view>

namespace aila::diagnostics {

enum class ExecutionStage {
    ContextInitialize, ContextRelease, DeviceShutdown, ModelLoad, ModelDestroy,
    SessionCreate, SessionDestroy, SessionReset, SessionTruncate, Forward,
    BackendForward, Sampling, LogitsRead, QueueWait, CopyH2DWait, CopyD2HWait,
    WorkspaceRelease, UsmFree, DecodeBatch
};

enum class ExecutionPhase { None, Prefill, Decode };
enum class TraceMemory { None, Device, PinnedHost };

// Zero IDs / -1 positions inherit the enclosing scope. IDs belong to this
// trace only, not to the public IPC handle namespace. No prompt/token content.
struct ExecutionFields {
    uint64_t modelId = 0;
    uint64_t sessionId = 0;
    ExecutionPhase phase = ExecutionPhase::None;
    int64_t tokenCount = -1;
    int64_t position = -1;
    int64_t targetPosition = -1;
    uint64_t bytes = 0;
    TraceMemory memory = TraceMemory::None;
};

class ExecutionTraceSink {
public:
    virtual ~ExecutionTraceSink() = default;
    // Success means the entire record and its newline were written and the
    // file flush completed. Errors may leave a partial tail; never retry it.
    virtual void WriteAndFlush(std::string_view record) = 0;
};

class ExecutionScope;

class ExecutionTrace {
public:
    static constexpr size_t MaxBytes = 64 * 1024 * 1024;

    // A unique, exclusively created execution-*.jsonl in the given directory.
    explicit ExecutionTrace(const std::filesystem::path& directory);
    // Injectable sink for CPU fault/lifetime tests; same encoder and state.
    explicit ExecutionTrace(std::unique_ptr<ExecutionTraceSink> sink,
                            size_t maxBytes = MaxBytes);
    ~ExecutionTrace();
    ExecutionTrace(const ExecutionTrace&) = delete;
    ExecutionTrace& operator=(const ExecutionTrace&) = delete;

    // Reads AILA_EXECUTION_TRACE_DIR once. Empty/unset creates no writer/file.
    static std::unique_ptr<ExecutionTrace> FromEnvironment();
    // Survives writer destruction, including a failed final trace_close flush.
    // Never reset within a process: Worker shutdown must not publish success.
    static bool ProcessFailed() noexcept;
    uint64_t NextObjectId() noexcept;
    bool Failed() const noexcept;
    void Check() const;
    const std::filesystem::path& Path() const noexcept;

private:
    friend class ExecutionScope;
    struct Impl;
    std::unique_ptr<Impl> impl_;
    uint64_t NextScopeId() noexcept;
    void Record(const char* event, uint64_t scope, uint64_t parent,
                const char* operation, const ExecutionFields& fields,
                const char* status) noexcept;
};

// Logging cannot throw while unwinding or prevent an existing queue drain or
// resource free. A failed writer is checked at safe submission boundaries.
// Explicit Complete is required: destruction alone records an error end.
class ExecutionScope {
public:
    ExecutionScope(ExecutionTrace* trace, ExecutionStage stage,
                   ExecutionFields fields = {}) noexcept;
    ~ExecutionScope();
    ExecutionScope(const ExecutionScope&) = delete;
    ExecutionScope& operator=(const ExecutionScope&) = delete;
    void Complete() noexcept;

private:
    static thread_local ExecutionScope* current_;
    ExecutionTrace* trace_ = nullptr;
    ExecutionScope* previous_ = nullptr;
    uint64_t id_ = 0;
    uint64_t parent_ = 0;
    const char* operation_ = nullptr;
    ExecutionFields fields_;
    bool completed_ = false;
};

} // namespace aila::diagnostics
