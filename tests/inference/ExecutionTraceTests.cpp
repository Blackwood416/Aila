#include "core/ExecutionTrace.hpp"
#include "simdjson.h"

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <future>
#include <iostream>
#include <iterator>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <thread>
#include <utility>
#include <vector>

namespace {

using namespace aila::diagnostics;

void Expect(bool value, const std::string& message) {
    if (!value) throw std::runtime_error(message);
}

template <typename Exception, typename Work>
void ExpectThrows(Work&& work, const std::string& message) {
    bool caught = false;
    try { work(); }
    catch (const Exception&) { caught = true; }
    Expect(caught, message);
}

class TempDirectory {
public:
    TempDirectory() {
        static std::atomic<uint64_t> serial{1};
        const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
        const auto base = std::filesystem::absolute(std::filesystem::temp_directory_path());
        for (unsigned attempt = 0; attempt < 32; ++attempt) {
            const auto candidate = base / ("aila-execution-trace-tests-" +
                std::to_string(stamp) + "-" + std::to_string(serial.fetch_add(1)));
            std::error_code error;
            if (std::filesystem::create_directory(candidate, error)) {
                path_ = candidate;
                return;
            }
            if (error && error != std::errc::file_exists)
                throw std::filesystem::filesystem_error("Create test directory", candidate, error);
        }
        throw std::runtime_error("Cannot create a unique test directory");
    }
    ~TempDirectory() {
        // Only this successfully, exclusively created test directory is owned.
        std::error_code ignored;
        std::filesystem::remove_all(path_, ignored);
    }
    TempDirectory(const TempDirectory&) = delete;
    TempDirectory& operator=(const TempDirectory&) = delete;
    const std::filesystem::path& Path() const { return path_; }
private:
    std::filesystem::path path_;
};

std::string ReadFile(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::binary);
    Expect(file.is_open(), "Cannot read trace while its writer is alive: " + path.string());
    std::string text{std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
    Expect(!file.bad(), "Trace file read failed");
    return text;
}

struct Record {
    std::string traceId, event, operation, status;
    uint64_t sequence = 0, pid = 0, tid = 0, scope = 0, parent = 0;
    int64_t utcNs = 0, steadyNs = 0;
    ExecutionFields fields;
};

uint64_t Unsigned(simdjson::dom::element object, const char* key) {
    uint64_t value = 0;
    Expect(object[key].get_uint64().get(value) == simdjson::SUCCESS,
           std::string("Missing or invalid unsigned trace field: ") + key);
    return value;
}

int64_t Signed(simdjson::dom::element object, const char* key) {
    int64_t value = 0;
    Expect(object[key].get_int64().get(value) == simdjson::SUCCESS,
           std::string("Missing or invalid signed trace field: ") + key);
    return value;
}

std::string String(simdjson::dom::element object, const char* key) {
    std::string_view value;
    Expect(object[key].get_string().get(value) == simdjson::SUCCESS,
           std::string("Missing or invalid string trace field: ") + key);
    return std::string(value);
}

std::vector<Record> ParseRecords(const std::string& text) {
    Expect(!text.empty() && text.back() == '\n', "Trace has an empty or incomplete JSONL tail");
    std::vector<Record> records;
    simdjson::dom::parser parser;
    size_t offset = 0;
    while (offset < text.size()) {
        const size_t end = text.find('\n', offset);
        Expect(end != std::string::npos && end != offset, "Trace contains an incomplete or empty line");
        simdjson::dom::element object;
        const auto error = parser.parse(text.data() + offset, end - offset).get(object);
        Expect(error == simdjson::SUCCESS, "Trace line is not a complete JSON document");
        simdjson::dom::object members;
        Expect(object.get_object().get(members) == simdjson::SUCCESS, "Trace record is not an object");
        std::set<std::string> keys;
        for (const auto member : members)
            Expect(keys.emplace(member.key).second, "Trace contains a duplicate JSON field");
        Expect(keys.size() == 20, "Unexpected execution trace schema fields");
        Expect(Unsigned(object, "schema_version") == 1, "Unexpected trace schema version");

        Record record;
        record.traceId = String(object, "trace_id");
        record.event = String(object, "event");
        record.operation = String(object, "operation");
        record.status = String(object, "status");
        record.sequence = Unsigned(object, "seq");
        record.pid = Unsigned(object, "pid");
        record.tid = Unsigned(object, "tid");
        record.scope = Unsigned(object, "scope_id");
        record.parent = Unsigned(object, "parent_scope_id");
        record.utcNs = Signed(object, "utc_ns");
        record.steadyNs = Signed(object, "steady_ns");
        record.fields.modelId = Unsigned(object, "model_id");
        record.fields.sessionId = Unsigned(object, "session_id");
        record.fields.tokenCount = Signed(object, "token_count");
        record.fields.position = Signed(object, "position");
        record.fields.targetPosition = Signed(object, "target_position");
        record.fields.bytes = Unsigned(object, "bytes");
        const auto phase = String(object, "phase");
        Expect(phase == "none" || phase == "prefill" || phase == "decode", "Invalid trace phase");
        record.fields.phase = phase == "prefill" ? ExecutionPhase::Prefill :
            phase == "decode" ? ExecutionPhase::Decode : ExecutionPhase::None;
        const auto memory = String(object, "memory");
        Expect(memory == "none" || memory == "device" || memory == "pinned_host", "Invalid memory kind");
        record.fields.memory = memory == "device" ? TraceMemory::Device :
            memory == "pinned_host" ? TraceMemory::PinnedHost : TraceMemory::None;

        Expect(record.sequence == records.size() + 1, "Trace sequence is missing, duplicated or out of order");
        Expect(!record.traceId.empty() && record.pid && record.tid && record.utcNs > 0 && record.steadyNs >= 0,
               "Trace identity or timestamp is invalid");
        Expect(record.event == "trace_open" || record.event == "begin" ||
                   record.event == "end" || record.event == "trace_close", "Unknown trace event");
        Expect(record.status == "ok" || record.status == "error", "Unknown trace status");
        if (record.event == "begin" || record.event == "end") {
            Expect(record.scope != 0 && record.operation != "trace", "Scope identity is missing");
        } else {
            Expect(record.scope == 0 && record.parent == 0 && record.operation == "trace",
                   "Trace boundary has scope metadata");
        }
        if (!records.empty()) {
            Expect(record.traceId == records.front().traceId && record.pid == records.front().pid,
                   "Records from different writers/processes were interleaved");
            Expect(record.steadyNs >= records.back().steadyNs, "Serialized trace time moved backwards");
        }
        records.push_back(std::move(record));
        offset = end + 1;
    }
    Expect(records.front().event == "trace_open", "Trace does not start with trace_open");
    return records;
}

const Record& Find(const std::vector<Record>& records, const char* event, const char* operation) {
    const Record* found = nullptr;
    for (const auto& record : records) {
        if (record.event == event && record.operation == operation) {
            Expect(!found, "Unexpected duplicate operation in test trace");
            found = &record;
        }
    }
    Expect(found != nullptr, std::string("Missing trace record: ") + event + "/" + operation);
    return *found;
}

void ExpectFields(const ExecutionFields& actual, const ExecutionFields& expected,
                  const std::string& message) {
    Expect(actual.modelId == expected.modelId && actual.sessionId == expected.sessionId &&
               actual.phase == expected.phase && actual.tokenCount == expected.tokenCount &&
               actual.position == expected.position && actual.targetPosition == expected.targetPosition &&
               actual.bytes == expected.bytes && actual.memory == expected.memory, message);
}

void ExpectCompleteScopes(const std::vector<Record>& records) {
    Expect(records.back().event == "trace_close", "Writer did not emit trace_close");
    std::map<uint64_t, const Record*> begins;
    std::set<uint64_t> ended;
    for (const auto& record : records) {
        if (record.event == "begin") {
            Expect(begins.emplace(record.scope, &record).second, "Scope ID was reused");
            if (record.parent) {
                const auto parent = begins.find(record.parent);
                Expect(parent != begins.end() && !ended.count(record.parent) &&
                           parent->second->tid == record.tid, "Scope inherited an inactive or foreign-thread parent");
            }
        } else if (record.event == "end") {
            const auto begin = begins.find(record.scope);
            Expect(begin != begins.end() && ended.emplace(record.scope).second,
                   "Scope ended more than once or without beginning");
            Expect(record.parent == begin->second->parent && record.tid == begin->second->tid &&
                       record.operation == begin->second->operation, "Scope metadata changed between begin/end");
            ExpectFields(record.fields, begin->second->fields, "Scope fields changed between begin/end");
        }
    }
    Expect(begins.size() == ended.size(), "A completed CPU trace contains an unterminated scope");
}

enum class FailureKind { None, Write, PartialWrite, Flush };

struct SinkState {
    std::mutex mutex;
    std::atomic<unsigned> active{0};
    std::atomic<bool> overlapped{false};
    size_t calls = 0;
    size_t failAt = 0;
    FailureKind failure = FailureKind::None;
    std::string text;
    std::vector<std::string> fullRecords;
};

class RecordingSink final : public ExecutionTraceSink {
public:
    explicit RecordingSink(std::shared_ptr<SinkState> state) : state_(std::move(state)) {}
    void WriteAndFlush(std::string_view record) override {
        struct ActiveCall {
            SinkState& state;
            explicit ActiveCall(SinkState& value) : state(value) {
                if (state.active.fetch_add(1) != 0) state.overlapped.store(true);
            }
            ~ActiveCall() { state.active.fetch_sub(1); }
        } active(*state_);
        // A lock in the test sink prevents undefined behavior if the writer
        // regresses; the separate active-call counter detects that regression.
        std::this_thread::yield();
        std::lock_guard<std::mutex> lock(state_->mutex);
        const bool fail = ++state_->calls == state_->failAt;
        if (fail && state_->failure == FailureKind::Write)
            throw std::runtime_error("injected write failure");
        if (fail && state_->failure == FailureKind::PartialWrite) {
            state_->text.append(record.data(), record.size() / 2);
            throw std::runtime_error("injected partial write failure");
        }
        state_->text.append(record.data(), record.size());
        state_->fullRecords.emplace_back(record);
        if (fail && state_->failure == FailureKind::Flush)
            throw std::runtime_error("injected flush failure");
    }
private:
    std::shared_ptr<SinkState> state_;
};

std::unique_ptr<ExecutionTraceSink> Sink(const std::shared_ptr<SinkState>& state) {
    return std::make_unique<RecordingSink>(state);
}

void TestRealFileIsReadableBeforeDestruction() {
    TempDirectory directory;
    std::filesystem::path path;
    {
        ExecutionTrace trace(directory.Path());
        path = trace.Path();
        Expect(path.is_absolute() && path.extension() == ".jsonl", "Trace path is not an absolute JSONL file");
        Expect(ParseRecords(ReadFile(path)).size() == 1, "trace_open was not flushed at construction");
        {
            ExecutionScope scope(&trace, ExecutionStage::Forward);
            const auto begun = ParseRecords(ReadFile(path));
            Expect(begun.size() == 2 && begun.back().event == "begin", "begin requires scope destruction to become visible");
            scope.Complete();
            const auto completed = ParseRecords(ReadFile(path));
            Expect(completed.size() == 3 && completed.back().event == "end" && completed.back().status == "ok",
                   "Successful end was not flushed while the scope/writer were alive");
            scope.Complete();
        }
        Expect(ParseRecords(ReadFile(path)).size() == 3, "Complete or destruction emitted a duplicate end");
        Expect(!trace.Failed(), "Healthy file writer latched a failure");
        trace.Check();
    }
    const auto records = ParseRecords(ReadFile(path));
    Expect(records.size() == 4, "Unexpected real-file trace record count");
    ExpectCompleteScopes(records);
}

void TestUniqueFilesPreserveExistingData() {
    TempDirectory directory;
    const auto sentinel = directory.Path() / "execution-existing.jsonl";
    const std::string sentinelText = "existing diagnostic evidence\n";
    {
        std::ofstream file(sentinel, std::ios::binary);
        file << sentinelText;
        file.close();
        Expect(!file.fail(), "Cannot create existing-file fixture");
    }
    std::vector<std::unique_ptr<ExecutionTrace>> writers;
    std::map<std::filesystem::path, std::string> snapshots;
    std::set<std::string> traceIds;
    for (unsigned i = 0; i < 4; ++i) {
        auto writer = std::make_unique<ExecutionTrace>(directory.Path());
        const auto text = ReadFile(writer->Path());
        const auto records = ParseRecords(text);
        Expect(snapshots.emplace(writer->Path(), text).second, "Writers reused a filename");
        Expect(traceIds.emplace(records.front().traceId).second, "Writers reused a trace ID");
        for (const auto& snapshot : snapshots)
            Expect(ReadFile(snapshot.first) == snapshot.second, "Creating a trace overwrote a live trace");
        writers.push_back(std::move(writer));
    }
    writers.clear();
    for (auto& snapshot : snapshots) {
        snapshot.second = ReadFile(snapshot.first);
        const auto records = ParseRecords(snapshot.second);
        Expect(records.size() == 2, "An empty trace contains another writer's records");
        ExpectCompleteScopes(records);
    }
    { ExecutionTrace later(directory.Path()); }
    for (const auto& snapshot : snapshots)
        Expect(ReadFile(snapshot.first) == snapshot.second, "Creating a trace overwrote closed evidence");
    Expect(ReadFile(sentinel) == sentinelText, "A pre-existing file was modified");
}

void TestNestedFieldsAndWriterLocalParents() {
    auto state = std::make_shared<SinkState>();
    auto foreignState = std::make_shared<SinkState>();
    ExecutionFields outer, local, foreign;
    {
        ExecutionTrace trace(Sink(state));
        ExecutionTrace other(Sink(foreignState));
        outer.modelId = trace.NextObjectId();
        outer.sessionId = trace.NextObjectId();
        foreign.modelId = other.NextObjectId();
        foreign.sessionId = other.NextObjectId();
        Expect(outer.modelId == 1 && outer.sessionId == 2 && foreign.modelId == 1 && foreign.sessionId == 2,
               "Object IDs are not local to each writer");
        outer.phase = ExecutionPhase::Prefill;
        outer.tokenCount = 16;
        outer.position = 12;
        outer.targetPosition = 28;
        outer.bytes = 8192;
        outer.memory = TraceMemory::Device;
        foreign.phase = ExecutionPhase::Decode;
        foreign.tokenCount = 1;
        foreign.position = 90;
        foreign.targetPosition = 91;
        local.sessionId = trace.NextObjectId();
        local.phase = ExecutionPhase::Decode;
        local.tokenCount = 0;
        local.position = 0;
        local.targetPosition = 0;
        local.bytes = 4096;
        local.memory = TraceMemory::PinnedHost;
        ExecutionScope root(&trace, ExecutionStage::Forward, outer);
        {
            ExecutionScope foreignRoot(&other, ExecutionStage::Forward, foreign);
            {
                ExecutionScope child(&trace, ExecutionStage::BackendForward, local);
                { ExecutionScope wait(&trace, ExecutionStage::QueueWait); wait.Complete(); }
                child.Complete();
                // A completed scope can still be on the C++ stack. It must
                // not parent the next operation or supply inherited fields.
                { ExecutionScope next(&trace, ExecutionStage::Sampling); next.Complete(); }
            }
            foreignRoot.Complete();
        }
        root.Complete();
        { ExecutionScope detached(&trace, ExecutionStage::LogitsRead); detached.Complete(); }
        Expect(trace.NextObjectId() == 4 && other.NextObjectId() == 3, "Scope creation consumed object IDs");
    }
    const auto records = ParseRecords(state->text);
    const auto foreignRecords = ParseRecords(foreignState->text);
    ExpectCompleteScopes(records);
    ExpectCompleteScopes(foreignRecords);
    const auto& root = Find(records, "begin", "forward");
    const auto& child = Find(records, "begin", "backend_forward");
    const auto& wait = Find(records, "begin", "queue_wait");
    const auto& next = Find(records, "begin", "sampling");
    const auto& detached = Find(records, "begin", "logits_read");
    Expect(root.parent == 0 && child.parent == root.scope && wait.parent == child.scope && next.parent == root.scope,
           "Nested scopes selected a foreign writer or completed parent");
    ExpectFields(root.fields, outer, "Root fields were changed");
    local.modelId = outer.modelId;
    ExpectFields(child.fields, local, "Explicit zero fields or inherited model ID were lost");
    local.bytes = 0;
    local.memory = TraceMemory::None;
    ExpectFields(wait.fields, local, "Nearest-parent fields were lost or allocation metadata was inherited");
    outer.bytes = 0;
    outer.memory = TraceMemory::None;
    ExpectFields(next.fields, outer, "Completed or foreign parent supplied fields");
    Expect(detached.parent == 0, "A completed root remained a parent");
    ExpectFields(detached.fields, {}, "A completed root supplied fields");
    const auto& foreignRoot = Find(foreignRecords, "begin", "forward");
    Expect(foreignRoot.parent == 0 && foreignRoot.scope == 1 && root.scope == 1,
           "Scope IDs or parents leaked between writers");
    ExpectFields(foreignRoot.fields, foreign, "Foreign writer inherited local fields");
}

void TestConcurrentRecordsAndThreadLocalParents() {
    constexpr size_t threadCount = 4;
    constexpr size_t iterations = 24;
    auto state = std::make_shared<SinkState>();
    std::vector<std::vector<ExecutionFields>> expected(threadCount);
    std::vector<std::exception_ptr> errors(threadCount);
    {
        ExecutionTrace trace(Sink(state));
        ExecutionFields mainFields;
        mainFields.modelId = trace.NextObjectId();
        mainFields.sessionId = trace.NextObjectId();
        ExecutionScope mainScope(&trace, ExecutionStage::ModelLoad, mainFields);
        std::promise<void> start;
        const auto gate = start.get_future().share();
        std::vector<std::thread> threads;
        threads.reserve(threadCount);
        try {
            for (size_t worker = 0; worker < threadCount; ++worker) {
                threads.emplace_back([&, worker, gate] {
                    try {
                        gate.wait();
                        expected[worker].reserve(iterations);
                        for (size_t iteration = 0; iteration < iterations; ++iteration) {
                            ExecutionFields fields;
                            fields.modelId = trace.NextObjectId();
                            fields.sessionId = trace.NextObjectId();
                            fields.phase = worker % 2 ? ExecutionPhase::Decode : ExecutionPhase::Prefill;
                            fields.tokenCount = static_cast<int64_t>(iteration + 1);
                            fields.position = static_cast<int64_t>(worker + 1);
                            fields.targetPosition = fields.position + fields.tokenCount;
                            expected[worker].push_back(fields);
                            ExecutionScope forward(&trace, ExecutionStage::Forward, fields);
                            { ExecutionScope wait(&trace, ExecutionStage::QueueWait); wait.Complete(); }
                            forward.Complete();
                        }
                    } catch (...) { errors[worker] = std::current_exception(); }
                });
            }
        } catch (...) {
            start.set_value();
            for (auto& thread : threads) thread.join();
            throw;
        }
        start.set_value();
        for (auto& thread : threads) thread.join();
        for (const auto& error : errors) if (error) std::rethrow_exception(error);
        Expect(trace.NextObjectId() == 3 + threadCount * iterations * 2, "Concurrent object IDs were lost");
        mainScope.Complete();
        trace.Check();
    }
    Expect(!state->overlapped.load(), "The writer called its sink concurrently");
    const auto records = ParseRecords(state->text);
    Expect(records.size() == 4 + threadCount * iterations * 4, "Concurrent trace lost records");
    ExpectCompleteScopes(records);
    std::map<uint64_t, ExecutionFields> bySession;
    std::set<uint64_t> objectIds{1, 2};
    for (const auto& worker : expected) {
        for (const auto& fields : worker) {
            Expect(objectIds.emplace(fields.modelId).second && objectIds.emplace(fields.sessionId).second,
                   "Concurrent object IDs were reused");
            Expect(bySession.emplace(fields.sessionId, fields).second, "Concurrent sessions reused an ID");
        }
    }
    std::map<int64_t, uint64_t> workerThreads;
    std::set<uint64_t> threadIds;
    const auto& main = Find(records, "begin", "model_load");
    for (const auto& record : records) {
        if (record.event != "begin" || record.operation == "model_load") continue;
        const auto expectedFields = bySession.find(record.fields.sessionId);
        Expect(expectedFields != bySession.end(), "Worker scope inherited an unrelated session");
        ExpectFields(record.fields, expectedFields->second, "Fields leaked between CPU threads");
        Expect(record.tid != main.tid, "Worker scope has the main thread's identity");
        const auto worker = workerThreads.emplace(record.fields.position, record.tid);
        Expect(worker.second || worker.first->second == record.tid, "Worker identity changed mid-trace");
        threadIds.insert(record.tid);
        if (record.operation == "forward")
            Expect(record.parent == 0, "A worker inherited the main or another worker's active scope");
        else
            Expect(record.operation == "queue_wait" && record.parent != 0, "Worker child lost its parent");
    }
    Expect(workerThreads.size() == threadCount && threadIds.size() == threadCount, "Thread IDs are not unique");
}

class ExternalFailure final : public std::runtime_error {
public:
    ExternalFailure() : std::runtime_error("original external cleanup failure") {}
    int Marker() const { return 73; }
};

void TestExceptionUnwindingAndExplicitCompletion() {
    auto state = std::make_shared<SinkState>();
    {
        ExecutionTrace trace(Sink(state));
        bool preserved = false;
        try {
            ExecutionScope scope(&trace, ExecutionStage::Forward);
            throw ExternalFailure();
        } catch (const ExternalFailure& error) {
            preserved = error.Marker() == 73 && std::string(error.what()) == "original external cleanup failure";
        }
        Expect(preserved && !trace.Failed(), "Healthy tracing changed the original exception");
        { ExecutionScope scope(&trace, ExecutionStage::SessionReset); scope.Complete(); scope.Complete(); }
        { ExecutionScope scope(&trace, ExecutionStage::SessionTruncate); }
        trace.Check();
    }
    const auto records = ParseRecords(state->text);
    Expect(records.size() == 8, "Explicit completion or unwinding emitted duplicate ends");
    ExpectCompleteScopes(records);
    Expect(Find(records, "end", "forward").status == "error", "Unwinding was recorded as success");
    Expect(Find(records, "end", "session_reset").status == "ok", "Complete did not record success");
    Expect(Find(records, "end", "session_truncate").status == "error", "Missing Complete was recorded as success");
}

std::shared_ptr<SinkState> CaptureFinalCloseFlushFailure() {
    // ProcessFailed is monotonic. This fixture must run before the other
    // injected failures, or in its own helper process.
    Expect(!ExecutionTrace::ProcessFailed(), "Final-close fixture requires a healthy initial process latch");
    auto state = std::make_shared<SinkState>();
    state->failure = FailureKind::Flush;
    state->failAt = 4; // trace_open, begin, successful end, then trace_close.
    int cleanups = 0;
    {
        ExecutionTrace trace(Sink(state));
        {
            ExecutionScope scope(&trace, ExecutionStage::SessionDestroy);
            const auto cleanup = [&] { ++cleanups; };
            cleanup();
            scope.Complete();
        }
        Expect(cleanups == 1 && state->calls == 3 && !trace.Failed() && !ExecutionTrace::ProcessFailed(),
               "Final-close fixture failed before cleanup and its successful end");
        trace.Check();
    }
    Expect(state->calls == 4, "Final close was missing or retried after its flush failure");
    Expect(ExecutionTrace::ProcessFailed(), "Final-close flush failure disappeared with the writer");
    const auto records = ParseRecords(state->text);
    Expect(records.size() == 4 && records.back().event == "trace_close" && records.back().status == "ok",
           "Final-close fixture did not leave a fully written close record");
    ExpectCompleteScopes(records);
    return state;
}

void TestFinalCloseFlushFailureSurvivesWriter() {
    const auto failed = CaptureFinalCloseFlushFailure();
    auto healthy = std::make_shared<SinkState>();
    {
        ExecutionTrace trace(Sink(healthy));
        Expect(!trace.Failed() && ExecutionTrace::ProcessFailed(),
               "A healthy new writer cleared the process failure or inherited another writer's local failure");
        trace.Check();
    }
    ExpectCompleteScopes(ParseRecords(healthy->text));
    Expect(ExecutionTrace::ProcessFailed() && failed->calls == 4,
           "Later healthy writer destruction cleared or retried the final-close failure");
}

enum class FailureBoundary { Begin, Complete, Unwind };

void ExerciseSinkFailure(FailureKind kind, FailureBoundary boundary) {
    auto state = std::make_shared<SinkState>();
    state->failure = kind;
    state->failAt = boundary == FailureBoundary::Begin ? 2 : 3;
    {
        ExecutionTrace trace(Sink(state));
        int cleanups = 0;
        bool preserved = false;
        const auto cleanup = [&] {
            ++cleanups;
            if (boundary == FailureBoundary::Unwind) throw ExternalFailure();
        };
        try {
            ExecutionScope scope(&trace, ExecutionStage::SessionDestroy);
            cleanup();
            scope.Complete();
            scope.Complete();
        } catch (const ExternalFailure& error) {
            preserved = error.Marker() == 73 && std::string(error.what()) == "original external cleanup failure";
        }
        Expect(cleanups == 1, "Trace failure skipped or retried cleanup");
        Expect(preserved == (boundary == FailureBoundary::Unwind), "Logging replaced or swallowed the original exception");
        Expect(trace.Failed(), "Sink failure was not latched");
        bool submitted = false;
        ExpectThrows<std::runtime_error>([&] { trace.Check(); submitted = true; },
                                        "Check allowed work after a sink failure");
        Expect(!submitted, "A failed trace admitted subsequent work");
        { ExecutionScope later(&trace, ExecutionStage::QueueWait); later.Complete(); }
        ExpectThrows<std::runtime_error>([&] { trace.Check(); }, "Trace failure was cleared");
        Expect(state->calls == state->failAt, "Logging retried after a failed write/flush");
    }
    Expect(state->calls == state->failAt, "Writer destruction retried a failed sink");
    std::string completePrefix;
    for (const auto& record : state->fullRecords) completePrefix += record;
    const size_t expectedFullRecords = state->failAt - (kind == FailureKind::Flush ? 0 : 1);
    Expect(ParseRecords(completePrefix).size() == expectedFullRecords, "Failure corrupted earlier complete records");
    if (kind == FailureKind::PartialWrite) {
        Expect(state->text.size() > completePrefix.size() && state->text.back() != '\n' &&
                   state->text.compare(0, completePrefix.size(), completePrefix) == 0,
               "Partial write fixture did not retain a single incomplete tail");
    } else {
        Expect(state->text == completePrefix, "Failure caused an unexpected partial record");
    }
}

void ExerciseConcurrentSinkFailure(FailureKind kind) {
    auto state = std::make_shared<SinkState>();
    state->failure = kind;
    state->failAt = 3;
    {
        ExecutionTrace trace(Sink(state));
        std::promise<void> start;
        const auto gate = start.get_future().share();
        std::vector<std::thread> threads;
        threads.reserve(6);
        try {
            for (unsigned worker = 0; worker < 6; ++worker) {
                threads.emplace_back([&trace, gate] {
                    gate.wait();
                    for (unsigned iteration = 0; iteration < 8; ++iteration) {
                        ExecutionScope scope(&trace, ExecutionStage::Forward);
                        { ExecutionScope wait(&trace, ExecutionStage::QueueWait); wait.Complete(); }
                        scope.Complete();
                    }
                });
            }
        } catch (...) {
            start.set_value();
            for (auto& thread : threads) thread.join();
            throw;
        }
        start.set_value();
        for (auto& thread : threads) thread.join();
        Expect(trace.Failed(), "Concurrent sink failure was not latched");
        Expect(state->calls == state->failAt, "Another thread wrote between sink failure and its failure latch");
        ExpectThrows<std::runtime_error>([&] { trace.Check(); }, "Concurrent failure allowed later work");
    }
    Expect(!state->overlapped.load(), "Concurrent failure bypassed the writer's sink lock");
    Expect(state->calls == state->failAt, "A failed concurrent trace wrote a close record");
    std::string prefix;
    for (const auto& record : state->fullRecords) prefix += record;
    ParseRecords(prefix);
    if (kind == FailureKind::PartialWrite)
        Expect(state->text.size() > prefix.size() && state->text.back() != '\n',
               "Another thread appended after the incomplete tail");
}

void TestSinkFailuresPreserveCleanupAndFirstException() {
    for (const auto kind : {FailureKind::Write, FailureKind::PartialWrite, FailureKind::Flush}) {
        for (const auto boundary : {FailureBoundary::Begin, FailureBoundary::Complete, FailureBoundary::Unwind})
            ExerciseSinkFailure(kind, boundary);
        ExerciseConcurrentSinkFailure(kind);
    }
}

void TestByteCapFailsClosedWithoutRetry() {
    constexpr size_t cap = 2048;
    auto state = std::make_shared<SinkState>();
    size_t callsAtFailure = 0;
    {
        ExecutionTrace trace(Sink(state), cap);
        size_t cleanups = 0;
        while (!trace.Failed() && cleanups < 32) {
            ExecutionScope scope(&trace, ExecutionStage::WorkspaceRelease);
            ++cleanups;
            scope.Complete();
        }
        Expect(trace.Failed() && cleanups > 0 && cleanups < 32, "Byte cap was not enforced");
        Expect(!state->text.empty() && state->text.size() <= cap, "Writer exceeded its byte cap");
        callsAtFailure = state->calls;
        ExpectThrows<std::runtime_error>([&] { trace.Check(); }, "Byte cap failure allowed later work");
        { ExecutionScope later(&trace, ExecutionStage::UsmFree); later.Complete(); }
    }
    Expect(state->calls == callsAtFailure, "Byte cap failure retried the sink during cleanup");
    Expect(ParseRecords(state->text).size() == callsAtFailure, "Byte cap cut a JSON record in the sink");
}

void TestConstructionFailsBeforeWork() {
    ExpectThrows<std::invalid_argument>([] {
        ExecutionTrace trace(std::unique_ptr<ExecutionTraceSink>{});
    }, "Null sink was accepted");
    ExpectThrows<std::invalid_argument>([] {
        ExecutionTrace trace(std::filesystem::path{});
    }, "Empty directory was accepted");
    for (const size_t cap : {size_t{0}, ExecutionTrace::MaxBytes + 1}) {
        auto state = std::make_shared<SinkState>();
        ExpectThrows<std::invalid_argument>([&] { ExecutionTrace trace(Sink(state), cap); },
                                            "Invalid byte cap was accepted");
        Expect(state->calls == 0, "Invalid configuration wrote to a sink");
    }
    auto capped = std::make_shared<SinkState>();
    ExpectThrows<std::runtime_error>([&] { ExecutionTrace trace(Sink(capped), 1); },
                                    "Construction survived failure to record trace_open");
    Expect(capped->calls == 0, "Too-small initial cap passed an oversized record to the sink");
    for (const auto kind : {FailureKind::Write, FailureKind::PartialWrite, FailureKind::Flush}) {
        auto state = std::make_shared<SinkState>();
        state->failure = kind;
        state->failAt = 1;
        ExpectThrows<std::runtime_error>([&] { ExecutionTrace trace(Sink(state)); },
                                        "Construction ignored an initial sink failure");
        Expect(state->calls == 1, "Failed construction retried or closed the sink with another record");
    }
}

void TestDisabledScopesLeaveEnabledParentsUntouched() {
    auto state = std::make_shared<SinkState>();
    ExecutionFields fields;
    fields.modelId = 7;
    fields.sessionId = 8;
    fields.phase = ExecutionPhase::Decode;
    fields.tokenCount = 1;
    fields.position = 4;
    fields.targetPosition = 5;
    {
        ExecutionScope disabledRoot(nullptr, ExecutionStage::DeviceShutdown);
        disabledRoot.Complete();
        ExecutionTrace trace(Sink(state));
        ExecutionScope root(&trace, ExecutionStage::Forward, fields);
        {
            ExecutionFields unrelated;
            unrelated.modelId = 999;
            unrelated.sessionId = 1000;
            ExecutionScope disabled(nullptr, ExecutionStage::SessionReset, unrelated);
            { ExecutionScope child(&trace, ExecutionStage::QueueWait); child.Complete(); }
            disabled.Complete();
            disabled.Complete();
        }
        { ExecutionScope child(&trace, ExecutionStage::Sampling); child.Complete(); }
        root.Complete();
        trace.Check();
    }
    const auto records = ParseRecords(state->text);
    Expect(records.size() == 8, "Disabled scopes emitted trace records");
    ExpectCompleteScopes(records);
    const auto& root = Find(records, "begin", "forward");
    const auto& during = Find(records, "begin", "queue_wait");
    const auto& after = Find(records, "begin", "sampling");
    Expect(root.scope == 1 && during.scope == 2 && after.scope == 3 &&
               during.parent == root.scope && after.parent == root.scope,
           "Disabled scopes changed IDs or the enabled parent stack");
    ExpectFields(during.fields, fields, "Disabled scope supplied inherited fields");
    ExpectFields(after.fields, fields, "Disabled scope destruction erased the enabled parent");
}

ExecutionFields HelperFields(ExecutionTrace& trace) {
    ExecutionFields fields;
    fields.modelId = trace.NextObjectId();
    fields.sessionId = trace.NextObjectId();
    fields.phase = ExecutionPhase::Decode;
    fields.tokenCount = 1;
    fields.position = 8;
    fields.targetPosition = 9;
    return fields;
}

void EmitTrace(const std::filesystem::path& directory) {
    std::filesystem::path path;
    {
        ExecutionTrace trace(directory);
        path = trace.Path();
        const auto fields = HelperFields(trace);
        {
            ExecutionScope forward(&trace, ExecutionStage::Forward, fields);
            {
                ExecutionScope backend(&trace, ExecutionStage::BackendForward);
                { ExecutionScope wait(&trace, ExecutionStage::QueueWait); wait.Complete(); }
                backend.Complete();
            }
            { ExecutionScope sampling(&trace, ExecutionStage::Sampling); sampling.Complete(); }
            forward.Complete();
        }
        ExecutionFields session;
        session.modelId = fields.modelId;
        session.sessionId = fields.sessionId;
        {
            ExecutionScope destroy(&trace, ExecutionStage::SessionDestroy, session);
            { ExecutionScope wait(&trace, ExecutionStage::QueueWait); wait.Complete(); }
            {
                ExecutionScope release(&trace, ExecutionStage::WorkspaceRelease);
                ExecutionFields allocation;
                allocation.bytes = 4096;
                allocation.memory = TraceMemory::Device;
                { ExecutionScope free(&trace, ExecutionStage::UsmFree, allocation); free.Complete(); }
                release.Complete();
            }
            destroy.Complete();
        }
        trace.Check();
    }
    ExpectCompleteScopes(ParseRecords(ReadFile(path)));
    std::cout << "TRACE_PATH=" << path.string() << std::endl;
}

int EmitCloseFailureTrace(const std::filesystem::path& directory) {
    const auto state = CaptureFinalCloseFlushFailure();
    const auto records = ParseRecords(state->text);
    std::filesystem::create_directories(directory);
    const auto path = std::filesystem::absolute(directory) / (records.front().traceId + ".jsonl");
    Expect(!std::filesystem::exists(path), "Close-failure helper refuses to overwrite existing evidence");
    // Save the exact bytes accepted by the injected sink, including the close
    // whose flush reported failure. This is fixture export, not a sink retry.
    std::ofstream output(path, std::ios::binary);
    output.write(state->text.data(), static_cast<std::streamsize>(state->text.size()));
    output.flush();
    Expect(output.good(), "Could not export close-failure fixture bytes");
    output.close();
    Expect(!output.fail(), "Could not close the exported fixture file");
    Expect(ReadFile(path) == state->text, "Exported close-failure evidence changed");
    std::cout << "TRACE_PATH=" << path.string() << std::endl;
    return ExecutionTrace::ProcessFailed() ? 5 : 0;
}

[[noreturn]] void EmitInterruptedTrace(const std::filesystem::path& directory) {
    ExecutionTrace trace(directory);
    ExecutionScope forward(&trace, ExecutionStage::Forward, HelperFields(trace));
    ExecutionScope wait(&trace, ExecutionStage::QueueWait);
    trace.Check();
    std::cout << "TRACE_PATH=" << trace.Path().string() << std::endl;
    // CPU-only interruption fixture: no scope/writer destructors and no close
    // record. The analyzer must still see both already-flushed begin records.
    std::_Exit(0);
}

} // namespace

int main(int argc, char** argv) {
    if (argc != 1) {
        if (argc != 3 || (std::string_view(argv[1]) != "--emit-trace" &&
                          std::string_view(argv[1]) != "--emit-close-failure-trace" &&
                          std::string_view(argv[1]) != "--emit-interrupted-trace")) {
            std::cerr << "Usage: AilaExecutionTraceTests [--emit-trace|--emit-close-failure-trace|--emit-interrupted-trace <directory>]\n";
            return 2;
        }
        try {
            const auto directory = std::filesystem::u8path(argv[2]);
            if (std::string_view(argv[1]) == "--emit-interrupted-trace") EmitInterruptedTrace(directory);
            if (std::string_view(argv[1]) == "--emit-close-failure-trace") return EmitCloseFailureTrace(directory);
            EmitTrace(directory);
            return 0;
        } catch (const std::exception& error) {
            std::cerr << "Trace helper failed: " << error.what() << '\n';
            return 1;
        }
    }
    const std::vector<std::pair<const char*, void (*)()>> tests{
        {"real trace records are readable before writer destruction", TestRealFileIsReadableBeforeDestruction},
        {"unique files preserve existing diagnostic evidence", TestUniqueFilesPreserveExistingData},
        {"nested fields and parents remain writer-local", TestNestedFieldsAndWriterLocalParents},
        {"concurrent JSONL records have ordered sequences and thread-local parents", TestConcurrentRecordsAndThreadLocalParents},
        {"unwinding is an error and Complete emits exactly one successful end", TestExceptionUnwindingAndExplicitCompletion},
        {"final-close flush failure survives writer destruction and healthy later writers", TestFinalCloseFlushFailureSurvivesWriter},
        {"sink failures preserve cleanup and the original exception", TestSinkFailuresPreserveCleanupAndFirstException},
        {"byte cap blocks later work without retrying the sink", TestByteCapFailsClosedWithoutRetry},
        {"construction fails before work on invalid configuration or trace_open failure", TestConstructionFailsBeforeWork},
        {"disabled scopes leave enabled parents and fields untouched", TestDisabledScopesLeaveEnabledParentsUntouched},
    };
    size_t failures = 0;
    for (const auto& test : tests) {
        try {
            test.second();
            std::cout << "PASS: " << test.first << std::endl;
        } catch (const std::exception& error) {
            ++failures;
            std::cerr << "FAIL: " << test.first << ": " << error.what() << std::endl;
        } catch (...) {
            ++failures;
            std::cerr << "FAIL: " << test.first << ": non-standard exception\n";
        }
    }
    std::cout << "AilaExecutionTraceTests: " << tests.size() - failures << " passed, " << failures << " failed\n";
    return failures ? 1 : 0;
}
