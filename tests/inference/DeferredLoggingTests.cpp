#include "profile/Profiling.hpp"

#include <algorithm>
#include <exception>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

using aila::LogLevel;
using aila::ScopedLogDeferral;

constexpr const char* kOverflowPrefix = "[Aila] Deferred log buffer limit reached;";

void Expect(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

bool IsOverflow(const std::string& message) {
    return message.rfind(kOverflowPrefix, 0) == 0;
}

struct Record {
    int level;
    std::string message;
    std::thread::id thread;
};

struct Capture {
    mutable std::mutex mutex;
    std::vector<Record> records;

    static void Callback(int level, const char* message, void* data) {
        auto& capture = *static_cast<Capture*>(data);
        std::lock_guard<std::mutex> lock(capture.mutex);
        capture.records.push_back({level, message, std::this_thread::get_id()});
    }

    std::vector<Record> Snapshot() const {
        std::lock_guard<std::mutex> lock(mutex);
        return records;
    }
};

struct ResetLogging {
    ResetLogging() { aila::set_log_level(LogLevel::Info); }
    ~ResetLogging() {
        aila::set_log_callback(nullptr);
        aila::set_log_level(LogLevel::Info);
    }
};

void TestNestedScopesAndRegistrationReplacement() {
    Capture previous;
    Capture current;
    ResetLogging reset;
    aila::set_log_callback(Capture::Callback, &previous);
    {
        ScopedLogDeferral outer;
        aila::log(LogLevel::Info, "outer-first");
        {
            ScopedLogDeferral inner;
            aila::log(LogLevel::Info, "inner");
        }
        Expect(previous.Snapshot().empty(), "inner scope flushed the outer scope's logs");
        aila::set_log_callback(Capture::Callback, &current);
        aila::log(LogLevel::Warning, "outer-last");
        Expect(current.Snapshot().empty(), "log callback ran before the outer scope exited");
    }
    Expect(previous.Snapshot().empty(), "queue retained an obsolete callback/user_data");
    const auto records = current.Snapshot();
    Expect(records.size() == 3, "nested scopes lost or duplicated a record");
    Expect(records[0].message == "outer-first" && records[1].message == "inner" &&
               records[2].message == "outer-last", "nested scope order changed");
    Expect(records[2].level == static_cast<int>(LogLevel::Warning),
           "deferred record lost its severity");
}

void TestCallbackReplacementDuringFlush() {
    struct Switch {
        Capture* destination;
        int calls = 0;
        static void Callback(int, const char*, void* data) {
            auto& state = *static_cast<Switch*>(data);
            ++state.calls;
            aila::set_log_callback(Capture::Callback, state.destination);
        }
    };
    Capture destination;
    Switch state{&destination};
    ResetLogging reset;
    aila::set_log_callback(Switch::Callback, &state);
    {
        ScopedLogDeferral scope;
        aila::log(LogLevel::Info, "switch");
        aila::log(LogLevel::Info, "new-registration");
    }
    const auto records = destination.Snapshot();
    Expect(state.calls == 1 && records.size() == 1 &&
               records[0].message == "new-registration",
           "flush reused a callback registration after it was replaced");
}

void TestThreadIsolationAndLegacyDelivery() {
    Capture capture;
    ResetLogging reset;
    aila::set_log_callback(Capture::Callback, &capture);
    const auto caller = std::this_thread::get_id();
    std::thread::id workerId;
    std::exception_ptr workerFailure;
    {
        ScopedLogDeferral callerScope;
        aila::log(LogLevel::Info, "caller-pending");
        std::thread worker([&] {
            try {
                workerId = std::this_thread::get_id();
                aila::log(LogLevel::Info, "worker-immediate");
                Expect(capture.Snapshot().size() == 1,
                       "another thread's scope deferred a legacy log");
                {
                    ScopedLogDeferral workerScope;
                    aila::log(LogLevel::Info, "worker-deferred");
                    Expect(capture.Snapshot().size() == 1,
                           "worker scope did not defer its own log");
                }
                Expect(capture.Snapshot().size() == 2,
                       "worker scope did not flush independently");
            } catch (...) {
                workerFailure = std::current_exception();
            }
        });
        worker.join();
        if (workerFailure) std::rethrow_exception(workerFailure);
        Expect(capture.Snapshot().size() == 2, "caller scope flushed prematurely");
    }
    const auto records = capture.Snapshot();
    Expect(records.size() == 3 && records[0].message == "worker-immediate" &&
               records[1].message == "worker-deferred" &&
               records[2].message == "caller-pending", "cross-thread delivery order changed");
    Expect(records[0].thread == workerId && records[1].thread == workerId &&
               records[2].thread == caller, "deferral moved callbacks to another thread");
}

void TestGetterReentryAfterUnlock() {
    struct State {
        std::mutex mutex;
        int value = 0;
        int observed = 0;
        int calls = 0;
        bool sawLockedState = false;

        static void Callback(int, const char*, void* data) {
            auto& state = *static_cast<State*>(data);
            ++state.calls;
            // A failed try_lock detects the getter deadlock without hanging
            // the test process when the scope ordering regresses.
            std::unique_lock<std::mutex> lock(state.mutex, std::try_to_lock);
            if (!lock.owns_lock()) {
                state.sawLockedState = true;
                return;
            }
            state.observed = state.value;
        }
    } state;
    ResetLogging reset;
    aila::set_log_callback(State::Callback, &state);
    {
        ScopedLogDeferral scope;
        std::lock_guard<std::mutex> lock(state.mutex);
        state.value = 17;
        aila::log(LogLevel::Info, "locked operation");
        Expect(state.calls == 0, "getter callback ran while the resource lock was held");
    }
    Expect(state.calls == 1 && !state.sawLockedState && state.observed == 17,
           "callback could not reenter the getter after unlock");
}

void TestThrowingCallbacksAndUnwinding() {
    struct Throwing {
        int calls = 0;
        static void Callback(int, const char*, void* data) {
            ++static_cast<Throwing*>(data)->calls;
            throw std::runtime_error("callback failure");
        }
    } state;
    Capture recovered;
    ResetLogging reset;
    aila::set_log_callback(Throwing::Callback, &state);
    {
        ScopedLogDeferral scope;
        aila::log(LogLevel::Info, "first");
        aila::log(LogLevel::Info, "second");
    }
    Expect(state.calls == 2, "throwing callback prevented later deferred deliveries");
    bool caughtOperation = false;
    try {
        ScopedLogDeferral scope;
        aila::log(LogLevel::Info, "unwinding");
        throw std::runtime_error("operation failure");
    } catch (const std::runtime_error& error) {
        caughtOperation = std::string(error.what()) == "operation failure";
    }
    Expect(caughtOperation && state.calls == 3,
           "deferred logging replaced an in-flight operation exception");
    bool legacyThrew = false;
    try { aila::log(LogLevel::Info, "legacy"); }
    catch (const std::runtime_error&) { legacyThrew = true; }
    Expect(legacyThrew, "scope support changed the unscoped exception behavior");
    aila::set_log_callback(Capture::Callback, &recovered);
    {
        ScopedLogDeferral scope;
        aila::log(LogLevel::Info, "recovered");
    }
    Expect(recovered.Snapshot().size() == 1,
           "callback exception left thread-local deferral active");
}

void TestBoundedBuffer(bool largeMessages) {
    Capture capture;
    ResetLogging reset;
    aila::set_log_callback(Capture::Callback, &capture);
    const std::string message = largeMessages ? std::string(2047, 'x') : std::string{};
    const std::size_t attempted = largeMessages ? 300 : ScopedLogDeferral::MaxBufferedMessages + 10;
    {
        ScopedLogDeferral scope;
        for (std::size_t i = 0; i < attempted; ++i) {
            aila::log(LogLevel::Info, "%s", message.c_str());
        }
        Expect(capture.Snapshot().empty(), "overflow delivered logs before unlock");
    }
    const auto records = capture.Snapshot();
    std::size_t summaries = 0;
    std::size_t payloadBytes = 0;
    std::size_t dataRecords = 0;
    for (const auto& record : records) {
        if (IsOverflow(record.message)) {
            ++summaries;
            Expect(record.level == static_cast<int>(LogLevel::Warning),
                   "overflow summary has the wrong severity");
        } else {
            ++dataRecords;
            payloadBytes += record.message.size() + 1;
            Expect(record.message == message, "overflow truncated or corrupted a queued record");
        }
    }
    Expect(dataRecords > 0 && dataRecords < attempted, "bounded queue did not drop overflow");
    Expect(dataRecords <= ScopedLogDeferral::MaxBufferedMessages &&
               payloadBytes <= ScopedLogDeferral::MaxBufferedBytes,
           "deferred output exceeded its stated bounds");
    Expect(summaries == 1 && IsOverflow(records.back().message),
           "overflow did not emit exactly one final summary");
    {
        ScopedLogDeferral scope;
        aila::log(LogLevel::Info, "after-overflow");
    }
    const auto after = capture.Snapshot();
    Expect(after.size() == records.size() + 1 && after.back().message == "after-overflow",
           "overflow state leaked into the next scope");
}

void TestRecursiveLoggingIsBounded() {
    struct Recursive {
        std::size_t calls = 0;
        std::size_t summaries = 0;
        int depth = 0;
        int maximumDepth = 0;

        static void Callback(int, const char* message, void* data) {
            auto& state = *static_cast<Recursive*>(data);
            ++state.calls;
            ++state.depth;
            state.maximumDepth = std::max(state.maximumDepth, state.depth);
            if (IsOverflow(message)) ++state.summaries;
            {
                ScopedLogDeferral nested;
                aila::log(LogLevel::Info, "callback-generated");
            }
            --state.depth;
        }
    } state;
    ResetLogging reset;
    aila::set_log_callback(Recursive::Callback, &state);
    {
        ScopedLogDeferral scope;
        aila::log(LogLevel::Info, "initial");
    }
    Expect(state.calls > 1 && state.calls <= ScopedLogDeferral::MaxBufferedMessages + 1,
           "recursive callback logging did not respect the delivery bound");
    Expect(state.maximumDepth == 1, "deferred callbacks recursively invoked themselves");
    Expect(state.summaries == 1, "recursive logging did not report overflow once");
}

void TestFilteringUsesEmissionLevel() {
    Capture capture;
    ResetLogging reset;
    aila::set_log_callback(Capture::Callback, &capture);
    {
        ScopedLogDeferral scope;
        aila::log(LogLevel::Debug, "filtered");
        aila::log(LogLevel::Info, "accepted");
        aila::set_log_level(LogLevel::Error);
    }
    const auto records = capture.Snapshot();
    Expect(records.size() == 1 && records[0].message == "accepted",
           "deferral changed filtering from emission time to delivery time");
}

} // namespace

int main() {
    try {
        TestNestedScopesAndRegistrationReplacement();
        TestCallbackReplacementDuringFlush();
        TestThreadIsolationAndLegacyDelivery();
        TestGetterReentryAfterUnlock();
        TestThrowingCallbacksAndUnwinding();
        TestBoundedBuffer(true);
        TestBoundedBuffer(false);
        TestRecursiveLoggingIsBounded();
        TestFilteringUsesEmissionLevel();
        std::cout << "Deferred logging tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Deferred logging test failure: " << error.what() << '\n';
        return 1;
    }
}
