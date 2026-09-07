#pragma once

#include "inference/Session.hpp"
#include "ipc/IpcProtocol.hpp"
#include "ipc/InferenceProtocol.hpp"

#include <chrono>
#include <functional>
#include <memory>
#include <string>
#include <string_view>

namespace aila::worker {

struct InferenceModelOptions {
    int maxSequenceLength = 4096;
    size_t maxSessions = 4;
    size_t maxSessionCacheBytes = 0;
};

// The protocol/controller is ordinary C++. Only the real factory adapter
// includes Model.hpp and initializes a SYCL device. Tests supply CPU sessions.
class InferenceModel {
public:
    virtual ~InferenceModel() = default;
    virtual std::shared_ptr<inference::SessionState> CreateSession() = 0;
    virtual int VocabSize() const noexcept = 0;
    virtual int MaxSequenceLength() const noexcept = 0;
    virtual size_t SessionReservationBytes() const noexcept = 0;
};

class InferenceModelFactory {
public:
    virtual ~InferenceModelFactory() = default;
    virtual std::shared_ptr<InferenceModel> LoadQwen3(
        const std::string& directory, InferenceModelOptions options) = 0;
    // Called on the lifecycle executor after resource destruction. A recorded
    // asynchronous error must propagate even when a destructor could not throw.
    virtual std::string RuntimeFaultMessage() const { return {}; }
};

std::unique_ptr<InferenceModelFactory> MakeSharedInferenceModelFactory();

struct InferenceWorkerOutput {
    ipc::Frame frame;
    // The dedicated writer calls this exactly once after a successful complete
    // frame write. Its own in-flight slot remains reserved until then.
    std::function<void()> written;
};

// One control/bridge thread, one lifecycle executor and one Scheduler. Pipe
// writers/readers live outside this class; no IO or user callback runs on the
// GPU submission thread. Enqueue never waits for a GPU step or an output reader.
class InferenceWorker final {
public:
    InferenceWorker(std::string epoch, std::unique_ptr<InferenceModelFactory> factory);
    ~InferenceWorker();
    InferenceWorker(const InferenceWorker&) = delete;
    InferenceWorker& operator=(const InferenceWorker&) = delete;

    bool Enqueue(ipc::Frame command, std::string& error);
    bool ReadResponse(InferenceWorkerOutput& output, std::chrono::milliseconds timeout);
    bool ReadEvent(InferenceWorkerOutput& output, std::chrono::milliseconds timeout);
    void Log(int level, std::string_view message) noexcept;
    void TransportFailed(std::string_view message) noexcept;
    bool Closed() const noexcept;
    bool Failed() const noexcept;
    std::string FailureMessage() const;
    ipc::v2::Limits Limits() const;
    void Stop() noexcept;

private:
    struct State;
    std::shared_ptr<State> state;
};

} // namespace aila::worker
