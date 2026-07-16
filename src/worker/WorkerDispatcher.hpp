#pragma once

#include "ipc/IpcProtocol.hpp"

#include <memory>
#include <string>

namespace aila::worker {

class WorkerEngineApi {
public:
    virtual ~WorkerEngineApi() = default;

    virtual int init(const std::string& model, int max_seq_len) = 0;
    virtual void reset_context() = 0;
    virtual int context_length() const = 0;
    virtual int last_error_code() const = 0;
    virtual std::string last_error_message() const = 0;
};

class WorkerDispatcher {
public:
    explicit WorkerDispatcher(std::unique_ptr<WorkerEngineApi> engine);

    ipc::Frame dispatch(const ipc::Frame& request, bool& should_shutdown);

private:
    std::unique_ptr<WorkerEngineApi> engine_;
    bool initialized_ = false;
};

} // namespace aila::worker
