#pragma once

#include "ipc/InferenceProtocol.hpp"
#include "ipc/Win32Pipe.hpp"
#include "worker/InferenceWorker.hpp"

namespace aila::worker {

// Identity is prepared with the same executable/runtime/PATH information as
// v1. This entry point creates a fresh OS-random epoch before the hello frame.
int RunInferenceWorker(HANDLE commandRead, HANDLE responseWrite, HANDLE eventWrite,
                       ipc::v2::Hello identity, std::unique_ptr<InferenceModelFactory> factory);

} // namespace aila::worker
