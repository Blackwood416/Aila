#pragma once

// Shared stage labels for the token-graph device profiler.
//
// The profiler itself lives in Qwen4ExpWholeTokenGraph.cpp because it owns the
// queue pointer and the event list. Layer bodies recorded from other
// translation units (the QSA body in Qwen4ExpPrefill.cpp) still need to mark
// their sub-stages, so the labels are declared here and `mark()` forwards into
// the single profiler instance. Marks from both places land in one timeline,
// which is what makes a QSA sub-stage span directly comparable with the
// whole-token group spans around it.
//
// A captured command graph exposes no per-node events, so nothing here has any
// effect unless the body is executed un-captured with
// AILA_Q4E_TOKEN_GRAPH_NODE_PROF=1 on a profiling queue
// (AILA_PROFILE_Q4E_COLD_DEVICE=1). Marks are ext_oneapi_submit_barrier calls,
// so they serialise the stream: use the per-stage shares, not the absolute
// total.

namespace aila::token_graph_stage_prof {

// Values must stay contiguous and must follow the last whole-token group label
// declared in Qwen4ExpWholeTokenGraph.cpp. A static_assert there guards drift.
enum QsaStage : int {
    kQkvProj = 8,
    kIndexerQk,
    kPrepareQkv,
    kPrepareIndexerQ,
    kCompressBlock,
    kBlockScores,
    kSelectBlocks,
    kValidateBlocks,
    kSparseAttend,
    kOProj,
    kQsaStageEnd,
};

// Records a mark for `stage`. No-op unless the profiler is enabled.
void mark(int stage);

}  // namespace aila::token_graph_stage_prof
