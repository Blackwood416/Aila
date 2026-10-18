#pragma once

#include <cstdint>

namespace aila {

// The first production graph body is intentionally narrow. WholeToken remains
// a target, not an enabled path.
enum class TokenGraphBody : std::uint8_t {
    QsaLayerProof = 0,
    WholeToken = 1,
};

enum class TokenGraphAttention : std::uint8_t {
    QsaDenseContiguous = 0,
    QsaSparse512BlockNative = 1,
};

enum class TokenGraphPhase : std::uint8_t {
    Empty = 0,
    Capturing = 1,
    Ready = 2,
    InFlight = 3,
    Aborting = 4,
    Poisoned = 5,
};

// A key is valid only for the exact object identities and generations used
// during capture. Mutable per-token control data does not belong here.
struct TokenGraphKey {
    const void* backend = nullptr;
    const void* session = nullptr;
    // Backend-assigned monotonic identity. Unlike the object address, this
    // cannot alias a destroyed Session whose generation restarted at one.
    std::uint64_t session_instance_id = 0;
    std::uint64_t session_generation = 0;
    std::uint64_t session_state_generation = 0;
    std::uint64_t runtime_generation = 0;
    std::int32_t runtime_capacity = 0;
    std::uint64_t residency_generation = 0;
    std::uint64_t descriptor_generation = 0;
    // Shared PLE workspace layout and the Session-scoped fixed graph view.
    std::uint64_t ple_workspace_generation = 0;
    std::uint64_t ple_graph_view_generation = 0;
    std::uint64_t topology_generation = 0;
    std::uint64_t topology_flags = 0;
    TokenGraphBody body = TokenGraphBody::QsaLayerProof;
    TokenGraphAttention attention = TokenGraphAttention::QsaDenseContiguous;
};

// POD state only. Executable ownership, events, rings and queues stay out of
// this header until ForwardTokenGraph is implemented.
struct TokenGraphLease {
    TokenGraphKey key{};
    TokenGraphPhase phase = TokenGraphPhase::Empty;
    std::uint32_t next_sequence = 0;
    std::uint32_t last_completed_sequence = 0;
    bool executable_ready = false;
    bool abort_requested = false;
};

}  // namespace aila
