#pragma once

#include "models/Qwen4ExpTokenGraph.hpp"

#include <sycl/sycl.hpp>

#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <utility>

namespace aila {

// Stable wire geometry copied from the proven P3 two-layer probe. Keep these
// explicit so a layout change is a compile-time contract change.
inline constexpr std::uint32_t kTokenGraphRouteRanks = 10;
inline constexpr std::uint32_t kTokenGraphResultWidth = 8;
inline constexpr std::uint32_t kTokenGraphResultValues =
    kTokenGraphRouteRanks * kTokenGraphResultWidth;
inline constexpr std::uint32_t kTokenGraphHiddenElements = 2560;
inline constexpr std::uint32_t kTokenGraphPacketSlots = 16;

enum class TokenGraphGateState : std::uint32_t {
    Pending = 0,
    Valid = 1,
    Timeout = 2,
    Stale = 3,
    ChannelMismatch = 4,
    PacketMismatch = 5,
    PacketShape = 6,
};

enum class TokenGraphSequenceStatus : std::uint8_t {
    Current = 0,
    NotPublished = 1,
    Stale = 2,
    Ahead = 3,
};

// Sequence zero is the reset value. Runtime launch refuses to wrap, so a
// non-zero expected sequence remains unambiguous for the whole graph lifetime.
constexpr TokenGraphSequenceStatus token_graph_sequence_status(
    std::uint32_t expected, std::uint32_t observed) noexcept {
    if (observed == expected) return TokenGraphSequenceStatus::Current;
    if (observed == 0) return TokenGraphSequenceStatus::NotPublished;
    const std::uint32_t distance = observed - expected;
    return distance >= 0x80000000u ? TokenGraphSequenceStatus::Stale
                                   : TokenGraphSequenceStatus::Ahead;
}

constexpr bool token_graph_sequence_is_current(
    std::uint32_t expected, std::uint32_t observed) noexcept {
    return token_graph_sequence_status(expected, observed) ==
           TokenGraphSequenceStatus::Current;
}

constexpr bool token_graph_sequence_is_stale(
    std::uint32_t expected, std::uint32_t observed) noexcept {
    return token_graph_sequence_status(expected, observed) ==
           TokenGraphSequenceStatus::Stale;
}

// These PODs preserve the exact P3 byte layout. They contain data only: GPU
// wait/copy kernels remain in the future graph body implementation.
struct alignas(64) TokenGraphControlRing {
    std::uint32_t route_sequence;
    std::uint32_t cpu_flag;
    std::uint32_t result_sequence;
    std::uint32_t done_sequence;
    std::uint32_t timeout_count;
    std::uint32_t stale_count;
    std::uint32_t mismatch_count;
    std::uint32_t host_route_error_count;
    std::uint64_t timeout_spins;
    std::uint32_t channel_mismatch_count;
    std::uint32_t packet_sequence_mismatch_count;
    std::uint32_t packet_shape_error_count;
    std::uint32_t gate_abort_count;
    std::uint64_t reserved[1];
};

struct alignas(64) TokenGraphRouteRing {
    std::int32_t ids[kTokenGraphRouteRanks];
    float weights[kTokenGraphRouteRanks];
    std::uint32_t reserved[12];
};

struct alignas(64) TokenGraphResultRing {
    float values[kTokenGraphResultValues];
    std::uint32_t reserved[16];
};

struct alignas(64) TokenGraphHostRing {
    TokenGraphControlRing control;
    TokenGraphRouteRing route;
    TokenGraphResultRing result;
};

// input_bf16 stores raw BF16 bit patterns so the wire layout stays independent
// of the SYCL bfloat16 wrapper type.
struct alignas(64) TokenGraphCpuPacketRing {
    std::uint32_t ready_sequence;
    std::uint32_t rows;
    std::uint32_t reserved[14];
    std::int32_t rank_to_row[kTokenGraphPacketSlots];
    float values[kTokenGraphPacketSlots * kTokenGraphHiddenElements];
    std::uint16_t input_bf16[kTokenGraphHiddenElements];
};

struct alignas(64) TokenGraphDeviceEpoch {
    std::uint32_t sequence;
    std::uint32_t gate_state;
    std::uint32_t sticky_abort;
    std::uint32_t reserved[13];
};

static_assert(sizeof(TokenGraphGateState) == 4);
static_assert(sizeof(TokenGraphControlRing) == 64);
static_assert(sizeof(TokenGraphRouteRing) == 128);
static_assert(sizeof(TokenGraphResultRing) == 384);
static_assert(sizeof(TokenGraphHostRing) == 576);
static_assert(sizeof(TokenGraphCpuPacketRing) == 169088);
static_assert(sizeof(TokenGraphDeviceEpoch) == 64);
static_assert(offsetof(TokenGraphHostRing, route) == 64);
static_assert(offsetof(TokenGraphHostRing, result) == 192);
static_assert(offsetof(TokenGraphCpuPacketRing, rank_to_row) == 64);
static_assert(offsetof(TokenGraphCpuPacketRing, values) == 128);
static_assert(offsetof(TokenGraphCpuPacketRing, input_bf16) == 163968);
static_assert(std::is_trivially_copyable<TokenGraphControlRing>::value);
static_assert(std::is_trivially_copyable<TokenGraphRouteRing>::value);
static_assert(std::is_trivially_copyable<TokenGraphResultRing>::value);
static_assert(std::is_trivially_copyable<TokenGraphHostRing>::value);
static_assert(std::is_trivially_copyable<TokenGraphCpuPacketRing>::value);
static_assert(std::is_trivially_copyable<TokenGraphDeviceEpoch>::value);
static_assert(std::is_standard_layout<TokenGraphHostRing>::value);
static_assert(std::is_standard_layout<TokenGraphCpuPacketRing>::value);

constexpr bool token_graph_gate_is_valid(
    const TokenGraphDeviceEpoch& epoch) noexcept {
    return epoch.gate_state ==
               static_cast<std::uint32_t>(TokenGraphGateState::Valid) &&
           epoch.sticky_abort == 0;
}

// Active ownership is host-thread confined. TokenGraphLease remains the POD
// diagnostic copy; this object is never copied or moved.
class TokenGraphRuntime {
public:
    TokenGraphRuntime() noexcept = default;
    TokenGraphRuntime(const TokenGraphRuntime&) = delete;
    TokenGraphRuntime& operator=(const TokenGraphRuntime&) = delete;
    TokenGraphRuntime(TokenGraphRuntime&&) = delete;
    TokenGraphRuntime& operator=(TokenGraphRuntime&&) = delete;

    // Empty/Ready -> Capturing. Re-capturing from Ready requires the caller to
    // have drained and destroyed the previous executable first.
    bool BeginCapture(const TokenGraphKey& key) noexcept;
    bool MarkExecutableReady() noexcept;

    // Ready -> InFlight. The sequence must equal next_sequence and must not
    // wrap. A failed launch leaves the current state unchanged.
    bool Launch(std::uint32_t sequence, sycl::event event) noexcept;

    // InFlight -> Ready only when sequence matches and gate_valid is true.
    // A sequence mismatch or rejected gate enters Aborting and returns false.
    bool Complete(std::uint32_t sequence, bool gate_valid) noexcept;

    // Enters Aborting without waiting or destroying the event. The caller must
    // drain the executable/event before Poison() and resource release.
    bool Abort() noexcept;

    // Aborting/any state -> Poisoned. Poisoning is permanent.
    bool Poison() noexcept;

    TokenGraphLease Snapshot() const noexcept;

    const TokenGraphKey& key() const noexcept { return key_; }
    TokenGraphPhase phase() const noexcept { return phase_; }
    std::uint32_t next_sequence() const noexcept { return nextSequence_; }
    std::uint32_t last_completed_sequence() const noexcept {
        return lastCompletedSequence_;
    }
    const sycl::event& in_flight_event() const noexcept {
        return inFlightEvent_;
    }
    bool executable_ready() const noexcept { return executableReady_; }
    bool gate_valid() const noexcept { return gateValid_; }
    bool abort_requested() const noexcept { return abortRequested_; }

private:
    bool EnterAborting() noexcept;

    TokenGraphKey key_{};
    TokenGraphPhase phase_ = TokenGraphPhase::Empty;
    sycl::event inFlightEvent_{};
    std::uint32_t nextSequence_ = 1;
    std::uint32_t lastCompletedSequence_ = 0;
    bool gateValid_ = false;
    bool abortRequested_ = false;
    bool executableReady_ = false;
};

static_assert(!std::is_copy_constructible<TokenGraphRuntime>::value);
static_assert(!std::is_copy_assignable<TokenGraphRuntime>::value);
static_assert(!std::is_move_constructible<TokenGraphRuntime>::value);
static_assert(!std::is_move_assignable<TokenGraphRuntime>::value);

inline bool TokenGraphRuntime::BeginCapture(const TokenGraphKey& key) noexcept {
    if (phase_ != TokenGraphPhase::Empty && phase_ != TokenGraphPhase::Ready)
        return false;
    key_ = key;
    phase_ = TokenGraphPhase::Capturing;
    inFlightEvent_ = sycl::event{};
    nextSequence_ = 1;
    lastCompletedSequence_ = 0;
    gateValid_ = false;
    abortRequested_ = false;
    executableReady_ = false;
    return true;
}

inline bool TokenGraphRuntime::MarkExecutableReady() noexcept {
    if (phase_ != TokenGraphPhase::Capturing) return false;
    phase_ = TokenGraphPhase::Ready;
    gateValid_ = false;
    executableReady_ = true;
    return true;
}

inline bool TokenGraphRuntime::Launch(
    std::uint32_t sequence, sycl::event event) noexcept {
    if (phase_ != TokenGraphPhase::Ready || !executableReady_ ||
        abortRequested_ || sequence == 0 ||
        sequence != nextSequence_ || sequence == 0xffffffffu) {
        return false;
    }
    inFlightEvent_ = std::move(event);
    phase_ = TokenGraphPhase::InFlight;
    gateValid_ = false;
    ++nextSequence_;
    return true;
}

inline bool TokenGraphRuntime::Complete(
    std::uint32_t sequence, bool gate_valid) noexcept {
    const std::uint32_t expected = nextSequence_ - 1u;
    if (phase_ != TokenGraphPhase::InFlight || abortRequested_ ||
        sequence != expected || !gate_valid) {
        EnterAborting();
        return false;
    }
    phase_ = TokenGraphPhase::Ready;
    inFlightEvent_ = sycl::event{};
    lastCompletedSequence_ = sequence;
    gateValid_ = true;
    return true;
}

inline bool TokenGraphRuntime::Abort() noexcept {
    if (phase_ == TokenGraphPhase::Poisoned) return false;
    return EnterAborting();
}

inline bool TokenGraphRuntime::Poison() noexcept {
    if (phase_ == TokenGraphPhase::Poisoned) return false;
    EnterAborting();
    phase_ = TokenGraphPhase::Poisoned;
    gateValid_ = false;
    abortRequested_ = true;
    executableReady_ = false;
    return true;
}

inline TokenGraphLease TokenGraphRuntime::Snapshot() const noexcept {
    TokenGraphLease lease{};
    lease.key = key_;
    lease.phase = phase_;
    lease.next_sequence = nextSequence_;
    lease.last_completed_sequence = lastCompletedSequence_;
    lease.executable_ready = executableReady_;
    lease.abort_requested = abortRequested_;
    return lease;
}

inline bool TokenGraphRuntime::EnterAborting() noexcept {
    if (phase_ == TokenGraphPhase::Aborting ||
        phase_ == TokenGraphPhase::Poisoned) {
        return false;
    }
    phase_ = TokenGraphPhase::Aborting;
    gateValid_ = false;
    abortRequested_ = true;
    executableReady_ = false;
    return true;
}

}  // namespace aila
