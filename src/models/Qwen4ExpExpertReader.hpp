#pragma once

#include "utils/SafeTensors.hpp"

#include <atomic>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace aila {

enum class ExpertReaderType : uint8_t {
    Mmap = 0,
    OverlappedReadFile = 1
};

struct ExpertSlotGeometry {
    size_t gu_packed_bytes = 0;
    size_t gu_absmax_count = 0;
    size_t gu_absmax_bytes = 0;
    size_t dn_packed_bytes = 0;
    size_t dn_absmax_count = 0;
    size_t dn_absmax_bytes = 0;
    size_t gu_packed_offset = 0;
    size_t gu_absmax_offset = 0;
    size_t dn_packed_offset = 0;
    size_t dn_absmax_offset = 0;
    size_t slot_bytes = 0;
    bool compact_nested = false;
};

struct ExpertLayerSource {
    Bnb4BitHostWeightRef gate_up{};
    Bnb4BitHostWeightRef down{};
    bool valid = false;
};

struct ExpertReaderStats {
    uint64_t read_count = 0;
    // Completed staging payload, not physical file I/O (nested scales may expand).
    uint64_t read_bytes = 0;
    uint64_t read_ns = 0;
    uint64_t prefetch_calls = 0;
    uint64_t cancel_calls = 0;
    uint64_t fallback_mmap_reads = 0;
};

class IExpertReader {
public:
    virtual ~IExpertReader() = default;
    virtual ExpertReaderType type() const = 0;
    virtual const char* name() const = 0;

    // Read [part, nparts) slice of expert (layer, expert_id) into staging_slot.
    // If nparts == 1, reads the full expert.
    virtual void read_expert_slice(
        int layer,
        int expert_id,
        uint8_t* staging_slot,
        int part = 0,
        int nparts = 1,
        uint64_t slot_gen = 0) = 0;

    // Optional advisory prefetch of expert data (e.g. PrefetchVirtualMemory)
    virtual void prefetch_expert(int layer, int expert_id) {}

    // Cancel in-flight I/O operations
    virtual void cancel() {}
    virtual void reset_cancelled() {}
    virtual bool is_cancelled() const { return false; }

    // Update layer source bindings
    virtual void set_layer_source(int layer, const ExpertLayerSource& src) = 0;

    // Race-free approximate snapshot while reading; exact after readers join.
    // Reset only at a quiescent measurement boundary.
    virtual ExpertReaderStats stats() const = 0;
    virtual void reset_stats() = 0;
};

// Factory functions
std::unique_ptr<IExpertReader> CreateMmapExpertReader(
    const ExpertSlotGeometry& geom,
    const std::vector<ExpertLayerSource>& layers,
    bool enable_virtual_prefetch = false);

std::unique_ptr<IExpertReader> CreateOverlappedReadFileExpertReader(
    const ExpertSlotGeometry& geom,
    const std::vector<std::string>& shard_paths,
    const std::vector<ExpertLayerSource>& layers);

std::unique_ptr<IExpertReader> CreateExpertReader(
    ExpertReaderType type,
    const ExpertSlotGeometry& geom,
    const std::vector<std::string>& shard_paths,
    const std::vector<ExpertLayerSource>& layers);

} // namespace aila
