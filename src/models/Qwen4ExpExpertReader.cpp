#include "models/Qwen4ExpExpertReader.hpp"
#include "profile/Profiling.hpp"
#include "utils/EnvUtils.hpp"

#ifdef _WIN32
#include <windows.h>
#include <memoryapi.h>
#endif

#include <algorithm>
#include <chrono>
#include <cstring>
#include <mutex>
#include <stdexcept>

namespace aila {

namespace {

inline void part_range(size_t len, int part, int nparts, size_t& begin, size_t& end) {
    begin = len * static_cast<size_t>(part) / static_cast<size_t>(nparts);
    end = len * static_cast<size_t>(part + 1) / static_cast<size_t>(nparts);
}

void mmap_fill_matrix(const Bnb4BitHostWeightRef& ref, int expert_id,
                      uint8_t* packed_dst, size_t packed_bytes,
                      uint8_t* absmax_dst, size_t absmax_count, bool compact_nested,
                      int part, int nparts) {
    size_t b0 = 0, b1 = 0;
    part_range(packed_bytes, part, nparts, b0, b1);
    if (ref.packed_weight.valid()) {
        const uint8_t* src = ref.packed_weight.ptr + static_cast<size_t>(expert_id) * packed_bytes;
        std::memcpy(packed_dst + b0, src + b0, b1 - b0);
    } else {
        std::memset(packed_dst + b0, 0, b1 - b0);
    }

    size_t a0 = 0, a1 = 0;
    part_range(absmax_count, part, nparts, a0, a1);
    if (compact_nested) {
        if (ref.absmax.valid()) {
            const size_t expert_base = static_cast<size_t>(expert_id) * absmax_count;
            std::memcpy(absmax_dst + a0, ref.absmax_codes_ptr() + expert_base + a0, a1 - a0);
        } else {
            std::memset(absmax_dst + a0, 0, a1 - a0);
        }
        return;
    }
    float* absmax_f32 = reinterpret_cast<float*>(absmax_dst);
    if (!ref.absmax.valid()) {
        std::memset(absmax_f32 + a0, 0, (a1 - a0) * sizeof(float));
        return;
    }
    const size_t expert_base = static_cast<size_t>(expert_id) * absmax_count;
    if (!ref.quant_state.nested) {
        std::memcpy(absmax_f32 + a0, ref.absmax_f32_ptr() + expert_base + a0, (a1 - a0) * sizeof(float));
        return;
    }
    const uint8_t* codes = ref.absmax_codes_ptr() + expert_base;
    const float* nested_absmax = ref.nested_absmax_ptr();
    const float* nested_quant_map = ref.nested_quant_map_ptr();
    const size_t nested_blocksize = ref.quant_state.nested_blocksize > 0
                                        ? static_cast<size_t>(ref.quant_state.nested_blocksize) : 256;
    const float nested_offset = ref.quant_state.nested_offset;
    size_t i = a0;
    while (i < a1) {
        const size_t global = expert_base + i;
        const float scale = nested_absmax[global / nested_blocksize];
        const size_t run_end = std::min(a1, i + (nested_blocksize - global % nested_blocksize));
        for (; i < run_end; ++i) {
            absmax_f32[i] = nested_quant_map[codes[i]] * scale + nested_offset;
        }
    }
}

#ifdef _WIN32
struct OverlappedIoContext {
    OVERLAPPED ov{};
    HANDLE hEvent = NULL;

    OverlappedIoContext() {
        hEvent = CreateEventA(NULL, TRUE, FALSE, NULL);
        ov.hEvent = hEvent;
    }
    ~OverlappedIoContext() {
        if (hEvent) {
            CloseHandle(hEvent);
            hEvent = NULL;
        }
    }
};

static thread_local OverlappedIoContext t_io_ctx;
#endif

} // namespace

// ============================================================
// MmapExpertReader
// ============================================================
class MmapExpertReader : public IExpertReader {
public:
    MmapExpertReader(const ExpertSlotGeometry& geom,
                     const std::vector<ExpertLayerSource>& layers,
                     bool enable_virtual_prefetch)
        : geom_(geom), layers_(layers), enable_virtual_prefetch_(enable_virtual_prefetch) {}

    ExpertReaderType type() const override { return ExpertReaderType::Mmap; }
    const char* name() const override { return "mmap"; }

    void read_expert_slice(int layer, int expert_id, uint8_t* staging_slot,
                           int part, int nparts, uint64_t /*slot_gen*/) override {
        if (cancelled_.load(std::memory_order_acquire)) return;
        const auto t0 = std::chrono::high_resolution_clock::now();
        if (layer < 0 || layer >= static_cast<int>(layers_.size()) || !layers_[layer].valid) {
            size_t b0 = 0, b1 = 0;
            part_range(geom_.slot_bytes, part, nparts, b0, b1);
            std::memset(staging_slot + b0, 0, b1 - b0);
            return;
        }
        const auto& lw = layers_[layer];
        mmap_fill_matrix(lw.gate_up, expert_id,
                         staging_slot + geom_.gu_packed_offset, geom_.gu_packed_bytes,
                         staging_slot + geom_.gu_absmax_offset, geom_.gu_absmax_count, geom_.compact_nested,
                         part, nparts);
        if (cancelled_.load(std::memory_order_acquire)) return;
        mmap_fill_matrix(lw.down, expert_id,
                         staging_slot + geom_.dn_packed_offset, geom_.dn_packed_bytes,
                         staging_slot + geom_.dn_absmax_offset, geom_.dn_absmax_count, geom_.compact_nested,
                         part, nparts);

        const auto t1 = std::chrono::high_resolution_clock::now();
        stats_.read_count++;
        stats_.read_bytes += geom_.slot_bytes / static_cast<size_t>(std::max(1, nparts));
        stats_.read_ns += static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count());
    }

    void cancel() override {
        stats_.cancel_calls++;
        cancelled_.store(true, std::memory_order_release);
    }

    void reset_cancelled() override {
        cancelled_.store(false, std::memory_order_release);
    }

    bool is_cancelled() const override {
        return cancelled_.load(std::memory_order_acquire);
    }

    void prefetch_expert(int layer, int expert_id) override {
#ifdef _WIN32
        if (enable_virtual_prefetch_) {
            if (layer < 0 || layer >= static_cast<int>(layers_.size()) || !layers_[layer].valid) return;
            const auto& lw = layers_[layer];
            WIN32_MEMORY_RANGE_ENTRY ranges[4];
            int n_ranges = 0;
            if (lw.gate_up.packed_weight.valid()) {
                ranges[n_ranges].VirtualAddress = const_cast<uint8_t*>(
                    lw.gate_up.packed_weight.ptr + static_cast<size_t>(expert_id) * geom_.gu_packed_bytes);
                ranges[n_ranges].NumberOfBytes = geom_.gu_packed_bytes;
                n_ranges++;
            }
            if (lw.gate_up.absmax.valid()) {
                ranges[n_ranges].VirtualAddress = const_cast<uint8_t*>(
                    lw.gate_up.absmax_codes_ptr() + static_cast<size_t>(expert_id) * geom_.gu_absmax_count);
                ranges[n_ranges].NumberOfBytes = geom_.gu_absmax_bytes;
                n_ranges++;
            }
            if (lw.down.packed_weight.valid()) {
                ranges[n_ranges].VirtualAddress = const_cast<uint8_t*>(
                    lw.down.packed_weight.ptr + static_cast<size_t>(expert_id) * geom_.dn_packed_bytes);
                ranges[n_ranges].NumberOfBytes = geom_.dn_packed_bytes;
                n_ranges++;
            }
            if (lw.down.absmax.valid()) {
                ranges[n_ranges].VirtualAddress = const_cast<uint8_t*>(
                    lw.down.absmax_codes_ptr() + static_cast<size_t>(expert_id) * geom_.dn_absmax_count);
                ranges[n_ranges].NumberOfBytes = geom_.dn_absmax_bytes;
                n_ranges++;
            }
            if (n_ranges > 0) {
                PrefetchVirtualMemory(GetCurrentProcess(), n_ranges, ranges, 0);
                stats_.prefetch_calls++;
            }
        }
#endif
    }

    void set_layer_source(int layer, const ExpertLayerSource& src) override {
        if (layer >= static_cast<int>(layers_.size())) {
            layers_.resize(layer + 1);
        }
        layers_[layer] = src;
    }

    ExpertReaderStats stats() const override { return stats_; }
    void reset_stats() override { stats_ = {}; }

private:
    ExpertSlotGeometry geom_{};
    std::vector<ExpertLayerSource> layers_;
    bool enable_virtual_prefetch_ = false;
    std::atomic<bool> cancelled_{false};
    ExpertReaderStats stats_{};
};

// ============================================================
// OverlappedReadFileExpertReader
// ============================================================
class OverlappedReadFileExpertReader : public IExpertReader {
public:
    OverlappedReadFileExpertReader(const ExpertSlotGeometry& geom,
                                   const std::vector<std::string>& shard_paths,
                                   const std::vector<ExpertLayerSource>& layers)
        : geom_(geom), shard_paths_(shard_paths), layers_(layers) {
#ifdef _WIN32
        shard_handles_.resize(shard_paths_.size(), INVALID_HANDLE_VALUE);
        for (size_t i = 0; i < shard_paths_.size(); ++i) {
            HANDLE h = CreateFileA(shard_paths_[i].c_str(),
                                   GENERIC_READ,
                                   FILE_SHARE_READ,
                                   NULL,
                                   OPEN_EXISTING,
                                   FILE_ATTRIBUTE_NORMAL | FILE_FLAG_OVERLAPPED,
                                   NULL);
            if (h != INVALID_HANDLE_VALUE) {
                shard_handles_[i] = h;
            } else {
                AILA_LOG_WARN("[ExpertReader] Failed to open shard %zu for overlapped ReadFile: %s",
                              i, shard_paths_[i].c_str());
            }
        }
#endif
    }

    ~OverlappedReadFileExpertReader() override {
#ifdef _WIN32
        for (HANDLE h : shard_handles_) {
            if (h != INVALID_HANDLE_VALUE) {
                CloseHandle(h);
            }
        }
        shard_handles_.clear();
#endif
    }

    ExpertReaderType type() const override { return ExpertReaderType::OverlappedReadFile; }
    const char* name() const override { return "readfile_overlapped"; }

    void read_expert_slice(int layer, int expert_id, uint8_t* staging_slot,
                           int part, int nparts, uint64_t /*slot_gen*/) override {
        if (cancelled_.load(std::memory_order_acquire)) return;
        const auto t0 = std::chrono::high_resolution_clock::now();
        if (layer < 0 || layer >= static_cast<int>(layers_.size()) || !layers_[layer].valid) {
            size_t b0 = 0, b1 = 0;
            part_range(geom_.slot_bytes, part, nparts, b0, b1);
            std::memset(staging_slot + b0, 0, b1 - b0);
            return;
        }

        const auto& lw = layers_[layer];
        const auto& gu_p_span = lw.gate_up.packed_weight.file_span;
        const auto& gu_a_span = lw.gate_up.absmax.file_span;
        const auto& dn_p_span = lw.down.packed_weight.file_span;
        const auto& dn_a_span = lw.down.absmax.file_span;

        // Fallback to mmap if any required FileSpan is invalid or handle missing
        const bool can_readfile = gu_p_span.valid() && gu_a_span.valid() &&
                                  dn_p_span.valid() && dn_a_span.valid() &&
                                  has_valid_handle(gu_p_span.shard_id) &&
                                  has_valid_handle(gu_a_span.shard_id) &&
                                  has_valid_handle(dn_p_span.shard_id) &&
                                  has_valid_handle(dn_a_span.shard_id);

        if (!can_readfile) {
            mmap_fill_matrix(lw.gate_up, expert_id,
                             staging_slot + geom_.gu_packed_offset, geom_.gu_packed_bytes,
                             staging_slot + geom_.gu_absmax_offset, geom_.gu_absmax_count, geom_.compact_nested,
                             part, nparts);
            if (cancelled_.load(std::memory_order_acquire)) return;
            mmap_fill_matrix(lw.down, expert_id,
                             staging_slot + geom_.dn_packed_offset, geom_.dn_packed_bytes,
                             staging_slot + geom_.dn_absmax_offset, geom_.dn_absmax_count, geom_.compact_nested,
                             part, nparts);
            stats_.fallback_mmap_reads++;
            const auto t1 = std::chrono::high_resolution_clock::now();
            stats_.read_count++;
            stats_.read_bytes += geom_.slot_bytes / static_cast<size_t>(std::max(1, nparts));
            stats_.read_ns += static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count());
            return;
        }

        // 1. Gate/Up packed
        {
            size_t b0 = 0, b1 = 0;
            part_range(geom_.gu_packed_bytes, part, nparts, b0, b1);
            const uint64_t offset = gu_p_span.absolute_offset +
                static_cast<uint64_t>(expert_id) * geom_.gu_packed_bytes + b0;
            if (!read_segment(gu_p_span.shard_id, offset, staging_slot + geom_.gu_packed_offset + b0, b1 - b0)) {
                return;
            }
        }
        if (cancelled_.load(std::memory_order_acquire)) return;

        // 2. Gate/Up absmax
        {
            size_t a0 = 0, a1 = 0;
            part_range(geom_.gu_absmax_count, part, nparts, a0, a1);
            if (geom_.compact_nested) {
                const uint64_t offset = gu_a_span.absolute_offset +
                    static_cast<uint64_t>(expert_id) * geom_.gu_absmax_count + a0;
                if (!read_segment(gu_a_span.shard_id, offset, staging_slot + geom_.gu_absmax_offset + a0, a1 - a0)) {
                    return;
                }
            } else if (lw.gate_up.quant_state.nested) {
                // Non-compact nested: read u8 codes into local buffer and expand to float absmax
                uint8_t temp_codes[512];
                const size_t count = a1 - a0;
                const size_t chunk_limit = sizeof(temp_codes);
                size_t done = 0;
                while (done < count) {
                    if (cancelled_.load(std::memory_order_acquire)) return;
                    const size_t cur_chunk = std::min(count - done, chunk_limit);
                    const uint64_t offset = gu_a_span.absolute_offset +
                        static_cast<uint64_t>(expert_id) * geom_.gu_absmax_count + a0 + done;
                    if (!read_segment(gu_a_span.shard_id, offset, temp_codes, cur_chunk)) {
                        return;
                    }

                    float* absmax_f32 = reinterpret_cast<float*>(staging_slot + geom_.gu_absmax_offset);
                    const float* nested_absmax = lw.gate_up.nested_absmax_ptr();
                    const float* nested_quant_map = lw.gate_up.nested_quant_map_ptr();
                    const size_t nested_blocksize = lw.gate_up.quant_state.nested_blocksize > 0
                        ? static_cast<size_t>(lw.gate_up.quant_state.nested_blocksize) : 256;
                    const float nested_offset = lw.gate_up.quant_state.nested_offset;
                    const size_t expert_base = static_cast<size_t>(expert_id) * geom_.gu_absmax_count;

                    for (size_t c = 0; c < cur_chunk; ++c) {
                        const size_t global = expert_base + a0 + done + c;
                        const float scale = nested_absmax[global / nested_blocksize];
                        absmax_f32[a0 + done + c] = nested_quant_map[temp_codes[c]] * scale + nested_offset;
                    }
                    done += cur_chunk;
                }
            } else {
                // Non-nested: absmax is already float32
                float* absmax_f32 = reinterpret_cast<float*>(staging_slot + geom_.gu_absmax_offset);
                const size_t float_bytes = (a1 - a0) * sizeof(float);
                const uint64_t offset = gu_a_span.absolute_offset +
                    (static_cast<uint64_t>(expert_id) * geom_.gu_absmax_count + a0) * sizeof(float);
                if (!read_segment(gu_a_span.shard_id, offset, reinterpret_cast<uint8_t*>(absmax_f32 + a0), float_bytes)) {
                    return;
                }
            }
        }
        if (cancelled_.load(std::memory_order_acquire)) return;

        // 3. Down packed
        {
            size_t b0 = 0, b1 = 0;
            part_range(geom_.dn_packed_bytes, part, nparts, b0, b1);
            const uint64_t offset = dn_p_span.absolute_offset +
                static_cast<uint64_t>(expert_id) * geom_.dn_packed_bytes + b0;
            if (!read_segment(dn_p_span.shard_id, offset, staging_slot + geom_.dn_packed_offset + b0, b1 - b0)) {
                return;
            }
        }
        if (cancelled_.load(std::memory_order_acquire)) return;

        // 4. Down absmax
        {
            size_t a0 = 0, a1 = 0;
            part_range(geom_.dn_absmax_count, part, nparts, a0, a1);
            if (geom_.compact_nested) {
                const uint64_t offset = dn_a_span.absolute_offset +
                    static_cast<uint64_t>(expert_id) * geom_.dn_absmax_count + a0;
                if (!read_segment(dn_a_span.shard_id, offset, staging_slot + geom_.dn_absmax_offset + a0, a1 - a0)) {
                    return;
                }
            } else if (lw.down.quant_state.nested) {
                uint8_t temp_codes[512];
                const size_t count = a1 - a0;
                const size_t chunk_limit = sizeof(temp_codes);
                size_t done = 0;
                while (done < count) {
                    if (cancelled_.load(std::memory_order_acquire)) return;
                    const size_t cur_chunk = std::min(count - done, chunk_limit);
                    const uint64_t offset = dn_a_span.absolute_offset +
                        static_cast<uint64_t>(expert_id) * geom_.dn_absmax_count + a0 + done;
                    if (!read_segment(dn_a_span.shard_id, offset, temp_codes, cur_chunk)) {
                        return;
                    }

                    float* absmax_f32 = reinterpret_cast<float*>(staging_slot + geom_.dn_absmax_offset);
                    const float* nested_absmax = lw.down.nested_absmax_ptr();
                    const float* nested_quant_map = lw.down.nested_quant_map_ptr();
                    const size_t nested_blocksize = lw.down.quant_state.nested_blocksize > 0
                        ? static_cast<size_t>(lw.down.quant_state.nested_blocksize) : 256;
                    const float nested_offset = lw.down.quant_state.nested_offset;
                    const size_t expert_base = static_cast<size_t>(expert_id) * geom_.dn_absmax_count;

                    for (size_t c = 0; c < cur_chunk; ++c) {
                        const size_t global = expert_base + a0 + done + c;
                        const float scale = nested_absmax[global / nested_blocksize];
                        absmax_f32[a0 + done + c] = nested_quant_map[temp_codes[c]] * scale + nested_offset;
                    }
                    done += cur_chunk;
                }
            } else {
                float* absmax_f32 = reinterpret_cast<float*>(staging_slot + geom_.dn_absmax_offset);
                const size_t float_bytes = (a1 - a0) * sizeof(float);
                const uint64_t offset = dn_a_span.absolute_offset +
                    (static_cast<uint64_t>(expert_id) * geom_.dn_absmax_count + a0) * sizeof(float);
                if (!read_segment(dn_a_span.shard_id, offset, reinterpret_cast<uint8_t*>(absmax_f32 + a0), float_bytes)) {
                    return;
                }
            }
        }

        const auto t1 = std::chrono::high_resolution_clock::now();
        stats_.read_count++;
        stats_.read_bytes += geom_.slot_bytes / static_cast<size_t>(std::max(1, nparts));
        stats_.read_ns += static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count());
    }

    void cancel() override {
#ifdef _WIN32
        stats_.cancel_calls++;
        cancelled_.store(true, std::memory_order_release);
        for (HANDLE h : shard_handles_) {
            if (h != INVALID_HANDLE_VALUE) {
                CancelIoEx(h, NULL);
            }
        }
#endif
    }

    void reset_cancelled() override {
        cancelled_.store(false, std::memory_order_release);
    }

    bool is_cancelled() const override {
        return cancelled_.load(std::memory_order_acquire);
    }

    void set_layer_source(int layer, const ExpertLayerSource& src) override {
        if (layer >= static_cast<int>(layers_.size())) {
            layers_.resize(layer + 1);
        }
        layers_[layer] = src;
    }

    ExpertReaderStats stats() const override { return stats_; }
    void reset_stats() override { stats_ = {}; }

private:
    bool has_valid_handle(int shard_id) const {
#ifdef _WIN32
        return shard_id >= 0 && shard_id < static_cast<int>(shard_handles_.size()) &&
               shard_handles_[shard_id] != INVALID_HANDLE_VALUE;
#else
        return false;
#endif
    }

    bool read_segment(int shard_id, uint64_t file_offset, uint8_t* dst, size_t bytes_to_read) {
        if (bytes_to_read == 0) return true;
        if (cancelled_.load(std::memory_order_acquire)) return false;
#ifdef _WIN32
        HANDLE h = shard_handles_[shard_id];
        auto& io_ctx = t_io_ctx;
        uint64_t cur_offset = file_offset;
        uint8_t* cur_dst = dst;
        size_t remaining = bytes_to_read;

        while (remaining > 0) {
            if (cancelled_.load(std::memory_order_acquire)) return false;
            io_ctx.ov.Offset = static_cast<DWORD>(cur_offset & 0xFFFFFFFF);
            io_ctx.ov.OffsetHigh = static_cast<DWORD>((cur_offset >> 32) & 0xFFFFFFFF);
            ResetEvent(io_ctx.hEvent);

            const DWORD to_read = static_cast<DWORD>(std::min<size_t>(remaining, 64 * 1024 * 1024));
            DWORD bytes_read = 0;
            BOOL ok = ReadFile(h, cur_dst, to_read, &bytes_read, &io_ctx.ov);
            if (!ok) {
                DWORD err = GetLastError();
                if (err == ERROR_IO_PENDING) {
                    if (!GetOverlappedResult(h, &io_ctx.ov, &bytes_read, TRUE)) {
                        DWORD res_err = GetLastError();
                        if (res_err == ERROR_OPERATION_ABORTED) return false;
                        throw std::runtime_error("GetOverlappedResult failed: " + std::to_string(res_err));
                    }
                } else if (err == ERROR_OPERATION_ABORTED) {
                    return false;
                } else {
                    throw std::runtime_error("ReadFile failed with error: " + std::to_string(err));
                }
            } else {
                if (!GetOverlappedResult(h, &io_ctx.ov, &bytes_read, FALSE)) {
                    DWORD res_err = GetLastError();
                    if (res_err == ERROR_OPERATION_ABORTED) return false;
                }
            }
            if (bytes_read == 0) break; // EOF
            cur_offset += bytes_read;
            cur_dst += bytes_read;
            remaining -= bytes_read;
        }
        return true;
#else
        std::memset(dst, 0, bytes_to_read);
        return true;
#endif
    }

    ExpertSlotGeometry geom_{};
    std::vector<std::string> shard_paths_;
#ifdef _WIN32
    std::vector<HANDLE> shard_handles_;
#endif
    std::vector<ExpertLayerSource> layers_;
    std::atomic<bool> cancelled_{false};
    ExpertReaderStats stats_{};
};

// ============================================================
// Factory Functions
// ============================================================
std::unique_ptr<IExpertReader> CreateMmapExpertReader(
    const ExpertSlotGeometry& geom,
    const std::vector<ExpertLayerSource>& layers,
    bool enable_virtual_prefetch) {
    return std::make_unique<MmapExpertReader>(geom, layers, enable_virtual_prefetch);
}

std::unique_ptr<IExpertReader> CreateOverlappedReadFileExpertReader(
    const ExpertSlotGeometry& geom,
    const std::vector<std::string>& shard_paths,
    const std::vector<ExpertLayerSource>& layers) {
    return std::make_unique<OverlappedReadFileExpertReader>(geom, shard_paths, layers);
}

std::unique_ptr<IExpertReader> CreateExpertReader(
    ExpertReaderType type,
    const ExpertSlotGeometry& geom,
    const std::vector<std::string>& shard_paths,
    const std::vector<ExpertLayerSource>& layers) {
    if (type == ExpertReaderType::OverlappedReadFile) {
        return CreateOverlappedReadFileExpertReader(geom, shard_paths, layers);
    }
    const bool virtual_prefetch = aila::env::read_flag("AILA_Q4E_MMAP_PREFETCH", false);
    return CreateMmapExpertReader(geom, layers, virtual_prefetch);
}

} // namespace aila
