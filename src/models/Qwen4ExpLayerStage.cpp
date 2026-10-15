#include "models/Qwen4ExpLayerStage.hpp"

#include "profile/Profiling.hpp"
#include "utils/EnvUtils.hpp"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <stdexcept>

namespace aila {

namespace {
using Clock = std::chrono::steady_clock;

double ms_since(Clock::time_point a) {
    return std::chrono::duration<double, std::milli>(Clock::now() - a).count();
}
} // namespace

Qwen4ExpLayerStage::~Qwen4ExpLayerStage() { Release(); }

bool Qwen4ExpLayerStage::Allocate(Context& ctx, const ExpertSlotGeometry& geom,
                                  const LayerStageConfig& cfg, std::string* error_message) {
    if (IsAllocated()) {
        if (error_message) *error_message = "Qwen4ExpLayerStage is already allocated";
        return false;
    }
    LayerStageConfig resolved = cfg;
    if (resolved.fill_threads <= 0) {
        resolved.fill_threads =
            std::clamp(aila::env::read_int("AILA_Q4E_LAYER_STAGE_THREADS", 6), 1, 32);
    } else {
        resolved.fill_threads = std::clamp(resolved.fill_threads, 1, 32);
    }
    if (geom.slot_bytes == 0 || cfg.max_slots < 1 ||
        cfg.max_slots > cfg.num_experts_per_layer ||
        cfg.chunk_slots < 1 || cfg.chunk_slots > cfg.max_slots ||
        cfg.fill_banks < 2 || cfg.fill_banks > 8) {
        if (error_message) *error_message = "Invalid Qwen4ExpLayerStage configuration";
        return false;
    }
    ctx_ = &ctx;
    geom_ = geom;
    config_ = resolved;
    try {
        arena_bytes_ = static_cast<size_t>(config_.max_slots) * geom_.slot_bytes;
        arena_dev_ = static_cast<uint8_t*>(ctx.alloc_device(arena_bytes_));
        host_bytes_ = static_cast<size_t>(config_.fill_banks) *
                      static_cast<size_t>(config_.chunk_slots) * geom_.slot_bytes;
        host_banks_ = static_cast<uint8_t*>(ctx.alloc_host(host_bytes_));
    } catch (const std::exception& e) {
        if (arena_dev_) { ctx.free_device(arena_dev_); arena_dev_ = nullptr; }
        if (host_banks_) { ctx.free_host(host_banks_); host_banks_ = nullptr; }
        arena_bytes_ = host_bytes_ = 0;
        ctx_ = nullptr;
        if (error_message) *error_message = e.what();
        return false;
    }
    slot_of_.assign(static_cast<size_t>(config_.num_experts_per_layer), -1);
    epoch_ids_.reserve(static_cast<size_t>(config_.max_slots));
    bank_events_.assign(static_cast<size_t>(config_.fill_banks), sycl::event{});
    bank_event_valid_.assign(static_cast<size_t>(config_.fill_banks), false);
    worker_ = std::thread([this] { WorkerLoop(); });
    if (config_.fill_threads > 1) {
        try {
            fill_workers_.reserve(static_cast<size_t>(config_.fill_threads) - 1);
            for (int i = 1; i < config_.fill_threads; ++i)
                fill_workers_.emplace_back([this, i] { FillLoop(i); });
        } catch (const std::exception& e) {
            // The pool is optional; keep the serial fill on partial spawn.
            fill_quit_.store(true, std::memory_order_release);
            cv_fill_.notify_all();
            for (auto& t : fill_workers_) if (t.joinable()) t.join();
            fill_workers_.clear();
            fill_quit_.store(false, std::memory_order_release);
            config_.fill_threads = 1;
            AILA_LOG_WARN("[Q4E LayerStage] fill pool spawn failed (%s); serial fill", e.what());
        }
    }
    return true;
}

void Qwen4ExpLayerStage::Release() noexcept {
    if (!worker_.joinable() && !arena_dev_ && !host_banks_) return;
    WaitIdle();
    {
        std::lock_guard<std::mutex> lk(mu_);
        quit_ = true;
    }
    cv_job_.notify_all();
    if (worker_.joinable()) worker_.join();
    {
        std::lock_guard<std::mutex> lk(mu_fill_);
        fill_quit_.store(true, std::memory_order_release);
    }
    cv_fill_.notify_all();
    for (auto& t : fill_workers_) if (t.joinable()) t.join();
    fill_workers_.clear();
    fill_quit_.store(false, std::memory_order_release);
    fill_bank_ = nullptr;
    fill_sources_ = nullptr;
    fill_count_ = 0;
    if (ctx_) {
        if (arena_dev_) { ctx_->free_device(arena_dev_); arena_dev_ = nullptr; }
        if (host_banks_) { ctx_->free_host(host_banks_); host_banks_ = nullptr; }
    } else {
        arena_dev_ = nullptr;
        host_banks_ = nullptr;
    }
    arena_bytes_ = host_bytes_ = 0;
    ctx_ = nullptr;
    copy_q_ = nullptr;
    current_epoch_ = 0;
    current_layer_ = -1;
    last_event_valid_ = false;
    arena_read_ = false;
    {
        std::lock_guard<std::mutex> lk(mu_);
        failed_ = false;
        error_.clear();
        stats_ = LayerStageStats{};
    }
}

void Qwen4ExpLayerStage::FillChunkShare(int index, int threads, uint8_t* bank,
                                        const uint8_t* const* sources, int count,
                                        size_t slot_bytes) {
    if (!bank || !sources || count < 1 || threads < 1) return;
    // Contiguous ranges give each thread whole slots to stream; the ranges are
    // disjoint, so the copies never inspect the same bytes.
    const int begin = (count * index) / threads;
    const int end = (count * (index + 1)) / threads;
    for (int i = begin; i < end; ++i)
        std::memcpy(bank + static_cast<size_t>(i) * slot_bytes, sources[i], slot_bytes);
}

void Qwen4ExpLayerStage::FillLoop(int index) {
    uint64_t seen = 0;
    for (;;) {
        uint8_t* bank = nullptr;
        const uint8_t* const* sources = nullptr;
        int count = 0;
        {
            std::unique_lock<std::mutex> lk(mu_fill_);
            cv_fill_.wait(lk, [&] {
                return fill_quit_.load(std::memory_order_acquire) ||
                       fill_gen_.load(std::memory_order_acquire) != seen;
            });
            if (fill_quit_.load(std::memory_order_acquire)) return;
            seen = fill_gen_.load(std::memory_order_acquire);
            bank = fill_bank_;
            sources = fill_sources_;
            count = fill_count_;
        }
        FillChunkShare(index, config_.fill_threads, bank, sources, count, geom_.slot_bytes);
        if (fill_remaining_.fetch_sub(1, std::memory_order_acq_rel) == 1) {
            // Wake the staging worker: the chunk's host bank is complete.
            std::lock_guard<std::mutex> lk(mu_fill_);
            cv_fill_done_.notify_all();
        }
    }
}

std::vector<int> Qwen4ExpLayerStage::ColdExperts(const Qwen4ExpAdaptiveCache& cache, int layer) {
    std::vector<int> out;
    const int n = cache.NumExpertsPerLayer();
    out.reserve(static_cast<size_t>(n));
    for (int e = 0; e < n; ++e)
        if (cache.GetSlot(layer, e) < 0) out.push_back(e);
    return out;
}

uint64_t Qwen4ExpLayerStage::Begin(int layer, const std::vector<int>& expert_ids,
                                   const std::vector<const uint8_t*>& sources,
                                   sycl::queue& copy_q, sycl::event arena_reader) {
    if (!IsAllocated()) throw std::logic_error("Qwen4ExpLayerStage is not allocated");
    if (expert_ids.size() != sources.size())
        throw std::invalid_argument("Qwen4ExpLayerStage source count mismatch");
    if (static_cast<int>(expert_ids.size()) > config_.max_slots)
        throw std::invalid_argument("Qwen4ExpLayerStage epoch exceeds the arena capacity");
    if (!copy_q.has_property<sycl::property::queue::in_order>())
        throw std::invalid_argument("Qwen4ExpLayerStage copy queue must be in order");
    if (copy_q_ && copy_q_ != &copy_q)
        throw std::invalid_argument("Qwen4ExpLayerStage copy queue changed between epochs");

    std::lock_guard<std::mutex> lk(mu_);
    if (job_active_.load()) throw std::logic_error("Qwen4ExpLayerStage epoch still in flight");
    if (failed_) throw std::runtime_error("Qwen4ExpLayerStage previous epoch failed: " + error_);
    copy_q_ = &copy_q;

    const uint64_t epoch = ++epoch_serial_;
    // Build the epoch map on the calling thread; the worker never touches it.
    std::fill(slot_of_.begin(), slot_of_.end(), -1);
    epoch_ids_.clear();
    in_ids_.clear();
    in_sources_.clear();
    uint64_t skipped = 0;
    for (size_t i = 0; i < expert_ids.size(); ++i) {
        const int e = expert_ids[i];
        const uint8_t* src = sources[i];
        if (e < 0 || e >= config_.num_experts_per_layer || !src) {
            ++skipped;
            continue;
        }
        slot_of_[static_cast<size_t>(e)] = static_cast<int32_t>(in_ids_.size());
        epoch_ids_.push_back(e);
        in_ids_.push_back(e);
        in_sources_.push_back(src);
    }
    current_epoch_ = epoch;
    current_layer_ = layer;
    in_layer_ = layer;
    in_epoch_ = epoch;
    // Gate the first H2D behind the previous epoch's last arena consumer.
    in_has_reader_ = false;
    if (arena_reader != sycl::event{}) {
        in_reader_ = arena_reader;
        in_has_reader_ = true;
    } else if (arena_read_) {
        in_reader_ = arena_reader_;
        in_has_reader_ = true;
    }
    arena_read_ = false;
    arena_reader_ = sycl::event{};
    last_event_ = sycl::event{};
    last_event_valid_ = false;
    chunk_events_.assign(
        static_cast<size_t>((static_cast<int>(in_ids_.size()) + config_.chunk_slots - 1) /
                            config_.chunk_slots),
        sycl::event{});
    stats_.skipped_no_source += skipped;  // mu_ is already held
    job_present_ = true;
    job_active_.store(true);
    cv_job_.notify_one();
    return epoch;
}

uint64_t Qwen4ExpLayerStage::BeginFromArena(int layer, const std::vector<int>& expert_ids,
                                            const Qwen4ExpHostArena& arena, sycl::queue& copy_q,
                                            sycl::event arena_reader) {
    std::vector<const uint8_t*> sources;
    sources.reserve(expert_ids.size());
    for (int e : expert_ids) sources.push_back(arena.GetExpertPointer(layer, e));
    return Begin(layer, expert_ids, sources, copy_q, arena_reader);
}

bool Qwen4ExpLayerStage::IsReady(uint64_t epoch) const noexcept {
    return epoch != 0 && done_epoch_.load(std::memory_order_acquire) >= epoch;
}

void Qwen4ExpLayerStage::WaitReady(uint64_t epoch) {
    if (epoch == 0) return;
    std::unique_lock<std::mutex> lk(mu_);
    cv_done_.wait(lk, [&] {
        return done_epoch_.load(std::memory_order_acquire) >= epoch || quit_;
    });
    if (failed_) throw std::runtime_error("Qwen4ExpLayerStage: " + error_);
    if (done_epoch_.load(std::memory_order_acquire) < epoch)
        throw std::runtime_error("Qwen4ExpLayerStage cancelled");
}

sycl::event Qwen4ExpLayerStage::ReadyEvent(uint64_t epoch) const {
    std::lock_guard<std::mutex> lk(mu_);
    if (epoch == 0 || epoch != current_epoch_ || failed_ || !last_event_valid_)
        return sycl::event{};
    return last_event_;
}

sycl::event Qwen4ExpLayerStage::ChunkEvent(uint64_t epoch, int chunk) const {
    if (epoch == 0 || epoch != current_epoch_ || chunk < 0 ||
        chunk >= static_cast<int>(chunk_events_.size()))
        return sycl::event{};
    return chunk_events_[static_cast<size_t>(chunk)];
}

int Qwen4ExpLayerStage::ChunkCount(uint64_t epoch) const {
    if (epoch == 0 || epoch != current_epoch_) return 0;
    return static_cast<int>(chunk_events_.size());
}

void Qwen4ExpLayerStage::RecordArenaConsumer(uint64_t epoch, const sycl::event& consumer) {
    std::lock_guard<std::mutex> lk(mu_);
    if (epoch != current_epoch_)
        throw std::invalid_argument("Qwen4ExpLayerStage consumer epoch is stale");
    arena_read_ = true;
    arena_reader_ = consumer;
}

Qwen4ExpLayerStage::RouteDelta Qwen4ExpLayerStage::MissedExperts(
    uint64_t epoch, const int* route_ids, int count) const {
    RouteDelta delta;
    if (!route_ids || count <= 0 || epoch == 0 || epoch != current_epoch_) {
        for (int i = 0; i < count; ++i) delta.missed.push_back(route_ids[i]);
        return delta;
    }
    for (int i = 0; i < count; ++i) {
        const int e = route_ids[i];
        if (SlotOf(epoch, e) >= 0) {
            ++delta.staged_hits;
            continue;
        }
        bool seen = false;
        for (int m : delta.missed)
            if (m == e) { seen = true; break; }
        if (!seen) delta.missed.push_back(e);
    }
    return delta;
}

int Qwen4ExpLayerStage::SlotOf(uint64_t epoch, int expert_id) const {
    if (epoch == 0 || epoch != current_epoch_ || expert_id < 0 ||
        expert_id >= config_.num_experts_per_layer)
        return -1;
    return slot_of_[static_cast<size_t>(expert_id)];
}

ops::MoeExpertSlotDev Qwen4ExpLayerStage::MakeSlot(const uint8_t* base, int expert,
                                                   const LayerNestedMetadata* m) const {
    ops::MoeExpertSlotDev d{};
    if (!base) return d;
    d.gate_up_packed = base + geom_.gu_packed_offset;
    d.down_packed = base + geom_.dn_packed_offset;
    d.gate_up_nested.codes = base + geom_.gu_absmax_offset;
    d.down_nested.codes = base + geom_.dn_absmax_offset;
    if (!m) return d;
    if (!m->valid || expert < 0 || !m->gu_nested_absmax_dev || !m->dn_nested_absmax_dev ||
        !m->gu_nested_quant_map_dev || !m->dn_nested_quant_map_dev ||
        m->gu_nested_blocksize != 256 || m->dn_nested_blocksize != 256 ||
        static_cast<size_t>(expert + 1) * geom_.gu_absmax_count / 256 > m->gu_nested_absmax.size() ||
        static_cast<size_t>(expert + 1) * geom_.dn_absmax_count / 256 > m->dn_nested_absmax.size())
        throw std::logic_error("Qwen4ExpLayerStage descriptor requires this expert's device nested metadata");
    d.gate_up_nested.nested_absmax =
        m->gu_nested_absmax_dev + static_cast<size_t>(expert) * geom_.gu_absmax_count / 256;
    d.down_nested.nested_absmax =
        m->dn_nested_absmax_dev + static_cast<size_t>(expert) * geom_.dn_absmax_count / 256;
    d.gate_up_nested.nested_quant_map = m->gu_nested_quant_map_dev;
    d.down_nested.nested_quant_map = m->dn_nested_quant_map_dev;
    d.gate_up_nested.offset = m->gu_nested_offset;
    d.down_nested.offset = m->dn_nested_offset;
    d.gate_up_nested.blocksize = d.down_nested.blocksize = 256;
    return d;
}

ops::MoeExpertSlotDev Qwen4ExpLayerStage::GetSlotDev(uint64_t epoch, int expert_id,
                                                     const LayerNestedMetadata* nested) const {
    const int slot = SlotOf(epoch, expert_id);
    if (slot < 0) return {};
    return MakeSlot(arena_dev_ + static_cast<size_t>(slot) * geom_.slot_bytes, expert_id,
                    nested);
}

void Qwen4ExpLayerStage::WaitIdle() {
    std::unique_lock<std::mutex> lk(mu_);
    cv_done_.wait(lk, [&] { return !job_active_.load(std::memory_order_acquire); });
}

void Qwen4ExpLayerStage::Reset() {
    WaitIdle();
    std::lock_guard<std::mutex> lk(mu_);
    current_epoch_ = 0;
    current_layer_ = -1;
    std::fill(slot_of_.begin(), slot_of_.end(), -1);
    epoch_ids_.clear();
    chunk_events_.clear();
    last_event_ = sycl::event{};
    last_event_valid_ = false;
}

LayerStageStats Qwen4ExpLayerStage::Stats() const {
    std::lock_guard<std::mutex> lk(mu_);
    return stats_;
}

void Qwen4ExpLayerStage::WorkerLoop() {
    std::unique_lock<std::mutex> lk(mu_);
    for (;;) {
        cv_job_.wait(lk, [&] { return quit_ || job_present_; });
        if (quit_ && !job_present_) return;
        work_ids_.swap(in_ids_);
        work_sources_.swap(in_sources_);
        work_layer_ = in_layer_;
        const uint64_t epoch = in_epoch_;
        const bool has_reader = in_has_reader_;
        const sycl::event reader = in_reader_;
        sycl::queue* q = copy_q_;
        job_present_ = false;
        lk.unlock();
        try {
            if (work_ids_.empty()) {
                std::lock_guard<std::mutex> lk2(mu_);
                stats_.epochs++;
                stats_.last_fill_ms = stats_.last_bank_wait_ms = 0.0;
                stats_.last_submit_ms = stats_.last_ready_ms = 0.0;
            } else if (q) {
                RunJob(epoch, work_layer_, has_reader, reader, *q);
            } else {
                throw std::logic_error("Qwen4ExpLayerStage has no copy queue");
            }
        } catch (const std::exception& e) {
            FailJob(epoch, e.what());
        } catch (...) {
            FailJob(epoch, "unknown layer stage failure");
        }
        lk.lock();
        job_active_.store(false, std::memory_order_release);
        cv_done_.notify_all();
    }
}

void Qwen4ExpLayerStage::RunJob(uint64_t epoch, int layer, bool has_reader,
                                const sycl::event& reader, sycl::queue& q) {
    const auto begin_tp = Clock::now();
    const int total = static_cast<int>(work_ids_.size());
    const int chunks =
        (total + config_.chunk_slots - 1) / config_.chunk_slots;
    const size_t chunk_stride =
        static_cast<size_t>(config_.chunk_slots) * geom_.slot_bytes;
    double bank_wait_ms = 0, fill_ms = 0, submit_ms = 0, ready_ms = 0;
    sycl::event last{};
    bool has_last = false;
    for (int c = 0; c < chunks; ++c) {
        const int s = c * config_.chunk_slots;
        const int n = std::min(config_.chunk_slots, total - s);
        const int b = c % config_.fill_banks;
        if (bank_event_valid_[static_cast<size_t>(b)]) {
            const auto t0 = Clock::now();
            bank_events_[static_cast<size_t>(b)].wait_and_throw();
            bank_wait_ms += ms_since(t0);
            bank_event_valid_[static_cast<size_t>(b)] = false;
        }
        uint8_t* bank = host_banks_ + static_cast<size_t>(b) * chunk_stride;
        {
            const auto t0 = Clock::now();
            const uint8_t* const* sources = work_sources_.data() + s;
            if (fill_workers_.empty()) {
                // Serial fill: the pre-pool behavior verbatim.
                for (int i = 0; i < n; ++i)
                    std::memcpy(bank + static_cast<size_t>(i) * geom_.slot_bytes,
                                sources[i], geom_.slot_bytes);
            } else {
                {
                    std::lock_guard<std::mutex> lk(mu_fill_);
                    fill_bank_ = bank;
                    fill_sources_ = sources;
                    fill_count_ = n;
                    fill_remaining_.store(static_cast<int>(fill_workers_.size()),
                                          std::memory_order_relaxed);
                    fill_gen_.fetch_add(1, std::memory_order_release);
                }
                cv_fill_.notify_all();
                // The staging worker owns share 0 so a 2-thread pool still
                // splits the chunk; it then waits for the helpers.
                FillChunkShare(0, config_.fill_threads, bank, sources, n, geom_.slot_bytes);
                std::unique_lock<std::mutex> lk(mu_fill_);
                cv_fill_done_.wait(lk, [&] {
                    return fill_remaining_.load(std::memory_order_acquire) == 0;
                });
            }
            fill_ms += ms_since(t0);
        }
        uint8_t* dst = arena_dev_ + static_cast<size_t>(s) * geom_.slot_bytes;
        const size_t bytes = static_cast<size_t>(n) * geom_.slot_bytes;
        sycl::event ev = q.submit([&](sycl::handler& h) {
            if (c == 0 && has_reader) h.depends_on(reader);
            h.memcpy(dst, bank, bytes);
        });
        bank_events_[static_cast<size_t>(b)] = ev;
        bank_event_valid_[static_cast<size_t>(b)] = true;
        if (c < static_cast<int>(chunk_events_.size())) chunk_events_[static_cast<size_t>(c)] = ev;
        last = ev;
        has_last = true;
        submit_ms = ms_since(begin_tp);
    }
    (void)layer;
    if (has_last) {
        const auto t0 = Clock::now();
        last.wait_and_throw();
        ready_ms = ms_since(begin_tp);
        (void)t0;
    } else {
        ready_ms = ms_since(begin_tp);
    }
    {
        std::lock_guard<std::mutex> lk(mu_);
        stats_.epochs++;
        const uint64_t staged_bytes =
            static_cast<uint64_t>(total) * static_cast<uint64_t>(geom_.slot_bytes);
        stats_.staged_experts += static_cast<uint64_t>(total);
        stats_.staged_bytes += staged_bytes;
        stats_.last_fill_ms = fill_ms;
        stats_.last_bank_wait_ms = bank_wait_ms;
        stats_.last_submit_ms = submit_ms;
        stats_.last_ready_ms = ready_ms;
    }
    last_event_ = last;
    last_event_valid_ = has_last;
    done_epoch_.store(epoch, std::memory_order_release);
}

void Qwen4ExpLayerStage::FailJob(uint64_t epoch, const std::string& message) {
    std::lock_guard<std::mutex> lk(mu_);
    failed_ = true;
    error_ = message;
    // Fail closed: nothing staged is served from a partially written arena.
    std::fill(slot_of_.begin(), slot_of_.end(), -1);
    epoch_ids_.clear();
    last_event_valid_ = false;
    done_epoch_.store(epoch, std::memory_order_release);
}

} // namespace aila
