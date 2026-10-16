// Qwen4Exp arena-source DMA bench (role P8, 2026-10-16).
//
// Question: production prefill streams every cold expert through
//   arena (private large pages) --memcpy--> pinned bank --DMA--> VRAM
// so each byte crosses PCIe once but is touched twice on the host plus a
// second host write. If the copy engine could read the large-page arena
// directly, the fill and the pinned bank would disappear. This bench measures
// the three source paths at production granularity on a small buffer (1 GiB
// by default) so the question is settled without a full model run:
//
//   bank  large-page source -> pinned bank memcpy -> H2D (today's path)
//   lg    large-page source -> H2D directly (the c1-e1 arm of 2026-10-13)
//   pin   pinned (sycl::malloc_host) source -> H2D directly
//
// All transfers use cgh.memcpy so large-page, pinned and device pointers take
// the same runtime path as the committed queue-overlap probe
// (tests/ops/Qwen4ExpKernelTests.cpp:4542-4591). Sources are allocated as N
// blocks of --block-mib (default 256 MiB, mirroring Qwen4ExpHostArena's
// per_chunk cap at src/models/Qwen4ExpHostArena.cpp:134) so the multi-block
// pinned-allocation pattern of AILA_Q4E_HOST_ARENA_PINNED=1 is exercised.
//
// Guards: the two sources carry different deterministic patterns; every arm is
// verified by full D2H readback + memcmp against its own source and a mismatch
// against the other source. No allocation inside the timed loop.
//
// Inputs:  --total-mib 1024 --block-mib 256 --group-slots 16 --banks 2
//          --reps 3 --arms bank,lg,pin
// Output:  [P8 ALLOC] [P8 TIME] [P8 VERIFY] [P8 MEDIAN] [P8 EXTRAP] [P8 DONE]
// ASCII only. GPU execution belongs to the coordinator; this file only needs a
// syntax check here.

#include "core/Context.hpp"

#include <sycl/sycl.hpp>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using Clock = std::chrono::steady_clock;

// Production expert slot bytes (see perf/qwen4exp/q4e_prefill_block_bench.cpp).
constexpr size_t kSlotBytes = 2534400;
// Bytes staged per 4096-token prefill block (P1 supply breakdown section 1).
constexpr double kRefBlockGb = 49.07;

uint32_t mix(uint32_t x) {
    x ^= x >> 16;
    x *= 0x7feb352dU;
    x ^= x >> 15;
    x *= 0x846ca68bU;
    return x ^ (x >> 16);
}
double ms_since(Clock::time_point a) {
    return std::chrono::duration<double, std::milli>(Clock::now() - a).count();
}
uint64_t span_ns(const sycl::event& e) {
    return e.get_profiling_info<sycl::info::event_profiling::command_end>() -
           e.get_profiling_info<sycl::info::event_profiling::command_start>();
}
double median(std::vector<double> v) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}
void check(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }

struct Params {
    size_t total_mib = 1024;
    size_t block_mib = 256;
    int group_slots = 16;
    int banks = 2;
    int reps = 3;
    bool arm_bank = true;
    bool arm_lg = true;
    bool arm_pin = true;
};

struct Source {
    std::vector<uint8_t*> blocks;
    size_t block_bytes = 0;
    size_t bytes = 0;
    bool ok = false;
};

struct Sample {
    double wall = 0, fill = 0, bankwait = 0, dma = 0;
    size_t bytes = 0;
};

Source allocate_blocks(Context& ctx, const Params& p, bool pinned) {
    Source s;
    s.block_bytes = p.block_mib << 20;
    const size_t blocks = std::max<size_t>(1, p.total_mib / p.block_mib);
    const auto t0 = Clock::now();
    for (size_t b = 0; b < blocks; ++b) {
        try {
            uint8_t* ptr = static_cast<uint8_t*>(
                pinned ? ctx.alloc_host(s.block_bytes) : ctx.alloc_large_host(s.block_bytes));
            s.blocks.push_back(ptr);
        } catch (const std::exception& e) {
            std::cout << "[P8 ALLOC] kind=" << (pinned ? "pin" : "large") << " block=" << b
                      << " ok=0 err=" << e.what() << '\n';
        }
    }
    s.bytes = s.blocks.size() * s.block_bytes;
    s.ok = !s.blocks.empty();
    std::cout << "[P8 ALLOC] kind=" << (pinned ? "pin" : "large") << " blocks=" << s.blocks.size()
              << "/" << blocks << " block_mib=" << p.block_mib << " bytes=" << s.bytes
              << " ms=" << ms_since(t0) << " pinned_current_mib="
              << double(ctx.current_pinned_host_bytes()) / 1048576.0 << " pinned_peak_mib="
              << double(ctx.peak_pinned_host_bytes()) / 1048576.0
              << " pinned_count=" << ctx.pinned_host_allocation_count() << '\n';
    return s;
}

void fill_blocks(const Source& s, uint32_t seed) {
    for (size_t bi = 0; bi < s.blocks.size(); ++bi) {
        auto* words = reinterpret_cast<uint32_t*>(s.blocks[bi]);
        const size_t count = s.block_bytes / sizeof(uint32_t);
        const uint32_t base = seed + uint32_t(bi) * 1000003u;
        for (size_t i = 0; i < count; ++i) words[i] = mix(uint32_t(i) + base);
    }
}

// arm 0 = bank (fill + H2D per ping-pong bank), 1 = direct large page,
// 2 = direct pinned. The bank arm waits the previous copy out of the same bank
// before refilling it exactly like the production reader fence.
void run_pass(sycl::queue& copy, uint8_t* dev, const Source& src, uint8_t* banks,
              const Params& p, int arm, Sample& s) {
    const size_t group_bytes = size_t(p.group_slots) * kSlotBytes;
    std::vector<sycl::event> events;
    events.reserve(src.blocks.size() * (src.block_bytes / group_bytes + 1));
    std::vector<sycl::event> bank_ev(size_t(p.banks));
    const auto t0 = Clock::now();
    size_t dev_off = 0;
    size_t batch = 0;
    for (uint8_t* blk : src.blocks) {
        for (size_t off = 0; off < src.block_bytes; off += group_bytes) {
            const size_t n = std::min(group_bytes, src.block_bytes - off);
            if (arm == 0) {
                const size_t bank = batch % size_t(p.banks);
                uint8_t* bank_host = banks + bank * group_bytes;
                if (bank_ev[bank] != sycl::event{}) {
                    const auto w0 = Clock::now();
                    bank_ev[bank].wait_and_throw();
                    s.bankwait += ms_since(w0);
                }
                const auto f0 = Clock::now();
                std::memcpy(bank_host, blk + off, n);
                s.fill += ms_since(f0);
                bank_ev[bank] = copy.submit(
                    [&](sycl::handler& cgh) { cgh.memcpy(dev + dev_off, bank_host, n); });
                events.push_back(bank_ev[bank]);
            } else {
                uint8_t* src_ptr = blk + off;
                events.push_back(copy.submit(
                    [&](sycl::handler& cgh) { cgh.memcpy(dev + dev_off, src_ptr, n); }));
            }
            dev_off += n;
            ++batch;
        }
    }
    copy.wait_and_throw();
    s.wall = ms_since(t0);
    for (const auto& e : events) s.dma += double(span_ns(e)) * 1e-6;
    s.bytes = dev_off;
}

struct VerifyResult {
    bool match_src = false;
    bool differs_other = false;
};

VerifyResult verify_arm(Context& ctx, uint8_t* dev, const Source& src, const Source& other,
                        std::vector<uint8_t>& scratch) {
    scratch.resize(src.bytes);
    ctx.memcpy_d2h(scratch.data(), dev, src.bytes);
    VerifyResult r;
    r.match_src = true;
    size_t off = 0;
    for (uint8_t* blk : src.blocks) {
        if (std::memcmp(scratch.data() + off, blk, src.block_bytes) != 0) {
            r.match_src = false;
            break;
        }
        off += src.block_bytes;
    }
    r.differs_other = std::memcmp(scratch.data(), other.blocks[0], 4096) != 0;
    return r;
}
}  // namespace

int main(int argc, char** argv) try {
    Params p;
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        if (i + 1 >= argc) throw std::runtime_error("missing value after " + key);
        const std::string value = argv[++i];
        if (key == "--total-mib") p.total_mib = size_t(std::stoull(value));
        else if (key == "--block-mib") p.block_mib = size_t(std::stoull(value));
        else if (key == "--group-slots") p.group_slots = std::stoi(value);
        else if (key == "--banks") p.banks = std::stoi(value);
        else if (key == "--reps") p.reps = std::stoi(value);
        else if (key == "--arms") {
            p.arm_bank = value.find("bank") != std::string::npos;
            p.arm_lg = value.find("lg") != std::string::npos;
            p.arm_pin = value.find("pin") != std::string::npos;
        } else {
            throw std::runtime_error("unknown flag " + key);
        }
    }
    check(p.total_mib > 0 && p.block_mib > 0, "total/block mib must be positive");
    check(p.group_slots > 0 && p.group_slots <= 64, "group slots 1..64");
    check(p.banks > 0 && p.banks <= 8, "banks 1..8");
    check(p.reps >= 1, "reps >= 1");

    Context ctx(nullptr, true);
    sycl::queue& q = ctx.queue();
    auto* copy = new sycl::queue(
        q.get_context(), q.get_device(),
        sycl::property_list{sycl::property::queue::in_order(),
                            sycl::ext::intel::property::queue::immediate_command_list(),
                            sycl::property::queue::enable_profiling()});
    struct Drain {
        Context& ctx;
        sycl::queue* copy;
        ~Drain() {
            try { copy->wait_and_throw(); } catch (...) {}
            try { ctx.synchronize(); } catch (...) {}
            delete copy;
        }
    } drain{ctx, copy};

    const size_t group_bytes = size_t(p.group_slots) * kSlotBytes;
    const size_t total_bytes = std::max<size_t>(1, p.total_mib / p.block_mib) * (p.block_mib << 20);
    auto* dev = static_cast<uint8_t*>(ctx.alloc_device(total_bytes));
    auto* bank_host = static_cast<uint8_t*>(ctx.alloc_host(size_t(p.banks) * group_bytes));

    const Source large = allocate_blocks(ctx, p, false);
    const Source pin = allocate_blocks(ctx, p, true);
    check(large.ok && pin.ok, "source allocation failed");
    check(large.bytes == pin.bytes, "source sizes differ");
    fill_blocks(large, 37);
    fill_blocks(pin, 91273);

    std::cout << std::unitbuf << std::fixed << std::setprecision(3)
              << "[P8 CONFIG] device=" << q.get_device().get_info<sycl::info::device::name>()
              << " total_mib=" << (total_bytes >> 20) << " block_mib=" << p.block_mib
              << " group_slots=" << p.group_slots << " group_bytes=" << group_bytes
              << " banks=" << p.banks << " reps=" << p.reps << " device_bytes="
              << ctx.current_allocated_bytes() << '\n';

    struct Arm {
        const char* name;
        int id;
        const Source* src;
        const Source* other;
        bool enabled;
    };
    const Arm arms[3] = {
        {"bank", 0, &large, &pin, p.arm_bank},
        {"lg", 1, &large, &pin, p.arm_lg},
        {"pin", 2, &pin, &large, p.arm_pin},
    };
    std::vector<uint8_t> scratch;
    bool all_verify = true;
    for (const Arm& arm : arms) {
        if (!arm.enabled) continue;
        bool arm_verify = true;
        std::vector<double> walls, fills, waits, dmas, gbps;
        for (int rep = -1; rep < p.reps; ++rep) {
            Sample s;
            run_pass(*copy, dev, *arm.src, bank_host, p, arm.id, s);
            if (rep < 0) continue;  // warmup
            const VerifyResult vr = verify_arm(ctx, dev, *arm.src, *arm.other, scratch);
            arm_verify = arm_verify && vr.match_src && vr.differs_other;
            walls.push_back(s.wall);
            fills.push_back(s.fill);
            waits.push_back(s.bankwait);
            dmas.push_back(s.dma);
            gbps.push_back(double(s.bytes) / (s.wall * 1e6));
            std::cout << "[P8 TIME] arm=" << arm.name << " sample=" << rep
                      << " wall_ms=" << s.wall << " fill_ms=" << s.fill
                      << " bankwait_ms=" << s.bankwait << " dma_ms=" << s.dma
                      << " wall_gbps=" << (double(s.bytes) / (s.wall * 1e6))
                      << " match_src=" << (vr.match_src ? 1 : 0)
                      << " differs_other=" << (vr.differs_other ? 1 : 0) << '\n';
        }
        const double wall = median(walls);
        std::cout << "[P8 VERIFY] arm=" << arm.name << " match_src="
                  << (arm_verify ? 1 : 0) << " samples=" << walls.size() << '\n';
        std::cout << "[P8 MEDIAN] arm=" << arm.name << " wall_ms=" << wall
                  << " fill_ms=" << median(fills) << " bankwait_ms=" << median(waits)
                  << " dma_ms=" << median(dmas) << " wall_gbps=" << median(gbps)
                  << " dma_gbps=" << (median(dmas) > 0
                                          ? double(large.bytes) / (median(dmas) * 1e6)
                                          : 0.0)
                  << " extrap_block_s=" << (kRefBlockGb / median(gbps)) << '\n';
        all_verify = all_verify && arm_verify;
    }
    std::cout << "[P8 GUARD] bitwise=" << (all_verify ? "pass" : "FAIL") << '\n';
    std::cout << "[P8 DONE]\n";
    if (!all_verify) return 1;
    return 0;
} catch (const std::exception& e) {
    std::cerr << "[P8 FAIL] " << e.what() << '\n';
    return 1;
}
