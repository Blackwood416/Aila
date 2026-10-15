// Qwen4Exp prefill block bench (role E1, 2026-10-15).
//
// Production-shaped supply split for 4096-token prefill blocks. One "block" =
// `--layers` layers x `--staged-per-layer` cold experts streamed through
// `--banks` x `--slots` ping-pong staging banks with the real production
// kernels (expert-major prefill gate_up/down, pair/single weight reuse).
//
// Buckets per pass:
//   fill_ms      pageable source -> pinned staging bank memcpy
//   copy_ms      copy-engine H2D spans (sum of per-batch memcpy events)
//   bankwait_ms  host blocking on the previous H2D into the same bank
//   gu_ms/down_ms production kernel device spans
//   wall_ms      host wall of the whole pass
//
// Arms:
//   copy    supply only: fill -> H2D through N banks, no readers/fences
//   block   production order: fill -> H2D -> per-batch kernels -> next bank,
//           including the reader fence (next H2D waits the bank's consumer)
//   kernel  all experts resident (no H2D), one gu+down launch pair per layer
//
// Guards: finite scan on written rows, sentinel tail check, bitwise repeat of
// the last two samples of each arm. ASCII only; no allocation in the timed
// loop; pageable fill source (mmap-like) on purpose.

#include "core/Context.hpp"
#include "core/Tensor.hpp"
#include "ops/MoeExpertOps.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {
using bf16 = sycl::ext::oneapi::bfloat16;
using Clock = std::chrono::steady_clock;

constexpr int H = 2560;
constexpr int I = 640;
constexpr int TOPK = 8;
constexpr int BLOCKSIZE = 64;
constexpr size_t Packed = 2 * size_t(I) * H / 2;
constexpr size_t Codes = 2 * size_t(I) * H / 64;
constexpr size_t Nested = Codes / 256;
constexpr size_t DownPacked = size_t(H) * I / 2;
constexpr size_t DownCodes = size_t(H) * I / 64;
constexpr size_t DownNested = DownCodes / 256;
constexpr size_t DownOffset = Packed + Codes;
constexpr size_t DownCodesOffset = DownOffset + DownPacked;
constexpr size_t SlotBytes = DownCodesOffset + DownCodes;
static_assert(SlotBytes == 2534400, "slot layout must match the production expert slot");

constexpr std::array<float, 16> Qmap{
    -1.f, -.6961928f, -.52507305f, -.3949175f, -.28444138f, -.18477343f,
    -.091050036f, 0.f, .0795803f, .1609302f, .2461123f, .33791524f,
    .44070983f, .562617f, .72295684f, 1.f};

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
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}
void check(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }

struct Params {
    int layers = 48;
    int staged_per_layer = 403;  // cold experts staged per layer (14K production arm)
    int cold_acts = 593;         // routed acts served by those cold experts
    int tokens = 4096;
    int banks = 2;
    int slots = 16;
    int reps = 3;
    bool fill = true;
    bool arm_copy = true;
    bool arm_block = true;
    bool arm_kernel = true;
};

struct Sample {
    double wall = 0, fill = 0, copy = 0, bankwait = 0, gu = 0, down = 0;
};

struct Buffers {
    Tensor staging;   // [banks][slots] expert slots (device)
    Tensor resident;  // [staged_per_layer] expert slots (device, kernel arm)
    Tensor metadata;  // nested absmax per slot + 256 LUT + 16 qmap
    Tensor slots_dev;
    Tensor tokens_dev;
    Tensor ranks_dev;
    Tensor x_dev;
    Tensor h_dev;
    Tensor down_dev;
    std::vector<ops::MoeExpertSlotDev> slots_host;
    std::vector<int> tokens_host;
    std::vector<int> ranks_host;
    std::vector<uint8_t> source;  // pageable fill source
    uint8_t* host_banks = nullptr;
};

int assign_share(int total, int index, int count) {
    return (total * (index + 1)) / count - (total * index) / count;
}

// Cold-act mix: pairs share one expert slot for two acts, singles are the
// remainder. Layout follows the production ALU order: [pairs 2*np][singles].
struct Mix {
    int singles = 0;
    int pairs = 0;
    int acts() const { return singles + 2 * pairs; }
};

void run_pass(Context& ctx, sycl::queue& copy, Buffers& b, const Params& p, int arm,
              Sample& s) {
    sycl::queue& q = ctx.queue();
    // arm 0 = copy only, 1 = block (fenced), 2 = kernel only (resident).
    const int batches = (p.staged_per_layer + p.slots - 1) / p.slots;
    const int layer_acts = p.cold_acts;
    const Mix layer_mix{2 * p.staged_per_layer - p.cold_acts,
                        p.cold_acts - p.staged_per_layer};
    std::vector<sycl::event> bank_copy(size_t(p.banks));
    std::vector<sycl::event> bank_reader(size_t(p.banks));
    std::vector<sycl::event> copies, kernels;
    copies.reserve(size_t(p.layers) * size_t(batches));
    kernels.reserve(size_t(p.layers) * size_t(batches) * 2);
    const auto t0 = Clock::now();
    for (int layer = 0; layer < p.layers; ++layer) {
        int act_base = 0;
        for (int bt = 0; bt < batches; ++bt) {
            const int n = std::min(p.slots, p.staged_per_layer - bt * p.slots);
            const int bank = bt % p.banks;
            const Mix mix{assign_share(layer_mix.singles, bt, batches),
                          assign_share(layer_mix.pairs, bt, batches)};
            const int acts = mix.acts();
            sycl::event ready{};
            if (arm != 2) {
                auto* bank_dev =
                    b.staging.data_as<uint8_t>() + size_t(bank) * size_t(p.slots) * SlotBytes;
                auto* bank_host = b.host_banks + size_t(bank) * size_t(p.slots) * SlotBytes;
                if (bank_copy[size_t(bank)] != sycl::event{}) {
                    const auto w0 = Clock::now();
                    bank_copy[size_t(bank)].wait_and_throw();
                    s.bankwait += ms_since(w0);
                }
                if (p.fill) {
                    const auto f0 = Clock::now();
                    std::memcpy(bank_host,
                                b.source.data() + size_t(bank) * size_t(p.slots) * SlotBytes,
                                size_t(n) * SlotBytes);
                    s.fill += ms_since(f0);
                }
                if (arm == 1 && bank_reader[size_t(bank)] != sycl::event{}) {
                    // Production: the device write waits its bank's last consumer.
                    ready = copy.submit([&](sycl::handler& cgh) {
                        cgh.depends_on(bank_reader[size_t(bank)]);
                        cgh.memcpy(bank_dev, bank_host, size_t(n) * SlotBytes);
                    });
                } else {
                    ready = copy.memcpy(bank_dev, bank_host, size_t(n) * SlotBytes);
                }
                bank_copy[size_t(bank)] = ready;
                copies.push_back(ready);
            }
            if (acts <= 0) continue;
            auto* slot_base = arm == 2
                                  ? b.resident.data_as<uint8_t>()
                                  : b.staging.data_as<uint8_t>() +
                                        size_t(bank) * size_t(p.slots) * SlotBytes;
            for (int i = 0; i < acts; ++i) {
                const int slot = i < 2 * mix.pairs ? i / 2 : mix.pairs + (i - 2 * mix.pairs);
                const int slot_clamped = std::min(slot, n - 1);
                const int slot_global = bt * p.slots + slot_clamped;
                auto* base = slot_base + size_t(slot_global) * SlotBytes;
                auto* meta = b.metadata.data_as<float>();
                auto* lut = meta + size_t(p.banks * p.slots) * (Nested + DownNested);
                auto& d = b.slots_host[size_t(act_base + i)];
                d.gate_up_packed = base;
                d.down_packed = base + DownOffset;
                d.gate_up_absmax = nullptr;
                d.down_absmax = nullptr;
                d.gate_up_nested = {base + Packed, meta + size_t(slot_global) * Nested, lut,
                                    .001f, 256};
                d.down_nested = {base + DownCodesOffset,
                                 meta + size_t(p.banks * p.slots) * Nested +
                                     size_t(slot_global) * DownNested,
                                 lut, .001f, 256};
                b.tokens_host[size_t(act_base + i)] = (act_base + i) / TOPK;
                b.ranks_host[size_t(act_base + i)] = (act_base + i) % TOPK;
            }
            q.memcpy(b.slots_dev.data_as<ops::MoeExpertSlotDev>() + act_base,
                     b.slots_host.data() + act_base,
                     size_t(acts) * sizeof(ops::MoeExpertSlotDev));
            q.memcpy(b.tokens_dev.data_as<int>() + act_base,
                     b.tokens_host.data() + act_base, size_t(acts) * sizeof(int));
            q.memcpy(b.ranks_dev.data_as<int>() + act_base,
                     b.ranks_host.data() + act_base, size_t(acts) * sizeof(int));
            auto gu = ops::moe_expert_major_prefill_gate_up(
                ctx, b.slots_dev.data_as<ops::MoeExpertSlotDev>() + act_base,
                b.tokens_dev.data_as<int>() + act_base, b.ranks_dev.data_as<int>() + act_base,
                nullptr, b.x_dev.data_as<bf16>(), b.h_dev.data_as<bf16>(), acts, TOPK,
                H, I, BLOCKSIZE, &ready, true, mix.pairs, 0, 0, true);
            auto down = ops::moe_expert_major_prefill_down(
                ctx, b.slots_dev.data_as<ops::MoeExpertSlotDev>() + act_base,
                b.tokens_dev.data_as<int>() + act_base, b.ranks_dev.data_as<int>() + act_base,
                nullptr, b.h_dev.data_as<bf16>(), b.down_dev.data_as<float>(), acts, TOPK,
                I, H, BLOCKSIZE, &gu, true, mix.pairs, 0, 0, false, true);
            kernels.push_back(gu);
            kernels.push_back(down);
            if (arm == 1) bank_reader[size_t(bank)] = down;
            act_base += acts;
        }
        check(act_base == layer_acts, "layer act layout mismatch");
    }
    q.wait_and_throw();
    copy.wait_and_throw();
    s.wall = ms_since(t0);
    for (const auto& e : copies) s.copy += double(span_ns(e)) * 1e-6;
    for (size_t i = 0; i + 1 < kernels.size(); i += 2) {
        s.gu += double(span_ns(kernels[i])) * 1e-6;
        s.down += double(span_ns(kernels[i + 1])) * 1e-6;
    }
}

void guard_outputs(Context& ctx, Buffers& b, const Params& p,
                   std::vector<bf16>& h_host, std::vector<float>& down_host,
                   std::vector<bf16>& h_reference, std::vector<float>& down_reference,
                   bool& has_reference) {
    h_host.resize(size_t(p.cold_acts) * I);
    down_host.resize(size_t(p.cold_acts) * H);
    ctx.memcpy_d2h(h_host.data(), b.h_dev.data_as<bf16>(), h_host.size() * sizeof(bf16));
    ctx.memcpy_d2h(down_host.data(), b.down_dev.data_as<float>(),
                   down_host.size() * sizeof(float));
    for (float v : down_host) check(std::isfinite(v), "non-finite down output");
    for (auto v : h_host) check(std::isfinite(float(v)), "non-finite h output");
    std::vector<bf16> h_tail(64);
    std::vector<float> down_tail(64);
    ctx.memcpy_d2h(h_tail.data(), b.h_dev.data_as<bf16>() + size_t(p.cold_acts) * I,
                   h_tail.size() * sizeof(bf16));
    ctx.memcpy_d2h(down_tail.data(), b.down_dev.data_as<float>() + size_t(p.cold_acts) * H,
                   down_tail.size() * sizeof(float));
    for (auto v : h_tail) check(float(v) == -9.f, "h sentinel overwritten");
    for (float v : down_tail) check(v == -17.f, "down sentinel overwritten");
    if (has_reference) {
        check(h_host.size() == h_reference.size() && down_host.size() == down_reference.size(),
              "reference shape mismatch");
        check(std::memcmp(h_host.data(), h_reference.data(), h_host.size() * sizeof(bf16)) == 0,
              "h is not bitwise reproducible");
        check(std::memcmp(down_host.data(), down_reference.data(),
                          down_host.size() * sizeof(float)) == 0,
              "down is not bitwise reproducible");
    }
    h_reference = h_host;
    down_reference = down_host;
    has_reference = true;
}

void poison(Context& ctx, Buffers& b) {
    ctx.queue().fill(b.h_dev.data_as<bf16>(), bf16(-9.f),
                     b.h_dev.size_bytes() / sizeof(bf16));
    ctx.queue().fill(b.down_dev.data_as<float>(), -17.f,
                     b.down_dev.size_bytes() / sizeof(float));
    ctx.queue().wait_and_throw();
}

Params parse(int argc, char** argv) {
    Params p;
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        if (key == "--no-fill") { p.fill = false; continue; }
        if (i + 1 >= argc) throw std::runtime_error("missing value after " + key);
        const std::string value = argv[++i];
        if (key == "--layers") p.layers = std::stoi(value);
        else if (key == "--staged-per-layer") p.staged_per_layer = std::stoi(value);
        else if (key == "--cold-acts") p.cold_acts = std::stoi(value);
        else if (key == "--tokens") p.tokens = std::stoi(value);
        else if (key == "--banks") p.banks = std::stoi(value);
        else if (key == "--slots") p.slots = std::stoi(value);
        else if (key == "--reps") p.reps = std::stoi(value);
        else if (key == "--arms") {
            p.arm_copy = value.find("copy") != std::string::npos;
            p.arm_block = value.find("block") != std::string::npos;
            p.arm_kernel = value.find("kernel") != std::string::npos;
        } else {
            throw std::runtime_error("unknown flag " + key);
        }
    }
    check(p.layers > 0 && p.banks > 0 && p.banks <= 8, "bad layers/banks");
    check(p.slots > 0 && p.slots <= 64, "bad slots (1..64)");
    check(p.staged_per_layer >= 1 && p.staged_per_layer <= p.slots * p.layers * 4,
          "bad staged-per-layer");
    check(p.cold_acts >= p.staged_per_layer && p.cold_acts <= 2 * p.staged_per_layer,
          "cold-acts must be within [staged, 2*staged]");
    check(p.tokens > 0 && p.cold_acts <= p.tokens * TOPK, "cold acts exceed the token map");
    check(p.reps >= 1, "reps >= 1");
    return p;
}
}  // namespace

int main(int argc, char** argv) try {
    const Params p = parse(argc, argv);
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
        uint8_t* host;
        ~Drain() {
            try { copy->wait_and_throw(); } catch (...) {}
            try { ctx.synchronize(); } catch (...) {}
            ctx.free_host(host);
            delete copy;
        }
    } drain{ctx, copy, nullptr};

    Buffers b;
    auto bytes = [&](size_t n) {
        return Tensor::allocate(ctx, {int64_t(n)}, dnnl::memory::data_type::u8);
    };
    const size_t host_bank_bytes = size_t(p.banks) * size_t(p.slots) * SlotBytes;
    b.staging = bytes(host_bank_bytes);
    b.resident = bytes(size_t(p.staged_per_layer) * SlotBytes);
    const size_t meta_floats =
        size_t(p.banks * p.slots) * (Nested + DownNested) + 256 + 16;
    b.metadata = bytes(meta_floats * sizeof(float));
    b.slots_dev = bytes(size_t(p.cold_acts) * sizeof(ops::MoeExpertSlotDev));
    b.tokens_dev = bytes(size_t(p.cold_acts) * sizeof(int));
    b.ranks_dev = bytes(size_t(p.cold_acts) * sizeof(int));
    b.x_dev = bytes(size_t(p.tokens) * H * sizeof(bf16));
    b.h_dev = bytes(size_t(p.cold_acts) * I * sizeof(bf16) + 64 * I * sizeof(bf16));
    b.down_dev =
        bytes(size_t(p.cold_acts) * H * sizeof(float) + 64 * H * sizeof(float));
    b.slots_host.resize(size_t(p.cold_acts));
    b.tokens_host.resize(size_t(p.cold_acts));
    b.ranks_host.resize(size_t(p.cold_acts));
    b.host_banks = static_cast<uint8_t*>(ctx.alloc_host(host_bank_bytes));
    drain.host = b.host_banks;
    b.source.resize(host_bank_bytes);
    for (size_t w = 0; w < b.source.size() / 4; ++w)
        reinterpret_cast<uint32_t*>(b.source.data())[w] = mix(uint32_t(w) + 37);

    std::vector<float> meta_host(meta_floats);
    for (size_t j = 0; j < size_t(p.banks * p.slots) * (Nested + DownNested); ++j)
        meta_host[j] = .035f + .00001f * float(j % 1000);
    for (int j = 0; j < 256; ++j)
        meta_host[size_t(p.banks * p.slots) * (Nested + DownNested) + j] =
            float(j + 1) / 256.f;
    std::copy(Qmap.begin(), Qmap.end(), meta_host.end() - 16);
    q.memcpy(b.metadata.data(), meta_host.data(), meta_host.size() * sizeof(float));
    std::vector<bf16> x(size_t(p.tokens) * H);
    for (size_t j = 0; j < x.size(); ++j)
        x[j] = bf16((int(mix(uint32_t(j) + 19) % 1001) - 500) / 1024.f);
    q.memcpy(b.x_dev.data(), x.data(), x.size() * sizeof(bf16));
    // Deterministic finite content for the resident pool (kernel-only arm).
    q.fill(b.resident.data_as<uint8_t>(), uint8_t(0x37),
           b.resident.size_bytes() / sizeof(uint8_t));
    q.wait_and_throw();

    std::cout << std::unitbuf << std::fixed << std::setprecision(3)
              << "[BLOCK CONFIG] device=" << q.get_device().get_info<sycl::info::device::name>()
              << " layers=" << p.layers << " staged_per_layer=" << p.staged_per_layer
              << " cold_acts=" << p.cold_acts << " tokens=" << p.tokens << " banks=" << p.banks
              << " slots=" << p.slots << " batches_per_layer="
              << (p.staged_per_layer + p.slots - 1) / p.slots
              << " pairs=" << p.cold_acts - p.staged_per_layer
              << " singles=" << 2 * p.staged_per_layer - p.cold_acts
              << " bytes_per_pass="
              << (uint64_t(p.layers) * uint64_t(p.staged_per_layer) * SlotBytes)
              << " device_bytes=" << ctx.current_allocated_bytes() << '\n';

    std::vector<bf16> h_host, h_ref;
    std::vector<float> down_host, down_ref;
    bool has_reference = false;
    const std::array<std::pair<const char*, int>, 3> arms{
        {{"copy", 0}, {"block", 1}, {"kernel", 2}}};
    for (const auto& entry : arms) {
        const char* name = entry.first;
        const int arm = entry.second;
        if ((arm == 0 && !p.arm_copy) || (arm == 1 && !p.arm_block) ||
            (arm == 2 && !p.arm_kernel))
            continue;
        has_reference = false;  // each arm compares only against its own repeats
        std::vector<double> walls, fills, cop, waits, gus, dns;
        for (int rep = -1; rep < p.reps; ++rep) {
            poison(ctx, b);
            Sample s;
            run_pass(ctx, *copy, b, p, arm, s);
            if (rep < 0) continue;  // warmup
            walls.push_back(s.wall);
            fills.push_back(s.fill);
            cop.push_back(s.copy);
            waits.push_back(s.bankwait);
            gus.push_back(s.gu);
            dns.push_back(s.down);
            if (arm != 0)
                guard_outputs(ctx, b, p, h_host, down_host, h_ref, down_ref,
                              has_reference);
            std::cout << "[BLOCK TIME] arm=" << name << " sample=" << rep
                      << " wall_ms=" << s.wall << " fill_ms=" << s.fill
                      << " copy_ms=" << s.copy << " bankwait_ms=" << s.bankwait
                      << " gu_ms=" << s.gu << " down_ms=" << s.down << '\n';
        }
        const double bytes = double(p.layers) * p.staged_per_layer * SlotBytes;
        std::cout << "[BLOCK MEDIAN] arm=" << name << " wall_ms=" << median(walls)
                  << " fill_ms=" << median(fills) << " copy_ms=" << median(cop)
                  << " bankwait_ms=" << median(waits) << " gu_ms=" << median(gus)
                  << " down_ms=" << median(dns) << " copy_gbps="
                  << (median(cop) > 0 ? bytes / (median(cop) * 1e6) : 0.0) << '\n';
    }
    std::cout << "[BLOCK GUARD] finite=pass sentinel=pass bitwise=pass\n";
    std::cout << "[BLOCK DONE]\n";
    return 0;
} catch (const std::exception& e) {
    std::cerr << "[BLOCK FAIL] " << e.what() << '\n';
    return 1;
}
