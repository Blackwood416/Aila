#pragma once

// B4-6 opt-in diagnostics: compact, order-independent fingerprints of the
// persistent device state that a continuation request resumes from. Used only
// by explicitly enabled probes (AILA_Q4E_MTP_BOUNDARY_DUMP); the default paths
// never call into this header, so it must stay allocation-light and side
// effect free apart from the D2H copies it performs.

#include "core/Context.hpp"
#include "core/Tensor.hpp"
#include "models/Qwen4ExpBnb4Backend.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace aila::q4ediag {

// FNV-1a over 8-byte words plus a masked tail: ~1 GB/s/thread is plenty for a
// probe and stays byte-exact (no float comparison, no tolerance).
inline uint64_t DigestBytes(uint64_t h, const void* data, size_t bytes) {
    const auto* p = static_cast<const uint8_t*>(data);
    while (bytes >= 8) {
        uint64_t w = 0;
        std::memcpy(&w, p, 8);
        h ^= w;
        h *= 1099511628211ull;
        p += 8;
        bytes -= 8;
    }
    if (bytes) {
        uint64_t w = 0;
        std::memcpy(&w, p, bytes);
        h ^= w;
        h *= 1099511628211ull;
    }
    return h;
}

inline uint64_t MixScalar(uint64_t h, int64_t value) {
    return DigestBytes(h, &value, sizeof(value));
}

// Hashes at most `rows` positions of a per-position tensor. A 3-D
// {heads, rows, cols} KV tensor is hashed per head (the position axis is not
// contiguous across heads). `rows <= 0` hashes the whole tensor.
inline uint64_t DigestTensorRows(Context& ctx, std::vector<uint8_t>& buf, uint64_t h,
                                 const Tensor& t, int64_t rows) {
    if (!t.valid()) return h;
    const size_t es = t.element_size();
    if (es == 0) return h;
    if (t.ndim() == 3) {
        const int64_t heads = t.shape(0), cap = t.shape(1), cols = t.shape(2);
        const int64_t n = (rows <= 0) ? cap : std::min<int64_t>(rows, cap);
        const size_t bytes = size_t(n) * size_t(cols) * es;
        if (bytes == 0) return h;
        for (int64_t head = 0; head < heads; ++head) {
            buf.resize(bytes);
            const auto* src = static_cast<const uint8_t*>(t.data()) +
                size_t(head) * size_t(cap) * size_t(cols) * es;
            ctx.memcpy_d2h(buf.data(), src, bytes);
            h = DigestBytes(h, buf.data(), bytes);
        }
        return h;
    }
    const int64_t cols = t.ndim() == 2 ? t.shape(1) : 1;
    const int64_t cap = t.ndim() == 2 ? t.shape(0) : t.numel();
    const int64_t n = (rows <= 0) ? cap : std::min<int64_t>(rows, cap);
    const size_t bytes = size_t(n) * size_t(cols) * es;
    if (bytes == 0) return h;
    buf.resize(bytes);
    ctx.memcpy_d2h(buf.data(), t.data(), bytes);
    return DigestBytes(h, buf.data(), bytes);
}

inline std::string Hex(uint64_t value) {
    char text[19];
    std::snprintf(text, sizeof(text), "%016llx", static_cast<unsigned long long>(value));
    return text;
}

struct SessionDigest {
    uint64_t kv = 0, kvValue = 0, kvScale = 0, indexerK = 0, indexerC = 0, indexerN = 0;
    uint64_t gdn = 0, gdnConv = 0, gdnHead = 0, ple = 0, scalars = 0;
    std::vector<uint64_t> kvLayers;
    std::vector<uint64_t> gdnLayers;
};

// Fingerprints the live (position < currentLength) portion of every persistent
// session tensor. This is exactly the region the next forward reads, so two
// fingerprints that match mean "same input state for the continuation".
// Compressed indexer blocks are hashed only up to num_compressed_blocks: the
// slots beyond that count hold not-yet-defined bytes.
inline SessionDigest DigestSession(Context& ctx, Qwen4ExpBnb4Backend::Session& s) {
    SessionDigest out;
    const int64_t visible = s.current_length_for_testing();
    std::vector<uint8_t> buf;
    auto& layers = s.layer_caches_for_testing();
    out.kvLayers.assign(layers.size(), 0);
    out.gdnLayers.assign(layers.size(), 0);
    for (size_t l = 0; l < layers.size(); ++l) {
        auto& c = layers[l];
        uint64_t kvLayer = 1469598103934665603ull;
        kvLayer = DigestTensorRows(ctx, buf, kvLayer, c.k, visible);
        out.kvLayers[l] = kvLayer;
        out.kv = DigestBytes(out.kv, &kvLayer, sizeof(kvLayer));
        uint64_t kvv = 1469598103934665603ull;
        kvv = DigestTensorRows(ctx, buf, kvv, c.v, visible);
        out.kvValue = DigestBytes(out.kvValue, &kvv, sizeof(kvv));
        out.kvScale = DigestTensorRows(ctx, buf, out.kvScale, c.k_scale, visible);
        out.kvScale = DigestTensorRows(ctx, buf, out.kvScale, c.v_scale, visible);
        out.indexerK = DigestTensorRows(ctx, buf, out.indexerK, c.indexer_k, visible);
        out.indexerC = DigestTensorRows(ctx, buf, out.indexerC, c.indexer_compressed_k,
                                        c.num_compressed_blocks);
        out.indexerN = MixScalar(out.indexerN, c.num_compressed_blocks);
        uint64_t gdn = 1469598103934665603ull;
        uint64_t gdnc = 1469598103934665603ull;
        gdn = DigestTensorRows(ctx, buf, gdn, c.linear_state, -1);
        gdnc = DigestTensorRows(ctx, buf, gdnc, c.linear_conv_state, -1);
        out.gdnConv = DigestBytes(out.gdnConv, &gdnc, sizeof(gdnc));
        out.gdnHead = MixScalar(out.gdnHead, c.linear_conv_head);
        gdn = DigestBytes(gdn, &gdnc, sizeof(gdnc));
        gdn = MixScalar(gdn, c.linear_conv_head);
        out.gdnLayers[l] = gdn;
        out.gdn = DigestBytes(out.gdn, &gdn, sizeof(gdn));
    }
    auto& ple = s.ple_state_for_testing();
    uint64_t pleHash = 1469598103934665603ull;
    pleHash = DigestTensorRows(ctx, buf, pleHash, ple.conv_state, -1);
    pleHash = MixScalar(pleHash, ple.conv_head);
    pleHash = MixScalar(pleHash, ple.prev_tokens[0]);
    pleHash = MixScalar(pleHash, ple.prev_tokens[1]);
    out.ple = pleHash;
    uint64_t scalars = 1469598103934665603ull;
    scalars = MixScalar(scalars, visible);
    scalars = MixScalar(scalars, s.consecutive_no_miss_layers());
    out.scalars = scalars;
    return out;
}

inline std::string DescribeSession(const SessionDigest& d) {
    std::string text;
    text.reserve(512);
    text += "kv=" + Hex(d.kv);
    text += " kvv=" + Hex(d.kvValue);
    text += " kvScale=" + Hex(d.kvScale);
    text += " idxk=" + Hex(d.indexerK);
    text += " idxc=" + Hex(d.indexerC);
    text += " idxn=" + Hex(d.indexerN);
    text += " gdn=" + Hex(d.gdn);
    text += " gdnc=" + Hex(d.gdnConv);
    text += " gdnH=" + Hex(d.gdnHead);
    text += " ple=" + Hex(d.ple);
    text += " scal=" + Hex(d.scalars);
    return text;
}

inline std::string DescribeLayerList(const std::vector<uint64_t>& values) {
    std::string text = "[";
    text.reserve(17 * values.size() + 2);
    for (size_t i = 0; i < values.size(); ++i) {
        if (i) text += ',';
        text += Hex(values[i]);
    }
    text += "]";
    return text;
}

// MTP draft state: visible KV rows plus the target hidden the next proposal is
// seeded with. Only the draft's own visible length is hashed, but the caller
// passes the target visible length so a stale draft (length mismatch) shows up
// as a distinct fingerprint rather than a silent pass.
template <typename MtpState>
inline std::string DescribeMtpState(Context& ctx, const MtpState& state, int64_t visibleRows) {
    uint64_t h = 1469598103934665603ull;
    h = MixScalar(h, state.length);
    h = MixScalar(h, state.valid ? 1 : 0);
    std::vector<uint8_t> buf;
    if (state.draft) {
        auto& cache = state.draft->cache;
        h = DigestTensorRows(ctx, buf, h, cache.k, visibleRows);
        h = DigestTensorRows(ctx, buf, h, cache.v, visibleRows);
        h = DigestTensorRows(ctx, buf, h, cache.indexer_k, visibleRows);
        h = DigestTensorRows(ctx, buf, h, cache.indexer_compressed_k, -1);
        h = MixScalar(h, cache.num_compressed_blocks);
    }
    h = DigestTensorRows(ctx, buf, h, state.lastHidden, -1);
    return Hex(h);
}

} // namespace aila::q4ediag
