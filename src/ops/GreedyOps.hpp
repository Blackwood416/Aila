#pragma once
#include "core/Context.hpp"
#include <cstdint>
#include <cstring>
#include <limits>

namespace ops {
struct alignas(16) GreedyResult {
    float value;
    int index;
    uint32_t nonfinite;
    uint32_t reserved;
};
constexpr int GreedyGroups = 64;

// reserved carries the top-2 logit margin (best value minus runner-up) as raw
// f32 bits, so the greedy scan stays allocation- and bandwidth-neutral.
inline float greedy_margin(const GreedyResult& r) {
    float margin = 0.0f;
    std::memcpy(&margin, &r.reserved, sizeof(margin));
    return margin;
}

// Two bounded launches, retained scratch, lowest-ID ties, and an
// integer BF16 exponent check that survives the normal fast-math build.
inline sycl::event greedy_bf16(Context& ctx, const sycl::ext::oneapi::bfloat16* logits, int vocab,
                          GreedyResult* partials, GreedyResult* result, int rows = 1) {
    if (vocab <= 0 || rows < 1 || rows > 4 || !logits || !partials || !result) throw std::invalid_argument("Invalid BF16 greedy buffers");
    constexpr int Wg = 256;
    ctx.queue().submit([&](sycl::handler& cgh) {
        sycl::local_accessor<float,1> values(sycl::range<1>(Wg),cgh);
        sycl::local_accessor<int,1> indices(sycl::range<1>(Wg),cgh);
        sycl::local_accessor<uint32_t,1> errors(sycl::range<1>(Wg),cgh);
        sycl::local_accessor<float,1> seconds(sycl::range<1>(Wg),cgh);
        cgh.parallel_for(sycl::nd_range<1>(rows*GreedyGroups*Wg,Wg),
            [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
                const int lid=int(item.get_local_id(0));
                const int row=int(item.get_group(0))/GreedyGroups;
                const int group=int(item.get_group(0))%GreedyGroups;
                const auto* rowLogits=logits+size_t(row)*vocab;
                float best=-std::numeric_limits<float>::max(); float second=-std::numeric_limits<float>::max();
                int index=std::numeric_limits<int>::max(); uint32_t bad=0;
                const auto* bits=reinterpret_cast<const uint16_t*>(rowLogits);
                for (int i=group*Wg+lid;i<vocab;i+=GreedyGroups*Wg) {
                    const bool invalid=(bits[i]&0x7f80u)==0x7f80u;
                    bad|=uint32_t(invalid);
                    if (!invalid) {
                        const float v=float(rowLogits[i]);
                        if (v>best || (v==best && i<index)) { second=best; best=v; index=i; }
                        else if (v>second) { second=v; }
                    }
                }
                values[lid]=best; indices[lid]=index; errors[lid]=bad; seconds[lid]=second;
                item.barrier(sycl::access::fence_space::local_space);
                for (int step=Wg/2;step;step/=2) {
                    if (lid<step) {
                        const float other=values[lid+step]; const int oi=indices[lid+step];
                        if (other>values[lid] || (other==values[lid] && oi<indices[lid])) {
                            seconds[lid]=sycl::max(values[lid],seconds[lid+step]);
                            values[lid]=other;indices[lid]=oi;
                        } else {
                            seconds[lid]=sycl::max(seconds[lid],other);
                        }
                        errors[lid]|=errors[lid+step];
                    }
                    item.barrier(sycl::access::fence_space::local_space);
                }
                if (!lid) partials[item.get_group(0)]={values[0],indices[0],errors[0],sycl::bit_cast<uint32_t>(seconds[0])};
            });
    });
    return ctx.queue().submit([&](sycl::handler& cgh) {
        sycl::local_accessor<float,1> values(sycl::range<1>(GreedyGroups),cgh);
        sycl::local_accessor<int,1> indices(sycl::range<1>(GreedyGroups),cgh);
        sycl::local_accessor<uint32_t,1> errors(sycl::range<1>(GreedyGroups),cgh);
        sycl::local_accessor<float,1> seconds(sycl::range<1>(GreedyGroups),cgh);
        cgh.parallel_for(sycl::nd_range<1>(rows*GreedyGroups,GreedyGroups),
            [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
                const int lid=int(item.get_local_id(0)), row=int(item.get_group(0));
                const auto p=partials[row*GreedyGroups+lid];
                values[lid]=p.value; indices[lid]=p.index; errors[lid]=p.nonfinite;
                seconds[lid]=sycl::bit_cast<float>(p.reserved);
                item.barrier(sycl::access::fence_space::local_space);
                for (int step=GreedyGroups/2;step;step/=2) {
                    if (lid<step) {
                        const float other=values[lid+step];const int oi=indices[lid+step];
                        if (other>values[lid] || (other==values[lid] && oi<indices[lid])) {
                            seconds[lid]=sycl::max(values[lid],seconds[lid+step]);
                            values[lid]=other;indices[lid]=oi;
                        } else {
                            seconds[lid]=sycl::max(seconds[lid],other);
                        }
                        errors[lid]|=errors[lid+step];
                    }
                    item.barrier(sycl::access::fence_space::local_space);
                }
                if (!lid) {
                    const float margin=(seconds[0]>-1e30f)?(values[0]-seconds[0]):100.0f;
                    result[row]={values[0],indices[0],errors[0],sycl::bit_cast<uint32_t>(margin)};
                }
            });
    });
}
} // namespace ops
