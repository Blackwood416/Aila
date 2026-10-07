#pragma once
#include "core/Context.hpp"
#include "core/Tensor.hpp"

namespace ops {
class Q4eAttentionPairPartials;
class Q4eAttentionPairMerge;
using Q4eAttentionPairBundle = sycl::kernel_bundle<sycl::bundle_state::executable>;
inline Q4eAttentionPairBundle prepare_q4e_attention_pair(Context& ctx) {
    return sycl::get_kernel_bundle<sycl::bundle_state::executable>(ctx.queue().get_context(),
        {ctx.queue().get_device()}, {sycl::get_kernel_id<Q4eAttentionPairPartials>(),sycl::get_kernel_id<Q4eAttentionPairMerge>()});
}
// Bounded Q4E experiment: two consecutive causal queries, BF16 K/V, Hq=24/Hkv=2/D=256.
// The caller may reuse the completed QKV/score phase allocation for partials.
inline void q4e_attention_decode_pair(Context& ctx, Tensor& q, Tensor& k, Tensor& v,
    Tensor& output, Tensor& scratch, int first_len, const Tensor& gate,
    const Q4eAttentionPairBundle* bundle = nullptr) {
    using value_t = sycl::ext::oneapi::bfloat16;
    constexpr int Heads=24, KvHeads=2, Dim=256, Tile=128, Wg=128, Stride=272;
    const int capacity=int(k.shape(1)), max_tiles=(capacity+Tile-1)/Tile;
    const int tiles=(first_len+1+Tile-1)/Tile;
    if(first_len<1024 || first_len+1>capacity || k.shape(0)!=KvHeads || k.shape(2)!=Dim ||
       k.dtype()!=dnnl::memory::data_type::bf16 || v.dtype()!=k.dtype() || v.numel()!=k.numel() ||
       q.dtype()!=k.dtype() || gate.dtype()!=k.dtype() || output.dtype()!=k.dtype() ||
       q.numel()<2*Heads*Dim || gate.numel()<2*Heads*Dim || output.numel()<2*Heads*Dim ||
       scratch.dtype()!=dnnl::memory::data_type::f32 || scratch.numel()<2*Heads*max_tiles*Stride)
        throw std::invalid_argument("Invalid Q4E attention pair geometry");
    const auto* qp=q.data_as<value_t>(); const auto* kp=k.data_as<value_t>();
    const auto* vp=v.data_as<value_t>(); const auto* gp=gate.data_as<value_t>();
    auto* op=output.data_as<value_t>();auto* partials=scratch.data_as<float>();
    ctx.queue().submit([&](sycl::handler& cgh) {
        if(bundle)cgh.use_kernel_bundle(*bundle);
        sycl::local_accessor<float,1> queries(2*Dim,cgh), logits(2*Tile,cgh), maxima(2,cgh);
        cgh.parallel_for<Q4eAttentionPairPartials>(sycl::nd_range<1>(Heads*tiles*Wg,Wg),
            [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
                const int head=item.get_group(0)/tiles,tile=item.get_group(0)%tiles;
                const int lid=item.get_local_id(0),kv=head/(Heads/KvHeads),start=tile*Tile;
                const int length0=sycl::max(0,sycl::min(Tile,first_len-start));
                const int length1=sycl::min(Tile,first_len+1-start);
                for(int d=lid;d<Dim;d+=Wg) {
                    queries[d]=float(qp[head*Dim+d]);
                    queries[Dim+d]=float(qp[Heads*Dim+head*Dim+d]);
                }
                item.barrier(sycl::access::fence_space::local_space);
                float sums[2]={0.0f,0.0f};
                if(lid<length1) {
                    const value_t* key=kp+(kv*capacity+start+lid)*Dim;
                    for(int d=0;d<Dim;d+=8) {
                        const auto values=*reinterpret_cast<const sycl::vec<value_t,8>*>(key+d);
                        #pragma unroll
                        for(int i=0;i<8;++i) {
                            const float w=float(values[i]);
                            sums[0]+=queries[d+i]*w;
                            sums[1]+=queries[Dim+d+i]*w;
                        }
                    }
                }
                sums[0]=lid<length0 ? sums[0]*0.0625f : -1e30f;
                sums[1]=lid<length1 ? sums[1]*0.0625f : -1e30f;
                #pragma unroll
                for(int r=0;r<2;++r) {
                    const float m=sycl::reduce_over_group(item.get_group(),sums[r],sycl::maximum<float>());
                    if(lid==0)maxima[r]=m;
                }
                item.barrier(sycl::access::fence_space::local_space);
                #pragma unroll
                for(int r=0;r<2;++r) {
                    const float e=lid<(r?length1:length0) ? sycl::native::exp(sums[r]-maxima[r]) : 0.0f;
                    logits[r*Tile+lid]=e;
                    const float total=sycl::reduce_over_group(item.get_group(),e,sycl::plus<float>());
                    if(lid==0) {
                        float* dst=partials+((r*Heads+head)*max_tiles+tile)*Stride;
                        dst[0]=maxima[r];dst[1]=total;
                    }
                }
                item.barrier(sycl::access::fence_space::local_space);
                for(int d=lid;d<Dim;d+=Wg) {
                    float a0=0.0f,a1=0.0f;
                    for(int i=0;i<length1;++i) {
                        const float value=float(vp[(kv*capacity+start+i)*Dim+d]);
                        if(i<length0)a0+=logits[i]*value;
                        a1+=logits[Tile+i]*value;
                    }
                    partials[(head*max_tiles+tile)*Stride+16+d]=a0;
                    partials[((Heads+head)*max_tiles+tile)*Stride+16+d]=a1;
                }
            });
    });
    ctx.queue().submit([&](sycl::handler& cgh) {
        if(bundle)cgh.use_kernel_bundle(*bundle);
        sycl::local_accessor<float,1> factors(tiles,cgh), total(1,cgh);
        cgh.parallel_for<Q4eAttentionPairMerge>(sycl::nd_range<1>(2*Heads*Wg,Wg),
            [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
                const int group=item.get_group(0),row=group/Heads,head=group%Heads,lid=item.get_local_id(0);
                const int count=(first_len+row+Tile-1)/Tile;
                const float* base=partials+group*max_tiles*Stride;
                if(lid==0) {
                    float m=-1e30f;
                    for(int t=0;t<count;++t)m=sycl::fmax(m,base[t*Stride]);
                    float sum=0.0f;
                    for(int t=0;t<count;++t){float f=sycl::native::exp(base[t*Stride]-m);factors[t]=f;sum+=f*base[t*Stride+1];}
                    total[0]=sum;
                }
                item.barrier(sycl::access::fence_space::local_space);
                const float inv=1.0f/total[0];
                for(int d=lid;d<Dim;d+=Wg) {
                    float acc=0.0f;
                    for(int t=0;t<count;++t)acc+=factors[t]*base[t*Stride+16+d];
                    const int offset=group*Dim+d;
                    float result=float(value_t(acc*inv));
                    const float s=1.0f/(1.0f+sycl::native::exp(-float(gp[offset])));
                    op[offset]=value_t(result*s);
                }
            });
    });
}
}
