#pragma once
#include "core/Context.hpp"
#include "core/Tensor.hpp"

namespace ops {
class Q4eAttentionPairShort;
class Q4eAttentionPairPartials;
class Q4eAttentionPairMerge;
using Q4eAttentionPairBundle = sycl::kernel_bundle<sycl::bundle_state::executable>;
inline Q4eAttentionPairBundle prepare_q4e_attention_pair(Context& ctx) {
    return sycl::get_kernel_bundle<sycl::bundle_state::executable>(ctx.queue().get_context(),
        {ctx.queue().get_device()}, {sycl::get_kernel_id<Q4eAttentionPairShort>(),sycl::get_kernel_id<Q4eAttentionPairPartials>(),sycl::get_kernel_id<Q4eAttentionPairMerge>()});
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
    if(first_len<1 || first_len+1>capacity || k.shape(0)!=KvHeads || k.shape(2)!=Dim ||
       k.dtype()!=dnnl::memory::data_type::bf16 || v.dtype()!=k.dtype() || v.numel()!=k.numel() ||
       q.dtype()!=k.dtype() || gate.dtype()!=k.dtype() || output.dtype()!=k.dtype() ||
       q.numel()<2*Heads*Dim || gate.numel()<2*Heads*Dim || output.numel()<2*Heads*Dim ||
       (first_len>=1024 && (scratch.dtype()!=dnnl::memory::data_type::f32 || scratch.numel()<2*Heads*max_tiles*Stride)))
        throw std::invalid_argument("Invalid Q4E attention pair geometry");
    const auto* qp=q.data_as<value_t>(); const auto* kp=k.data_as<value_t>();
    const auto* vp=v.data_as<value_t>(); const auto* gp=gate.data_as<value_t>();
    auto* op=output.data_as<value_t>();
    if(first_len<1024) {
        constexpr int ShortWg=512;
        const int padded0=((first_len+31)/32)*32;
        const int padded1=((first_len+1+31)/32)*32;
        ctx.queue().submit([&](sycl::handler& cgh) {
            if(bundle)cgh.use_kernel_bundle(*bundle);
            sycl::local_accessor<float,1> shared0(padded0,cgh), shared1(padded1,cgh);
            sycl::local_accessor<float,1> q_cache0(Dim,cgh), q_cache1(Dim,cgh);
            cgh.parallel_for<Q4eAttentionPairShort>(sycl::nd_range<1>(Heads*ShortWg,ShortWg),
                [=](sycl::nd_item<1> item) {
                    const int head=item.get_group(0), lid=item.get_local_id(0);
                    const int kv=head/(Heads/KvHeads);
                    for(int d=lid;d<Dim;d+=ShortWg) {
                        q_cache0[d]=float(qp[head*Dim+d]);
                        q_cache1[d]=float(qp[Heads*Dim+head*Dim+d]);
                    }
                    item.barrier(sycl::access::fence_space::local_space);
                    for(int t=lid;t<first_len+1;t+=ShortWg) {
                        float sum0=0.0f, sum1=0.0f;
                        const value_t* k_tok=kp+(size_t(kv)*capacity+t)*Dim;
                        for(int d=0;d<Dim;d+=8) {
                            const auto k_vec=*reinterpret_cast<const sycl::vec<value_t,8>*>(k_tok+d);
                            const float k0=float(k_vec[0]), k1=float(k_vec[1]), k2=float(k_vec[2]), k3=float(k_vec[3]);
                            const float k4=float(k_vec[4]), k5=float(k_vec[5]), k6=float(k_vec[6]), k7=float(k_vec[7]);
                            if(t<first_len) {
                                sum0+=q_cache0[d+0]*k0; sum0+=q_cache0[d+1]*k1;
                                sum0+=q_cache0[d+2]*k2; sum0+=q_cache0[d+3]*k3;
                                sum0+=q_cache0[d+4]*k4; sum0+=q_cache0[d+5]*k5;
                                sum0+=q_cache0[d+6]*k6; sum0+=q_cache0[d+7]*k7;
                            }
                            sum1+=q_cache1[d+0]*k0; sum1+=q_cache1[d+1]*k1;
                            sum1+=q_cache1[d+2]*k2; sum1+=q_cache1[d+3]*k3;
                            sum1+=q_cache1[d+4]*k4; sum1+=q_cache1[d+5]*k5;
                            sum1+=q_cache1[d+6]*k6; sum1+=q_cache1[d+7]*k7;
                        }
                        if(t<first_len)shared0[t]=sum0*0.0625f;
                        shared1[t]=sum1*0.0625f;
                    }
                    item.barrier(sycl::access::fence_space::local_space);
                    float max0=-1e30f, max1=-1e30f;
                    for(int t=lid;t<first_len;t+=ShortWg)max0=sycl::fmax(max0,shared0[t]);
                    for(int t=lid;t<first_len+1;t+=ShortWg)max1=sycl::fmax(max1,shared1[t]);
                    max0=sycl::reduce_over_group(item.get_group(),max0,sycl::maximum<float>());
                    max1=sycl::reduce_over_group(item.get_group(),max1,sycl::maximum<float>());
                    float sum_val0=0.0f, sum_val1=0.0f;
                    for(int t=lid;t<first_len;t+=ShortWg) {
                        const float e=sycl::native::exp(shared0[t]-max0);
                        shared0[t]=e; sum_val0+=e;
                    }
                    for(int t=lid;t<first_len+1;t+=ShortWg) {
                        const float e=sycl::native::exp(shared1[t]-max1);
                        shared1[t]=e; sum_val1+=e;
                    }
                    sum_val0=sycl::reduce_over_group(item.get_group(),sum_val0,sycl::plus<float>());
                    sum_val1=sycl::reduce_over_group(item.get_group(),sum_val1,sycl::plus<float>());
                    for(int t=lid;t<first_len;t+=ShortWg)shared0[t]=shared0[t]/sum_val0;
                    for(int t=lid;t<first_len+1;t+=ShortWg)shared1[t]=shared1[t]/sum_val1;
                    item.barrier(sycl::access::fence_space::local_space);
                    for(int d=lid;d<Dim;d+=ShortWg) {
                        float acc0=0.0f, acc1=0.0f;
                        const value_t* v_col=vp+size_t(kv)*capacity*Dim+d;
                        int t0=0;
                        for(;t0+7<first_len;t0+=8) {
                            const float v0=float(v_col[(t0+0)*Dim]), v1=float(v_col[(t0+1)*Dim]);
                            const float v2=float(v_col[(t0+2)*Dim]), v3=float(v_col[(t0+3)*Dim]);
                            const float v4=float(v_col[(t0+4)*Dim]), v5=float(v_col[(t0+5)*Dim]);
                            const float v6=float(v_col[(t0+6)*Dim]), v7=float(v_col[(t0+7)*Dim]);
                            acc0+=shared0[t0+0]*v0; acc0+=shared0[t0+1]*v1;
                            acc0+=shared0[t0+2]*v2; acc0+=shared0[t0+3]*v3;
                            acc0+=shared0[t0+4]*v4; acc0+=shared0[t0+5]*v5;
                            acc0+=shared0[t0+6]*v6; acc0+=shared0[t0+7]*v7;
                            acc1+=shared1[t0+0]*v0; acc1+=shared1[t0+1]*v1;
                            acc1+=shared1[t0+2]*v2; acc1+=shared1[t0+3]*v3;
                            acc1+=shared1[t0+4]*v4; acc1+=shared1[t0+5]*v5;
                            acc1+=shared1[t0+6]*v6; acc1+=shared1[t0+7]*v7;
                        }
                        int t1=t0;
                        for(;t0<first_len;++t0)acc0+=shared0[t0]*float(v_col[t0*Dim]);
                        if(t1+7<first_len+1) {
                            const float v0=float(v_col[(t1+0)*Dim]), v1=float(v_col[(t1+1)*Dim]);
                            const float v2=float(v_col[(t1+2)*Dim]), v3=float(v_col[(t1+3)*Dim]);
                            const float v4=float(v_col[(t1+4)*Dim]), v5=float(v_col[(t1+5)*Dim]);
                            const float v6=float(v_col[(t1+6)*Dim]), v7=float(v_col[(t1+7)*Dim]);
                            acc1+=shared1[t1+0]*v0; acc1+=shared1[t1+1]*v1;
                            acc1+=shared1[t1+2]*v2; acc1+=shared1[t1+3]*v3;
                            acc1+=shared1[t1+4]*v4; acc1+=shared1[t1+5]*v5;
                            acc1+=shared1[t1+6]*v6; acc1+=shared1[t1+7]*v7;
                            t1+=8;
                        }
                        for(;t1<first_len+1;++t1)acc1+=shared1[t1]*float(v_col[t1*Dim]);
                        const int off0=head*Dim+d, off1=Heads*Dim+head*Dim+d;
                        const float r0=float(value_t(acc0)), r1=float(value_t(acc1));
                        const float s0=1.0f/(1.0f+sycl::native::exp(-float(gp[off0])));
                        const float s1=1.0f/(1.0f+sycl::native::exp(-float(gp[off1])));
                        op[off0]=value_t(r0*s0);
                        op[off1]=value_t(r1*s1);
                    }
                });
        });
        return;
    }
    auto* partials=scratch.data_as<float>();
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
