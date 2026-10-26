// R2b test-only prototype. Packed BNB NF4 is unchanged; Q8 activations use
// symmetric 64-element blocks. One row/SG16, uint32 weight loads, integer LUT
// in SLM. dot_acc is a compiler idiom: native DP4A must be verified separately.
static constexpr int8_t kIntegerNf4[16] = {
    -127,-88,-67,-50,-36,-23,-12,0,10,20,31,43,56,71,92,127
};

// Independent exact-rational RTE oracle. Host fast-math may rewrite even a
// double division to reciprocal multiplication and perturb an exact half tie.
int integer_q8_reference(bf16 value,bf16 maximum) {
    uint16_t v,m;std::memcpy(&v,&value,2);std::memcpy(&m,&maximum,2);
    const int ve=(v>>7)&255,me=(m>>7)&255;
    const int shift=std::max(1,me)-std::max(1,ve);
    if((m&0x7fff)==0 || shift>=16)return 0;
    const uint32_t n=((v&127)+(ve?128:0))*127;
    const uint32_t d=((m&127)+(me?128:0))<<shift;
    uint32_t q=n/d,r=n%d;
    if(2*r>d || (2*r==d && (q&1)))++q;
    return (v&0x8000)?-int(q):int(q);
}

template<bool CorrectedReciprocal=false>
sycl::event integer_quantize(Context& ctx, const bf16* src, int8_t* dst,
                            float* scales, int rows, int width, int top_k,
                            const int* tokens=nullptr, const int* ranks=nullptr) {
    expect(width%64==0,"R2b Q8 block alignment");
    const int blocks=width/64;
    return ctx.queue().submit([&](sycl::handler& cgh) {
        cgh.parallel_for(sycl::nd_range<1>((size_t(rows)*blocks+7)/8*128,128),
            [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
                const int block=it.get_global_linear_id()/16, lane=it.get_local_linear_id()%16;
                if(block>=rows*blocks)return; // Entire subgroup, before collectives.
                const int row=block/blocks,k=(block%blocks)*64+lane*4;
                const int srcrow=tokens?tokens[row]*top_k+ranks[row]:row;
                sycl::vec<float,4> v;
                float a=0.f;
                for(int j=0;j<4;++j){
                    // Check raw BF16 bits BEFORE float conversion: fast-math can
                    // assume the conversion result is finite and erase a later test.
                    uint16_t bits=reinterpret_cast<const uint16_t*>(src)[srcrow*width+k+j];
                    if((bits&0x7f80U)==0x7f80U)bits=0;
                    const float f=float(sycl::bit_cast<bf16>(bits));
                    v[j]=f;a=sycl::fmax(a,sycl::fabs(f));
                }
                a=sycl::reduce_over_group(it.get_sub_group(),a,sycl::maximum<float>());
                const float scale=a/127.f;
                if(lane==0)scales[block]=scale;
                sycl::vec<int8_t,4> q;
                // BF16 * 127 is exact in f32. Explicit correctly-rounded division
                // prevents fast reciprocal approximations changing half-way ties.
                for(int j=0;j<4;++j){
                    if constexpr(CorrectedReciprocal){
                        // BF16 * small integer is exact in f32. Repair approximate
                        // reciprocal rounding by comparing exact half thresholds.
                        // /256 keeps products finite even near max BF16. Tiny
                        // blocks use the correctly-rounded division fallback.
                        if(a>0x1p-110f){
                            int mag=int(sycl::rint(sycl::clamp(sycl::fabs(v[j])*(127.f/a),0.f,127.f)));
                            const float value=sycl::fma(sycl::fabs(v[j]),254.f/256.f,0.f);
                            const float lower=sycl::fma(a,float(2*mag-1)/256.f,0.f);
                            const float upper=sycl::fma(a,float(2*mag+1)/256.f,0.f);
                            if(value>upper || (value==upper && (mag&1)))++mag;
                            else if(value<lower || (value==lower && (mag&1)))--mag;
                            q[j]=int8_t(v[j]<0.f?-mag:mag);
                        }else q[j]=int8_t(sycl::rint(a>0.f?sycl::ext::intel::math::fdiv_rn(v[j]*127.f,a):0.f));
                    }else q[j]=int8_t(sycl::rint(sycl::clamp(a>0.f?sycl::ext::intel::math::fdiv_rn(v[j]*127.f,a):0.f,-127.f,127.f)));
                }
                *reinterpret_cast<sycl::vec<int8_t,4>*>(dst+row*width+k)=q;
            });
    });
}

inline float integer_nested_scale(const ops::MoeNestedScaleDev& ns,size_t block) {
    const float product=sycl::fma(ns.nested_quant_map[ns.codes[block]],ns.nested_absmax[block/ns.blocksize],0.f);
    return sycl::fma(product,1.f,ns.offset);
}

template<class Lookup>
inline sycl::vec<int32_t,2> integer_unpack8(uint32_t packed,const Lookup& lut) {
    sycl::vec<int8_t,4> a,b;
    // Each byte is high nibble then low nibble (BNB), little-endian uint32.
    for(int j=0;j<4;++j) {
        a[j]=lut[(packed>>((j/2)*8+(j%2?0:4)))&15];
        b[j]=lut[(packed>>((j/2+2)*8+(j%2?0:4)))&15];
    }
    return {a.template as<int32_t>(),b.template as<int32_t>()};
}

template<bool Gate,int Rows=1>
sycl::event integer_projection(Context& ctx,const ops::MoeExpertSlotDev* slots,
    const int* tokens,const int* ranks,const int8_t* x,const float* xs,
    bf16* h,float* down,int acts,int top_k,int width,int outputs,
    float* diagnostic=nullptr) {
    expect(width%128==0 && outputs%8==0 && acts%Rows==0,"R2b projection alignment");
    return ctx.queue().submit([&](sycl::handler& cgh) {
        sycl::local_accessor<int8_t,1> lut(16,cgh);
        cgh.parallel_for(sycl::nd_range<1>(size_t(acts/Rows)*outputs*16,128),
            [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
                const int lid=it.get_local_linear_id(),lane=lid%16;
                if(lid<16)lut[lid]=kIntegerNf4[lid];
                it.barrier(sycl::access::fence_space::local_space);
                const int index=it.get_global_linear_id()/16,a=index/outputs*Rows,row=index%outputs;
                int xr[Rows],yr[Rows];float g[Rows]={},u[Rows]={};
#pragma unroll
                for(int r=0;r<Rows;++r){xr[r]=Gate?tokens[a+r]:a+r;yr[r]=tokens[a+r]*top_k+ranks[a+r];}
                // Fixture guarantees each quad belongs to the same resident expert.
                const auto slot=slots[a];
                const auto ns=Gate?slot.gate_up_nested:slot.down_nested;
                const uint8_t* w=Gate?slot.gate_up_packed:slot.down_packed;
                const int blocks=width/64;
                for(int word=lane;word<width/8;word+=16) {
                    const size_t bi=size_t(row)*blocks+word/8;
                    const float s=(lane==0||lane==8)?integer_nested_scale(ns,bi):0.f;
                    const float s0=sycl::group_broadcast(it.get_sub_group(),s,0);
                    const float s8=sycl::group_broadcast(it.get_sub_group(),s,8);
                    const uint32_t gw=*reinterpret_cast<const uint32_t*>(w+size_t(row)*width/2+word*4);
                    const auto gp=integer_unpack8(gw,lut);
                    float us0=0.f,us8=0.f;sycl::vec<int32_t,2> up;
                    if constexpr(Gate) {
                        const float us=(lane==0||lane==8)?integer_nested_scale(ns,bi+size_t(outputs)*blocks):0.f;
                        us0=sycl::group_broadcast(it.get_sub_group(),us,0);
                        us8=sycl::group_broadcast(it.get_sub_group(),us,8);
                        const uint32_t uw=*reinterpret_cast<const uint32_t*>(w+size_t(row+outputs)*width/2+word*4);
                        up=integer_unpack8(uw,lut);
                    }
#pragma unroll
                    for(int r=0;r<Rows;++r){
                        const float ax=xs[xr[r]*blocks+word/8]/127.f;
                        const auto qx=*reinterpret_cast<const sycl::vec<int32_t,2>*>(x+xr[r]*width+word*8);
                        const int gd=sycl::ext::oneapi::dot_acc(gp[1],qx[1],sycl::ext::oneapi::dot_acc(gp[0],qx[0],int32_t(0)));
                        g[r]=sycl::fma(float(gd),ax*(lane<8?s0:s8),g[r]);
                        if constexpr(Gate){
                            const int ud=sycl::ext::oneapi::dot_acc(up[1],qx[1],sycl::ext::oneapi::dot_acc(up[0],qx[0],int32_t(0)));
                            u[r]=sycl::fma(float(ud),ax*(lane<8?us0:us8),u[r]);
                        }
                    }
                }
#pragma unroll
                for(int r=0;r<Rows;++r){
                    g[r]=sycl::reduce_over_group(it.get_sub_group(),g[r],sycl::plus<float>());
                    if constexpr(Gate)u[r]=sycl::reduce_over_group(it.get_sub_group(),u[r],sycl::plus<float>());
                    if(lane==0) {
                        if constexpr(Gate) {
                            h[yr[r]*outputs+row]=bf16(g[r]/(1.f+sycl::native::exp(-g[r]))*u[r]);
                            if(diagnostic){diagnostic[(a+r)*2*outputs+row]=g[r];diagnostic[(a+r)*2*outputs+outputs+row]=u[r];}
                        } else down[yr[r]*outputs+row]=g[r];
                    }
                }
            });
    });
}

#include "Qwen4ExpDpasMoeProbe.hpp"

