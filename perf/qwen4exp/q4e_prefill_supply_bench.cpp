#include "core/Context.hpp"
#include "core/Tensor.hpp"
#include "ops/MoeExpertOps.hpp"
#include "utils/EnvUtils.hpp"
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace {
using bf16 = sycl::ext::oneapi::bfloat16;
using Clock = std::chrono::steady_clock;
constexpr int H=2560, I=640, E=256, M=8, R=10, A=E*M, T=(A+R-1)/R;
constexpr size_t Packed=2*I*H/2, Codes=2*I*H/64, Nested=Codes/256, Slot=2534400;
constexpr std::array<float,16> Qmap{-1.f,-.6961928f,-.52507305f,-.3949175f,-.28444138f,-.18477343f,
    -.091050036f,0.f,.0795803f,.1609302f,.2461123f,.33791524f,.44070983f,.562617f,.72295684f,1.f};
uint32_t mix(uint32_t x) { x^=x>>16; x*=0x7feb352dU; x^=x>>15; x*=0x846ca68bU; return x^(x>>16); }
uint64_t start(const sycl::event& e) { return e.get_profiling_info<sycl::info::event_profiling::command_start>(); }
uint64_t end(const sycl::event& e) { return e.get_profiling_info<sycl::info::event_profiling::command_end>(); }
double median(std::vector<double> a) { std::sort(a.begin(),a.end()); return a[a.size()/2]; }
struct Sample { double wall=0, kernel=0, barrier=0, copy=0; };
void check(bool ok, const char* message) { if(!ok) throw std::runtime_error(message); }

void run(bool smoke, bool sliced) {
    aila::env::enable_l0_copy_engine();
    Context ctx(nullptr,true);
    auto& q=ctx.queue();
    // Host inputs outlive the drain guard as well as every device tensor.
    std::vector<float> meta(E*Nested+256+16);
    std::vector<bf16> x(T*H), result(T*R*I+16), reference;
    std::vector<ops::MoeExpertSlotDev> slots(A); std::vector<int> tokens(A),ranks(A);
    auto bytes=[&](size_t n) { return Tensor::allocate(ctx,{int64_t(n)},dnnl::memory::data_type::u8); };
    auto pool=bytes(E*Slot), background=bytes(E*Slot), metadata=bytes((E*Nested+256+16)*4);
    auto sd=bytes(A*sizeof(ops::MoeExpertSlotDev)), td=bytes(A*4), rd=bytes(A*4);
    auto xd=bytes(T*H*2), hd=bytes((T*R*I+16)*2);
    // Declared after tensors so exceptional teardown drains both queues before buffers die.
    sycl::queue copy(q.get_context(),q.get_device(),sycl::property_list{sycl::property::queue::in_order(),
        sycl::ext::intel::property::queue::immediate_command_list(),sycl::property::queue::enable_profiling()});
    auto* host=static_cast<uint8_t*>(ctx.alloc_host(E*Slot));
    struct Drain { Context& ctx; sycl::queue& copy; void* host; ~Drain() {
        try {copy.wait_and_throw();} catch(...) {} try {ctx.synchronize();} catch(...) {} ctx.free_host(host);
    }} drain{ctx,copy,host};
    for(size_t w=0;w<E*Slot/4;++w) reinterpret_cast<uint32_t*>(host)[w]=mix(uint32_t(w)+37);
    for(int j=0;j<E*Nested;++j) meta[j]=.035f+.00001f*float(j%1000);
    for(int j=0;j<256;++j) meta[E*Nested+j]=float(j+1)/256.f;
    std::copy(Qmap.begin(),Qmap.end(),meta.begin()+E*Nested+256);
    for(size_t j=0;j<x.size();++j) x[j]=bf16((int(mix(uint32_t(j)+19)%1001)-500)/1024.f);
    for(int a=0;a<A;++a) {
        const int e=a/M; auto& s=slots[a];
        s.gate_up_packed=pool.data_as<uint8_t>()+e*Slot;
        s.gate_up_nested={s.gate_up_packed+Packed,metadata.data_as<float>()+e*Nested,
            metadata.data_as<float>()+E*Nested,.001f,256};
        tokens[a]=a/R; ranks[a]=a%R;
    }
    q.memcpy(pool.data(),host,E*Slot); q.memcpy(metadata.data(),meta.data(),meta.size()*4);
    q.memcpy(sd.data(),slots.data(),slots.size()*sizeof(slots[0]));
    q.memcpy(td.data(),tokens.data(),A*4); q.memcpy(rd.data(),ranks.data(),A*4);
    q.memcpy(xd.data(),x.data(),x.size()*2); q.fill(hd.data_as<bf16>(),bf16(-9.f),result.size()); q.wait_and_throw();
    std::cout<<std::unitbuf<<std::fixed<<std::setprecision(5)<<"[SUPPLY CONFIG] device="
        <<q.get_device().get_info<sycl::info::device::name>()<<" driver="
        <<q.get_device().get_info<sycl::info::device::driver_version>()
        <<" experts="<<E<<" rows_per_expert="<<M<<" H="<<H<<" I="<<I<<" slot_bytes="<<Slot
        <<" device_bytes="<<ctx.current_allocated_bytes()<<" host_bytes="<<ctx.current_pinned_host_bytes()<<'\n';
    // 0: resident; 1: unrelated background H2D; 2: own payload, direct dependency;
    // 3: same DAG with an explicit in-order compute barrier before the kernel.
    // 4/5: background/direct dependency with one contiguous copy per batch.
    // 6/7: background/direct dependency with expert payloads sliced at 512 KiB.
    std::vector<sycl::event> kernels, copies, barriers;
    kernels.reserve(E); copies.reserve(E*5); barriers.reserve(E);
    auto execute=[&](int batch,int mode) {
        kernels.clear(); copies.clear(); barriers.clear();
        const auto t0=Clock::now();
        for(int e=0;e<E;e+=batch) {
            sycl::event ready{};
            const bool unrelated=mode==1||mode==4||mode==6;
            if(mode==4||mode==5) {
                auto* dst=(unrelated?background:pool).data_as<uint8_t>()+e*Slot;
                ready=copy.memcpy(dst,host+e*Slot,std::min(batch,E-e)*Slot); copies.push_back(ready);
            } else if(mode) for(int j=e;j<std::min(E,e+batch);++j) {
                uint8_t* dst=(unrelated?background:pool).data_as<uint8_t>()+j*Slot;
                const size_t slice=mode>=6?512*1024:Slot;
                for(size_t offset=0;offset<Slot;offset+=slice) {
                    ready=copy.memcpy(dst+offset,host+j*Slot+offset,std::min(slice,Slot-offset));
                    copies.push_back(ready);
                }
            }
            const sycl::event* dependency=(mode==2||mode==5||mode==7)?&ready:nullptr;
            if(mode==3) barriers.push_back(q.ext_oneapi_submit_barrier({ready}));
            kernels.push_back(ops::moe_expert_major_prefill_gate_up_4token(ctx,
                sd.data_as<ops::MoeExpertSlotDev>()+e*M,td.data_as<int>()+e*M,rd.data_as<int>()+e*M,
                metadata.data_as<float>()+E*Nested+256,xd.data_as<bf16>(),hd.data_as<bf16>(),
                std::min(batch,E-e)*M/4,R,H,I,64,dependency,true));
        }
        q.wait_and_throw(); copy.wait_and_throw();
        Sample s; s.wall=std::chrono::duration<double,std::milli>(Clock::now()-t0).count();
        for(const auto& ev:kernels) s.kernel+=(end(ev)-start(ev))*1e-6;
        for(const auto& ev:barriers) s.barrier+=(end(ev)-start(ev))*1e-6;
        for(const auto& ev:copies) s.copy+=(end(ev)-start(ev))*1e-6;
        return s;
    };
    execute(E,0); q.memcpy(result.data(),hd.data(),result.size()*2).wait_and_throw(); reference=result;
    int checked=0; double max_abs=0;
    for(int e:{0,37,255}) for(int a=0;a<M;++a) for(int n:{0,319,639}) {
        double g=0,u=0; const int t=tokens[e*M+a];
        auto weight=[&](int row,int k) {
            const size_t idx=size_t(row)*H+k; const auto b=host[e*Slot+idx/2];
            const auto code=host[e*Slot+Packed+idx/64];
            return Qmap[(b>>(idx%2?0:4))&15]*(meta[e*Nested+idx/64/256]*meta[E*Nested+code]+.001f);
        };
        for(int k=0;k<H;++k) {g+=float(x[t*H+k])*weight(n,k);u+=float(x[t*H+k])*weight(n+I,k);}
        const float ref=float(bf16(float(g/(1+std::exp(-g))*u)));
        const float actual=float(result[(e*M+a)*I+n]); const double err=std::abs(ref-actual);
        check(std::isfinite(actual)&&err<=.001+.006*std::abs(ref),"CPU gate/up reference mismatch");
        max_abs=std::max(max_abs,err); ++checked;
    }
    for(size_t j=A*I;j<result.size();++j) check(float(result[j])==-9.f,"Output guard overwritten");
    for(int batch:{256,64,32}) for(int mode=0;mode<8;++mode) {
        execute(batch,mode); q.memcpy(result.data(),hd.data(),result.size()*2).wait_and_throw();
        check(std::memcmp(result.data(),reference.data(),result.size()*2)==0,"Scheduling changed output bits");
    }
    std::cout<<"[SUPPLY CHECK] cpu_samples="<<checked<<" max_abs="<<max_abs<<" schedules=24 bitwise=pass guards=pass\n";
    const int samples=smoke?1:20, repeats=smoke?1:3;
    const std::vector<int> batches=sliced?std::vector<int>{32}:std::vector<int>{256,64,32};
    const std::vector<int> modes=sliced?std::vector<int>{0,1,2,6,7}:std::vector<int>{0,1,2,3,4,5};
    for(int rep=0;rep<repeats;++rep) for(int batch:batches) {
        std::array<std::vector<Sample>,8> data;
        for(int warm=0;warm<3;++warm) for(int m:modes) execute(batch,m);
        for(int n=0;n<samples;++n) for(size_t order=0;order<modes.size();++order) {
            const int m=modes[(n+order+rep)%modes.size()]; data[m].push_back(execute(batch,m));
        }
        for(int mode:modes) {
            std::vector<double> w,k,b,c;
            for(auto s:data[mode]) {w.push_back(s.wall);k.push_back(s.kernel);b.push_back(s.barrier);c.push_back(s.copy);}
            auto ordered=w;std::sort(ordered.begin(),ordered.end());
            std::cout<<"[SUPPLY TIME] repeat="<<rep<<" batch="<<batch<<" mode="<<mode<<" samples="<<samples
                <<" wall_ms="<<median(w)<<" kernel_ms="<<median(k)<<" barrier_ms="<<median(b)<<" copy_ms="<<median(c)
                <<" wall_p10="<<ordered[ordered.size()/10]<<" wall_p90="<<ordered[ordered.size()*9/10]<<'\n';
        }
    }
}
// Full gate/up -> down chain, one scheduling variable. Host payloads are
// prefilled: this isolates DMA/compute competition, not ReadFile throughput.
void run_pacing(int rows, bool smoke, bool ab = false) {
    constexpr int batch=32, guard=16;
    constexpr size_t DownPacked=H*I/2, DownCodes=H*I/64, DownNested=DownCodes/256;
    constexpr size_t DownOffset=Packed+Codes, DownCodesOffset=DownOffset+DownPacked;
    static_assert(DownCodesOffset+DownCodes==Slot);
    const int acts=E*rows, tokens=(acts+R-1)/R;
    const size_t h_count=size_t(tokens)*R*I+guard, d_count=size_t(tokens)*R*H+guard;
    aila::env::enable_l0_copy_engine();
    Context ctx(nullptr,true);
    auto& q=ctx.queue();
    const size_t scale_count=E*(Nested+DownNested), lut_offset=scale_count, qmap_offset=lut_offset+256;
    std::vector<float> meta(qmap_offset+16), actual_down(d_count), reference_down;
    std::vector<bf16> x(size_t(tokens)*H), actual_h(h_count), reference_h;
    std::vector<ops::MoeExpertSlotDev> slots(acts);
    std::vector<int> token_ids(acts), ranks(acts);
    auto bytes=[&](size_t n) {return Tensor::allocate(ctx,{int64_t(n)},dnnl::memory::data_type::u8);};
    auto pool=bytes(E*Slot), metadata=bytes(meta.size()*4), sd=bytes(acts*sizeof(slots[0]));
    auto td=bytes(acts*4), rd=bytes(acts*4), xd=bytes(x.size()*2), hd=bytes(h_count*2), dd=bytes(d_count*4);
    sycl::queue copy(q.get_context(),q.get_device(),sycl::property_list{sycl::property::queue::in_order(),
        sycl::ext::intel::property::queue::immediate_command_list(),sycl::property::queue::enable_profiling()});
    auto* host=static_cast<uint8_t*>(ctx.alloc_host(E*Slot));
    struct Drain {Context& ctx;sycl::queue& copy;void* host;~Drain(){
        try {copy.wait_and_throw();}catch(...){} try{ctx.synchronize();}catch(...){}ctx.free_host(host);
    }} drain{ctx,copy,host};
    for(size_t w=0;w<E*Slot/4;++w)reinterpret_cast<uint32_t*>(host)[w]=mix(uint32_t(w)+37);
    for(size_t j=0;j<scale_count;++j)meta[j]=.035f+.00001f*float(j%1000);
    for(int j=0;j<256;++j)meta[lut_offset+j]=float(j+1)/256.f;
    std::copy(Qmap.begin(),Qmap.end(),meta.begin()+qmap_offset);
    for(size_t j=0;j<x.size();++j)x[j]=bf16((int(mix(uint32_t(j)+19)%1001)-500)/1024.f);
    for(int a=0;a<acts;++a){
        const int e=a/rows, m=a%rows, logical=m*E+e;
        // Unique expert per token/rank, with four arbitrary token rows sharing
        // one expert, as in the production expert-major route.
        token_ids[a]=logical/R;ranks[a]=logical%R;
        auto& s=slots[a];auto* base=pool.data_as<uint8_t>()+size_t(e)*Slot;
        s.gate_up_packed=base;s.down_packed=base+DownOffset;
        s.gate_up_nested={base+Packed,metadata.data_as<float>()+e*Nested,
            metadata.data_as<float>()+lut_offset,.001f,256};
        s.down_nested={base+DownCodesOffset,metadata.data_as<float>()+E*Nested+e*DownNested,
            metadata.data_as<float>()+lut_offset,.001f,256};
    }
    q.memcpy(metadata.data(),meta.data(),meta.size()*4);
    q.memcpy(sd.data(),slots.data(),acts*sizeof(slots[0]));q.memcpy(td.data(),token_ids.data(),acts*4);
    q.memcpy(rd.data(),ranks.data(),acts*4);q.memcpy(xd.data(),x.data(),x.size()*2);q.wait_and_throw();
    std::vector<sycl::event> kernels,copies;
    kernels.reserve(2*E/batch);copies.reserve(E);
    auto poison=[&]{q.fill(hd.data_as<bf16>(),bf16(-9),h_count);q.fill(dd.data_as<float>(),-17.f,d_count);q.wait_and_throw();};
    auto execute=[&](bool paced, bool fixed){
        kernels.clear();copies.clear();sycl::event last{};
        const auto t0=Clock::now();
        for(int e=0;e<E;e+=batch){
            // No host wait. CPU source preparation could run while the previous
            // group's GPU work is active; only its DMA receives this dependency.
            if(paced && e)copy.ext_oneapi_submit_barrier({last});
            sycl::event ready;
            for(int j=e;j<e+batch;++j){
                ready=copy.memcpy(pool.data_as<uint8_t>()+size_t(j)*Slot,host+size_t(j)*Slot,Slot);
                copies.push_back(ready);
            }
            auto gu=ops::moe_expert_major_prefill_gate_up_4token(ctx,
                sd.data_as<ops::MoeExpertSlotDev>()+e*rows,td.data_as<int>()+e*rows,rd.data_as<int>()+e*rows,
                metadata.data_as<float>()+qmap_offset,xd.data_as<bf16>(),hd.data_as<bf16>(),
                batch*rows/4,R,H,I,64,&ready,true,fixed);
            last=ops::moe_expert_major_prefill_down_4token(ctx,
                sd.data_as<ops::MoeExpertSlotDev>()+e*rows,td.data_as<int>()+e*rows,rd.data_as<int>()+e*rows,
                metadata.data_as<float>()+qmap_offset,hd.data_as<bf16>(),dd.data_as<float>(),
                batch*rows/4,R,I,H,64,&gu,true,fixed);
            kernels.push_back(gu);kernels.push_back(last);
        }
        q.wait_and_throw();copy.wait_and_throw();
        Sample s;s.wall=std::chrono::duration<double,std::milli>(Clock::now()-t0).count();
        // Deferred extraction, after the natural end of the complete chain.
        for(const auto& ev:kernels)s.kernel+=(end(ev)-start(ev))*1e-6;
        for(const auto& ev:copies)s.copy+=(end(ev)-start(ev))*1e-6;
        return s;
    };
    poison();execute(false,false);
    q.memcpy(actual_h.data(),hd.data(),h_count*2);q.memcpy(actual_down.data(),dd.data(),d_count*4);q.wait_and_throw();
    reference_h=actual_h;reference_down=actual_down;
    for(size_t j=0;j<size_t(acts)*I;++j)check(std::isfinite(float(actual_h[j])),"Nonfinite gate/up");
    for(size_t j=0;j<size_t(acts)*H;++j)check(std::isfinite(actual_down[j]),"Nonfinite down");
    for(size_t j=size_t(acts)*I;j<h_count;++j)check(float(actual_h[j])==-9,"Gate/up guard overwritten");
    for(size_t j=size_t(acts)*H;j<d_count;++j)check(actual_down[j]==-17,"Down guard overwritten");
    double max_h_error=0,max_d_error=0;int cpu_samples=0;
    auto weight=[&](int e,bool down,int n,int k){
        const size_t index=size_t(n)*(down?I:H)+k;
        const auto* base=host+size_t(e)*Slot;
        const auto packed=base[(down?DownOffset:0)+index/2];
        const auto code=base[(down?DownCodesOffset:Packed)+index/64];
        const size_t scale=(down?E*Nested+e*DownNested:e*Nested)+index/64/256;
        return Qmap[(packed>>(index%2?0:4))&15]*(meta[scale]*meta[lut_offset+code]+.001f);
    };
    for(int e:{0,37,255})for(int m:{0,rows-1}){
        const int a=e*rows+m,t=token_ids[a],r=ranks[a];
        std::vector<bf16> cpu_h(I);
        for(int n=0;n<I;++n){
            double g=0,u=0;for(int k=0;k<H;++k){g+=float(x[size_t(t)*H+k])*weight(e,false,n,k);u+=float(x[size_t(t)*H+k])*weight(e,false,n+I,k);}
            cpu_h[n]=bf16(float(g/(1+std::exp(-g))*u));
            const float got=float(actual_h[(size_t(t)*R+r)*I+n]),ref=float(cpu_h[n]);
            max_h_error=std::max(max_h_error,double(std::abs(got-ref)));
            check(std::abs(got-ref)<=.001f+.006f*std::abs(ref),"CPU gate/up reference mismatch");
        }
        for(int n:{0,1279,2559}){
            double value=0;for(int k=0;k<I;++k)value+=float(cpu_h[k])*weight(e,true,n,k);
            const float got=actual_down[(size_t(t)*R+r)*H+n];
            max_d_error=std::max(max_d_error,std::abs(got-value));
            check(std::abs(got-value)<=.001+.006*std::abs(value),"CPU full-chain reference mismatch");++cpu_samples;
        }
    }
    poison();execute(true,false);
    q.memcpy(actual_h.data(),hd.data(),h_count*2);q.memcpy(actual_down.data(),dd.data(),d_count*4);q.wait_and_throw();
    check(!std::memcmp(actual_h.data(),reference_h.data(),h_count*2),"Pacing changed gate/up or guards");
    check(!std::memcmp(actual_down.data(),reference_down.data(),d_count*4),"Pacing changed down or guards");
    poison();execute(false,true);
    q.memcpy(actual_h.data(),hd.data(),h_count*2);q.memcpy(actual_down.data(),dd.data(),d_count*4);q.wait_and_throw();
    check(!std::memcmp(actual_h.data(),reference_h.data(),h_count*2),"Fixed-256 specialization changed gate/up bits");
    check(!std::memcmp(actual_down.data(),reference_down.data(),d_count*4),"Fixed-256 specialization changed down bits");
    std::cout<<std::unitbuf<<std::fixed<<std::setprecision(6)<<"[PACING CONFIG] rows="<<rows<<" batch="<<batch
        <<" experts="<<E<<" copy_bytes="<<E*Slot<<" device="<<q.get_device().get_info<sycl::info::device::name>()
        <<" driver="<<q.get_device().get_info<sycl::info::device::driver_version>()<<'\n';
    std::cout<<"[PACING CHECK] rows="<<rows<<" cpu_down_samples="<<cpu_samples<<" max_h_abs="<<max_h_error
        <<" max_down_abs="<<max_d_error<<" bitwise=1 guards=1\n";
    const int repeats=smoke?1:3,samples=smoke?1:20;
    for(int rep=0;rep<repeats;++rep){
        std::array<std::vector<Sample>,2> data;
        if(ab){
            for(int warm=0;warm<3;++warm){execute(false,false);execute(false,true);}
            for(int n=0;n<samples;++n)for(int order=0;order<2;++order){
                int which=(n+order+rep)%2;data[which].push_back(execute(false,which==1));
            }
            const char* names[2]={"generic","fixed256"};
            for(int arm=0;arm<2;++arm){
                std::vector<double>w,k,c;for(auto s:data[arm]){w.push_back(s.wall);k.push_back(s.kernel);c.push_back(s.copy);}
                auto sorted=w;std::sort(sorted.begin(),sorted.end());
                std::cout<<"[PACING AB] rows="<<rows<<" repeat="<<rep<<" arm="<<names[arm]
                    <<" samples="<<samples<<" wall_ms="<<median(w)<<" kernel_ms="<<median(k)
                    <<" copy_ms="<<median(c)<<" p10_ms="<<sorted[sorted.size()/10]
                    <<" p90_ms="<<sorted[sorted.size()*9/10]<<" raw_ms=";
                for(size_t n=0;n<w.size();++n)std::cout<<(n?",":"")<<w[n];std::cout<<'\n';
            }
            continue;
        }
        for(int warm=0;warm<3;++warm){execute(false,false);execute(true,false);}
        for(int n=0;n<samples;++n)for(int order=0;order<2;++order){
            int mode=(n+order+rep)%2;data[mode].push_back(execute(mode!=0,false));
        }
        for(int mode=0;mode<2;++mode){
            std::vector<double>w,k,c;for(auto s:data[mode]){w.push_back(s.wall);k.push_back(s.kernel);c.push_back(s.copy);}
            auto sorted=w;std::sort(sorted.begin(),sorted.end());
            std::cout<<"[PACING TIME] rows="<<rows<<" repeat="<<rep<<" paced="<<mode<<" samples="<<samples
                <<" wall_ms="<<median(w)<<" kernel_ms="<<median(k)<<" copy_ms="<<median(c)
                <<" p10_ms="<<sorted[sorted.size()/10]<<" p90_ms="<<sorted[sorted.size()*9/10]<<" raw_ms=";
            for(size_t n=0;n<w.size();++n)std::cout<<(n?",":"")<<w[n];std::cout<<'\n';
        }
    }
}
}
int main(int argc,char** argv) {
    try {
        check(argc==2,"Expected --smoke, --bench or --sliced");
        const std::string option=argv[1];
        if(option=="--pacing" || option=="--pacing-smoke"){
            run_pacing(8,option=="--pacing-smoke");run_pacing(80,option=="--pacing-smoke");return 0;
        }
        if(option=="--pacingab"){run_pacing(8,false,true);run_pacing(80,false,true);return 0;}
        check(option=="--smoke"||option=="--bench"||option=="--sliced","Unknown supply probe option");
        run(option=="--smoke",option=="--sliced");return 0;
    }
    catch(const std::exception& e) {std::cerr<<"[SUPPLY FAIL] "<<e.what()<<'\n';return 1;}
}
