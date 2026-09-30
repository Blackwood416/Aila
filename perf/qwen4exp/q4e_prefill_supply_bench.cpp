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
}
int main(int argc,char** argv) {
    try {
        check(argc==2,"Expected --smoke, --bench or --sliced");
        const std::string option=argv[1];
        check(option=="--smoke"||option=="--bench"||option=="--sliced","Unknown supply probe option");
        run(option=="--smoke",option=="--sliced");return 0;
    }
    catch(const std::exception& e) {std::cerr<<"[SUPPLY FAIL] "<<e.what()<<'\n';return 1;}
}
