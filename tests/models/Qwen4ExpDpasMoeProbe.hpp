// R2b DPAS hot-bucket prototype, ordinary SYCL SG8 (not ESIMD).
// W[N,K] row-major s8; X is B[K/4,M,4] VNNI. Each 64-K integer
// partial has its own weight and activation scale; no full-K rescaling.
namespace r2b_matrix=sycl::ext::oneapi::experimental::matrix;
namespace r2b_intel_matrix=sycl::ext::intel::experimental::matrix;

template<bool Gate>
sycl::event integer_dpas_stage(Context& ctx,const ops::MoeExpertSlotDev* slots,
    int8_t* w,float* scale,int experts,int rows,int width,int outputs) {
    const int N=outputs*(Gate?2:1);
    return ctx.queue().submit([&](sycl::handler& cgh){
        sycl::local_accessor<int8_t,1> lut(16,cgh);
        cgh.parallel_for(sycl::nd_range<1>(size_t(experts)*N*width/8,128),
            [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
                const int lid=it.get_local_linear_id();if(lid<16)lut[lid]=kIntegerNf4[lid];
                it.barrier(sycl::access::fence_space::local_space);
                const size_t i=it.get_global_linear_id(),e=i/(N*width/8),offset=i%(N*width/8);
                const auto slot=slots[e*rows];const auto ns=Gate?slot.gate_up_nested:slot.down_nested;
                const uint8_t* p=Gate?slot.gate_up_packed:slot.down_packed;
                const auto q=integer_unpack8(*reinterpret_cast<const uint32_t*>(p+offset*4),lut);
                *reinterpret_cast<sycl::vec<int32_t,2>*>(w+i*8)=q;
                if(offset%8==0)scale[i/8]=integer_nested_scale(ns,offset/8);
            });
    });
}

// Gather active rows and pack four adjacent K bytes. Dense output also feeds
// the oneDNN comparison without an unmeasured host transformation.
template<bool Gather,bool Pad=false>
sycl::event integer_dpas_pack(Context& ctx,const int8_t* src,const float* scales,
    int8_t* vnni,int8_t* dense,float* gathered_scales,const int* tokens,
    int experts,int rows,int width) {
    const int stride_rows=Pad?((rows+31)/32)*32:rows;
    return ctx.queue().parallel_for(sycl::range<1>(size_t(experts)*rows*width/4),
        [=](sycl::id<1> index) [[sycl::reqd_sub_group_size(16)]] {
            const int a=index[0]/(width/4),word=index[0]%(width/4),e=a/rows,m=a%rows;
            const int source=Gather?tokens[a]:a;
            const auto value=reinterpret_cast<const uint32_t*>(src)[source*(width/4)+word];
            if(vnni)reinterpret_cast<uint32_t*>(vnni)[e*(width/4)*stride_rows+word*stride_rows+m]=value;
            if(dense)reinterpret_cast<uint32_t*>(dense)[a*(width/4)+word]=value;
            if(gathered_scales && word%16==0)gathered_scales[a*(width/64)+word/16]=scales[source*(width/64)+word/16];
        });
}

template<bool Gate,int WG=8,bool Pad=false>
sycl::event integer_dpas_projection(Context& ctx,const int8_t* weights,const float* ws,
    const int8_t* vnni,const float* xs,const int* tokens,const int* ranks,
    bf16* h,float* down,int experts,int rows,int width,int outputs,int top_k,
    float* diagnostic=nullptr) {
    expect(rows%8==0 && width%64==0 && outputs%8==0,"R2b DPAS shape");
    const int N=outputs*(Gate?2:1),blocks=width/64,stride_rows=Pad?((rows+31)/32)*32:rows;
    const size_t groups=size_t(experts)*(rows/8)*(outputs/8);
    return ctx.queue().parallel_for(sycl::nd_range<1>(groups*8,WG),
        [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(8)]] {
            using namespace r2b_matrix;
            auto sg=it.get_sub_group();const int tile=it.get_global_linear_id()/8;
            const int n=(tile%(outputs/8))*8,m=(tile/(outputs/8)%(rows/8))*8;
            const int e=tile/((outputs/8)*(rows/8));
            joint_matrix<sycl::sub_group,int8_t,use::a,8,32,layout::row_major> a;
            joint_matrix<sycl::sub_group,int8_t,use::b,32,8,layout::ext_intel_packed> b;
            joint_matrix<sycl::sub_group,int32_t,use::accumulator,8,8> g,u;
            joint_matrix<sycl::sub_group,float,use::accumulator,8,8> sum,up,scaled;
            joint_matrix_fill(sg,scaled,0.f);joint_matrix_fill(sg,sum,0.f);if constexpr(Gate)joint_matrix_fill(sg,up,0.f);
            for(int block=0;block<blocks;++block){
                joint_matrix_fill(sg,g,0);if constexpr(Gate)joint_matrix_fill(sg,u,0);
#pragma unroll
                for(int k=0;k<64;k+=32){
                    joint_matrix_load(sg,a,sycl::address_space_cast<sycl::access::address_space::global_space,sycl::access::decorated::no>(weights+(e*N+n)*width+block*64+k),width);
                    joint_matrix_load(sg,b,sycl::address_space_cast<sycl::access::address_space::global_space,sycl::access::decorated::no>(vnni+e*stride_rows*width+(block*16+k/4)*stride_rows*4+m*4),stride_rows*4);
                    joint_matrix_mad(sg,g,a,b,g);
                    if constexpr(Gate){
                        joint_matrix_load(sg,a,sycl::address_space_cast<sycl::access::address_space::global_space,sycl::access::decorated::no>(weights+(e*N+n+outputs)*width+block*64+k),width);
                        joint_matrix_mad(sg,u,a,b,u);
                    }
                }
                r2b_intel_matrix::joint_matrix_apply(sg,scaled,[&](float& s,size_t r,size_t c){
                    s=ws[(e*N+n+r)*blocks+block]*xs[(e*rows+m+c)*blocks+block]/127.f;
                });
                joint_matrix_apply(sg,scaled,g,[](float& s,int32_t& v){s*=float(v);});
                joint_matrix_apply(sg,sum,scaled,[](float& acc,float& s){acc+=s;});
                if constexpr(Gate){
                    r2b_intel_matrix::joint_matrix_apply(sg,scaled,[&](float& s,size_t r,size_t c){
                        s=ws[(e*N+n+r+outputs)*blocks+block]*xs[(e*rows+m+c)*blocks+block]/127.f;
                    });
                    joint_matrix_apply(sg,scaled,u,[](float& s,int32_t& v){s*=float(v);});
                    joint_matrix_apply(sg,up,scaled,[](float& acc,float& s){acc+=s;});
                }
            }
            if constexpr(Gate){
                if(diagnostic){
                    r2b_intel_matrix::joint_matrix_apply(sg,sum,[&](float& v,size_t r,size_t c){diagnostic[(e*rows+m+c)*2*outputs+n+r]=v;});
                    r2b_intel_matrix::joint_matrix_apply(sg,up,[&](float& v,size_t r,size_t c){diagnostic[(e*rows+m+c)*2*outputs+outputs+n+r]=v;});
                }
                joint_matrix_apply(sg,sum,up,[](float& g,float& u){g=g/(1.f+sycl::native::exp(-g))*u;});
            }
            r2b_intel_matrix::joint_matrix_apply(sg,sum,[&](float& v,size_t r,size_t c){
                const int act=e*rows+m+c;const size_t dest=(size_t(tokens[act])*top_k+ranks[act])*outputs+n+r;
                if constexpr(Gate)h[dest]=bf16(v);else down[dest]=v;
            });
        });
}

void test_integer_dpas_smoke(){
    Context ctx;constexpr int M=8,N=8,K=128,B=2;
    auto alloc=[&](int bytes){return Tensor::allocate(ctx,{bytes},dnnl::memory::data_type::u8);};
    auto w=alloc(N*K),x=alloc(M*K),v=alloc(M*K),ws=alloc(N*B*4),xs=alloc(M*B*4),t=alloc(M*4),r=alloc(M*4),out=alloc(M*N*4);
    struct Drain{Context& c;~Drain(){try{c.synchronize();}catch(...){}}}drain{ctx};
    std::vector<int8_t> wh(N*K),xh(M*K);std::vector<float> sw(N*B),sx(M*B),got(M*N);
    std::vector<int> tok(M),rank(M,0);std::mt19937 rng(71009);std::uniform_int_distribution<int> dist(-127,127);
    for(auto& a:wh)a=int8_t(dist(rng));for(auto& a:xh)a=int8_t(dist(rng));
    for(int i=0;i<N*B;++i)sw[i]=float(i+1)/32;for(int i=0;i<M*B;++i)sx[i]=float(i+3)/256;
    for(int i=0;i<M;++i)tok[i]=i*3%M;
    ctx.memcpy_h2d(w.data(),wh.data(),wh.size());ctx.memcpy_h2d(x.data(),xh.data(),xh.size());
    ctx.memcpy_h2d(ws.data(),sw.data(),sw.size()*4);ctx.memcpy_h2d(xs.data(),sx.data(),sx.size()*4);
    ctx.memcpy_h2d(t.data(),tok.data(),M*4);ctx.memcpy_h2d(r.data(),rank.data(),M*4);
    integer_dpas_pack<false>(ctx,x.data_as<int8_t>(),nullptr,v.data_as<int8_t>(),nullptr,nullptr,nullptr,1,M,K);
    integer_dpas_projection<false>(ctx,w.data_as<int8_t>(),ws.data_as<float>(),v.data_as<int8_t>(),xs.data_as<float>(),t.data_as<int>(),r.data_as<int>(),nullptr,out.data_as<float>(),1,M,K,N,1);
    ctx.memcpy_d2h(got.data(),out.data(),got.size()*4);
    double maxerr=0;for(int m=0;m<M;++m)for(int n=0;n<N;++n){double ref=0;
        for(int b=0;b<B;++b){int sum=0;for(int k=0;k<64;++k)sum+=int(wh[n*K+b*64+k])*int(xh[m*K+b*64+k]);ref+=double(sum)*sw[n*B+b]*sx[m*B+b]/127.;}
        const double d=std::abs(got[tok[m]*N+n]-ref);maxerr=std::max(maxerr,d);expect(d<=1e-5+1e-5*std::abs(ref),"DPAS smoke signed/layout/scales");}
    std::cout<<"[R2B DPAS SMOKE] M=8 N=8 K=128 blocks=2 signed_s8_s8=1 errors=0 max_abs="<<maxerr<<"\n";
}


// Same block64 integer semantics through mature library matmuls. Each K block
// is an independent batch; the mandatory partial materialization/reduction is
// included in full-path timing (not an unscaled full-K INT8 comparison).
struct IntegerDpasLibrary {
    std::array<std::unique_ptr<dnnl::matmul>,2> mm;
    std::array<std::unordered_map<int,dnnl::memory>,2> args;
    Tensor partial,scratch;
    IntegerDpasLibrary(Context& ctx,int E,int M,int H,int I,int8_t* w,int8_t* x,int8_t* mid){
        const auto s8=dnnl::memory::data_type::s8,s32=dnnl::memory::data_type::s32;
        using md=dnnl::memory::desc;using dims=dnnl::memory::dims;
        partial=Tensor::allocate(ctx,{int64_t(E)*(H/64)*2*I*M},s32);
        std::array<dnnl::matmul::primitive_desc,2> pd;
        std::array<md,2> wm,xm,cm;
        dnnl::primitive_attr attr;attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
        attr.set_fpmath_mode(dnnl::fpmath_mode::strict);
        size_t scratch_size=1;
        for(int mode=0;mode<2;++mode){const int N=mode?H:2*I,K=mode?I:H,B=K/64;
            wm[mode]=md({E,B,N,64},s8,dims{int64_t(N)*K,64,K,1});
            xm[mode]=md({E,B,64,M},s8,dims{int64_t(M)*K,64,1,K});
            cm[mode]=md({E,B,N,M},s32,dnnl::memory::format_tag::abcd);
            pd[mode]=dnnl::matmul::primitive_desc(ctx.engine(),wm[mode],xm[mode],cm[mode],attr);
            scratch_size=std::max(scratch_size,pd[mode].scratchpad_desc().get_size());
        }
        scratch=Tensor::allocate(ctx,{int64_t(scratch_size)},dnnl::memory::data_type::u8);
        auto mem=[&](const md& desc,void* ptr){return dnnl::sycl_interop::make_memory(desc,ctx.engine(),dnnl::sycl_interop::memory_kind::usm,ptr);};
        for(int mode=0;mode<2;++mode){
            mm[mode]=std::make_unique<dnnl::matmul>(pd[mode]);
            args[mode]={{DNNL_ARG_SRC,mem(wm[mode],w)},{DNNL_ARG_WEIGHTS,mem(xm[mode],mode?mid:x)},{DNNL_ARG_DST,mem(cm[mode],partial.data())}};
            if(pd[mode].scratchpad_desc().get_size())args[mode].emplace(DNNL_ARG_SCRATCHPAD,mem(pd[mode].scratchpad_desc(),scratch.data()));
            std::cout<<"[R2B DPAS LIBRARY] M="<<M<<" projection="<<mode<<" impl="<<pd[mode].impl_info_str()<<"\n";
        }
        std::cout<<"[R2B DPAS LIBRARY MEMORY] M="<<M<<" partial_bytes="<<partial.size_bytes()<<" scratch_bytes="<<scratch_size<<"\n";
    }
    sycl::event Run(Context& ctx,int mode){return dnnl::sycl_interop::execute(*mm[mode],ctx.stream(),args[mode]);}
};

template<bool Gate>
sycl::event integer_dpas_library_reduce(Context& ctx,const int32_t* partial,const float* ws,
    const float* xs,const int* tokens,const int* ranks,bf16* h,float* down,
    int experts,int rows,int width,int outputs,int top_k,float* diagnostic=nullptr){
    const int B=width/64,N=outputs*(Gate?2:1);
    return ctx.queue().parallel_for(sycl::range<1>(size_t(experts)*rows*outputs),
        [=](sycl::id<1> index) [[sycl::reqd_sub_group_size(16)]] {
            const int act=index[0]/outputs,n=index[0]%outputs,e=act/rows,m=act%rows;
            float g=0.f,u=0.f;
            for(int b=0;b<B;++b){const float sx=xs[act*B+b]/127.f;
                g=sycl::fma(float(partial[((e*B+b)*N+n)*rows+m]),ws[(e*N+n)*B+b]*sx,g);
                if constexpr(Gate)u=sycl::fma(float(partial[((e*B+b)*N+n+outputs)*rows+m]),ws[(e*N+n+outputs)*B+b]*sx,u);
            }
            const size_t dest=(size_t(tokens[act])*top_k+ranks[act])*outputs+n;
            if constexpr(Gate){
                h[dest]=bf16(g/(1.f+sycl::native::exp(-g))*u);
                if(diagnostic){diagnostic[act*2*outputs+n]=g;diagnostic[act*2*outputs+outputs+n]=u;}
            }else down[dest]=g;
        });
}
