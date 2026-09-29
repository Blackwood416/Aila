#include "GdnChunkOps.hpp"
#include <sycl/ext/oneapi/matrix/matrix.hpp>

namespace ops {
namespace {
constexpr int D=128, T=16, P=3*D+2, M=T+D, LD=D+1;
using namespace sycl::ext::oneapi::experimental::matrix;

void prepare_inputs(Context& ctx, DeltaNetGeometry g, DeltaNetWeights w,
    const float* history, int head, const bf16* input, float* prepared, int count) {
    ctx.queue().parallel_for(sycl::nd_range<1>(size_t(count)*g.v_heads*D,D),
        [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
            const int group=int(item.get_group(0)), t=group/g.v_heads, h=group%g.v_heads;
            const int j=int(item.get_local_id(0)), hk=h/(g.v_heads/g.q_heads);
            const auto* row=input+size_t(t)*g.all_dim;
            const auto conv=[&](int c) {
                const auto* weight=w.conv_weight+c*4;
                float value=0;
                for(int s=0;s<3;++s) {
                    const int prior=t-3+s;
                    const float x=prior>=0?float(input[size_t(prior)*g.all_dim+c]):history[((head+prior+3)%3)*g.qkv_dim+c];
                    value+=x*float(weight[s]);
                }
                value+=float(row[c])*float(weight[3]);
                return value/(1.f+sycl::native::exp(-value));
            };
            const float q=conv(hk*D+j), k=conv(g.q_dim+hk*D+j), v=conv(2*g.q_dim+h*D+j);
            const float qs=sycl::reduce_over_group(item.get_group(),q*q,sycl::plus<float>());
            const float ks=sycl::reduce_over_group(item.get_group(),k*k,sycl::plus<float>());
            auto* dst=prepared+size_t(group)*P;
            dst[j]=q/sycl::sqrt(qs+g.eps); dst[D+j]=k/sycl::sqrt(ks+g.eps); dst[2*D+j]=v;
            if(j==0) {
                const float a=float(row[g.qkv_dim+g.z_dim+h])+float(w.dt_bias[h]);
                const float sp=a>20.f?a:a< -20.f?sycl::native::exp(a):sycl::log1p(sycl::native::exp(a));
                dst[3*D]=1.f/(1.f+sycl::native::exp(-float(row[g.qkv_dim+g.z_dim+g.v_heads+h])));
                dst[3*D+1]=sycl::native::exp(-sycl::native::exp(w.a_log[h])*sp);
            }
        });
}

// d_i + sum_{j<i} beta_i*(a_{j+1}...a_i)*(k_i.k_j)*d_j
//   = beta_i*(v_i - (a_0...a_i)*k_i*S0).
// Keep this solve, Q/K, S0 and gate products in FP32. Only the final products
// are optionally rounded to BF16. No full head state is held in SLM/registers.
void solve_chunk(Context& ctx, const float* p, float* state, float* matrix,
    float* delta, float* base, float* decay, int heads, int start, int n) {
    ctx.queue().submit([&](sycl::handler& cgh) {
        // Avoid mapping different token rows onto the same SLM bank in KKT.
        sycl::local_accessor<float,1> q(T*LD,cgh), k(T*LD,cgh), l(T*T,cgh);
        sycl::local_accessor<float,1> a(T,cgh), beta(T,cgh), prefix(T,cgh);
        cgh.parallel_for(sycl::nd_range<1>(size_t(heads)*D,D),
            [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
                const int h=int(item.get_group(0)), v=int(item.get_local_id(0));
                for(int t=0;t<T;++t) {
                    if(t<n) {
                        const auto* row=p+(size_t(start+t)*heads+h)*P;
                        q[t*LD+v]=row[v]; k[t*LD+v]=row[D+v];
                    } else {q[t*LD+v]=0.f;k[t*LD+v]=0.f;}
                }
                if(v<T) {
                    if(v<n) {
                        const auto* row=p+(size_t(start+v)*heads+h)*P;
                        a[v]=row[3*D+1]; beta[v]=row[3*D];
                    } else {a[v]=1.f;beta[v]=0.f;}
                }
                item.barrier(sycl::access::fence_space::local_space);
                if(v==0) {
                    float prod=1.f; for(int t=0;t<T;++t) {prod*=a[t];prefix[t]=prod;}
                    decay[h]=prod;
                }
                for(int ij=v;ij<T*T;ij+=D) {
                    const int i=ij/T,j=ij%T;
                    float kk=0.f,qk=0.f,ratio=1.f;
                    if(j<=i && i<n) {
                        for(int d=0;d<D;++d) {kk+=k[i*LD+d]*k[j*LD+d];qk+=q[i*LD+d]*k[j*LD+d];}
                        for(int t=j+1;t<=i;++t) ratio*=a[t];
                    }
                    l[ij]=j<i?beta[i]*ratio*kk:0.f;
                    matrix[(size_t(h)*M+i)*T+j]=ratio*qk;
                }
                // Decayed K^T forms the lower block of the final matrix product.
                for(int t=0;t<T;++t) {
                    float ratio=1.f; for(int j=t+1;j<n;++j) ratio*=a[j];
                    matrix[(size_t(h)*M+T+v)*T+t]=k[t*LD+v]*ratio;
                }
                item.barrier(sycl::access::fence_space::local_space);
                float ks[T]={},qs[T]={},d[T]={};
                for(int j=0;j<D;++j) {
                    const float s=state[(size_t(h)*D+j)*D+v];
                    for(int t=0;t<T;++t) {ks[t]+=k[t*LD+j]*s; qs[t]+=q[t*LD+j]*s;}
                }
                for(int t=0;t<T;++t) {
                    const float val=t<n?p[(size_t(start+t)*heads+h)*P+2*D+v]:0.f;
                    float dv=beta[t]*(val-prefix[t]*ks[t]);
                    for(int j=0;j<t;++j) dv-=l[t*T+j]*d[j];
                    d[t]=dv;
                    const size_t idx=(size_t(h)*T+t)*D+v;
                    delta[idx]=dv; base[idx]=prefix[t]*qs[t];
                }
            });
    });
}

void product_f32(Context& ctx,const float* a,const float* d,const float* base,
    const float* decay,float* state,float* raw,int heads,int start,int n) {
    ctx.queue().parallel_for(sycl::nd_range<1>(size_t(heads)*M*D,D),
        [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
            const int row=int(item.get_group(0)),h=row/M,m=row%M,v=int(item.get_local_id(0));
            float sum=0; for(int t=0;t<T;++t) sum+=a[size_t(row)*T+t]*d[(size_t(h)*T+t)*D+v];
            if(m<T) {if(m<n) raw[(size_t(start+m)*heads+h)*D+v]=sum+base[(size_t(h)*T+m)*D+v];}
            else {const size_t i=(size_t(h)*D+m-T)*D+v;state[i]=decay[h]*state[i]+sum;}
        });
}

// [HYPOTHESIS] DG2 SG8, 16x32 output tile, single K16 DPAS step. FP32
// accumulator and residual state; BF16 only for A and delta operands.
void product_xmx(Context& ctx,const float* a,const float* d,const float* base,
    const float* decay,float* state,float* raw,int heads,int start,int n) {
    constexpr int BM=16,BN=32,WG=64, GN=D/BN, GM=M/BM;
    ctx.queue().submit([&](sycl::handler& cgh) {
        sycl::local_accessor<bf16,1> as(BM*T,cgh),bs(T*BN,cgh);
        sycl::local_accessor<float,1> cs(BM*BN,cgh);
        cgh.parallel_for(sycl::nd_range<1>(size_t(heads)*GM*GN*WG,WG),
            [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(8)]] {
                const int id=int(item.get_group(0)),h=id/(GM*GN),m0=(id/GN%GM)*BM,v0=(id%GN)*BN;
                const int lid=int(item.get_local_id(0)),sgid=lid/8,sr=sgid/4,sc=sgid%4;
                auto sg=item.get_sub_group();
                for(int i=lid;i<BM*T;i+=WG) as[i]=bf16(a[(size_t(h)*M+m0+i/T)*T+i%T]);
                for(int i=lid;i<T*BN;i+=WG) bs[i]=bf16(d[(size_t(h)*T+i/BN)*D+v0+i%BN]);
                item.barrier(sycl::access::fence_space::local_space);
                joint_matrix<sycl::sub_group,bf16,use::a,8,16,layout::row_major> ja;
                joint_matrix<sycl::sub_group,bf16,use::b,16,8,layout::row_major> jb;
                joint_matrix<sycl::sub_group,float,use::accumulator,8,8> jc;
                joint_matrix_fill(sg,jc,0.f);
                joint_matrix_load(sg,ja,as.get_multi_ptr<sycl::access::decorated::no>()+sr*8*T,T);
                joint_matrix_load(sg,jb,bs.get_multi_ptr<sycl::access::decorated::no>()+sc*8,BN);
                joint_matrix_mad(sg,jc,ja,jb,jc);
                joint_matrix_store(sg,jc,cs.get_multi_ptr<sycl::access::decorated::no>()+sgid*64,8,layout::row_major);
                item.barrier(sycl::access::fence_space::local_space);
                for(int i=lid;i<BM*BN;i+=WG) {
                    const int s=i/64,m=m0+(s/4)*8+(i%64)/8,v=v0+(s%4)*8+i%8;
                    if(m<T) {if(m<n) raw[(size_t(start+m)*heads+h)*D+v]=cs[i]+base[(size_t(h)*T+m)*D+v];}
                    else {const size_t j=(size_t(h)*D+m-T)*D+v;state[j]=decay[h]*state[j]+cs[i];}
                }
            });
    });
}

void finish(Context& ctx,DeltaNetGeometry g,DeltaNetWeights w,const bf16* input,
    const float* raw,bf16* output,int count) {
    ctx.queue().parallel_for(sycl::nd_range<1>(size_t(count)*g.v_heads*D,D),
        [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
            const int row=int(item.get_group(0)),j=int(item.get_local_id(0)),t=row/g.v_heads,h=row%g.v_heads;
            const float x=raw[size_t(row)*D+j]*g.scale;
            const float ss=sycl::reduce_over_group(item.get_group(),x*x,sycl::plus<float>());
            const float z=float(input[size_t(t)*g.all_dim+g.qkv_dim+h*D+j]);
            const float sig=1.f/(1.f+sycl::native::exp(-z));
            const float gate=g.gate_act==GateActivation::Silu?z*sig:sig;
            output[size_t(row)*D+j]=bf16(x/sycl::sqrt(ss/D+g.eps)*w.norm_weight[j]*gate);
        });
}
} // namespace

void GdnChunkScratch::Reserve(Context& ctx,int tokens,int valueHeads) {
    if(tokens<=capacity && valueHeads==heads) return;
    if(tokens<=0 || valueHeads<=0) throw std::invalid_argument("Invalid GDN chunk capacity");
    ctx.synchronize(); // growth is outside the execution hot path
    using f32=dnnl::memory::data_type;
    prepared=Tensor::allocate(ctx,{tokens,valueHeads,P},f32::f32);
    matrix=Tensor::allocate(ctx,{valueHeads,M,T},f32::f32);
    delta=Tensor::allocate(ctx,{valueHeads,T,D},f32::f32);
    base=Tensor::allocate(ctx,{valueHeads,T,D},f32::f32);
    decay=Tensor::allocate(ctx,{valueHeads},f32::f32);
    raw=Tensor::allocate(ctx,{tokens,valueHeads,D},f32::f32);
    capacity=tokens;heads=valueHeads;
}

bool try_linear_delta_prefill_chunked(Context& ctx,const DeltaNetGeometry& g,
    const DeltaNetWeights& w,DeltaNetState& state,GdnChunkScratch& s,
    const bf16* input,bf16* output,int count,bool xmx) {
    if(count<=0 || g.head_k_dim!=D || g.head_v_dim!=D || g.kernel!=4 || g.conv_rows!=3 ||
       g.q_heads<=0 || g.v_heads<=0 || g.v_heads%g.q_heads || s.capacity<count || s.heads!=g.v_heads ||
       !input || !output || !state.state || !state.conv_state || !w.conv_weight || !w.norm_weight || !w.a_log || !w.dt_bias)
        return false;
    const int head=((state.conv_head%3)+3)%3;
    prepare_inputs(ctx,g,w,state.conv_state,head,input,s.prepared.data_as<float>(),count);
    for(int start=0;start<count;start+=T) {
        const int n=std::min(T,count-start);
        solve_chunk(ctx,s.prepared.data_as<float>(),state.state,s.matrix.data_as<float>(),s.delta.data_as<float>(),
            s.base.data_as<float>(),s.decay.data_as<float>(),g.v_heads,start,n);
        auto product=xmx?product_xmx:product_f32;
        product(ctx,s.matrix.data_as<float>(),s.delta.data_as<float>(),s.base.data_as<float>(),s.decay.data_as<float>(),
            state.state,s.raw.data_as<float>(),g.v_heads,start,n);
    }
    commit_linear_conv_history(ctx,input,g.all_dim,state.conv_state,g.qkv_dim,3,head,count);
    finish(ctx,g,w,input,s.raw.data_as<float>(),output,count);
    state.conv_head=(head+count)%3; state.device_state_dirty=true;
    return true;
}
} // namespace ops
