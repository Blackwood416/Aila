#include "ops/GatedDeltaNetOps.hpp"
#include "ops/GdnChunkOps.hpp"
#include "utils/EnvUtils.hpp"
#include <array>
#include <chrono>
#include <cstring>
#include <iostream>
#include <memory>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

namespace {
void require(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }
using Clock = std::chrono::steady_clock;
struct Error {
    double err = 0, ref = 0, max = 0;
    void add(double a, double b) {
        require(std::isfinite(a) && std::isfinite(b), "Nonfinite GDN value");
        const double d = a - b; err += d*d; ref += a*a; max = std::max(max, std::abs(d));
    }
    double nrmse() const { return std::sqrt(err / std::max(ref, 1e-24)); }
};
double silu(double x) { return x / (1 + std::exp(-x)); }
double softplus(double x) { return x > 20 ? x : x < -20 ? std::exp(x) : std::log1p(std::exp(x)); }
double quantile(std::vector<double> v, double q) {
    std::sort(v.begin(), v.end()); return v[std::min(v.size()-1, size_t(q*(v.size()-1)))];
}

struct Fixture {
    Context& ctx;
    ops::DeltaNetGeometry g;
    int count, initial_head = 2;
    std::vector<bf16> input, cw, dt;
    std::vector<float> initial, history, nw, al;
    Tensor in, conv_w, norm_w, a_log, dt_bias, state, conv_state, out;
    ops::GdnChunkScratch scratch;
    Fixture(Context& c, int tokens, int qh, int vh) : ctx(c),
        g(ops::DeltaNetGeometry::create(qh, vh, 128, 128, 4, 1e-6f, ops::GateActivation::Sigmoid)), count(tokens) {
        std::mt19937 rng(4917);
        const auto rnd = [&]() { return (int(rng()%2001)-1000)/1000.f; };
        input.resize(size_t(count)*g.all_dim); for (auto& x : input) x = bf16(rnd()*.3f);
        initial.resize(size_t(vh)*128*128); for (auto& x : initial) x = rnd()*.02f;
        history.resize(3*g.qkv_dim); for (auto& x : history) x = rnd()*.2f;
        cw.resize(g.qkv_dim*4); for (auto& x : cw) x = bf16(rnd()*.4f);
        nw.resize(128); for (auto& x : nw) x = 1.f + rnd()*.1f;
        al.resize(vh); for (auto& x : al) x = -2.f + rnd()*.5f;
        dt.resize(vh); for (auto& x : dt) x = bf16(-1.f + rnd()*.5f);
        const auto alloc = [&](size_t n, dnnl::memory::data_type t) { return Tensor::allocate(ctx, {int64_t(n)}, t); };
        using type = dnnl::memory::data_type;
        in=alloc(input.size(),type::bf16); conv_w=alloc(cw.size(),type::bf16);
        norm_w=alloc(nw.size(),type::f32); a_log=alloc(al.size(),type::f32); dt_bias=alloc(dt.size(),type::bf16);
        state=alloc(initial.size(),type::f32); conv_state=alloc(history.size(),type::f32);
        out=alloc(size_t(count)*g.kv_dim+16,type::bf16);
        ctx.memcpy_h2d(in.data(),input.data(),in.size_bytes()); ctx.memcpy_h2d(conv_w.data(),cw.data(),conv_w.size_bytes());
        ctx.memcpy_h2d(norm_w.data(),nw.data(),norm_w.size_bytes()); ctx.memcpy_h2d(a_log.data(),al.data(),a_log.size_bytes());
        ctx.memcpy_h2d(dt_bias.data(),dt.data(),dt_bias.size_bytes());
        scratch.Reserve(ctx,count,vh);
    }
    ops::DeltaNetWeights weights() { return {conv_w.data_as<bf16>(),norm_w.data_as<float>(),a_log.data_as<float>(),dt_bias.data_as<bf16>()}; }
    ops::DeltaNetState reset() {
        ctx.memcpy_h2d(state.data(),initial.data(),state.size_bytes()); ctx.memcpy_h2d(conv_state.data(),history.data(),conv_state.size_bytes());
        std::vector<bf16> sentinel(out.numel(),bf16(-9.f)); ctx.memcpy_h2d(out.data(),sentinel.data(),out.size_bytes());
        return {state.data_as<float>(),conv_state.data_as<float>(),initial_head,false};
    }
};

struct Prepared { std::vector<double> q,k,v,beta,log_decay,history; };
Prepared prepare_cpu(const Fixture& f) {
    Prepared p; const auto& g=f.g;
    p.q.resize(size_t(f.count)*g.kv_dim); p.k.resize(p.q.size()); p.v.resize(p.q.size());
    p.beta.resize(f.count*g.v_heads); p.log_decay.resize(p.beta.size()); p.history.assign(f.history.begin(),f.history.end());
    std::vector<double> conv(g.qkv_dim); int ring=f.initial_head;
    for (int t=0;t<f.count;++t) {
        const auto* row=f.input.data()+size_t(t)*g.all_dim;
        for (int c=0;c<g.qkv_dim;++c) {
            double v=double(float(row[c]))*float(f.cw[c*4+3]);
            for (int j=0;j<3;++j) v+=p.history[((ring+j)%3)*g.qkv_dim+c]*float(f.cw[c*4+j]);
            conv[c]=silu(v);
        }
        for (int c=0;c<g.qkv_dim;++c) p.history[ring*g.qkv_dim+c]=float(row[c]); ring=(ring+1)%3;
        for (int h=0;h<g.v_heads;++h) {
            const int hk=h/(g.v_heads/g.q_heads); double qs=0,ks=0;
            for (int j=0;j<128;++j) { qs+=conv[hk*128+j]*conv[hk*128+j]; ks+=conv[g.q_dim+hk*128+j]*conv[g.q_dim+hk*128+j]; }
            const size_t base=(size_t(t)*g.v_heads+h)*128;
            for (int j=0;j<128;++j) {
                p.q[base+j]=conv[hk*128+j]/std::sqrt(qs+g.eps); p.k[base+j]=conv[g.q_dim+hk*128+j]/std::sqrt(ks+g.eps);
                p.v[base+j]=conv[2*g.q_dim+h*128+j];
            }
            const size_t scalar=size_t(t)*g.v_heads+h;
            p.beta[scalar]=1/(1+std::exp(-double(float(row[g.qkv_dim+g.z_dim+g.v_heads+h]))));
            p.log_decay[scalar]=-std::exp(double(f.al[h]))*softplus(double(float(row[g.qkv_dim+g.z_dim+h]))+float(f.dt[h]));
        }
    }
    return p;
}
struct CpuResult { std::vector<double> state,raw; };
CpuResult recurrent_cpu(const Fixture& f,const Prepared& p) {
    CpuResult r; r.state.assign(f.initial.begin(),f.initial.end()); r.raw.resize(size_t(f.count)*f.g.kv_dim);
    for (int t=0;t<f.count;++t) for (int h=0;h<f.g.v_heads;++h) {
        const size_t base=(size_t(t)*f.g.v_heads+h)*128; auto* s=r.state.data()+h*128*128;
        const double a=std::exp(p.log_decay[t*f.g.v_heads+h]), b=p.beta[t*f.g.v_heads+h];
        for (int v=0;v<128;++v) {
            double dot=0; for(int k=0;k<128;++k) dot+=a*s[k*128+v]*p.k[base+k];
            const double d=b*(p.v[base+v]-dot); double o=0;
            for(int k=0;k<128;++k) { s[k*128+v]=a*s[k*128+v]+p.k[base+k]*d; o+=s[k*128+v]*p.q[base+k]; }
            r.raw[base+v]=o;
        }
    }
    return r;
}

// Causal triangular system, with no inverse exp(prefix), even at underflow.
CpuResult chunk_cpu(const Fixture& f,const Prepared& p,int chunk) {
    CpuResult r; r.state.assign(f.initial.begin(),f.initial.end()); r.raw.resize(size_t(f.count)*f.g.kv_dim);
    for(int start=0;start<f.count;start+=chunk) for(int h=0;h<f.g.v_heads;++h) {
        const int n=std::min(chunk,f.count-start); std::vector<double> log(n),l(n*n),a(n*n),d(n*128);
        auto idx=[&](int t) {return (size_t(start+t)*f.g.v_heads+h)*128;};
        for(int i=0;i<n;++i) log[i]=(i?log[i-1]:0)+p.log_decay[(start+i)*f.g.v_heads+h];
        for(int i=0;i<n;++i) for(int j=0;j<=i;++j) {
            double kk=0,qk=0; for(int k=0;k<128;++k) {kk+=p.k[idx(i)+k]*p.k[idx(j)+k]; qk+=p.q[idx(i)+k]*p.k[idx(j)+k];}
            const double ratio=std::exp(log[i]-log[j]); l[i*n+j]=i==j?0:p.beta[(start+i)*f.g.v_heads+h]*ratio*kk; a[i*n+j]=ratio*qk;
        }
        auto* s=r.state.data()+h*128*128;
        for(int v=0;v<128;++v) {
            for(int i=0;i<n;++i) {
                double ks=0; for(int k=0;k<128;++k) ks+=p.k[idx(i)+k]*s[k*128+v];
                double value=p.beta[(start+i)*f.g.v_heads+h]*(p.v[idx(i)+v]-std::exp(log[i])*ks);
                for(int j=0;j<i;++j) value-=l[i*n+j]*d[j*128+v]; d[i*128+v]=value;
            }
            for(int i=0;i<n;++i) {
                double value=0; for(int k=0;k<128;++k) value+=p.q[idx(i)+k]*s[k*128+v]; value*=std::exp(log[i]);
                for(int j=0;j<=i;++j) value+=a[i*n+j]*d[j*128+v]; r.raw[idx(i)+v]=value;
            }
            for(int k=0;k<128;++k) {
                double value=std::exp(log[n-1])*s[k*128+v];
                for(int i=0;i<n;++i) value+=std::exp(log[n-1]-log[i])*p.k[idx(i)+k]*d[i*128+v]; s[k*128+v]=value;
            }
        }
    }
    return r;
}

Error check_gpu(Fixture& f,const Prepared& p,const CpuResult& ref) {
    std::vector<float> state(ref.state.size()),history(p.history.size()); std::vector<bf16> output(f.out.numel());
    f.ctx.memcpy_d2h(state.data(),f.state.data(),state.size()*4); f.ctx.memcpy_d2h(history.data(),f.conv_state.data(),history.size()*4);
    f.ctx.memcpy_d2h(output.data(),f.out.data(),output.size()*2); Error es,eo;
    for(size_t i=0;i<state.size();++i) es.add(ref.state[i],state[i]);
    for(size_t i=0;i<history.size();++i) require(history[i]==p.history[i],"Conv history changed");
    for(int t=0;t<f.count;++t) for(int h=0;h<f.g.v_heads;++h) {
        const size_t base=(size_t(t)*f.g.v_heads+h)*128; double ss=0;
        for(int v=0;v<128;++v) ss+=std::pow(ref.raw[base+v]*f.g.scale,2); const double norm=1/std::sqrt(ss/128+f.g.eps);
        for(int v=0;v<128;++v) {
            const double z=float(f.input[size_t(t)*f.g.all_dim+f.g.qkv_dim+h*128+v]);
            const double gate=f.g.gate_act==ops::GateActivation::Silu?silu(z):1/(1+std::exp(-z));
            const double expected=float(bf16(float(ref.raw[base+v]*f.g.scale*norm*f.nw[v]*gate))); eo.add(expected,float(output[base+v]));
        }
    }
    for(size_t i=ref.raw.size();i<output.size();++i) require(float(output[i])==-9.f,"GDN output guard overwritten");
    std::cout<<"[X4 CHECK] state_nrmse="<<es.nrmse()<<" state_max_abs="<<es.max<<" output_nrmse="<<eo.nrmse()<<" output_max_abs="<<eo.max<<'\n';
    require(es.nrmse()<.015 && eo.nrmse()<.015,"GDN numerical canary failed"); return es;
}

void baseline_bench(Context& ctx) {
    for(int n:{30,256}) {
        Fixture f(ctx,n,16,48);
        for(int series=0;series<3;++series) {
            std::vector<double> wall,span;
            for(int i=-3;i<20;++i) {
                auto st=f.reset(); auto w=f.weights(); const auto start=Clock::now(); auto a=ctx.queue().ext_oneapi_submit_barrier();
                ops::run_linear_delta_prefill_gpu_batched(ctx,f.g,w,st,f.in.data_as<bf16>(),f.out.data_as<bf16>(),n);
                auto b=ctx.queue().ext_oneapi_submit_barrier(); b.wait_and_throw();
                const double ms=std::chrono::duration<double,std::milli>(Clock::now()-start).count();
                if(i>=0) {wall.push_back(ms);span.push_back((b.get_profiling_info<sycl::info::event_profiling::command_start>()-a.get_profiling_info<sycl::info::event_profiling::command_end>())*1e-6);}
            }
            std::cout<<"[X4 BASELINE] count="<<n<<" series="<<series<<" wall_ms="<<quantile(wall,.5)<<" p10="<<quantile(wall,.1)<<" p90="<<quantile(wall,.9)<<" device_span_ms="<<quantile(span,.5)<<'\n';
        }
    }
}
// Independent-workgroup sweep for the batched DeltaNet prefill kernel. Each
// (vh, n) cell is one geometry; samples are collected round-robin so that
// frequency/thermal drift does not alias with a single cell. Reported
// per_head_us / per_head_token_ns use the wall median with the same
// reset+barrier timing convention as baseline_bench.
void mlp_sweep(Context& ctx) {
    const std::vector<int> vh_list{32,48,64,96};
    const std::vector<int> n_list{256,512,1024,2048,4096};
    struct Cell { int vh=0,n=0; std::unique_ptr<Fixture> f; std::vector<double> wall,span; };
    std::vector<Cell> cells; cells.reserve(vh_list.size()*n_list.size());
    for(int vh:vh_list) for(int n:n_list) {
        Cell c; c.vh=vh; c.n=n; c.f=std::make_unique<Fixture>(ctx,n,16,vh);
        cells.push_back(std::move(c));
    }
    const auto time_cell=[&](Cell& c) {
        auto st=c.f->reset(); auto w=c.f->weights(); const auto start=Clock::now();
        auto a=ctx.queue().ext_oneapi_submit_barrier();
        ops::run_linear_delta_prefill_gpu_batched(ctx,c.f->g,w,st,c.f->in.data_as<bf16>(),c.f->out.data_as<bf16>(),c.n);
        auto b=ctx.queue().ext_oneapi_submit_barrier(); b.wait_and_throw();
        c.wall.push_back(std::chrono::duration<double,std::milli>(Clock::now()-start).count());
        c.span.push_back((b.get_profiling_info<sycl::info::event_profiling::command_start>()-a.get_profiling_info<sycl::info::event_profiling::command_end>())*1e-6);
    };
    for(auto& c:cells) time_cell(c); // one warm-up per geometry
    for(int sample=0;sample<5;++sample) for(auto& c:cells) time_cell(c);
    for(const auto& c:cells) {
        const double med=quantile(c.wall,.5);
        std::cout<<"[X4 MLP SWEEP] vh="<<c.vh<<" n="<<c.n<<" median_ms="<<med
            <<" p10="<<quantile(c.wall,.1)<<" p90="<<quantile(c.wall,.9)
            <<" device_span_ms="<<quantile(c.span,.5)
            <<" per_head_us="<<med*1000.0/c.vh
            <<" per_head_token_ns="<<med*1e6/(double(c.vh)*c.n)<<'\n';
    }
}
// Bit-exact A/B of the batched prefill state walk against its software-
// pipelined (MLP/prefetch) variant. Both runs share the fixture, weights and
// inputs; only the launcher variant flag differs. Any difference in the final
// fp32 state, the bf16 output or the conv history fails the case.
void mlp_parity(Context& ctx) {
    struct Geom { int qh, vh; };
    const std::vector<Geom> geoms{{16,48},{16,96}};
    const std::vector<int> n_list{256,1024,4096};
    for(const auto& gm:geoms) for(int n:n_list) {
        Fixture f(ctx,n,gm.qh,gm.vh); auto w=f.weights();
        const auto run_once=[&](bool prefetch,std::vector<float>& st,
                                std::vector<float>& hist,std::vector<bf16>& out) {
            auto s=f.reset();
            ops::run_linear_delta_prefill_gpu_batched(ctx,f.g,w,s,f.in.data_as<bf16>(),
                f.out.data_as<bf16>(),n,false,prefetch);
            ctx.synchronize();
            st.resize(f.initial.size()); hist.resize(f.history.size()); out.resize(f.out.numel());
            ctx.memcpy_d2h(st.data(),f.state.data(),f.state.size_bytes());
            ctx.memcpy_d2h(hist.data(),f.conv_state.data(),f.conv_state.size_bytes());
            ctx.memcpy_d2h(out.data(),f.out.data(),f.out.size_bytes());
        };
        const auto first_diff=[](const auto& a,const auto& b)->long long {
            for(size_t i=0;i<a.size();++i) if(std::memcmp(&a[i],&b[i],sizeof(a[0]))!=0) return (long long)i;
            return -1;
        };
        std::vector<float> st0,st1,hist0,hist1; std::vector<bf16> out0,out1;
        run_once(false,st0,hist0,out0);
        run_once(true,st1,hist1,out1);
        const long long ds=first_diff(st0,st1),dh=first_diff(hist0,hist1),do_=first_diff(out0,out1);
        const bool ok=(ds<0 && dh<0 && do_<0);
        std::cout<<"[X4 MLP PARITY] qh="<<gm.qh<<" vh="<<gm.vh<<" n="<<n
            <<" state_first_diff="<<ds<<" conv_first_diff="<<dh<<" out_first_diff="<<do_
            <<" bitwise="<<(ok?"pass":"FAIL")<<'\n';
        require(ok,"MLP prefetch variant is not bit-identical");
    }
}
void chunk_bench(Context& ctx) {
    for(int n:{30,256}) {
        Fixture f(ctx,n,16,48); auto w=f.weights();
        const auto p=prepare_cpu(f); const auto ref=recurrent_cpu(f,p);
        auto run=[&](int mode,ops::DeltaNetState& st) {
            if(mode==0) ops::run_linear_delta_prefill_gpu_batched(ctx,f.g,w,st,f.in.data_as<bf16>(),f.out.data_as<bf16>(),n);
            else require(ops::try_linear_delta_prefill_chunked(ctx,f.g,w,st,f.scratch,f.in.data_as<bf16>(),f.out.data_as<bf16>(),n,mode==2),"Bench dispatch rejected");
        };
        for(int mode=0;mode<3;++mode) {
            auto st=f.reset();run(mode,st);
            std::cout<<"[X4 BENCH CHECK] count="<<n<<" mode="<<mode<<'\n';check_gpu(f,p,ref);
            for(int warm=0;warm<3;++warm) {st=f.reset();run(mode,st);ctx.synchronize();}
        }
        for(int series=0;series<3;++series) {
            std::array<std::vector<double>,3> wall,span;
            for(int sample=0;sample<20;++sample) for(int order=0;order<3;++order) {
                const int mode=(sample+order)%3;auto st=f.reset();
                const auto start=Clock::now();auto a=ctx.queue().ext_oneapi_submit_barrier();
                run(mode,st);auto b=ctx.queue().ext_oneapi_submit_barrier();b.wait_and_throw();
                wall[mode].push_back(std::chrono::duration<double,std::milli>(Clock::now()-start).count());
                span[mode].push_back((b.get_profiling_info<sycl::info::event_profiling::command_start>()-a.get_profiling_info<sycl::info::event_profiling::command_end>())*1e-6);
            }
            for(int mode=0;mode<3;++mode) {
                double sum=0,sq=0;for(double v:wall[mode]){sum+=v;sq+=v*v;}
                const double mean=sum/20,cv=std::sqrt(std::max(0.,sq/20-mean*mean))/mean;
                std::cout<<"[X4 BENCH] count="<<n<<" mode="<<mode<<" series="<<series<<" wall_ms="<<quantile(wall[mode],.5)
                    <<" p10="<<quantile(wall[mode],.1)<<" p90="<<quantile(wall[mode],.9)<<" device_span_ms="<<quantile(span[mode],.5)
                    <<" cv="<<cv<<" noisy="<<(cv>.1)<<'\n';
            }
        }
    }
}
void correctness(Context& ctx) {
    Fixture f(ctx,35,2,6); const auto p=prepare_cpu(f); const auto ref=recurrent_cpu(f,p);
    for(int c:{16,32}) {
        const auto cr=chunk_cpu(f,p,c); Error es,eo;
        for(size_t i=0;i<ref.state.size();++i) es.add(ref.state[i],cr.state[i]);
        for(size_t i=0;i<ref.raw.size();++i) eo.add(ref.raw[i],cr.raw[i]);
        require(es.nrmse()<1e-12 && eo.nrmse()<1e-12,"Chunk triangular derivation failed");
        std::cout<<"[X4 MATH] chunk="<<c<<" state_nrmse="<<es.nrmse()<<" raw_nrmse="<<eo.nrmse()<<'\n';
    }
    auto st=f.reset(); auto w=f.weights();
    ops::run_linear_delta_prefill_gpu_batched(ctx,f.g,w,st,f.in.data_as<bf16>(),f.out.data_as<bf16>(),f.count); check_gpu(f,p,ref);
    for(bool xmx:{false,true}) {
        st=f.reset();
        require(ops::try_linear_delta_prefill_chunked(ctx,f.g,w,st,f.scratch,f.in.data_as<bf16>(),f.out.data_as<bf16>(),f.count,xmx),"Chunk dispatch rejected");
        std::cout<<"[X4 CHUNK] xmx="<<xmx<<'\n'; check_gpu(f,p,ref);
        require(st.conv_head==(f.initial_head+f.count)%3 && st.device_state_dirty,"Chunk state metadata mismatch");
        for(int cut:{1,15,16,17,32}) {
            st=f.reset();
            require(ops::try_linear_delta_prefill_chunked(ctx,f.g,w,st,f.scratch,f.in.data_as<bf16>(),f.out.data_as<bf16>(),cut,xmx),"Prefix rejected");
            std::vector<float> saved(f.initial.size()),hist(f.history.size());
            ctx.memcpy_d2h(saved.data(),f.state.data(),f.state.size_bytes()); ctx.memcpy_d2h(hist.data(),f.conv_state.data(),f.conv_state.size_bytes());
            const int head=st.conv_head;
            auto suffix=[&]() {
                require(ops::try_linear_delta_prefill_chunked(ctx,f.g,w,st,f.scratch,f.in.data_as<bf16>()+size_t(cut)*f.g.all_dim,
                    f.out.data_as<bf16>()+size_t(cut)*f.g.kv_dim,f.count-cut,xmx),"Suffix rejected");
                ctx.synchronize();
            };
            suffix(); check_gpu(f,p,ref);
            std::vector<float> end(f.initial.size()),replay(end.size());
            std::vector<bf16> out(f.out.numel()),replayed(out.size());
            ctx.memcpy_d2h(end.data(),f.state.data(),f.state.size_bytes()); ctx.memcpy_d2h(out.data(),f.out.data(),f.out.size_bytes());
            ctx.memcpy_h2d(f.state.data(),saved.data(),f.state.size_bytes()); ctx.memcpy_h2d(f.conv_state.data(),hist.data(),f.conv_state.size_bytes());
            st.conv_head=head; suffix();
            ctx.memcpy_d2h(replay.data(),f.state.data(),f.state.size_bytes()); ctx.memcpy_d2h(replayed.data(),f.out.data(),f.out.size_bytes());
            require(std::memcmp(end.data(),replay.data(),f.state.size_bytes())==0,"Restored chunk state replay differs");
            require(std::memcmp(out.data(),replayed.data(),f.out.size_bytes())==0,"Restored chunk output replay differs");
            std::cout<<"[X4 CONTINUATION] xmx="<<xmx<<" cut="<<cut<<" restore_replay=bitwise guards=pass\n";
        }
    }
    auto invalid=f.g;invalid.head_k_dim=64;
    require(!ops::try_linear_delta_prefill_chunked(ctx,invalid,w,st,f.scratch,f.in.data_as<bf16>(),f.out.data_as<bf16>(),f.count,true),"Unsupported geometry accepted");
    require(!ops::try_linear_delta_prefill_chunked(ctx,f.g,w,st,f.scratch,f.in.data_as<bf16>(),f.out.data_as<bf16>(),f.count+1,true),"Insufficient scratch accepted");
    // Equal Q/K and V head counts, SiLU output gate, and both vanishing and
    // near-unit decay. These catch inverse-prefix and ring-history mistakes.
    for(float alog:{10.f,-20.f}) {
        Fixture edge(ctx,5,2,2);edge.g.gate_act=ops::GateActivation::Silu;
        std::fill(edge.al.begin(),edge.al.end(),alog);
        ctx.memcpy_h2d(edge.a_log.data(),edge.al.data(),edge.a_log.size_bytes());
        const auto ep=prepare_cpu(edge);const auto er=recurrent_cpu(edge,ep);
        const auto ec=chunk_cpu(edge,ep,16);Error math;
        for(size_t i=0;i<er.state.size();++i) math.add(er.state[i],ec.state[i]);
        require(math.nrmse()<1e-12,"Extreme decay math mismatch");
        for(bool xmx:{false,true}) {
            auto es=edge.reset();auto ew=edge.weights();
            require(ops::try_linear_delta_prefill_chunked(ctx,edge.g,ew,es,edge.scratch,edge.in.data_as<bf16>(),edge.out.data_as<bf16>(),5,xmx),"Decay edge rejected");
            std::cout<<"[X4 DECAY] a_log="<<alog<<" xmx="<<xmx<<'\n';check_gpu(edge,ep,er);
        }
    }
}
void profile_chunk(Context& ctx) {
    Fixture f(ctx,256,16,48);auto w=f.weights();
    for(int mode:{1,2}) for(int rep=0;rep<2;++rep) {
        auto st=f.reset();
        require(ops::try_linear_delta_prefill_chunked(ctx,f.g,w,st,f.scratch,f.in.data_as<bf16>(),f.out.data_as<bf16>(),f.count,mode==2),"Profile chunk rejected");
        ctx.synchronize();
    }
}
} // namespace
int main(int argc,char** argv) {
    std::cout<<std::unitbuf; aila::env::enable_l0_copy_engine();
    try {
        Context ctx(nullptr,true); std::cout<<"[X4 DEVICE] "<<ctx.queue().get_device().get_info<sycl::info::device::name>()<<'\n';
        if(argc==2 && std::string(argv[1])=="--baseline-bench") baseline_bench(ctx);
        else if(argc==2 && std::string(argv[1])=="--mlp-sweep") mlp_sweep(ctx);
        else if(argc==2 && std::string(argv[1])=="--mlp-parity") mlp_parity(ctx);
        else if(argc==2 && std::string(argv[1])=="--bench") chunk_bench(ctx);
        else if(argc==2 && std::string(argv[1])=="--profile-chunk") profile_chunk(ctx);
        else correctness(ctx);
        return 0;
    } catch(const std::exception& e) {std::cerr<<"[X4 FAIL] "<<e.what()<<'\n'; return 1;}
}
