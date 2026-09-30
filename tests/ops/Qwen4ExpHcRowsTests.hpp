// Included in the kernel test's anonymous namespace.
void test_hc_rows(Context& ctx, int groups, int dim, int rank, bool inject) {
    const int k = groups * dim, cap = 4, guard = 32;
    const bf16 sentinel(-7.0f);
    std::mt19937 rng(1049 + dim + rank);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<bf16> n(cap*k), wd(rank*k), wi(groups*k), wu(k*rank);
    for (auto& v : n) v = bf16(dist(rng));
    for (auto& v : wd) v = bf16(dist(rng)*0.02f);
    for (auto& v : wi) v = bf16(dist(rng)*0.02f);
    for (auto& v : wu) v = bf16(dist(rng)*0.2f);
    auto alloc = [&](int size) { return Tensor::allocate(ctx, {size}, dnnl::memory::data_type::bf16); };
    auto nd=alloc(cap*k+2*guard), wdd=alloc(rank*k), wid=alloc(groups*k), wud=alloc(k*rank);
    auto ad=alloc(cap*rank+2*guard), xd=alloc(cap*dim+2*guard);
    auto id=Tensor::allocate(ctx, {cap*groups+2*guard}, dnnl::memory::data_type::f32);
    auto up=alloc(cap*k), rawinj=alloc(cap*groups);
    auto* np=nd.data_as<bf16>()+guard; auto* ap=ad.data_as<bf16>()+guard;
    auto* xp=xd.data_as<bf16>()+guard; auto* ip=id.data_as<float>()+guard;
    auto nv=Tensor::view(ctx,np,{cap,k}), av=Tensor::view(ctx,ap,{cap,rank});
    auto xv=Tensor::view(ctx,xp,{cap,dim});
    Linear down, upward, injection;
    down.init(ctx,wdd,k,rank,false,false);
    upward.init(ctx,wud,rank,k,false,false);
    injection.init(ctx,wid,k,groups,false,false);
    struct Drain { Context& c; ~Drain() { try { c.synchronize(); } catch (...) {} } } drain{ctx};
    std::vector<bf16> nall(cap*k+2*guard,sentinel), aall(cap*rank+2*guard,sentinel), xall(cap*dim+2*guard,sentinel);
    std::vector<float> iall(cap*groups+2*guard,-7.0f);
    std::copy(n.begin(),n.end(),nall.begin()+guard);
    ctx.memcpy_h2d(nd.data(),nall.data(),nall.size()*sizeof(bf16));
    ctx.memcpy_h2d(wdd.data(),wd.data(),wd.size()*sizeof(bf16));
    ctx.memcpy_h2d(wid.data(),wi.data(),wi.size()*sizeof(bf16));
    ctx.memcpy_h2d(wud.data(),wu.data(),wu.size()*sizeof(bf16));
    auto reset = [&] {
        ctx.memcpy_h2d(ad.data(),aall.data(),aall.size()*sizeof(bf16));
        ctx.memcpy_h2d(xd.data(),xall.data(),xall.size()*sizeof(bf16));
        ctx.memcpy_h2d(id.data(),iall.data(),iall.size()*sizeof(float));
    };
    auto capture = [&](int rows, std::vector<bf16>& a, std::vector<bf16>& x, std::vector<float>& i) {
        a.resize(rows*rank); x.resize(rows*dim); i.resize(rows*groups);
        ctx.memcpy_d2h(a.data(),ap,a.size()*sizeof(bf16));
        ctx.memcpy_d2h(x.data(),xp,x.size()*sizeof(bf16));
        ctx.memcpy_d2h(i.data(),ip,i.size()*sizeof(float));
    };
    // Independent double CPU dot products with the model's intermediate BF16 rounding.
    auto rb=[](double v) { return float(bf16(float(v))); };
    auto sigmoid=[](float v) { return 1.0f/(1.0f+std::exp(-v)); };
    std::vector<float> ca(cap*rank), cx(cap*dim), ci(cap*groups,-7.0f);
    for (int t=0;t<cap;++t) {
        for (int r=0;r<rank;++r) {
            double dot=0; for (int j=0;j<k;++j) dot+=double(float(wd[r*k+j]))*float(n[t*k+j]);
            float v=rb(dot)/groups; ca[t*rank+r]=rb(v*sigmoid(v));
        }
        if (inject) for (int g=0;g<groups;++g) {
            double dot=0; for (int j=0;j<k;++j) dot+=double(float(wi[g*k+j]))*float(n[t*k+j]);
            ci[t*groups+g]=2*sigmoid(rb(dot)/groups);
        }
        for (int d=0;d<dim;++d) {
            float mixed=0;
            for (int g=0;g<groups;++g) {
                int r=g*dim+d; double dot=0;
                for (int j=0;j<rank;++j) dot+=double(float(wu[r*rank+j]))*ca[t*rank+j];
                mixed+=rb(sigmoid(rb(dot)))*float(n[t*k+r]);
            }
            cx[t*dim+d]=rb(mixed/groups);
        }
    }
    reset();
    for (int t=0;t<cap;++t) {
        ops::hc_decode_down_inject(ctx,np+t*k,wdd.data_as<bf16>(),inject?wid.data_as<bf16>():nullptr,
            ap+t*rank,ip+t*groups,rank,dim,groups);
        ops::hc_decode_up_mix(ctx,ap+t*rank,wud.data_as<bf16>(),np+t*k,xp+t*dim,rank,dim,groups);
    }
    std::vector<bf16> b1a,b1x; std::vector<float> b1i;
    capture(cap,b1a,b1x,b1i);
    check_close(b1a,ca,"HC B1 CPU act",1e-2f); check_close(b1x,cx,"HC B1 CPU x",1e-2f);
    check_close_f32(b1i,ci,"HC B1 CPU inject",1e-2f);
    for (int rows : {1,2,3,4,1}) {
        reset();
        const auto live=ctx.device_allocation_count();
        ops::hc_decode_down_inject_rows(ctx,np,wdd.data_as<bf16>(),inject?wid.data_as<bf16>():nullptr,
            ap,ip,rank,dim,groups,rows);
        ops::hc_decode_up_mix_rows(ctx,ap,wud.data_as<bf16>(),np,xp,rank,dim,groups,rows);
        std::vector<bf16> a,x; std::vector<float> i; capture(rows,a,x,i);
        if (ctx.device_allocation_count()!=live ||
            std::memcmp(a.data(),b1a.data(),a.size()*sizeof(bf16)) ||
            std::memcmp(x.data(),b1x.data(),x.size()*sizeof(bf16)) ||
            std::memcmp(i.data(),b1i.data(),i.size()*sizeof(float)))
            throw std::runtime_error("HC rows B1 parity or live allocation count failed");
        auto check_guard=[&](auto& tensor, const auto& expected, int used) {
            auto actual=expected;
            ctx.memcpy_d2h(actual.data(),tensor.data(),actual.size()*sizeof(actual[0]));
            for (size_t j=0;j<actual.size();++j)
                if ((j<size_t(guard) || j>=size_t(guard+used)) && actual[j]!=expected[j])
                    throw std::runtime_error("HC rows guard/input mutation");
        };
        check_guard(ad,aall,rows*rank); check_guard(xd,xall,rows*dim);
        check_guard(id,iall,inject?rows*groups:0); check_guard(nd,nall,0);
        // Same dense BF16 semantics via oneDNN matmul, raw weights with strides [1,K].
        // Keep CPU tolerance identical for the library and custom route.
        reset();
        down.forward(ctx,nv,av,rows);
        ctx.queue().parallel_for(sycl::range<1>(rows*rank),[=](sycl::id<1> j) {
            float v=float(ap[j])/groups; ap[j]=bf16(v/(1+sycl::native::exp(-v)));
        });
        upward.forward(ctx,av,up,rows);
        auto* u=up.data_as<bf16>();
        ctx.queue().parallel_for(sycl::range<1>(rows*k),[=](sycl::id<1> j) {
            u[j]=bf16(1/(1+sycl::native::exp(-float(u[j]))));
        });
        ops::hc_mix_reduce(ctx,np,u,xp,rows,dim,groups);
        if (inject) {
            injection.forward(ctx,nv,rawinj,rows); auto* ri=rawinj.data_as<bf16>();
            ctx.queue().parallel_for(sycl::range<1>(rows*groups),[=](sycl::id<1> j) {
                ip[j]=2/(1+sycl::native::exp(-float(ri[j])/groups));
            });
        }
        capture(rows,a,x,i);
        check_close(a,std::vector<float>(ca.begin(),ca.begin()+rows*rank),"HC oneDNN CPU act",1e-2f);
        check_close(x,std::vector<float>(cx.begin(),cx.begin()+rows*dim),"HC oneDNN CPU x",1e-2f);
        check_close_f32(i,std::vector<float>(ci.begin(),ci.begin()+rows*groups),"HC oneDNN CPU inject",1e-2f);
        std::cout<<"[PASS] HC rows="<<rows<<" groups="<<groups<<" dim="<<dim<<" rank="<<rank
                 <<" inject="<<inject<<" B1 bitwise, CPU abs_tol=0.01, guards, live USM unchanged\n";
    }
}

void test_hc_rows_suite(Context& ctx) {
    for (bool inject : {false,true}) test_hc_rows(ctx,4,2560,320,inject);
    test_hc_rows(ctx,2,256,45,true);
    test_hc_rows(ctx,3,31,35,true);
}
