// Diagnostic only: fixed allocations and identical Q/K/V across context lengths.
// Included inside Qwen4ExpKernelTests' anonymous namespace.
void test_qsa_selector_adversarial(Context& ctx) {
    constexpr int capacity = 32772, sentinel = -777;
    auto scores = Tensor::allocate(ctx, {9, capacity}, dnnl::memory::data_type::f32);
    auto output = Tensor::allocate(ctx, {9, 516}, dnnl::memory::data_type::s32);
    auto counts = Tensor::allocate(ctx, {9, 2}, dnnl::memory::data_type::s32);
    struct Drain { Context& c; ~Drain() { try { c.synchronize(); } catch (...) {} } } drain{ctx};
    const auto live = ctx.device_allocation_count();
    std::mt19937 rng(61001);
    auto fill = [&](std::vector<float>& values, int pattern) {
        for (size_t i = 0; i < values.size(); ++i) {
            switch (pattern) {
            case 0: values[i] = sycl::bit_cast<float>(std::uint32_t(rng())); break;
            case 1: values[i] = 1.0f; break;
            case 2: values[i] = sycl::bit_cast<float>(0x3f800000U + std::uint32_t(i % 11)); break;
            case 3: {
                const std::uint32_t bits[] = {0U, 0x80000000U, 0x7fc00001U, 0xffc00002U,
                    0x7f800000U, 0xff800000U, 1U, 0x80000001U};
                values[i] = sycl::bit_cast<float>(bits[i % 8]); break;
            }
            case 4: values[i] = float(int(i % 67) - 33); break;
            default: values[i] = sycl::bit_cast<float>(0x7fc00001U); break;
            }
        }
    };
    int cases = 0;
    for (int n : {0, 1, 511, 512, 513, 514, 515, 536, 1023, 1024, 2048, 2049, 8191, 8192, 8193, 16384, 32767, 32768, 32769}) {
        for (int pattern = 0; pattern < 6; ++pattern) {
            std::vector<float> host(std::max(1, n)); fill(host, pattern);
            ctx.memcpy_h2d(scores.data(), host.data(), host.size() * sizeof(float));
            for (int tail = 0; tail < 4; ++tail) {
                for (int topk : {1, 31, 512}) {
                    std::vector<int> got(516, sentinel), ref(516, sentinel);
                    ctx.memcpy_h2d(output.data(), got.data(), got.size() * sizeof(int));
                    const int actual = ops::qsa_select_topk_blocks(ctx, scores, n, topk, n * 4 + tail, 4, output);
                    const int expected = ops::qsa_select_topk_blocks_host(host.data(), n, topk, n * 4 + tail, 4, ref.data());
                    ctx.memcpy_d2h(got.data(), output.data(), got.size() * sizeof(int));
                    if (actual != expected || got != ref)
                        throw std::runtime_error("QSA adversarial single: n=" + std::to_string(n) +
                            " pattern=" + std::to_string(pattern) + " tail=" + std::to_string(tail) + " k=" + std::to_string(topk));
                    ++cases;
                }
            }
        }
    }
    for (int start : {0, 2044, 2048, 2052, 4091, 8187, 32763, 65527, 131063, 131067}) {
        for (int pattern = 0; pattern < 6; ++pattern) {
            constexpr int rows = 9, topk = 512, stride = topk + 4;
            const int score_stride = (start + rows) / 4 + 3;
            std::vector<float> host(rows * score_stride); fill(host, pattern);
            std::vector<int> got(rows * stride, sentinel), ref = got;
            std::vector<int> valid(rows * 2, sentinel);
            ctx.memcpy_h2d(scores.data(), host.data(), host.size() * sizeof(float));
            ctx.memcpy_h2d(output.data(), got.data(), got.size() * sizeof(int));
            ops::qsa_select_topk_blocks_batched(ctx, scores, output, rows, score_stride, start, topk, 4, &counts);
            ctx.memcpy_d2h(got.data(), output.data(), got.size() * sizeof(int));
            ctx.memcpy_d2h(valid.data(), counts.data(), valid.size() * sizeof(int));
            for (int row = 0; row < rows; ++row) {
                const int length = start + row + 1;
                const int expected = ops::qsa_select_topk_blocks_host(host.data() + row * score_stride,
                    length / 4, topk, length, 4, ref.data() + row * stride);
                if (valid[2 * row] != expected || valid[2 * row + 1] != length % 4)
                    throw std::runtime_error("QSA adversarial valid counts");
            }
            if (got != ref) throw std::runtime_error("QSA adversarial batch start=" + std::to_string(start) + " pattern=" + std::to_string(pattern));
            ++cases;
        }
    }
    if (ctx.device_allocation_count() != live) throw std::runtime_error("QSA selector allocated USM");
    std::cout << "[QSA SELECTOR PASS] cases=" << cases << " exact_ids=1 sentinel_guards=1 hot_allocations=0\n";
}

void test_qsa_selector_batches(Context& ctx, bool profile_only) {
    std::cout << std::unitbuf;
    int cases = 0;
    for (int start : {2044, 2052, 2143, 4095, 8191, 32767, 65535, 131008, 131071}) {
        for (int rows : {4, 64}) {
            if (profile_only && (start != 32767 || rows != 4)) continue;
            constexpr int topk = 512, stride = topk + 4;
            const int blocks = (start + rows) / 4;
            auto scores = Tensor::allocate(ctx, {rows, blocks}, dnnl::memory::data_type::f32);
            auto ids = Tensor::allocate(ctx, {rows, stride}, dnnl::memory::data_type::s32);
            struct Drain { Context& c; ~Drain() { try { c.synchronize(); } catch (...) {} } } drain{ctx};
            std::mt19937 rng(61002);
            std::vector<float> host(rows * blocks);
            for (auto& x : host) x = float(rng() % 100003) * 0.001f;
            ctx.memcpy_h2d(scores.data(), host.data(), host.size() * sizeof(float));
            auto run = [&] { ops::qsa_select_topk_blocks_batched(ctx, scores, ids, rows, blocks, start, topk, 4, nullptr); };
            for (int i = 0; i < (profile_only ? 1 : 5); ++i) run();
            ctx.synchronize();
            std::vector<int> got(rows * stride), ref(rows * stride);
            ctx.memcpy_d2h(got.data(), ids.data(), got.size() * sizeof(int));
            for (int row = 0; row < rows; ++row) {
                const int length = start + row + 1;
                const int n = ops::qsa_select_topk_blocks_host(host.data() + row * blocks, length / 4,
                    topk, length, 4, ref.data() + row * stride);
                for (int j = 0; j < n + length % 4; ++j)
                    if (got[row * stride + j] != ref[row * stride + j]) throw std::runtime_error("QSA batch benchmark IDs");
            }
            for (int round = 0; round < 3; ++round) {
                if (profile_only) { run(); ctx.synchronize(); continue; }
                std::vector<double> times;
                for (int i = 0; i < 20; ++i) {
                    const auto begin = std::chrono::steady_clock::now(); run(); ctx.synchronize();
                    times.push_back(std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - begin).count());
                }
                std::sort(times.begin(), times.end());
                const auto begin = std::chrono::steady_clock::now();
                for (int i = 0; i < 20; ++i) run();
                ctx.synchronize();
                const double queued = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - begin).count() / 20;
                std::cout << "[QSA BATCH COST] {\"start\":" << start << ",\"rows\":" << rows << ",\"round\":" << round
                    << ",\"median_us\":" << (times[9] + times[10]) / 2 << ",\"p10_us\":" << times[2]
                    << ",\"p90_us\":" << times[18] << ",\"queued_wall_us\":" << queued << "}\n";
            }
            ++cases;
        }
    }
    std::cout << "[QSA BATCH END] cases=" << cases << " exact_ids=1 profile_only=" << profile_only << "\n";
}

void test_qsa_decode_cost_curve(Context& ctx) {
    std::cout << std::unitbuf;
    constexpr int capacity=32768, heads=24, kvHeads=2, dim=256, topk=512;
    std::mt19937 rng(20261001);
    std::uniform_real_distribution<float> dist(-0.25f,0.25f);
    std::vector<bf16> qh(heads*dim), kh(kvHeads*capacity*dim), vh(kh.size());
    std::vector<bf16> iqh(4*128), ckh((capacity/4)*128);
    for(auto* values:{&qh,&kh,&vh,&iqh,&ckh}) for(auto& v:*values)v=bf16(dist(rng));
    auto alloc=[&](std::vector<int64_t> shape, dnnl::memory::data_type type=dnnl::memory::data_type::bf16){return Tensor::allocate(ctx,shape,type);};
    auto q=alloc({1,heads*dim}), k=alloc({kvHeads,capacity,dim}), v=alloc({kvHeads,capacity,dim});
    auto out=alloc({1,heads*dim}), iq=alloc({4,128}), ck=alloc({capacity/4,128});
    auto scores=alloc({capacity/4},dnnl::memory::data_type::f32);
    auto ids=alloc({topk+4},dnnl::memory::data_type::s32);
    auto attnScores=alloc({heads,capacity},dnnl::memory::data_type::f32);
    auto partials=alloc({heads,(capacity+127)/128,272},dnnl::memory::data_type::f32);
    struct Drain {Context& c;~Drain(){try{c.synchronize();}catch(...){}}} drain{ctx};
    ctx.memcpy_h2d(q.data(),qh.data(),qh.size()*sizeof(bf16));
    ctx.memcpy_h2d(k.data(),kh.data(),kh.size()*sizeof(bf16));
    ctx.memcpy_h2d(v.data(),vh.data(),vh.size()*sizeof(bf16));
    ctx.memcpy_h2d(iq.data(),iqh.data(),iqh.size()*sizeof(bf16));
    ctx.memcpy_h2d(ck.data(),ckh.data(),ckh.size()*sizeof(bf16));
    const auto live=ctx.device_allocation_count();
    for(int length:{512,2048,2051,2052,2056,2146,4096,8192,32768}) {
        const int blocks=length/4;
        const bool sparse=blocks>topk;
        auto score=[&]{ops::qsa_compute_block_scores_compressed(ctx,iq,ck,scores,blocks);};
        auto select=[&]{return ops::qsa_select_topk_blocks(ctx,scores,blocks,topk,length,4,ids);};
        score();const int selected=select();ctx.synchronize();
        std::vector<float> sh(blocks);std::vector<int> ih(selected+length%4), ref(ih.size());
        ctx.memcpy_d2h(sh.data(),scores.data(),sh.size()*sizeof(float));
        ctx.memcpy_d2h(ih.data(),ids.data(),ih.size()*sizeof(int));
        const int expected=ops::qsa_select_topk_blocks_host(sh.data(),blocks,topk,length,4,ref.data());
        if(expected!=selected || ih!=ref)throw std::runtime_error("QSA cost probe selection mismatch");
        auto attention=[&](bool indexed) {
            ops::attention_decode(ctx,q,k,v,out,attnScores,heads,kvHeads,dim,length,
                &partials,nullptr,nullptr,0,indexed?ids.data_as<int>():nullptr,indexed?selected:0);
        };
        // CPU f64 reference for one query head from each KV group; all outputs finite.
        auto verify=[&](bool indexed) {
            attention(indexed);ctx.synchronize();
            std::vector<bf16> got(heads*dim);ctx.memcpy_d2h(got.data(),out.data(),got.size()*sizeof(bf16));
            for(auto x:got)if(!std::isfinite(float(x)))throw std::runtime_error("QSA cost probe nonfinite");
            std::vector<int> active;
            if(indexed){for(int b=0;b<selected;++b)for(int j=0;j<4;++j)active.push_back(ih[b]*4+j);for(int j=0;j<length%4;++j)active.push_back(ih[selected+j]);}
            else for(int j=0;j<length;++j)active.push_back(j);
            double error=0;
            for(int h:{0,12}) {
                const int kv=h/12;std::vector<double> logits(active.size());double mx=-1e100,sum=0;
                for(size_t t=0;t<active.size();++t){double dot=0;for(int d=0;d<dim;++d)dot+=double(qh[h*dim+d])*double(kh[(size_t(kv)*capacity+active[t])*dim+d]);logits[t]=dot/16;mx=std::max(mx,logits[t]);}
                for(auto& x:logits){x=std::exp(x-mx);sum+=x;}
                for(int d=0;d<dim;++d){double value=0;for(size_t t=0;t<active.size();++t)value+=logits[t]*double(vh[(size_t(kv)*capacity+active[t])*dim+d]);error=std::max(error,std::abs(double(got[h*dim+d])-value/sum));}
            }
            if(error>2e-4)throw std::runtime_error("QSA cost probe CPU error="+std::to_string(error));
            std::cout<<"[QSA CHECK] length="<<length<<" indexed="<<indexed<<" selected="<<selected<<" max_abs="<<error<<" reference_heads=2 finite_heads=24\n";
        };
        verify(false);verify(true);
        auto measure=[&](const char* name,auto&& fn) {
            for(int i=0;i<5;++i)fn();ctx.synchronize();
            for(int round=0;round<3;++round) {
                std::vector<double> times;
                for(int i=0;i<20;++i){auto start=std::chrono::steady_clock::now();fn();ctx.synchronize();times.push_back(std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-start).count());}
                std::sort(times.begin(),times.end());
                auto start=std::chrono::steady_clock::now();for(int i=0;i<20;++i)fn();ctx.synchronize();
                const double queued=std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-start).count()/20;
                std::cout<<"[QSA COST] {\"length\":"<<length<<",\"stage\":\""<<name<<"\",\"round\":"<<round
                    <<",\"median_us\":"<<(times[9]+times[10])/2<<",\"p10_us\":"<<times[2]<<",\"p90_us\":"<<times[18]<<",\"queued_wall_us\":"<<queued<<"}\n";
            }
        };
        measure("score",score);measure("select",select);
        measure("dense_attention",[&]{attention(false);});
        measure("selected_attention",[&]{attention(true);});
        measure("routed_post_projection",[&]{if(sparse){score();select();}attention(sparse);});
        if(ctx.device_allocation_count()!=live)throw std::runtime_error("QSA cost probe live USM changed");
    }
    std::cout<<"[QSA COST END] model_free=1 device_time_measured=0 projections_norm_compression_excluded=1\n";
}
