// Included in the tolerance executable's anonymous namespace (after Access).
// B4-2 diagnostic: compare a multi-row (2/4) decode_verification window against
// the equivalent sequence of independent single-row (T1) forwards from the same
// checkpoint. Reports per-row logits and per-layer stage divergence so the first
// misaligned operator can be localized. Opt-in driver only; no production path.
void test_mtp_align(const std::string& directory) {
    std::cout << std::unitbuf << std::setprecision(10);
    int promptCount = aila::env::read_int("AILA_Q4E_MTP_PROMPT", 30);
    const int capacity = aila::env::read_int("AILA_Q4E_MTP_MAX_CONTEXT", 2048);
    const int rowsMax = aila::env::read_int("AILA_Q4E_MTP_ALIGN_ROWS", 4);
    const std::string promptFile = aila::env::read_string("AILA_Q4E_MTP_PROMPT_FILE", "");
    expect(capacity == 640 || capacity == 2048, "MTP align context must be 640 or 2048");
    expect((promptCount == 30 || promptCount == 512 || !promptFile.empty()),
           "MTP align prompt must be 30, 512 or a file");
    expect(rowsMax >= 2 && rowsMax <= 4, "MTP align rows must be 2..4");
    InferenceEngine engine;
    expect(engine.init(directory, capacity), "MTP align target init failed");
    auto& ctx = Access::Ctx(engine);
    auto& target = Access::Backend(engine);
    Tokenizer tokenizer;
    expect(tokenizer.load(directory), "MTP align tokenizer failed");
    std::vector<int> tokens;
    if (promptCount == 30 || !promptFile.empty()) {
        GenerationConfig config;
        config.do_sample = false;
        config.thinking_budget_tokens = 0;
        aila::chat::ChatRequest request;
        std::string error;
        std::string messages = R"([{"role":"user","content":"Write a short paragraph (about 80 words) describing a sunrise over the ocean."}])";
        if (!promptFile.empty()) {
            std::ifstream in(promptFile, std::ios::binary);
            expect(bool(in), "MTP align prompt file missing");
            messages.assign(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
        }
        expect(aila::chat::parse_chat_request_json(messages, config, request, &error), "MTP align prompt parse failed");
        aila::chat::ChatFormatTextResult rendered;
        aila::chat::ChatFormatter formatter;
        expect(formatter.render_text(engine.make_chat_format_input(request), true, rendered, &error), error.c_str());
        tokens = tokenizer.encode(rendered.text);
        if (promptFile.empty()) expect(tokens.size() == 30, "MTP align prompt identity changed");
        else { expect(tokens.size() > 0 && int(tokens.size()) + rowsMax + 2 <= capacity, "MTP align corpus exceeds context"); promptCount = int(tokens.size()); }
    } else {
        std::string text;
        while (tokenizer.encode(text).size() < size_t(promptCount))
            text += "The lighthouse overlooks the ocean. At sunrise the keeper watches the waves and writes in the logbook. ";
        tokens = tokenizer.encode(text);
        tokens.resize(promptCount);
    }
    const int P = promptCount;
    const int L = Access::LayerCount(target), H = Access::HiddenSize(target), V = Access::VocabSize(target);
    Tensor ids = Tensor::allocate(ctx, {capacity}, dnnl::memory::data_type::s32);
    Access::ReserveVerifyLogits(engine, 4);
    Access::PrepareVerification(target, ctx);
    Tensor attn1 = Tensor::allocate(ctx, {int64_t(L) * 1 * H}, dnnl::memory::data_type::bf16);
    Tensor router1 = Tensor::allocate(ctx, {int64_t(L) * 1 * H}, dnnl::memory::data_type::bf16);
    Tensor hidden1 = Tensor::allocate(ctx, {int64_t(L) * 1 * 10240}, dnnl::memory::data_type::bf16);
    Tensor attnIn1 = Tensor::allocate(ctx, {int64_t(L) * 1 * H}, dnnl::memory::data_type::bf16);
    Tensor attnRaw1 = Tensor::allocate(ctx, {int64_t(L) * 1 * 6144}, dnnl::memory::data_type::bf16);
    Tensor attnR = Tensor::allocate(ctx, {int64_t(L) * rowsMax * H}, dnnl::memory::data_type::bf16);
    Tensor routerR = Tensor::allocate(ctx, {int64_t(L) * rowsMax * H}, dnnl::memory::data_type::bf16);
    Tensor hiddenR = Tensor::allocate(ctx, {int64_t(L) * rowsMax * 10240}, dnnl::memory::data_type::bf16);
    Tensor attnInR = Tensor::allocate(ctx, {int64_t(L) * rowsMax * H}, dnnl::memory::data_type::bf16);
    Tensor attnRawR = Tensor::allocate(ctx, {int64_t(L) * rowsMax * 6144}, dnnl::memory::data_type::bf16);
    struct ProbeGuard {
        Qwen4ExpBnb4Backend& b;
        ~ProbeGuard() { Access::StageProbe(b, nullptr, nullptr, nullptr, nullptr, nullptr, 0); }
    } guard{target};

    auto& session = Access::Session(engine);
    auto& layerCaches = session.layer_caches_for_testing();
    auto& pleState = session.ple_state_for_testing();
    struct Span { void* runtime; size_t offset, bytes; };
    std::vector<Span> spans;
    size_t stateBytes = 0;
    for (auto& cache : layerCaches)
        for (auto* t : {&cache.linear_state, &cache.linear_conv_state})
            if (t->valid()) { spans.push_back({t->data(), stateBytes, t->size_bytes()}); stateBytes += t->size_bytes(); }
    spans.push_back({pleState.conv_state.data(), stateBytes, pleState.conv_state.size_bytes()});
    stateBytes += pleState.conv_state.size_bytes();
    Tensor checkpoint = Tensor::allocate(ctx, {int64_t(stateBytes)}, dnnl::memory::data_type::u8);
    std::vector<int> convHeads(layerCaches.size()), blocks(convHeads.size());
    std::array<int, 2> pleTokens{};
    int pleHead = 0, length = 0, noMiss = 0;
    auto capture = [&] {
        length = session.current_length_for_testing();
        noMiss = session.consecutive_no_miss_layers_for_testing();
        pleTokens = pleState.prev_tokens;
        pleHead = pleState.conv_head;
        for (size_t i = 0; i < convHeads.size(); ++i) {
            convHeads[i] = layerCaches[i].linear_conv_head;
            blocks[i] = layerCaches[i].num_compressed_blocks;
        }
        for (const auto& s : spans) ctx.queue().memcpy(checkpoint.data_as<uint8_t>() + s.offset, s.runtime, s.bytes);
    };
    auto restore = [&] {
        for (const auto& s : spans) ctx.queue().memcpy(s.runtime, checkpoint.data_as<uint8_t>() + s.offset, s.bytes);
        session.current_length_for_testing() = length;
        session.consecutive_no_miss_layers_for_testing() = noMiss;
        pleState.prev_tokens = pleTokens;
        pleState.conv_head = pleHead;
        for (size_t i = 0; i < convHeads.size(); ++i) {
            auto& c = layerCaches[i];
            c.linear_conv_head = convHeads[i];
            c.num_compressed_blocks = blocks[i];
            if (c.linear_state.valid()) c.device_state_dirty = true;
        }
    };
    auto argmax = [&](const bf16* host) {
        int best = 0;
        for (int i = 0; i < V; ++i) {
            expect(std::isfinite(float(host[i])), "MTP align nonfinite logits");
            if (float(host[i]) > float(host[best])) best = i;
        }
        return best;
    };
    std::vector<bf16> tmp(V);
    std::vector<bf16> refLog(size_t(rowsMax) * V), bat4Log(4 * size_t(V)), bat2Log(2 * size_t(V));
    std::vector<bf16> refAttn(size_t(L) * rowsMax * H), refRouter(size_t(L) * rowsMax * H);
    std::vector<bf16> refHidden(size_t(L) * rowsMax * 10240);
    std::vector<bf16> refAttnIn(size_t(L) * rowsMax * H), refAttnRaw(size_t(L) * rowsMax * 6144);
    std::vector<bf16> batAttn(size_t(L) * rowsMax * H), batRouter(size_t(L) * rowsMax * H);
    std::vector<bf16> batHidden(size_t(L) * rowsMax * 10240);
    std::vector<bf16> batAttnIn(size_t(L) * rowsMax * H), batAttnRaw(size_t(L) * rowsMax * 6144);

    std::vector<int> prompt(tokens.begin(), tokens.begin() + P);
    engine.reset_context();
    ctx.memcpy_h2d(ids.data(), prompt.data(), size_t(P) * sizeof(int));
    auto& promptLogits = Access::MtpTargetPrompt(engine, ids.data_as<int>(), P);
    ctx.synchronize();
    ctx.memcpy_d2h(tmp.data(), promptLogits.data_as<bf16>(), size_t(V) * sizeof(bf16));
    const int promptTop1 = argmax(tmp.data());
    std::vector<int> cont;
    {
        Access::StageProbe(target, attn1.data_as<bf16>(), router1.data_as<bf16>(), hidden1.data_as<bf16>(),
                           attnIn1.data_as<bf16>(), attnRaw1.data_as<bf16>(), 1);
        capture();
        ctx.synchronize();
        expect(length == P, "MTP align checkpoint anchor mismatch");
        int next = promptTop1;
        for (int r = 0; r < rowsMax; ++r) {
            cont.push_back(next);
            ctx.memcpy_h2d(ids.data(), &next, sizeof(int));
            auto& lg = Access::VerifyRows(engine, ids.data_as<int>(), 1);
            ctx.memcpy_d2h(refLog.data() + size_t(r) * V, lg.data_as<bf16>(), size_t(V) * sizeof(bf16));
            ctx.memcpy_d2h(refAttn.data() + size_t(r) * L * H, attn1.data(), size_t(L) * H * sizeof(bf16));
            ctx.memcpy_d2h(refRouter.data() + size_t(r) * L * H, router1.data(), size_t(L) * H * sizeof(bf16));
            ctx.memcpy_d2h(refHidden.data() + size_t(r) * L * 10240, hidden1.data(), size_t(L) * 10240 * sizeof(bf16));
            ctx.memcpy_d2h(refAttnIn.data() + size_t(r) * L * H, attnIn1.data(), size_t(L) * H * sizeof(bf16));
            ctx.memcpy_d2h(refAttnRaw.data() + size_t(r) * L * 6144, attnRaw1.data(), size_t(L) * 6144 * sizeof(bf16));
            next = argmax(refLog.data() + size_t(r) * V);
        }
    }
    std::vector<std::vector<float>> refStateF;
    for (size_t l = 0; l < layerCaches.size(); ++l) {
        auto& st = layerCaches[l].linear_state;
        std::vector<float> f;
        if (st.valid()) { f.resize(st.numel()); ctx.memcpy_d2h(f.data(), st.data(), st.size_bytes()); }
        refStateF.push_back(std::move(f));
    }
    auto runBatched = [&](int width, std::vector<bf16>& outLog, std::vector<bf16>& outAttn, std::vector<bf16>& outRouter, std::vector<bf16>& outHidden,
                          std::vector<bf16>& outAttnIn, std::vector<bf16>& outAttnRaw) {
        restore();
        ctx.synchronize();
        Access::StageProbe(target, attnR.data_as<bf16>(), routerR.data_as<bf16>(), hiddenR.data_as<bf16>(),
                           attnInR.data_as<bf16>(), attnRawR.data_as<bf16>(), rowsMax);
        ctx.memcpy_h2d(ids.data(), cont.data(), size_t(width) * sizeof(int));
        auto& lg = Access::VerifyRows(engine, ids.data_as<int>(), width);
        ctx.memcpy_d2h(outLog.data(), lg.data_as<bf16>(), size_t(width) * V * sizeof(bf16));
        ctx.memcpy_d2h(outAttn.data(), attnR.data(), size_t(L) * rowsMax * H * sizeof(bf16));
        ctx.memcpy_d2h(outRouter.data(), routerR.data(), size_t(L) * rowsMax * H * sizeof(bf16));
        ctx.memcpy_d2h(outHidden.data(), hiddenR.data(), size_t(L) * rowsMax * 10240 * sizeof(bf16));
        ctx.memcpy_d2h(outAttnIn.data(), attnInR.data(), size_t(L) * rowsMax * H * sizeof(bf16));
        ctx.memcpy_d2h(outAttnRaw.data(), attnRawR.data(), size_t(L) * rowsMax * 6144 * sizeof(bf16));
    };
    runBatched(4, bat4Log, batAttn, batRouter, batHidden, batAttnIn, batAttnRaw);
    std::vector<std::vector<float>> bstate;
    for (size_t l = 0; l < layerCaches.size(); ++l) {
        auto& st = layerCaches[l].linear_state;
        std::vector<float> f;
        if (st.valid()) { f.resize(st.numel()); ctx.memcpy_d2h(f.data(), st.data(), st.size_bytes()); }
        bstate.push_back(std::move(f));
    }
    std::vector<bf16> bat2Attn(size_t(L) * rowsMax * H), bat2Router(size_t(L) * rowsMax * H);
    std::vector<bf16> bat2Hidden(size_t(L) * rowsMax * 10240);
    std::vector<bf16> bat2AttnIn(size_t(L) * rowsMax * H), bat2AttnRaw(size_t(L) * rowsMax * 6144);
    runBatched(2, bat2Log, bat2Attn, bat2Router, bat2Hidden, bat2AttnIn, bat2AttnRaw);

    auto topk = [&](const bf16* x, int k, std::vector<int>& out) {
        out.resize(k);
        std::array<int, 5> t{};
        for (int i = 0; i < k; ++i) t[i] = i;
        for (int i = k; i < V; ++i)
            for (int j = 0; j < k; ++j)
                if (float(x[i]) > float(x[t[j]])) { for (int m = k - 1; m > j; --m) t[m] = t[m - 1]; t[j] = i; break; }
        for (int i = 0; i < k; ++i) out[i] = t[i];
    };
    auto logitCompare = [&](const bf16* ref, const bf16* cand) {
        size_t different = 0;
        float maxAbs = 0;
        for (int i = 0; i < V; ++i) {
            different += std::memcmp(&ref[i], &cand[i], sizeof(bf16)) != 0;
            maxAbs = std::max(maxAbs, std::abs(float(ref[i]) - float(cand[i])));
        }
        std::vector<int> r5, c5;
        topk(ref, 5, r5); topk(cand, 5, c5);
        int overlap = 0;
        for (int a = 0; a < 5; ++a) for (int b = 0; b < 5; ++b) if (r5[a] == c5[b]) ++overlap;
        return std::array<double, 4>{double(different), double(maxAbs), double(r5[0] == c5[0]), double(overlap)};
    };
    // Per-row localization over the [layers, rows, hidden] stage buffers.
    auto stageFirstLayer = [&](const std::vector<bf16>& ref, const std::vector<bf16>& cand, int row, int width) {
        for (int l = 0; l < L; ++l)
            for (int d = 0; d < width; ++d) {
                const bf16 a = ref[size_t(row) * L * width + size_t(l) * width + d];
                const bf16 b = cand[size_t(l) * rowsMax * width + size_t(row) * width + d];
                if (std::memcmp(&a, &b, sizeof(bf16)) != 0) return l;
            }
        return -1;
    };
    auto stageMaxAbs = [&](const std::vector<bf16>& ref, const std::vector<bf16>& cand, int row, int width) {
        float worst = 0;
        for (int l = 0; l < L; ++l)
            for (int d = 0; d < width; ++d)
                worst = std::max(worst, std::abs(float(ref[size_t(row) * L * width + size_t(l) * width + d]) -
                                                 float(cand[size_t(l) * rowsMax * width + size_t(row) * width + d])));
        return worst;
    };
    auto hiddenFirst = [&](const std::vector<bf16>& ref, const std::vector<bf16>& cand, int row) {
        for (int l = 0; l < L; ++l)
            for (int d = 0; d < 10240; ++d) {
                const bf16 a = ref[size_t(row) * L * 10240 + size_t(l) * 10240 + d];
                const bf16 b = cand[size_t(l) * rowsMax * 10240 + size_t(row) * 10240 + d];
                if (std::memcmp(&a, &b, sizeof(bf16)) != 0) return l;
            }
        return -1;
    };
    auto hiddenMaxAbs = [&](const std::vector<bf16>& ref, const std::vector<bf16>& cand, int row) {
        float worst = 0;
        for (int l = 0; l < L; ++l)
            for (int d = 0; d < 10240; ++d)
                worst = std::max(worst, std::abs(float(ref[size_t(row) * L * 10240 + size_t(l) * 10240 + d]) -
                                                 float(cand[size_t(l) * rowsMax * 10240 + size_t(row) * 10240 + d])));
        return worst;
    };
    std::cout << "[MTP ALIGN] {\"context\":" << capacity << ",\"prompt\":" << P << ",\"rows\":" << rowsMax
              << ",\"prompt_top1\":" << promptTop1 << ",\"start\":";
    std::cout << '['; for (size_t i = 0; i < cont.size(); ++i) { if (i) std::cout << ','; std::cout << cont[i]; } std::cout << "]}\n";
    for (int rows : {rowsMax, 2}) {
        const std::vector<bf16>& bl = rows == 4 ? bat4Log : bat2Log;
        const std::vector<bf16>& ba = rows == 4 ? batAttn : bat2Attn;
        const std::vector<bf16>& br = rows == 4 ? batRouter : bat2Router;
        const std::vector<bf16>& bh = rows == 4 ? batHidden : bat2Hidden;
        const std::vector<bf16>& bai = rows == 4 ? batAttnIn : bat2AttnIn;
        const std::vector<bf16>& bar = rows == 4 ? batAttnRaw : bat2AttnRaw;
        for (int r = 0; r < rows; ++r) {
            auto m = logitCompare(refLog.data() + size_t(r) * V, bl.data() + size_t(r) * V);
            const float aWorst = stageMaxAbs(refAttn, ba, r, H);
            const float rWorst = stageMaxAbs(refRouter, br, r, H);
            const int aRow = stageFirstLayer(refAttn, ba, r, H);
            const int rRow = stageFirstLayer(refRouter, br, r, H);
            const int aiRow = stageFirstLayer(refAttnIn, bai, r, H);
            const int arRow = stageFirstLayer(refAttnRaw, bar, r, 6144);
            const int hRow = hiddenFirst(refHidden, bh, r);
            const float hWorst = hiddenMaxAbs(refHidden, bh, r);
            std::cout << "[MTP ALIGN ROW] {\"width\":" << rows << ",\"row\":" << r
                      << ",\"logit_different\":" << m[0] << ",\"logit_max_abs\":" << m[1]
                      << ",\"top1_match\":" << int(m[2]) << ",\"top5_overlap\":" << int(m[3])
                      << ",\"hidden_first_diff_layer\":" << hRow
                      << ",\"attn_in_first_diff_layer\":" << aiRow
                      << ",\"attn_raw_first_diff_layer\":" << arRow
                      << ",\"attn_first_diff_layer\":" << aRow
                      << ",\"router_first_diff_layer\":" << rRow
                      << ",\"hidden_max_abs\":" << hWorst
                      << ",\"attn_max_abs\":" << aWorst << ",\"router_max_abs\":" << rWorst << "}\n";
        }
    }
    // GDN final-state comparison: T1(rowsMax) vs B(rowsMax).
    {
        for (size_t l = 0; l < layerCaches.size(); ++l) {
            if (bstate[l].empty()) continue;
            size_t different = 0;
            float maxAbs = 0;
            for (size_t i = 0; i < bstate[l].size(); ++i) {
                different += std::memcmp(&bstate[l][i], &refStateF[l][i], sizeof(float)) != 0;
                maxAbs = std::max(maxAbs, std::abs(bstate[l][i] - refStateF[l][i]));
            }
            if (different) std::cout << "[MTP ALIGN STATE] {\"layer\":" << l << ",\"different\":" << different
                                     << ",\"max_abs\":" << maxAbs << "}\n";
        }
    }
    std::cout << "[MTP ALIGN END]\n";
}
