#include "src/ops/Bnb4BitLinear.cpp"
#include "src/ops/HyperConnectionOps.cpp"
#include "src/ops/MoeExpertOps.cpp"
#include "src/ops/GatedDeltaNetOps.cpp"

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace aila::env {
int g_q35_prefill_step_override = -1;
bool g_kv_quant_override = false;
}

namespace fs = std::filesystem;

namespace {

struct IRTensor {
    uint16_t dtype = 0;
    std::vector<int64_t> shape;
    std::vector<uint8_t> payload;
};

template <typename T> T load(const std::vector<uint8_t>& bytes, size_t offset) {
    if (offset + sizeof(T) > bytes.size()) {
        throw std::runtime_error("truncated .irtensor");
    }
    T value{};
    std::memcpy(&value, bytes.data() + offset, sizeof(value));
    return value;
}

IRTensor read_tensor(const fs::path& path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) throw std::runtime_error("cannot open " + path.string());
    stream.seekg(0, std::ios::end);
    const auto length = stream.tellg();
    if (length < 48) throw std::runtime_error("invalid .irtensor length");
    stream.seekg(0, std::ios::beg);
    std::vector<uint8_t> bytes(static_cast<size_t>(length));
    stream.read(reinterpret_cast<char*>(bytes.data()), length);
    if (std::memcmp(bytes.data(), "IRTN", 4) != 0 ||
        load<uint16_t>(bytes, 4) != 1 || load<uint32_t>(bytes, 10) != 1 ||
        load<uint16_t>(bytes, 16) != 0) {
        throw std::runtime_error("unsupported .irtensor header");
    }
    IRTensor tensor;
    const uint16_t header_size = load<uint16_t>(bytes, 6);
    tensor.dtype = load<uint16_t>(bytes, 8);
    const uint16_t rank = load<uint16_t>(bytes, 14);
    const uint64_t payload_bytes = load<uint64_t>(bytes, 26);
    if (header_size > bytes.size() ||
        payload_bytes != bytes.size() - header_size ||
        header_size != 48 + 16 * rank) {
        throw std::runtime_error("malformed .irtensor size");
    }
    tensor.shape.resize(rank);
    for (uint16_t index = 0; index < rank; ++index) {
        tensor.shape[index] = load<int64_t>(bytes, 34 + 8 * index);
    }
    tensor.payload.assign(bytes.begin() + header_size, bytes.end());
    return tensor;
}

void store(std::vector<uint8_t>& bytes, size_t offset, const void* value,
           size_t count) {
    std::memcpy(bytes.data() + offset, value, count);
}

void write_tensor(const fs::path& path, const IRTensor& tensor) {
    const uint16_t rank = static_cast<uint16_t>(tensor.shape.size());
    const uint16_t header_size = static_cast<uint16_t>(48 + 16 * rank);
    uint64_t numel = 1;
    for (int64_t dimension : tensor.shape) {
        numel *= static_cast<uint64_t>(dimension);
    }
    std::vector<uint8_t> header(header_size, 0);
    store(header, 0, "IRTN", 4);
    const uint16_t version = 1, reserved = 0;
    const uint32_t flags = 1;
    const uint64_t payload_bytes = tensor.payload.size();
    store(header, 4, &version, 2);
    store(header, 6, &header_size, 2);
    store(header, 8, &tensor.dtype, 2);
    store(header, 10, &flags, 4);
    store(header, 14, &rank, 2);
    store(header, 16, &reserved, 2);
    store(header, 18, &numel, 8);
    store(header, 26, &payload_bytes, 8);
    int64_t stride = 1;
    std::vector<int64_t> strides(rank);
    for (int index = rank - 1; index >= 0; --index) {
        strides[static_cast<size_t>(index)] = stride;
        stride *= tensor.shape[static_cast<size_t>(index)];
    }
    for (uint16_t index = 0; index < rank; ++index) {
        store(header, 34 + 8 * index, &tensor.shape[index], 8);
        store(header, 34 + 8 * rank + 8 * index, &strides[index], 8);
    }
    const int64_t storage_offset = 0;
    store(header, 34 + 16 * rank, &storage_offset, 8);
    fs::create_directories(path.parent_path());
    std::ofstream output(path, std::ios::binary);
    output.write(reinterpret_cast<const char*>(header.data()), header.size());
    output.write(reinterpret_cast<const char*>(tensor.payload.data()),
                 tensor.payload.size());
    if (!output) throw std::runtime_error("failed to write output .irtensor");
}

} // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 4 || std::string(argv[2]) != "--output") {
            throw std::runtime_error(
                "usage: adapter <testcase> --output <directory>");
        }
        const fs::path testcase = fs::absolute(argv[1]);
        const fs::path output_directory = fs::absolute(argv[3]);

        Context context;

        // 1. Standard NF4 GEMM boundary execution (base operator in Qwen4Exp)
        if (fs::exists(testcase / "inputs" / "packed.irtensor")) {
            const IRTensor packed = read_tensor(testcase / "inputs" / "packed.irtensor");
            const IRTensor qmap = read_tensor(testcase / "inputs" / "qmap.irtensor");
            const IRTensor absmax = read_tensor(testcase / "inputs" / "absmax.irtensor");
            const IRTensor input = read_tensor(testcase / "inputs" / "input.irtensor");
            const IRTensor reference = read_tensor(testcase / "reference" / "output.irtensor");
            if (packed.dtype != 6 || qmap.dtype != 12 || absmax.dtype != 12 ||
                input.dtype != 11 || reference.dtype != 11 ||
                packed.shape.size() != 2 || qmap.shape != std::vector<int64_t>{16} ||
                absmax.shape.size() != 2 || input.shape.size() != 2 ||
                reference.shape.size() != 2) {
                throw std::runtime_error("unsupported Qwen4Exp NF4 testcase boundary");
            }
            const int M = static_cast<int>(input.shape[0]);
            const int K = static_cast<int>(input.shape[1]);
            const int N = static_cast<int>(packed.shape[0]);
            if (packed.shape[1] != K / 2 || reference.shape[0] != M ||
                reference.shape[1] != N || absmax.shape[0] != N ||
                K % absmax.shape[1] != 0) {
                throw std::runtime_error("inconsistent Qwen4Exp NF4 testcase shapes");
            }
            const int blocksize = K / static_cast<int>(absmax.shape[1]);

            Tensor packed_device = Tensor::allocate(context, packed.shape, dnnl::memory::data_type::u8);
            Tensor qmap_device = Tensor::allocate(context, qmap.shape, dnnl::memory::data_type::f32);
            Tensor absmax_device = Tensor::allocate(context, absmax.shape, dnnl::memory::data_type::f32);
            Tensor input_device = Tensor::allocate(context, input.shape, dnnl::memory::data_type::bf16);
            Tensor output_device = Tensor::allocate(context, reference.shape, dnnl::memory::data_type::bf16);

            context.memcpy_h2d(packed_device.data(), packed.payload.data(), packed.payload.size());
            context.memcpy_h2d(qmap_device.data(), qmap.payload.data(), qmap.payload.size());
            context.memcpy_h2d(absmax_device.data(), absmax.payload.data(), absmax.payload.size());
            context.memcpy_h2d(input_device.data(), input.payload.data(), input.payload.size());

            packed_nf4_gemm_bf16(
                context, static_cast<const uint8_t*>(packed_device.data()),
                static_cast<const float*>(qmap_device.data()),
                static_cast<const float*>(absmax_device.data()),
                static_cast<const bf16*>(input_device.data()),
                static_cast<bf16*>(output_device.data()), M, N, K, blocksize);
            context.synchronize();

            IRTensor result;
            result.dtype = 11;
            result.shape = reference.shape;
            result.payload.resize(output_device.size_bytes());
            context.memcpy_d2h(result.payload.data(), output_device.data(), result.payload.size());
            write_tensor(output_directory / "output.irtensor", result);
            return 0;
        }

        // 2. Grouped RMSNorm
        if (fs::exists(testcase / "inputs" / "hc_norm_x.irtensor")) {
            const IRTensor x = read_tensor(testcase / "inputs" / "hc_norm_x.irtensor");
            const int rows = static_cast<int>(x.shape[0]);
            const int total_dim = static_cast<int>(x.shape[1]);
            constexpr int groups = 4;
            const int group_dim = total_dim / groups;

            Tensor x_dev = Tensor::allocate(context, x.shape, dnnl::memory::data_type::bf16);
            Tensor out_dev = Tensor::allocate(context, x.shape, dnnl::memory::data_type::bf16);
            context.memcpy_h2d(x_dev.data(), x.payload.data(), x.payload.size());

            Tensor w_dev;
            const float* w_ptr = nullptr;
            if (fs::exists(testcase / "inputs" / "hc_norm_w.irtensor")) {
                const IRTensor w = read_tensor(testcase / "inputs" / "hc_norm_w.irtensor");
                w_dev = Tensor::allocate(context, w.shape, dnnl::memory::data_type::f32);
                context.memcpy_h2d(w_dev.data(), w.payload.data(), w.payload.size());
                w_ptr = static_cast<const float*>(w_dev.data());
            }

            ops::grouped_rms_norm(
                context, static_cast<const bf16*>(x_dev.data()),
                w_ptr, 1e-6f, static_cast<bf16*>(out_dev.data()),
                rows, groups, group_dim, /*plus_one=*/true);
            context.synchronize();

            IRTensor result;
            result.dtype = x.dtype;
            result.shape = x.shape;
            result.payload.resize(out_dev.size_bytes());
            context.memcpy_d2h(result.payload.data(), out_dev.data(), result.payload.size());
            write_tensor(output_directory / "output.irtensor", result);
            return 0;
        }

        // 3. HC Mix Reduce
        if (fs::exists(testcase / "inputs" / "hc_mix_n.irtensor")) {
            const IRTensor N = read_tensor(testcase / "inputs" / "hc_mix_n.irtensor");
            const IRTensor mix = read_tensor(testcase / "inputs" / "hc_mix_val.irtensor");
            const int rows = static_cast<int>(N.shape[0]);
            constexpr int groups = 4;
            const int group_dim = static_cast<int>(N.shape[1]) / groups;

            Tensor N_dev = Tensor::allocate(context, N.shape, dnnl::memory::data_type::bf16);
            Tensor mix_dev = Tensor::allocate(context, mix.shape, dnnl::memory::data_type::bf16);
            Tensor out_dev = Tensor::allocate(context, {rows, group_dim}, dnnl::memory::data_type::bf16);

            context.memcpy_h2d(N_dev.data(), N.payload.data(), N.payload.size());
            context.memcpy_h2d(mix_dev.data(), mix.payload.data(), mix.payload.size());

            ops::hc_mix_reduce(
                context, static_cast<const bf16*>(N_dev.data()),
                static_cast<const bf16*>(mix_dev.data()),
                static_cast<bf16*>(out_dev.data()),
                rows, group_dim, groups);
            context.synchronize();

            IRTensor result;
            result.dtype = N.dtype;
            result.shape = {rows, group_dim};
            result.payload.resize(out_dev.size_bytes());
            context.memcpy_d2h(result.payload.data(), out_dev.data(), result.payload.size());
            write_tensor(output_directory / "output.irtensor", result);
            return 0;
        }

        // 4. HC Inject Add
        if (fs::exists(testcase / "inputs" / "hc_inject_h.irtensor")) {
            const IRTensor H = read_tensor(testcase / "inputs" / "hc_inject_h.irtensor");
            const IRTensor y = read_tensor(testcase / "inputs" / "hc_inject_y.irtensor");
            const IRTensor inj = read_tensor(testcase / "inputs" / "hc_inject_inj.irtensor");
            const int rows = static_cast<int>(H.shape[0]);
            constexpr int groups = 4;
            const int group_dim = static_cast<int>(H.shape[1]) / groups;

            Tensor H_dev = Tensor::allocate(context, H.shape, dnnl::memory::data_type::bf16);
            Tensor y_dev = Tensor::allocate(context, y.shape, dnnl::memory::data_type::bf16);
            Tensor inj_dev = Tensor::allocate(context, inj.shape, dnnl::memory::data_type::f32);

            context.memcpy_h2d(H_dev.data(), H.payload.data(), H.payload.size());
            context.memcpy_h2d(y_dev.data(), y.payload.data(), y.payload.size());
            context.memcpy_h2d(inj_dev.data(), inj.payload.data(), inj.payload.size());

            ops::hc_inject_add(
                context, static_cast<bf16*>(H_dev.data()),
                static_cast<const bf16*>(y_dev.data()),
                static_cast<const float*>(inj_dev.data()),
                rows, group_dim, groups);
            context.synchronize();

            IRTensor result;
            result.dtype = H.dtype;
            result.shape = H.shape;
            result.payload.resize(H_dev.size_bytes());
            context.memcpy_d2h(result.payload.data(), H_dev.data(), result.payload.size());
            write_tensor(output_directory / "output.irtensor", result);
            return 0;
        }

        // 5. PLE Gate Value
        if (fs::exists(testcase / "inputs" / "ple_k.irtensor")) {
            const IRTensor K = read_tensor(testcase / "inputs" / "ple_k.irtensor");
            const IRTensor Q = read_tensor(testcase / "inputs" / "ple_q.irtensor");
            const IRTensor V = read_tensor(testcase / "inputs" / "ple_v.irtensor");
            const int rows = static_cast<int>(K.shape[0]);
            constexpr int hc_count = 4;
            constexpr int hidden_size = 2560;

            Tensor K_dev = Tensor::allocate(context, K.shape, dnnl::memory::data_type::bf16);
            Tensor Q_dev = Tensor::allocate(context, Q.shape, dnnl::memory::data_type::bf16);
            Tensor V_dev = Tensor::allocate(context, V.shape, dnnl::memory::data_type::bf16);
            Tensor GV_dev = Tensor::allocate(context, K.shape, dnnl::memory::data_type::bf16);

            context.memcpy_h2d(K_dev.data(), K.payload.data(), K.payload.size());
            context.memcpy_h2d(Q_dev.data(), Q.payload.data(), Q.payload.size());
            context.memcpy_h2d(V_dev.data(), V.payload.data(), V.payload.size());

            ops::ple_gate_value(
                context, static_cast<const bf16*>(K_dev.data()),
                static_cast<const bf16*>(Q_dev.data()),
                static_cast<const bf16*>(V_dev.data()),
                static_cast<bf16*>(GV_dev.data()),
                rows, hc_count, hidden_size);
            context.synchronize();

            IRTensor result;
            result.dtype = K.dtype;
            result.shape = K.shape;
            result.payload.resize(GV_dev.size_bytes());
            context.memcpy_d2h(result.payload.data(), GV_dev.data(), result.payload.size());
            write_tensor(output_directory / "output.irtensor", result);
            return 0;
        }

        // 6. Router Softmax TopK
        if (fs::exists(testcase / "inputs" / "router_scores.irtensor")) {
            const IRTensor scores = read_tensor(testcase / "inputs" / "router_scores.irtensor");
            const int tokens = static_cast<int>(scores.shape[0]);
            const int num_experts = static_cast<int>(scores.shape[1]);
            constexpr int top_k = 10;

            Tensor scores_dev = Tensor::allocate(context, scores.shape, dnnl::memory::data_type::bf16);
            Tensor ids_dev = Tensor::allocate(context, {tokens, top_k}, dnnl::memory::data_type::s32);
            Tensor weights_dev = Tensor::allocate(context, {tokens, top_k}, dnnl::memory::data_type::f32);
            context.memcpy_h2d(scores_dev.data(), scores.payload.data(), scores.payload.size());

            ops::router_softmax_topk_device(
                context, static_cast<const bf16*>(scores_dev.data()), nullptr,
                static_cast<int*>(ids_dev.data()),
                static_cast<float*>(weights_dev.data()),
                tokens, num_experts, top_k, /*norm_topk_prob=*/true);
            context.synchronize();

            IRTensor result_w;
            result_w.dtype = 12; // float32
            result_w.shape = {tokens, top_k};
            result_w.payload.resize(weights_dev.size_bytes());
            context.memcpy_d2h(result_w.payload.data(), weights_dev.data(), result_w.payload.size());
            write_tensor(output_directory / "output.irtensor", result_w);
            return 0;
        }

        // Pass-through fallback for initial scaffold verification
        if (fs::exists(testcase / "inputs" / "input.irtensor")) {
            const IRTensor input = read_tensor(testcase / "inputs" / "input.irtensor");
            write_tensor(output_directory / "output.irtensor", input);
            return 0;
        }

        throw std::runtime_error("No supported input tensors found in: " + testcase.string());
    } catch (const std::exception& error) {
        std::cerr << "Qwen4ExpInferRefAdapter error: " << error.what() << '\n';
        return 1;
    }
}
