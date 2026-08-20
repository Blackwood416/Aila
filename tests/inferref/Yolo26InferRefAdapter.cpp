#include "core/Context.hpp"
#include "core/Tensor.hpp"

#include <dnnl.hpp>

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace fs = std::filesystem;

namespace {

struct IRTensor {
    uint16_t dtype = 0;
    std::vector<int64_t> shape;
    std::vector<uint8_t> payload;
};

template <typename T> T load(const std::vector<uint8_t>& bytes, size_t offset) {
    if (offset + sizeof(T) > bytes.size()) throw std::runtime_error("truncated .irtensor");
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
    if (std::memcmp(bytes.data(), "IRTN", 4) != 0 || load<uint16_t>(bytes, 4) != 1 ||
        load<uint32_t>(bytes, 10) != 1 || load<uint16_t>(bytes, 16) != 0) {
        throw std::runtime_error("unsupported .irtensor header");
    }
    IRTensor tensor;
    const uint16_t header_size = load<uint16_t>(bytes, 6);
    tensor.dtype = load<uint16_t>(bytes, 8);
    const uint16_t rank = load<uint16_t>(bytes, 14);
    const uint64_t payload_bytes = load<uint64_t>(bytes, 26);
    if (header_size > bytes.size() || payload_bytes != bytes.size() - header_size ||
        header_size != 48 + 16 * rank) throw std::runtime_error("malformed .irtensor size");
    tensor.shape.resize(rank);
    for (uint16_t index = 0; index < rank; ++index) {
        tensor.shape[index] = load<int64_t>(bytes, 34 + 8 * index);
    }
    tensor.payload.assign(bytes.begin() + header_size, bytes.end());
    return tensor;
}

void store(std::vector<uint8_t>& bytes, size_t offset, const void* value, size_t count) {
    std::memcpy(bytes.data() + offset, value, count);
}

void write_tensor(const fs::path& path, const IRTensor& tensor) {
    const uint16_t rank = static_cast<uint16_t>(tensor.shape.size());
    const uint16_t header_size = static_cast<uint16_t>(48 + 16 * rank);
    uint64_t numel = 1;
    for (int64_t dimension : tensor.shape) numel *= static_cast<uint64_t>(dimension);
    std::vector<uint8_t> header(header_size, 0);
    store(header, 0, "IRTN", 4);
    const uint16_t version = 1, reserved = 0;
    const uint32_t flags = 1;
    const uint64_t payload_bytes = tensor.payload.size();
    store(header, 4, &version, 2); store(header, 6, &header_size, 2);
    store(header, 8, &tensor.dtype, 2); store(header, 10, &flags, 4);
    store(header, 14, &rank, 2); store(header, 16, &reserved, 2);
    store(header, 18, &numel, 8); store(header, 26, &payload_bytes, 8);
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
    output.write(reinterpret_cast<const char*>(tensor.payload.data()), tensor.payload.size());
    if (!output) throw std::runtime_error("failed to write output .irtensor");
}

} // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 4 || std::string(argv[2]) != "--output") {
            throw std::runtime_error("usage: adapter <testcase> --output <directory>");
        }
        const fs::path testcase = fs::absolute(argv[1]);
        const fs::path output_directory = fs::absolute(argv[3]);
        const IRTensor input_host = read_tensor(testcase / "inputs" / "input.irtensor");
        const IRTensor weight_host = read_tensor(testcase / "inputs" / "weight.irtensor");
        const IRTensor bias_host = read_tensor(testcase / "inputs" / "bias.irtensor");
        const IRTensor output_shape = read_tensor(testcase / "reference" / "output.irtensor");
        if (input_host.dtype != 10 || weight_host.dtype != 10 || bias_host.dtype != 10 ||
            input_host.shape.size() != 4 || weight_host.shape.size() != 4 ||
            bias_host.shape.size() != 1 || output_shape.shape.size() != 4) {
            throw std::runtime_error("adapter supports float16 NCHW ConvSiLU only");
        }

        Context context;
        Tensor input = Tensor::allocate(context, input_host.shape, dnnl::memory::data_type::f16);
        Tensor weights = Tensor::allocate(context, weight_host.shape, dnnl::memory::data_type::f16);
        Tensor bias = Tensor::allocate(context, bias_host.shape, dnnl::memory::data_type::f16);
        Tensor output = Tensor::allocate(context, output_shape.shape, dnnl::memory::data_type::f16);
        context.memcpy_h2d(input.data(), input_host.payload.data(), input_host.payload.size());
        context.memcpy_h2d(weights.data(), weight_host.payload.data(), weight_host.payload.size());
        context.memcpy_h2d(bias.data(), bias_host.payload.data(), bias_host.payload.size());

        const int64_t in_channels = input_host.shape[1];
        const int64_t out_channels = weight_host.shape[0];
        const int64_t kernel = weight_host.shape[2];
        const int64_t groups = in_channels / weight_host.shape[1];
        const int64_t pad = kernel / 2;
        const int64_t stride = (input_host.shape[2] + 2 * pad - kernel) /
                               (output_shape.shape[2] - 1);
        auto src_md = dnnl::memory::desc(input_host.shape, dnnl::memory::data_type::f16,
                                         dnnl::memory::format_tag::nchw);
        auto dst_md = dnnl::memory::desc(output_shape.shape, dnnl::memory::data_type::f16,
                                         dnnl::memory::format_tag::nchw);
        dnnl::memory::desc weights_md = groups == 1
            ? dnnl::memory::desc(weight_host.shape, dnnl::memory::data_type::f16,
                                 dnnl::memory::format_tag::oihw)
            : dnnl::memory::desc({groups, out_channels / groups, in_channels / groups, kernel, kernel},
                                 dnnl::memory::data_type::f16, dnnl::memory::format_tag::goihw);
        auto bias_md = dnnl::memory::desc(bias_host.shape, dnnl::memory::data_type::f16,
                                          dnnl::memory::format_tag::a);
        dnnl::post_ops post_ops;
        post_ops.append_eltwise(dnnl::algorithm::eltwise_swish, 1.0f, 0.0f);
        dnnl::primitive_attr attr;
        attr.set_post_ops(post_ops);
        auto descriptor = dnnl::convolution_forward::primitive_desc(
            context.engine(), dnnl::prop_kind::forward_inference,
            dnnl::algorithm::convolution_direct, src_md, weights_md, bias_md, dst_md,
            {stride, stride}, {0, 0}, {pad, pad}, {pad, pad}, attr);
        dnnl::convolution_forward primitive(descriptor);
        std::unordered_map<int, dnnl::memory> arguments{
            {DNNL_ARG_SRC, input.make_dnnl_memory(src_md)},
            {DNNL_ARG_WEIGHTS, weights.make_dnnl_memory(weights_md)},
            {DNNL_ARG_BIAS, bias.make_dnnl_memory(bias_md)},
            {DNNL_ARG_DST, output.make_dnnl_memory(dst_md)}};
        primitive.execute(context.stream(), arguments);
        context.synchronize();
        IRTensor result;
        result.dtype = 10; result.shape = output_shape.shape;
        result.payload.resize(output.size_bytes());
        context.memcpy_d2h(result.payload.data(), output.data(), result.payload.size());
        write_tensor(output_directory / "output.irtensor", result);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
