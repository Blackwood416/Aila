#include "src/ops/Bnb4BitLinear.cpp"

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
    const uint16_t version = 1;
    const uint16_t reserved = 0;
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
        const IRTensor packed =
            read_tensor(testcase / "inputs" / "packed.irtensor");
        const IRTensor qmap =
            read_tensor(testcase / "inputs" / "qmap.irtensor");
        const IRTensor absmax =
            read_tensor(testcase / "inputs" / "absmax.irtensor");
        const IRTensor input =
            read_tensor(testcase / "inputs" / "input.irtensor");
        const IRTensor reference =
            read_tensor(testcase / "reference" / "output.irtensor");
        if (packed.dtype != 6 || qmap.dtype != 12 || absmax.dtype != 12 ||
            input.dtype != 11 || reference.dtype != 11 ||
            packed.shape.size() != 2 || qmap.shape != std::vector<int64_t>{16} ||
            absmax.shape.size() != 2 || input.shape.size() != 2 ||
            reference.shape.size() != 2) {
            throw std::runtime_error("unsupported NF4 testcase boundary");
        }
        const int M = static_cast<int>(input.shape[0]);
        const int K = static_cast<int>(input.shape[1]);
        const int N = static_cast<int>(packed.shape[0]);
        if (packed.shape[1] != K / 2 || reference.shape[0] != M ||
            reference.shape[1] != N || absmax.shape[0] != N ||
            K % absmax.shape[1] != 0) {
            throw std::runtime_error("inconsistent NF4 testcase shapes");
        }
        const int blocksize = K / static_cast<int>(absmax.shape[1]);

        Context context;
        Tensor packed_device = Tensor::allocate(
            context, packed.shape, dnnl::memory::data_type::u8);
        Tensor qmap_device = Tensor::allocate(
            context, qmap.shape, dnnl::memory::data_type::f32);
        Tensor absmax_device = Tensor::allocate(
            context, absmax.shape, dnnl::memory::data_type::f32);
        Tensor input_device = Tensor::allocate(
            context, input.shape, dnnl::memory::data_type::bf16);
        Tensor output_device = Tensor::allocate(
            context, reference.shape, dnnl::memory::data_type::bf16);
        context.memcpy_h2d(packed_device.data(), packed.payload.data(),
                           packed.payload.size());
        context.memcpy_h2d(qmap_device.data(), qmap.payload.data(),
                           qmap.payload.size());
        context.memcpy_h2d(absmax_device.data(), absmax.payload.data(),
                           absmax.payload.size());
        context.memcpy_h2d(input_device.data(), input.payload.data(),
                           input.payload.size());
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
        context.memcpy_d2h(result.payload.data(), output_device.data(),
                           result.payload.size());
        write_tensor(output_directory / "output.irtensor", result);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
