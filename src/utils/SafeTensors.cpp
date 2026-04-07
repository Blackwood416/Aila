#include "SafeTensors.hpp"
#include "profile/Profiling.hpp"
#include <stdexcept>
#include <algorithm>
#include <filesystem>
#include <unordered_set>
#include <fstream>
#include <iterator>

// ============================================================
// ModelWeights 实现
// ============================================================

Tensor& ModelWeights::get(const std::string& name) {
    auto it = tensors_.find(name);
    if (it == tensors_.end()) {
        throw std::runtime_error("Weight not found: " + name);
    }
    return it->second;
}

const Tensor& ModelWeights::get(const std::string& name) const {
    auto it = tensors_.find(name);
    if (it == tensors_.end()) {
        throw std::runtime_error("Weight not found: " + name);
    }
    return it->second;
}

bool ModelWeights::has(const std::string& name) const {
    return tensors_.find(name) != tensors_.end();
}

void ModelWeights::put(const std::string& name, Tensor tensor) {
    tensors_.emplace(name, std::move(tensor));
}

void ModelWeights::replace(const std::string& name, Tensor tensor) {
    tensors_.erase(name);
    tensors_.emplace(name, std::move(tensor));
}

std::vector<std::string> ModelWeights::names() const {
    std::vector<std::string> result;
    result.reserve(tensors_.size());
    for (const auto& pair : tensors_) {
        result.push_back(pair.first);
    }
    std::sort(result.begin(), result.end());
    return result;
}

// ============================================================
// DataType -> oneDNN data_type 转换
// ============================================================

dnnl::memory::data_type to_dnnl_dtype(DataType dt) {
    switch (dt) {
        case DT_F16:  return dnnl::memory::data_type::f16;
        case DT_BF16: return dnnl::memory::data_type::bf16;
        case DT_F32:  return dnnl::memory::data_type::f32;
        case DT_F64:  return dnnl::memory::data_type::f64;
        case DT_U8:   return dnnl::memory::data_type::u8;
        case DT_S8:   return dnnl::memory::data_type::s8;
        default:
            throw std::runtime_error("Unsupported data type");
    }
}

// ============================================================
// format_tag 辅助
// ============================================================

static dnnl::memory::format_tag get_format_tag(int ndims) {
    switch (ndims) {
        case 1: return dnnl::memory::format_tag::a;
        case 2: return dnnl::memory::format_tag::ab;
        case 3: return dnnl::memory::format_tag::abc;
        case 4: return dnnl::memory::format_tag::abcd;
        case 5: return dnnl::memory::format_tag::abcde;
        case 6: return dnnl::memory::format_tag::abcdef;
        default: return dnnl::memory::format_tag::a;
    }
}

// ============================================================
// 解析 safetensors JSON header
// ============================================================

std::vector<std::string> ParseHeader(const std::string& header,
                                     std::unordered_map<std::string, TensorMeta>& metadata) {
    std::vector<std::string> layer_names;
    try {
        simdjson::padded_string padded_json(header);
        simdjson::ondemand::parser parser;
        simdjson::ondemand::document doc = parser.iterate(padded_json);

        for (auto field : doc.get_object()) {
            std::string_view key = field.unescaped_key();
            if (key == "__metadata__") continue;

            simdjson::ondemand::object tensor_info = field.value().get_object();
            std::string_view dtype = tensor_info["dtype"];

            TensorMeta meta;
            meta.shape.ndims = 0;

            // dtype 转换
            if (dtype == "F16" || dtype == "float16") {
                meta.dtype = DT_F16;
            } else if (dtype == "BF16" || dtype == "bfloat16") {
                meta.dtype = DT_BF16;
            } else if (dtype == "F32" || dtype == "float32") {
                meta.dtype = DT_F32;
            } else if (dtype == "F64" || dtype == "float64") {
                meta.dtype = DT_F64;
            } else if (dtype == "U8" || dtype == "uint8") {
                meta.dtype = DT_U8;
            } else if (dtype == "S8" || dtype == "int8") {
                meta.dtype = DT_S8;
            } else {
                meta.dtype = DT_UNKNOWN;
                AILA_LOG_WARN("Unknown dtype: %.*s", (int)dtype.size(), dtype.data());
            }

            for (int64_t dim : tensor_info["shape"].get_array()) {
                meta.shape.push_back(dim);
            }

            auto offsets = tensor_info["data_offsets"].get_array();
            auto it = offsets.begin();
            meta.byte_offset_start = static_cast<size_t>(int64_t(*it));
            ++it;
            meta.byte_offset_end = static_cast<size_t>(int64_t(*it));

            std::string layer_name(key);
            metadata[layer_name] = meta;
            layer_names.push_back(layer_name);
        }
    } catch (const std::exception& e) {
        AILA_LOG_ERROR("[ParseHeader] Failed: %s", e.what());
    }
    return layer_names;
}

// ============================================================
// 将单个 tensor 从 mmap 加载到 GPU, 返回 Tensor 对象
// ============================================================

static Tensor LoadTensorToGPU(const uint8_t* mmap_base_ptr,
                               const TensorMeta& meta,
                               Context& ctx) {
    size_t size_in_bytes = meta.byte_offset_end - meta.byte_offset_start;
    const void* host_src_ptr = mmap_base_ptr + meta.byte_offset_start;

    // 构造 shape vector
    std::vector<int64_t> shape(meta.shape.dims, meta.shape.dims + meta.shape.ndims);
    dnnl::memory::data_type dnnl_dtype = to_dnnl_dtype(meta.dtype);
    // Use owning tensor so replacement/free lifecycle is correct.
    Tensor t = Tensor::allocate(ctx, shape, dnnl_dtype);
    ctx.memcpy_h2d(t.data(), host_src_ptr, size_in_bytes);

    return t;
}

// ============================================================
// 主加载函数
// ============================================================

ModelWeights LoadSafetensors(const std::string& path, Context& ctx) {
    ModelWeights weights;

    try {
        // 打开 mmap 文件
        weights.mmap_file_ = std::make_unique<MemoryMappedFile>(path);
        const uint8_t* raw_ptr = weights.mmap_file_->data();

        // 解析 header
        uint64_t header_size = *reinterpret_cast<const uint64_t*>(raw_ptr);
        std::string json_str(reinterpret_cast<const char*>(raw_ptr + 8), header_size);
        AILA_LOG_INFO("[SafeTensors] Header size: %llu bytes", (unsigned long long)header_size);

        const uint8_t* tensor_data_start = raw_ptr + 8 + header_size;

        std::unordered_map<std::string, TensorMeta> metadata;
        auto layer_names = ParseHeader(json_str, metadata);
        AILA_LOG_INFO("[SafeTensors] Found %zu tensors", layer_names.size());

        // 加载所有 tensor 到 GPU
        size_t total_bytes = 0;
        int success = 0;
        int fail = 0;

        for (const auto& name : layer_names) {
            const auto& meta = metadata[name];
            try {
                Tensor t = LoadTensorToGPU(tensor_data_start, meta, ctx);
                size_t bytes = meta.byte_offset_end - meta.byte_offset_start;
                total_bytes += bytes;
                weights.put(name, std::move(t));
                success++;
            } catch (const std::exception& e) {
                AILA_LOG_ERROR("  [FAIL] %s: %s", name.c_str(), e.what());
                fail++;
            }
        }

        AILA_LOG_INFO("[SafeTensors] Loaded %d tensors (%.2f MB) to GPU%s",
                      success, total_bytes / (1024.0 * 1024.0),
                      (fail > 0 ? (" , " + std::to_string(fail) + " failed").c_str() : ""));

    } catch (const std::exception& e) {
        AILA_LOG_ERROR("[SafeTensors] Load failed: %s", e.what());
        throw;
    }

    return weights;
}

namespace {

std::string read_text_file(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) {
        return "";
    }
    return std::string((std::istreambuf_iterator<char>(in)),
                       std::istreambuf_iterator<char>());
}

std::vector<std::string> parse_sharded_safetensors_index(const std::string& index_path) {
    std::string text = read_text_file(index_path);
    if (text.empty()) {
        throw std::runtime_error("Empty or unreadable safetensors index: " + index_path);
    }

    simdjson::dom::parser parser;
    simdjson::dom::element root = parser.parse(text);

    simdjson::dom::element weight_map_elem;
    if (root.at_key("weight_map").get(weight_map_elem) != simdjson::SUCCESS) {
        throw std::runtime_error("Invalid safetensors index: missing weight_map");
    }

    simdjson::dom::object weight_map;
    if (weight_map_elem.get_object().get(weight_map) != simdjson::SUCCESS) {
        throw std::runtime_error("Invalid safetensors index: weight_map is not an object");
    }

    std::unordered_set<std::string> unique_shards;
    for (auto field : weight_map) {
        std::string_view shard_sv;
        if (field.value.get_string().get(shard_sv) == simdjson::SUCCESS) {
            unique_shards.emplace(shard_sv);
        }
    }

    std::vector<std::string> shards(unique_shards.begin(), unique_shards.end());
    std::sort(shards.begin(), shards.end());
    return shards;
}

} // namespace

ModelWeights LoadModelWeightsFromDir(const std::string& model_dir, Context& ctx) {
    namespace fs = std::filesystem;

    fs::path dir(model_dir);
    fs::path single = dir / "model.safetensors";
    if (fs::exists(single)) {
        AILA_LOG_INFO("[SafeTensors] Loading single-file weights: %s", single.string().c_str());
        return LoadSafetensors(single.string(), ctx);
    }

    fs::path index = dir / "model.safetensors.index.json";
    if (!fs::exists(index)) {
        throw std::runtime_error("No model.safetensors or model.safetensors.index.json found in: " + model_dir);
    }

    auto shards = parse_sharded_safetensors_index(index.string());
    if (shards.empty()) {
        throw std::runtime_error("No shard entries found in: " + index.string());
    }

    AILA_LOG_INFO("[SafeTensors] Loading sharded weights from %zu shard(s)", shards.size());

    ModelWeights merged;
    for (const auto& shard_name : shards) {
        fs::path shard_path = dir / shard_name;
        if (!fs::exists(shard_path)) {
            throw std::runtime_error("Missing safetensors shard: " + shard_path.string());
        }

        AILA_LOG_INFO("[SafeTensors] Loading shard: %s", shard_path.string().c_str());
        ModelWeights shard_weights = LoadSafetensors(shard_path.string(), ctx);
        auto tensor_names = shard_weights.names();

        for (const auto& name : tensor_names) {
            if (merged.has(name)) {
                merged.replace(name, std::move(shard_weights.get(name)));
            } else {
                merged.put(name, std::move(shard_weights.get(name)));
            }
        }
    }

    AILA_LOG_INFO("[SafeTensors] Sharded load complete: %zu tensors", merged.size());
    return merged;
}
