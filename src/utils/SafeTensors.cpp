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

void ModelWeights::erase(const std::string& name) {
    tensors_.erase(name);
}

ModelWeightsEraseStats ModelWeights::erase_with_prefix(const std::string& prefix) {
    ModelWeightsEraseStats stats;
    if (prefix.empty()) {
        return stats;
    }

    std::vector<std::string> erased_names;
    for (const auto& pair : tensors_) {
        if (pair.first.rfind(prefix, 0) == 0) {
            stats.bytes += pair.second.size_bytes();
            erased_names.push_back(pair.first);
        }
    }

    stats.count = erased_names.size();
    for (const std::string& name : erased_names) {
        tensors_.erase(name);
    }
    return stats;
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

void set_error(std::string* error_message, const std::string& message) {
    if (error_message) {
        *error_message = message;
    }
}

bool read_json_int64(simdjson::dom::element root, const char* key, int64_t& out) {
    simdjson::dom::element elem;
    if (root.at_key(key).get(elem) != simdjson::SUCCESS) {
        return false;
    }
    return elem.get_int64().get(out) == simdjson::SUCCESS;
}

bool read_json_float(simdjson::dom::element root, const char* key, float& out) {
    simdjson::dom::element elem;
    if (root.at_key(key).get(elem) != simdjson::SUCCESS) {
        return false;
    }

    double value = 0.0;
    if (elem.get_double().get(value) == simdjson::SUCCESS) {
        out = static_cast<float>(value);
        return true;
    }

    int64_t int_value = 0;
    if (elem.get_int64().get(int_value) == simdjson::SUCCESS) {
        out = static_cast<float>(int_value);
        return true;
    }
    return false;
}

std::string read_json_string(simdjson::dom::element root, const char* key) {
    simdjson::dom::element elem;
    if (root.at_key(key).get(elem) != simdjson::SUCCESS) {
        return "";
    }
    std::string_view sv;
    if (elem.get_string().get(sv) != simdjson::SUCCESS) {
        return "";
    }
    return std::string(sv);
}

bool parse_bnb_4bit_quant_state_json(const std::string& json_text,
                                     Bnb4BitQuantState& out,
                                     std::string* error_message) {
    out = {};
    try {
        simdjson::dom::parser parser;
        simdjson::dom::element root = parser.parse(json_text);

        out.quant_type = read_json_string(root, "quant_type");
        out.dtype = read_json_string(root, "dtype");

        int64_t blocksize = 0;
        if (!read_json_int64(root, "blocksize", blocksize)) {
            set_error(error_message, "bitsandbytes quant_state is missing blocksize");
            return false;
        }
        out.blocksize = static_cast<int>(blocksize);

        simdjson::dom::element shape_elem;
        if (root.at_key("shape").get(shape_elem) != simdjson::SUCCESS) {
            set_error(error_message, "bitsandbytes quant_state is missing shape");
            return false;
        }
        out.shape.clear();
        for (auto value : shape_elem.get_array()) {
            int64_t dim = 0;
            if (value.get_int64().get(dim) != simdjson::SUCCESS) {
                set_error(error_message, "bitsandbytes quant_state shape contains a non-integer value");
                return false;
            }
            out.shape.push_back(dim);
        }

        int64_t nested_blocksize = 0;
        if (read_json_int64(root, "nested_blocksize", nested_blocksize)) {
            out.nested = true;
            out.nested_blocksize = static_cast<int>(nested_blocksize);
        }
        std::string nested_dtype = read_json_string(root, "nested_dtype");
        if (!nested_dtype.empty()) {
            out.nested = true;
            out.nested_dtype = nested_dtype;
        }
        float nested_offset = 0.0f;
        if (read_json_float(root, "nested_offset", nested_offset)) {
            out.nested = true;
            out.nested_offset = nested_offset;
        }
        return true;
    } catch (const std::exception& e) {
        set_error(error_message, std::string("parse bitsandbytes quant_state failed: ") + e.what());
        return false;
    }
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

bool LoadBnb4BitWeightRef(Context& ctx,
                          ModelWeights& weights,
                          const std::string& name,
                          Bnb4BitWeightRef& out,
                          std::string* error_message) {
    out = {};
    out.name = name;

    if (!weights.has(name)) {
        set_error(error_message, "Missing bitsandbytes packed weight tensor: " + name);
        return false;
    }

    const std::string absmax_name = name + ".absmax";
    const std::string quant_map_name = name + ".quant_map";
    const std::string nested_absmax_name = name + ".nested_absmax";
    const std::string nested_quant_map_name = name + ".nested_quant_map";
    const std::string packed_quant_state_name = name + ".quant_state.bitsandbytes__nf4";

    if (!weights.has(absmax_name) || !weights.has(quant_map_name) || !weights.has(packed_quant_state_name)) {
        set_error(error_message,
                  "Missing required bitsandbytes side tensors for weight: " + name);
        return false;
    }

    out.packed_weight = &weights.get(name);
    out.absmax = &weights.get(absmax_name);
    out.quant_map = &weights.get(quant_map_name);
    out.packed_quant_state = &weights.get(packed_quant_state_name);
    if (weights.has(nested_absmax_name)) {
        out.nested_absmax = &weights.get(nested_absmax_name);
    }
    if (weights.has(nested_quant_map_name)) {
        out.nested_quant_map = &weights.get(nested_quant_map_name);
    }

    if (out.packed_weight->dtype() != dnnl::memory::data_type::u8) {
        set_error(error_message, "bitsandbytes packed weight must be uint8: " + name);
        return false;
    }
    if (out.quant_map->dtype() != dnnl::memory::data_type::f32) {
        set_error(error_message, "bitsandbytes quant_map must be float32: " + quant_map_name);
        return false;
    }
    if (out.packed_quant_state->dtype() != dnnl::memory::data_type::u8) {
        set_error(error_message, "bitsandbytes packed quant_state must be uint8: " + packed_quant_state_name);
        return false;
    }
    if (out.absmax->dtype() != dnnl::memory::data_type::u8 &&
        out.absmax->dtype() != dnnl::memory::data_type::f32) {
        set_error(error_message, "bitsandbytes absmax must be uint8 or float32: " + absmax_name);
        return false;
    }

    std::string packed_state_json(static_cast<size_t>(out.packed_quant_state->numel()), '\0');
    ctx.memcpy_d2h(packed_state_json.data(), out.packed_quant_state->data(), packed_state_json.size());
    if (!parse_bnb_4bit_quant_state_json(packed_state_json, out.quant_state, error_message)) {
        return false;
    }

    if (!out.valid()) {
        set_error(error_message, "Invalid bitsandbytes weight view: " + name);
        return false;
    }
    if (out.quant_state.quant_type != "nf4") {
        set_error(error_message, "Only NF4 bitsandbytes weights are supported: " + name);
        return false;
    }
    if (out.quant_state.blocksize <= 0) {
        set_error(error_message, "bitsandbytes blocksize must be positive: " + name);
        return false;
    }

    const int64_t logical_numel = out.logical_numel();
    if (logical_numel <= 0) {
        set_error(error_message, "bitsandbytes logical weight shape is invalid: " + name);
        return false;
    }
    if (out.packed_num_bytes() * 2 != logical_numel) {
        set_error(error_message, "bitsandbytes packed weight size does not match logical shape: " + name);
        return false;
    }

    const int64_t block_count = (logical_numel + out.quant_state.blocksize - 1) / out.quant_state.blocksize;
    if (out.absmax->numel() != block_count) {
        set_error(error_message, "bitsandbytes absmax length does not match block count: " + name);
        return false;
    }
    if (out.quant_map->numel() < 16) {
        set_error(error_message, "bitsandbytes quant_map must contain at least 16 entries: " + name);
        return false;
    }

    if (out.quant_state.nested) {
        if (out.nested_absmax == nullptr || out.nested_quant_map == nullptr) {
            set_error(error_message, "bitsandbytes nested quantization tensors are missing: " + name);
            return false;
        }
        if (out.nested_absmax->dtype() != dnnl::memory::data_type::f32 ||
            out.nested_quant_map->dtype() != dnnl::memory::data_type::f32) {
            set_error(error_message, "bitsandbytes nested tensors must be float32: " + name);
            return false;
        }
        if (out.quant_state.nested_blocksize <= 0) {
            set_error(error_message, "bitsandbytes nested blocksize must be positive: " + name);
            return false;
        }
        const int64_t nested_block_count =
            (block_count + out.quant_state.nested_blocksize - 1) / out.quant_state.nested_blocksize;
        if (out.nested_absmax->numel() != nested_block_count) {
            set_error(error_message, "bitsandbytes nested_absmax length does not match nested block count: " + name);
            return false;
        }
    } else if (out.absmax->dtype() != dnnl::memory::data_type::f32) {
        set_error(error_message, "Non-nested bitsandbytes absmax must be float32: " + name);
        return false;
    }

    return true;
}

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
