#include "models/cpu/CpuSafetensorsStore.hpp"

#include "simdjson.h"

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <unordered_set>

namespace {

struct TensorMeta {
    CpuDataType dtype = CpuDataType::Unknown;
    std::vector<int64_t> shape;
    size_t byte_offset_start = 0;
    size_t byte_offset_end = 0;
};

void set_error(std::string* error, const std::string& message) {
    if (error) {
        *error = message;
    }
}

std::string read_text_file(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) {
        return "";
    }
    return std::string((std::istreambuf_iterator<char>(in)),
                       std::istreambuf_iterator<char>());
}

CpuDataType parse_dtype(std::string_view dtype) {
    if (dtype == "F16" || dtype == "float16") {
        return CpuDataType::F16;
    }
    if (dtype == "BF16" || dtype == "bfloat16") {
        return CpuDataType::BF16;
    }
    if (dtype == "F32" || dtype == "float32") {
        return CpuDataType::F32;
    }
    if (dtype == "F64" || dtype == "float64") {
        return CpuDataType::F64;
    }
    if (dtype == "U8" || dtype == "uint8") {
        return CpuDataType::U8;
    }
    if (dtype == "S8" || dtype == "int8") {
        return CpuDataType::S8;
    }
    return CpuDataType::Unknown;
}

bool parse_header(const std::string& header,
                  std::vector<std::pair<std::string, TensorMeta>>& tensors,
                  std::string* error) {
    tensors.clear();
    try {
        simdjson::padded_string padded(header);
        simdjson::ondemand::parser parser;
        simdjson::ondemand::document doc = parser.iterate(padded);

        for (auto field : doc.get_object()) {
            std::string_view key = field.unescaped_key();
            if (key == "__metadata__") {
                continue;
            }

            simdjson::ondemand::object tensor_info = field.value().get_object();
            std::string_view dtype_text = tensor_info["dtype"];
            TensorMeta meta;
            meta.dtype = parse_dtype(dtype_text);

            for (int64_t dim : tensor_info["shape"].get_array()) {
                meta.shape.push_back(dim);
            }

            auto offsets = tensor_info["data_offsets"].get_array();
            auto it = offsets.begin();
            if (it == offsets.end()) {
                set_error(error, "safetensors tensor is missing data_offsets start: " +
                                     std::string(key));
                return false;
            }
            meta.byte_offset_start = static_cast<size_t>(int64_t(*it));
            ++it;
            if (it == offsets.end()) {
                set_error(error, "safetensors tensor is missing data_offsets end: " +
                                     std::string(key));
                return false;
            }
            meta.byte_offset_end = static_cast<size_t>(int64_t(*it));

            if (meta.byte_offset_end < meta.byte_offset_start) {
                set_error(error, "safetensors tensor has invalid data_offsets: " +
                                     std::string(key));
                return false;
            }

            tensors.emplace_back(std::string(key), std::move(meta));
        }
        return true;
    } catch (const std::exception& e) {
        set_error(error, std::string("parse safetensors header failed: ") + e.what());
        return false;
    }
}

std::vector<std::string> parse_sharded_safetensors_index(
    const std::filesystem::path& index_path) {
    const std::string text = read_text_file(index_path);
    if (text.empty()) {
        throw std::runtime_error("Empty or unreadable safetensors index: " +
                                 index_path.string());
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
        std::string_view shard;
        if (field.value.get_string().get(shard) == simdjson::SUCCESS) {
            unique_shards.emplace(shard);
        }
    }

    std::vector<std::string> shards(unique_shards.begin(), unique_shards.end());
    std::sort(shards.begin(), shards.end());
    return shards;
}

uint64_t read_u64_le(const uint8_t* data) {
    uint64_t value = 0;
    for (int i = 0; i < 8; ++i) {
        value |= static_cast<uint64_t>(data[i]) << (i * 8);
    }
    return value;
}

}  // namespace

bool CpuSafetensorsStore::load_from_dir(const std::string& model_dir,
                                        std::string* error) {
    clear();
    if (error) {
        error->clear();
    }

    namespace fs = std::filesystem;
    try {
        const fs::path dir(model_dir);
        const fs::path single = dir / "model.safetensors";
        if (fs::exists(single)) {
            return load_file(single.string(), error);
        }

        const fs::path index = dir / "model.safetensors.index.json";
        if (!fs::exists(index)) {
            set_error(error,
                      "No model.safetensors or model.safetensors.index.json found in: " +
                          model_dir);
            return false;
        }

        const std::vector<std::string> shards =
            parse_sharded_safetensors_index(index);
        if (shards.empty()) {
            set_error(error, "No shard entries found in: " + index.string());
            return false;
        }

        for (const std::string& shard : shards) {
            const fs::path shard_path = dir / shard;
            if (!fs::exists(shard_path)) {
                set_error(error, "Missing safetensors shard: " + shard_path.string());
                clear();
                return false;
            }
            if (!load_file(shard_path.string(), error)) {
                clear();
                return false;
            }
        }
        return true;
    } catch (const std::exception& e) {
        clear();
        set_error(error, std::string("load CPU safetensors store failed: ") + e.what());
        return false;
    }
}

bool CpuSafetensorsStore::has(const std::string& name) const {
    return tensors_.find(name) != tensors_.end();
}

const CpuTensorView& CpuSafetensorsStore::get(const std::string& name) const {
    auto it = tensors_.find(name);
    if (it == tensors_.end()) {
        throw std::runtime_error("CPU safetensors tensor not found: " + name);
    }
    return it->second;
}

std::vector<std::string> CpuSafetensorsStore::names() const {
    std::vector<std::string> result;
    result.reserve(tensors_.size());
    for (const auto& pair : tensors_) {
        result.push_back(pair.first);
    }
    std::sort(result.begin(), result.end());
    return result;
}

bool CpuSafetensorsStore::load_file(const std::string& path, std::string* error) {
    auto mapped = std::make_unique<MemoryMappedFile>(path);
    if (mapped->size() < 8) {
        set_error(error, "safetensors file is too small: " + path);
        return false;
    }

    const uint8_t* raw = mapped->data();
    const uint64_t header_size = read_u64_le(raw);
    if (header_size > mapped->size() - 8) {
        set_error(error, "safetensors header exceeds file size: " + path);
        return false;
    }

    const auto header_bytes = static_cast<size_t>(header_size);
    const std::string header(reinterpret_cast<const char*>(raw + 8), header_bytes);
    const uint8_t* tensor_data_start = raw + 8 + header_bytes;
    const size_t tensor_data_bytes = mapped->size() - 8 - header_bytes;

    std::vector<std::pair<std::string, TensorMeta>> parsed;
    if (!parse_header(header, parsed, error)) {
        return false;
    }

    for (const auto& item : parsed) {
        const std::string& name = item.first;
        const TensorMeta& meta = item.second;
        if (meta.byte_offset_end > tensor_data_bytes) {
            set_error(error, "safetensors tensor data exceeds file size: " + name);
            return false;
        }

        CpuTensorView view;
        view.name = name;
        view.dtype = meta.dtype;
        view.shape = meta.shape;
        view.data = tensor_data_start + meta.byte_offset_start;
        view.bytes = meta.byte_offset_end - meta.byte_offset_start;
        tensors_[name] = std::move(view);
    }

    mapped_files_.push_back(std::move(mapped));
    return true;
}

void CpuSafetensorsStore::clear() {
    tensors_.clear();
    mapped_files_.clear();
}
