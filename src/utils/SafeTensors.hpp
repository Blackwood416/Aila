#pragma once
#include <string>
#include <unordered_map>
#include <vector>
#include <memory>
#include "MemoryMappedFile.hpp"
#include "simdjson.h"
#include "../core/Context.hpp"
#include "../core/Tensor.hpp"
#include "oneapi/dnnl/dnnl.hpp"
#include "oneapi/dnnl/dnnl_sycl.hpp"
#include <sycl/sycl.hpp>

// ============================================================
// Tensor 元数据 (从 safetensors JSON header 解析)
// ============================================================
struct TensorShape {
    static constexpr int MAX_DIMS = 8;
    int64_t dims[MAX_DIMS];
    int ndims = 0;

    void push_back(int64_t dim) {
        if (ndims < MAX_DIMS) {
            dims[ndims++] = dim;
        }
    }
};

enum DataType {
    DT_F16, DT_BF16, DT_F32, DT_F64, DT_U8, DT_S8, DT_I32, DT_I64, DT_UNKNOWN
};

struct TensorMeta {
    TensorShape shape;
    DataType dtype;
    size_t byte_offset_start;
    size_t byte_offset_end;
};

struct ModelWeightsEraseStats {
    size_t count = 0;
    size_t bytes = 0;
};

struct Bnb4BitQuantState {
    std::string quant_type;
    std::string dtype;
    std::vector<int64_t> shape;
    int blocksize = 0;
    bool nested = false;
    int nested_blocksize = 0;
    std::string nested_dtype;
    float nested_offset = 0.0f;
};

struct FileSpan {
    int32_t shard_id = -1;
    uint64_t absolute_offset = 0;
    size_t bytes = 0;
    bool valid() const { return shard_id >= 0 && bytes > 0; }
};

struct HostTensorView {
    const uint8_t* ptr = nullptr;
    std::vector<int64_t> shape;
    DataType dtype = DT_UNKNOWN;
    size_t bytes = 0;
    FileSpan file_span{};
    const uint8_t* data() const { return ptr; }
    bool valid() const { return ptr != nullptr; }
    int64_t numel() const {
        if (shape.empty()) return 0;
        int64_t n = 1;
        for (auto d : shape) n *= d;
        return n;
    }
};

struct HostTensorEntry {
    size_t shard_index = 0;
    TensorMeta meta{};
    const uint8_t* tensor_data_start = nullptr;
    const uint8_t* data_ptr = nullptr;
    FileSpan file_span{};
};

struct Bnb4BitHostWeightRef {
    std::string name;
    HostTensorView packed_weight{};
    HostTensorView absmax{};
    HostTensorView quant_map{};
    HostTensorView nested_absmax{};
    HostTensorView nested_quant_map{};
    HostTensorView packed_quant_state{};
    Bnb4BitQuantState quant_state{};

    bool valid() const {
        return packed_weight.valid() && absmax.valid() && quant_map.valid() &&
               (quant_state.shape.size() == 2 || quant_state.shape.size() == 3);
    }
    int64_t logical_out_features() const {
        if (quant_state.shape.size() == 3) return quant_state.shape[0] * quant_state.shape[1];
        return quant_state.shape.size() > 0 ? quant_state.shape[0] : 0;
    }
    int64_t logical_in_features() const {
        return quant_state.shape.size() > 0 ? quant_state.shape.back() : 0;
    }
    int64_t logical_numel() const {
        int64_t numel = 1;
        for (int64_t dim : quant_state.shape) numel *= dim;
        return numel;
    }
    int64_t packed_num_bytes() const {
        return packed_weight.valid() ? static_cast<int64_t>(packed_weight.bytes) : 0;
    }
    const uint8_t* packed_ptr() const { return packed_weight.ptr; }
    const uint8_t* absmax_codes_ptr() const { return absmax.ptr; }
    const float* absmax_f32_ptr() const { return reinterpret_cast<const float*>(absmax.ptr); }
    const float* quant_map_ptr() const { return reinterpret_cast<const float*>(quant_map.ptr); }
    const float* nested_absmax_ptr() const { return reinterpret_cast<const float*>(nested_absmax.ptr); }
    const float* nested_quant_map_ptr() const { return reinterpret_cast<const float*>(nested_quant_map.ptr); }
};

struct Bnb4BitWeightRef {
    std::string name;
    Tensor* packed_weight = nullptr;
    Tensor* absmax = nullptr;
    Tensor* quant_map = nullptr;
    Tensor* nested_absmax = nullptr;
    Tensor* nested_quant_map = nullptr;
    Tensor* packed_quant_state = nullptr;
    Bnb4BitQuantState quant_state{};

    bool valid() const {
        return packed_weight != nullptr && absmax != nullptr && quant_map != nullptr &&
               (quant_state.shape.size() == 2 || quant_state.shape.size() == 3);
    }

    int64_t logical_out_features() const {
        if (quant_state.shape.size() == 3) {
            return quant_state.shape[0] * quant_state.shape[1];
        }
        return quant_state.shape.size() > 0 ? quant_state.shape[0] : 0;
    }

    int64_t logical_in_features() const {
        return quant_state.shape.size() > 0 ? quant_state.shape.back() : 0;
    }

    int64_t logical_numel() const {
        int64_t numel = 1;
        for (int64_t dim : quant_state.shape) {
            numel *= dim;
        }
        return numel;
    }

    int64_t packed_num_bytes() const {
        return packed_weight ? packed_weight->numel() : 0;
    }
};

// ============================================================
// ModelWeights: 封装模型所有权重 (name -> Tensor GPU 映射)
// ============================================================
class ModelWeights {
public:
    ModelWeights() = default;
    ModelWeights(ModelWeights&&) = default;
    ModelWeights& operator=(ModelWeights&&) = default;
    ModelWeights(const ModelWeights&) = delete;
    ModelWeights& operator=(const ModelWeights&) = delete;

    // 按名称获取 tensor 引用，找不到时抛异常
    Tensor& get(const std::string& name);
    const Tensor& get(const std::string& name) const;

    // 检查是否存在
    bool has(const std::string& name) const;

    // 存入 tensor
    void put(const std::string& name, Tensor tensor);

    // 替换已有 tensor (用于预处理)
    void replace(const std::string& name, Tensor tensor);

    // 删除已有 tensor，释放其设备内存
    void erase(const std::string& name);
    ModelWeightsEraseStats erase_with_prefix(const std::string& prefix);

    // 获取所有名称
    std::vector<std::string> names() const;

    size_t size() const { return tensors_.size(); }

    bool has_host(const std::string& name) const;
    HostTensorView host_view(const std::string& name) const;
    std::vector<std::string> host_names() const;
    size_t host_size() const;
    // Unmaps shard files that hold no host tensor matching any keep_prefix.
    // Returns the number of shard mappings released.
    size_t release_unused_shards(const std::vector<std::string>& keep_prefixes);

    Tensor& upload(const std::string& name);
    Tensor& upload(const std::string& name, Context& ctx);

    void set_context(Context* ctx) { ctx_ = ctx; }
    Context* context() const { return ctx_; }

    const std::vector<std::unique_ptr<MemoryMappedFile>>& mmaps() const { return mmaps_; }
    const std::unordered_map<std::string, HostTensorEntry>& host_index() const { return host_index_; }
    const std::vector<std::string>& shard_paths() const { return shard_paths_; }

private:
    std::unordered_map<std::string, Tensor> tensors_;

    // 保持 mmap 文件存活 (权重数据来自 mmap, GPU 拷贝后可释放)
    friend ModelWeights LoadSafetensors(const std::string& path, Context& ctx);
    std::vector<std::unique_ptr<MemoryMappedFile>> mmaps_;
    std::vector<std::string> shard_paths_;
    std::unordered_map<std::string, HostTensorEntry> host_index_;
    Context* ctx_ = nullptr;

    friend ModelWeights LoadModelWeightsLazy(const std::string& model_dir, Context* ctx);
    std::unique_ptr<MemoryMappedFile> mmap_file_;
};

// ============================================================
// 公共 API
// ============================================================

// 加载 safetensors 文件，所有权重上传到 GPU，返回 ModelWeights
ModelWeights LoadSafetensors(const std::string& path, Context& ctx);

// 从模型目录自动加载权重:
// 1) 优先 model.safetensors
// 2) 否则使用 model.safetensors.index.json + 分片 safetensors
ModelWeights LoadModelWeightsFromDir(const std::string& model_dir, Context& ctx);

ModelWeights LoadModelWeightsLazy(const std::string& model_dir, Context* ctx = nullptr);
ModelWeights LoadModelWeightsLazy(const std::string& model_dir, Context& ctx);

bool parse_bnb_4bit_quant_state_json(const std::string& json_text,
                                     Bnb4BitQuantState& out,
                                     std::string* error_message = nullptr);

bool LoadBnb4BitHostRef(const ModelWeights& weights,
                        const std::string& name,
                        Bnb4BitHostWeightRef& out,
                        std::string* error_message = nullptr,
                        int64_t in_features_hint = 0);

// 从 bitsandbytes 4-bit checkpoint 条目组装运行时视图
bool LoadBnb4BitWeightRef(Context& ctx,
                          ModelWeights& weights,
                          const std::string& name,
                          Bnb4BitWeightRef& out,
                          std::string* error_message = nullptr,
                          int64_t in_features_hint = 0);

// 内部工具函数
std::vector<std::string> ParseHeader(const std::string& header,
                                     std::unordered_map<std::string, TensorMeta>& metadata);

dnnl::memory::data_type to_dnnl_dtype(DataType dt);
