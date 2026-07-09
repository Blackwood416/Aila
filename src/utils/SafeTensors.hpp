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
    DT_F16, DT_BF16, DT_F32, DT_F64, DT_U8, DT_S8, DT_UNKNOWN
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
               packed_quant_state != nullptr && quant_state.shape.size() == 2;
    }

    int64_t logical_out_features() const {
        return quant_state.shape.size() > 0 ? quant_state.shape[0] : 0;
    }

    int64_t logical_in_features() const {
        return quant_state.shape.size() > 1 ? quant_state.shape[1] : 0;
    }

    int64_t logical_numel() const {
        return logical_out_features() * logical_in_features();
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

private:
    std::unordered_map<std::string, Tensor> tensors_;

    // 保持 mmap 文件存活 (权重数据来自 mmap, GPU 拷贝后可释放)
    friend ModelWeights LoadSafetensors(const std::string& path, Context& ctx);
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

// 从 bitsandbytes 4-bit checkpoint 条目组装运行时视图
bool LoadBnb4BitWeightRef(Context& ctx,
                          ModelWeights& weights,
                          const std::string& name,
                          Bnb4BitWeightRef& out,
                          std::string* error_message = nullptr);

// 内部工具函数
std::vector<std::string> ParseHeader(const std::string& header,
                                     std::unordered_map<std::string, TensorMeta>& metadata);

dnnl::memory::data_type to_dnnl_dtype(DataType dt);
