#pragma once

#include "../core/Context.hpp"
#include "../core/Tensor.hpp"
#include "../utils/SafeTensors.hpp"
#include "oneapi/dnnl/dnnl.hpp"
#include "oneapi/dnnl/dnnl_sycl.hpp"
#include <string>
#include <unordered_map>
#include <vector>

struct Bnb4BitLinearScratch {
    Tensor input_bf16;
    Tensor output_bf16;
    Tensor weight_bf16;

    int seq_capacity = 0;
    int input_dim_capacity = 0;
    int output_dim_capacity = 0;
    int weight_in_capacity = 0;
    int weight_out_capacity = 0;

    void ensure(Context& ctx, int seq_len, int in_features, int out_features, bool need_weight);
};

class Bnb4BitLinear {
public:
    Bnb4BitLinear() = default;

    bool init(Context& ctx, const Bnb4BitWeightRef& weight, std::string* error_message = nullptr);
    bool init_fused_rows(Context& ctx,
                         const Bnb4BitLinear& first,
                         const Bnb4BitLinear& second,
                         std::string* error_message = nullptr);
    bool init_fused_rows(Context& ctx,
                         const Bnb4BitLinear& first,
                         const Bnb4BitLinear& second,
                         const Bnb4BitLinear& third,
                         std::string* error_message = nullptr);
    bool init_fused_rows(Context& ctx,
                         const Bnb4BitLinear& first,
                         const Bnb4BitLinear& second,
                         const Bnb4BitLinear& third,
                         const Bnb4BitLinear& fourth,
                         std::string* error_message = nullptr);
    void forward(Context& ctx,
                 Bnb4BitLinearScratch& scratch,
                 Tensor& input,
                 Tensor& output,
                 int seq_len);
    static bool try_forward_decode_qkv(Context& ctx,
                                       Bnb4BitLinear& q_proj,
                                       Bnb4BitLinear& k_proj,
                                       Bnb4BitLinear& v_proj,
                                       Tensor& input,
                                       Tensor& q_output,
                                       Tensor& k_output,
                                       Tensor& v_output);
    static bool try_forward_decode_gate_up(Context& ctx,
                                           Bnb4BitLinear& gate_proj,
                                           Bnb4BitLinear& up_proj,
                                           Tensor& input,
                                           Tensor& gate_output,
                                           Tensor& up_output);

    int in_features() const { return in_features_; }
    int out_features() const { return out_features_; }

private:
    bool init_fused_rows_impl(Context& ctx,
                              const std::vector<const Bnb4BitLinear*>& sources,
                              std::string* error_message);
    void finish_init(Context& ctx);
    void ensure_primitive(Context& ctx, int seq_len);
    void dequantize_weight(Context& ctx, Tensor& weight_bf16_view);

    Bnb4BitWeightRef weight_{};
    Tensor owned_packed_weight_;
    Tensor absmax_f32_;
    Tensor cached_weight_bf16_;
    int in_features_ = 0;
    int out_features_ = 0;
    bool force_dequant_cache_ = false;
    bool release_quant_after_cache_ = false;
    bool cache_dequantized_weight_ = false;
    bool cached_weight_ready_ = false;

    dnnl::matmul decode_prim_;
    dnnl::memory::desc decode_src_md_;
    dnnl::memory::desc decode_weight_md_;
    dnnl::memory::desc decode_dst_md_;
    bool decode_inited_ = false;

    dnnl::memory decode_src_mem_;
    dnnl::memory decode_weight_mem_;
    dnnl::memory decode_dst_mem_;
    dnnl::memory decode_scratchpad_mem_;
    Tensor decode_scratchpad_;
    bool decode_mem_inited_ = false;
    void* decode_src_ptr_ = nullptr;
    void* decode_weight_ptr_ = nullptr;
    void* decode_dst_ptr_ = nullptr;
    std::unordered_map<int, dnnl::memory> decode_args_;

    struct CachedPrimitive {
        dnnl::matmul prim;
        dnnl::memory::desc src_md;
        dnnl::memory::desc weight_md;
        dnnl::memory::desc dst_md;
        dnnl::memory src_mem;
        dnnl::memory weight_mem;
        dnnl::memory dst_mem;
        dnnl::memory scratchpad_mem;
        Tensor scratchpad;
        bool mem_inited = false;
        void* src_ptr = nullptr;
        void* weight_ptr = nullptr;
        void* dst_ptr = nullptr;
        std::unordered_map<int, dnnl::memory> args;
    };
    std::unordered_map<int, CachedPrimitive> prim_cache_;
};
