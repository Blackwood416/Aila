#include "LoraLoader.hpp"
#include "LoraConfig.hpp"
#include "profile/Profiling.hpp"
#include "utils/MemoryMappedFile.hpp"
#include "simdjson.h"
#include <string>
#include <vector>
#include <unordered_map>
#include <cstring>
#include <cstdint>
#include <memory>

using bf16 = sycl::ext::oneapi::bfloat16;

namespace aila {
namespace lora {

// ============================================================
// PEFT key → Aila base weight name
// "base_model.model.model.layers.N.self_attn.q_proj.lora_A.weight"
//   → "model.layers.N.self_attn.q_proj.weight"
// ============================================================

std::string LoraLoader::peft_key_to_base_name(const std::string& lora_key) {
    const std::string peft_prefix = "base_model.model.";
    const std::string lora_a_suffix = ".lora_A.weight";
    const std::string lora_b_suffix = ".lora_B.weight";

    if (lora_key.size() <= peft_prefix.size()) return "";
    if (lora_key.compare(0, peft_prefix.size(), peft_prefix) != 0) return "";

    std::string result = lora_key.substr(peft_prefix.size());

    if (result.size() > lora_a_suffix.size() &&
        result.compare(result.size() - lora_a_suffix.size(),
                       lora_a_suffix.size(), lora_a_suffix) == 0) {
        result = result.substr(0, result.size() - lora_a_suffix.size()) + ".weight";
    } else if (result.size() > lora_b_suffix.size() &&
               result.compare(result.size() - lora_b_suffix.size(),
                              lora_b_suffix.size(), lora_b_suffix) == 0) {
        result = result.substr(0, result.size() - lora_b_suffix.size()) + ".weight";
    } else {
        return "";
    }
    return result;
}

// ============================================================
// Load LoRA adapter from safetensors directory
// ============================================================

bool LoraLoader::load(const std::string& lora_dir, LoraAdapter& adapter,
                      std::string* error_message) {
    // 1. Parse adapter_config.json
    adapter.config = LoraConfig::from_json_file(lora_dir, error_message);
    if (adapter.config.r <= 0) {
        if (error_message) *error_message = "Invalid LoRA rank from adapter_config.json";
        return false;
    }

    AILA_LOG_INFO("[LoRA] adapter_config: r=%d alpha=%d scaling=%.1f target_modules=%zu",
                  adapter.config.r, adapter.config.lora_alpha,
                  adapter.config.scaling, adapter.config.target_modules.size());

    // 2. Memory-map adapter_model.safetensors
    std::string safetensors_path = lora_dir;
    if (safetensors_path.back() != '/' && safetensors_path.back() != '\\') {
        safetensors_path += "/";
    }
    safetensors_path += "adapter_model.safetensors";

    std::unique_ptr<MemoryMappedFile> mmap_file;
    try {
        mmap_file = std::make_unique<MemoryMappedFile>(safetensors_path);
    } catch (const std::exception& e) {
        if (error_message) *error_message = std::string("Cannot mmap adapter_model.safetensors: ") + e.what();
        return false;
    }

    const uint8_t* raw_ptr = mmap_file->data();
    if (!raw_ptr) {
        if (error_message) *error_message = "adapter_model.safetensors is empty";
        return false;
    }

    // 3. Parse safetensors JSON header
    uint64_t header_size = *reinterpret_cast<const uint64_t*>(raw_ptr);
    std::string json_str(reinterpret_cast<const char*>(raw_ptr + 8), header_size);
    const uint8_t* data_start = raw_ptr + 8 + header_size;

    // Parse header JSON via simdjson
    struct TensorEntry {
        std::string name;
        std::vector<int64_t> shape;
        size_t offset_start;
        size_t offset_end;
    };
    std::vector<TensorEntry> entries;

    try {
        simdjson::padded_string padded_json(json_str);
        simdjson::ondemand::parser parser;
        simdjson::ondemand::document doc = parser.iterate(padded_json);

        for (auto field : doc.get_object()) {
            std::string_view key = field.unescaped_key();
            if (key == "__metadata__") continue;

            simdjson::ondemand::object info = field.value().get_object();

            TensorEntry entry;
            entry.name = std::string(key);

            for (int64_t dim : info["shape"].get_array()) {
                entry.shape.push_back(dim);
            }

            auto offsets = info["data_offsets"].get_array();
            auto it = offsets.begin();
            entry.offset_start = static_cast<size_t>(int64_t(*it));
            ++it;
            entry.offset_end = static_cast<size_t>(int64_t(*it));

            entries.push_back(std::move(entry));
        }
    } catch (const std::exception& e) {
        if (error_message)
            *error_message = std::string("Safetensors header parse failed: ") + e.what();
        return false;
    }

    AILA_LOG_INFO("[LoRA] adapter_model.safetensors: %zu tensors", entries.size());

    // 4. Group lora_A and lora_B tensors by base weight name
    std::unordered_map<std::string, const TensorEntry*> a_entries; // base_name → entry
    std::unordered_map<std::string, const TensorEntry*> b_entries;

    for (const auto& entry : entries) {
        if (entry.name.find(".lora_A.weight") != std::string::npos) {
            std::string base = peft_key_to_base_name(entry.name);
            if (!base.empty()) a_entries[base] = &entry;
        } else if (entry.name.find(".lora_B.weight") != std::string::npos) {
            std::string base = peft_key_to_base_name(entry.name);
            if (!base.empty()) b_entries[base] = &entry;
        }
    }

    // 5. Build LoraPair for each matched A/B pair
    int skipped = 0;
    for (const auto& [base_name, a_entry] : a_entries) {
        auto b_it = b_entries.find(base_name);
        if (b_it == b_entries.end()) {
            skipped++;
            continue;
        }
        const TensorEntry* b_entry = b_it->second;

        // Validate shapes: lora_A = [r, in_features], lora_B = [out_features, r]
        if (a_entry->shape.size() != 2 || b_entry->shape.size() != 2) {
            skipped++;
            continue;
        }

        LoraPair pair;
        pair.weight_name = base_name;
        pair.r = static_cast<int>(a_entry->shape[0]);
        pair.in_features = static_cast<int>(a_entry->shape[1]);
        pair.out_features = static_cast<int>(b_entry->shape[0]);

        if (static_cast<int>(b_entry->shape[1]) != pair.r) {
            AILA_LOG_WARN("[LoRA] Rank mismatch for %s: A rank=%d, B rank=%lld — skipping",
                          base_name.c_str(), pair.r,
                          (long long)b_entry->shape[1]);
            skipped++;
            continue;
        }

        size_t a_count = a_entry->offset_end - a_entry->offset_start;
        size_t b_count = b_entry->offset_end - b_entry->offset_start;
        size_t a_floats = a_count / sizeof(float);
        size_t b_floats = b_count / sizeof(float);

        pair.lora_a.resize(a_floats);
        pair.lora_b.resize(b_floats);
        std::memcpy(pair.lora_a.data(), data_start + a_entry->offset_start, a_count);
        std::memcpy(pair.lora_b.data(), data_start + b_entry->offset_start, b_count);

        adapter.pairs.push_back(std::move(pair));
    }

    AILA_LOG_INFO("[LoRA] Parsed %zu LoRA pairs%s",
                  adapter.pairs.size(),
                  (skipped > 0 ? (" (" + std::to_string(skipped) + " skipped)").c_str() : ""));
    return true;
}

// ============================================================
// Merge LoRA deltas into ModelWeights (dense bf16 models only)
// ============================================================

int LoraLoader::merge_into_weights(Context& ctx, const LoraAdapter& adapter,
                                   ModelWeights& weights, std::string* error_message) {
    int merged = 0;

    for (const auto& pair : adapter.pairs) {
        if (!weights.has(pair.weight_name)) {
            AILA_LOG_WARN("[LoRA] Weight '%s' not found in model — skipping",
                          pair.weight_name.c_str());
            continue;
        }

        Tensor& weight = weights.get(pair.weight_name);

        // Validate shape: weight should be [out_features, in_features] in bf16
        if (weight.ndim() != 2) {
            AILA_LOG_WARN("[LoRA] Weight '%s' is not 2D — skipping",
                          pair.weight_name.c_str());
            continue;
        }
        int64_t w_out = weight.shape(0);
        int64_t w_in = weight.shape(1);

        if (w_out != pair.out_features || w_in != pair.in_features) {
            AILA_LOG_WARN("[LoRA] Shape mismatch for '%s': weight=[%lld,%lld] lora=[%d,%d] — skipping",
                          pair.weight_name.c_str(),
                          (long long)w_out, (long long)w_in,
                          pair.out_features, pair.in_features);
            continue;
        }

        // Compute delta = lora_B @ lora_A * scaling on CPU (F32).
        int r = pair.r;
        int out_f = pair.out_features;
        int in_f = pair.in_features;
        std::vector<float> delta_f32(static_cast<size_t>(out_f) * in_f);

        for (int i = 0; i < out_f; ++i) {
            for (int j = 0; j < in_f; ++j) {
                float sum = 0.0f;
                for (int k = 0; k < r; ++k) {
                    sum += pair.lora_b[static_cast<size_t>(i) * r + k] *
                           pair.lora_a[static_cast<size_t>(k) * in_f + j];
                }
                delta_f32[static_cast<size_t>(i) * in_f + j] = sum * adapter.config.scaling;
            }
        }

        // Convert to bf16 on CPU
        size_t numel = static_cast<size_t>(out_f) * in_f;
        std::vector<bf16> delta_bf16(numel);
        for (size_t i = 0; i < numel; ++i) {
            delta_bf16[i] = bf16(delta_f32[i]);
        }

        // Upload delta to GPU temporary tensor
        std::vector<int64_t> delta_shape = {static_cast<int64_t>(out_f), static_cast<int64_t>(in_f)};
        Tensor delta_tensor = Tensor::allocate(ctx, delta_shape, weight.dtype());
        ctx.memcpy_h2d(delta_tensor.data(), delta_bf16.data(), numel * sizeof(bf16));

        // GPU elementwise addition: weight += delta (vec8 optimized)
        {
            using vec8 = sycl::vec<bf16, 8>;
            bf16* w_ptr = static_cast<bf16*>(weight.data());
            bf16* d_ptr = static_cast<bf16*>(delta_tensor.data());

            if (numel % 8 == 0) {
                vec8* w_v = reinterpret_cast<vec8*>(w_ptr);
                vec8* d_v = reinterpret_cast<vec8*>(d_ptr);
                int n8 = static_cast<int>(numel / 8);
                ctx.queue().parallel_for(sycl::range<1>(n8),
                    [=](sycl::id<1> idx) {
                        vec8 wv = w_v[idx];
                        vec8 dv = d_v[idx];
                        vec8 ov;
                        for (int k = 0; k < 8; ++k) {
                            ov[k] = bf16(static_cast<float>(wv[k]) + static_cast<float>(dv[k]));
                        }
                        w_v[idx] = ov;
                    });
            } else {
                ctx.queue().parallel_for(sycl::range<1>(numel),
                    [=](sycl::id<1> idx) {
                        int i = idx[0];
                        w_ptr[i] = bf16(static_cast<float>(w_ptr[i]) + static_cast<float>(d_ptr[i]));
                    });
            }
        }

        merged++;
    }

    ctx.synchronize();
    AILA_LOG_INFO("[LoRA] Merged %d weight matrices", merged);
    return merged;
}

} // namespace lora
} // namespace aila
