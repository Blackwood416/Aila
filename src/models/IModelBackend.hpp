#pragma once

#include "../core/Context.hpp"
#include "../core/Tensor.hpp"
#include "../utils/SafeTensors.hpp"
#include "engine/Types.hpp"
#include <string>

class IModelBackend {
public:
    virtual ~IModelBackend() = default;

    virtual bool load(Context& ctx,
                      ModelWeights& weights,
                      const ModelSpec& spec,
                      int max_seq_len,
                      std::string* error_message) = 0;

    virtual Tensor& forward(Context& ctx, const int* token_ids_device, int seq_len) = 0;
    virtual void reset() = 0;
    virtual void truncate_kv_cache(int new_len) = 0;
    virtual int max_seq_len() const = 0;
    virtual int vocab_size() const = 0;
    virtual ModelFamily family() const = 0;
};

