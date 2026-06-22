#pragma once

#include "alia_api.h"
#include "AliaAsrPipeline.hpp"
#include "AliaBackgroundPipeline.hpp"
#include "AliaForegroundPipeline.hpp"
#include "AliaTtsPipeline.hpp"
#include "ModelSlot.hpp"
#include "RuntimeContext.hpp"

#include <atomic>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

struct AliaContext {
    explicit AliaContext(int max_seq_len_in);
    void configure_model_slots();
    bool load_model_metadata();
    bool load_model_slots();

    int max_seq_len = 0;
    std::string last_error;
    std::string asr_model_dir;
    std::string vlm_4b_model_dir;
    std::string vlm_4b_lora_dir;
    std::string vlm_0_8b_model_dir;
    std::string tts_model_dir;

    std::unique_ptr<aila::alia::RuntimeContext> runtime;
    aila::alia::ModelSlot asr;
    aila::alia::ModelSlot foreground_vlm;
    aila::alia::ModelSlot background_vlm;
    aila::alia::ModelSlot tts;
    std::atomic<int> abort_mask{0};

    std::unique_ptr<aila::alia::AliaAsrPipeline> asr_pipeline;
    std::unique_ptr<aila::alia::AliaTtsPipeline> tts_pipeline;
    std::unique_ptr<aila::alia::AliaForegroundPipeline> foreground_pipeline;
    std::unique_ptr<aila::alia::AliaBackgroundPipeline> background_pipeline;
};
