#include "AliaContext.hpp"

#include "../utils/EnvUtils.hpp"

AliaContext::AliaContext(int max_seq_len_in)
    : max_seq_len(max_seq_len_in),
      runtime(std::make_unique<aila::alia::RuntimeContext>()),
      asr_pipeline(std::make_unique<aila::alia::AliaAsrPipeline>(&asr)),
      tts_pipeline(std::make_unique<aila::alia::AliaTtsPipeline>(&tts)),
      foreground_pipeline(std::make_unique<aila::alia::AliaForegroundPipeline>(
          &foreground_vlm, tts_pipeline.get(), asr_pipeline.get())),
      background_pipeline(std::make_unique<aila::alia::AliaBackgroundPipeline>(
          &background_vlm)) {}

void AliaContext::configure_model_slots() {
    asr.configure(aila::alia::ModelRole::Asr, asr_model_dir, &runtime->foreground());
    foreground_vlm.configure(aila::alia::ModelRole::ForegroundVlm, vlm_4b_model_dir,
                             &runtime->foreground());
    foreground_vlm.set_lora_dir(vlm_4b_lora_dir);
    background_vlm.configure(aila::alia::ModelRole::BackgroundVlm, vlm_0_8b_model_dir,
                             &runtime->background());
    tts.configure(aila::alia::ModelRole::Tts, tts_model_dir, &runtime->foreground());
}

bool AliaContext::load_model_metadata() {
    last_error.clear();

    auto load_slot = [&](aila::alia::ModelSlot& slot) {
        if (slot.load_metadata()) {
            return true;
        }
        last_error = slot.last_error();
        return false;
    };

    return load_slot(asr) &&
           load_slot(foreground_vlm) &&
           load_slot(background_vlm) &&
           load_slot(tts);
}

bool AliaContext::load_model_slots() {
    last_error.clear();

    auto load_slot = [&](aila::alia::ModelSlot& slot) {
        if (slot.load_model(max_seq_len)) {
            return true;
        }
        last_error = slot.last_error();
        return false;
    };

    if (!load_slot(asr) ||
        !load_slot(foreground_vlm) ||
        !load_slot(background_vlm) ||
        !load_slot(tts)) {
        return false;
    }

    if (foreground_pipeline &&
        aila::env::read_flag("AILA_FOREGROUND_VLM_WARMUP", true)) {
        std::string foreground_warmup_error;
        if (!foreground_pipeline->warmup_loaded_vlm(&foreground_warmup_error)) {
            last_error = foreground_warmup_error.empty()
                ? "failed to warm up Alia foreground VLM"
                : "failed to warm up Alia foreground VLM: " + foreground_warmup_error;
            return false;
        }
    }

    std::string reference_voice_error;
    if (tts_pipeline &&
        !tts_pipeline->preload_reference_voice(&reference_voice_error)) {
        last_error = reference_voice_error.empty()
            ? "failed to preload Alia TTS reference voice"
            : reference_voice_error;
        return false;
    }

    return true;
}
