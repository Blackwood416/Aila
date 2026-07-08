#include "AliaTtsTextChunker.hpp"

#include "../utils/EnvUtils.hpp"

#include <algorithm>
#include <cctype>

namespace aila::alia {
namespace {

const std::vector<std::string>& tts_chunk_boundary_markers() {
    static const std::vector<std::string> markers = {
        ".", "!", "?", ";", "\n",
        "\xE3\x80\x82", "\xEF\xBC\x81", "\xEF\xBC\x9F", "\xEF\xBC\x9B", "\xE2\x80\xA6",
    };
    return markers;
}

const std::vector<std::string>& tts_soft_chunk_boundary_markers() {
    static const std::vector<std::string> markers = {
        ",", ":", ")", "]",
        "\xEF\xBC\x8C", "\xE3\x80\x81", "\xEF\xBC\x9A", "\xEF\xBC\x89", "\xE3\x80\x91",
        " ",
    };
    return markers;
}

bool ends_with_tts_chunk_boundary(const std::string& text) {
    for (const std::string& marker : tts_chunk_boundary_markers()) {
        if (text.size() >= marker.size() &&
            text.compare(text.size() - marker.size(), marker.size(), marker) == 0) {
            return true;
        }
    }
    return false;
}

size_t last_tts_chunk_boundary(const std::string& text) {
    size_t cutoff = std::string::npos;
    for (const std::string& marker : tts_chunk_boundary_markers()) {
        size_t pos = text.find(marker);
        while (pos != std::string::npos) {
            cutoff = std::max(cutoff == std::string::npos ? 0 : cutoff,
                              pos + marker.size());
            pos = text.find(marker, pos + marker.size());
        }
    }
    return cutoff;
}

size_t last_tts_soft_chunk_boundary(const std::string& text,
                                    size_t min_cutoff,
                                    size_t max_cutoff) {
    if (max_cutoff == 0 || text.size() < max_cutoff) {
        return std::string::npos;
    }
    max_cutoff = std::min(max_cutoff, text.size());
    size_t cutoff = std::string::npos;
    for (const std::string& marker : tts_soft_chunk_boundary_markers()) {
        size_t pos = text.find(marker);
        while (pos != std::string::npos) {
            const size_t candidate = pos + marker.size();
            if (candidate > max_cutoff) {
                break;
            }
            if (candidate >= min_cutoff) {
                cutoff = std::max(cutoff == std::string::npos ? 0 : cutoff,
                                  candidate);
            }
            pos = text.find(marker, pos + marker.size());
        }
    }
    return cutoff;
}

TtsChunkDecisionReason waiting_reason(const std::string& buffer,
                                      int hard_min_chars,
                                      int soft_min_chars,
                                      bool low_latency_first_chunk,
                                      bool first_soft_boundary_flush) {
    if (buffer.empty()) {
        return TtsChunkDecisionReason::NoText;
    }
    if (static_cast<int>(buffer.size()) < hard_min_chars) {
        return TtsChunkDecisionReason::BelowHardMin;
    }
    if (low_latency_first_chunk &&
        first_soft_boundary_flush &&
        static_cast<int>(buffer.size()) < soft_min_chars) {
        return TtsChunkDecisionReason::BelowSoftMin;
    }
    return TtsChunkDecisionReason::NoBoundary;
}

}  // namespace

const char* tts_chunk_decision_reason_name(TtsChunkDecisionReason reason) {
    switch (reason) {
    case TtsChunkDecisionReason::NoText:
        return "no_text";
    case TtsChunkDecisionReason::NoBoundary:
        return "no_boundary";
    case TtsChunkDecisionReason::BelowHardMin:
        return "below_hard_min";
    case TtsChunkDecisionReason::BelowSoftMin:
        return "below_soft_min";
    case TtsChunkDecisionReason::HardBoundaryFlush:
        return "hard_boundary_flush";
    case TtsChunkDecisionReason::SoftBoundaryFlush:
        return "soft_boundary_flush";
    case TtsChunkDecisionReason::SoftMax:
        return "soft_max";
    case TtsChunkDecisionReason::Force:
        return "force";
    case TtsChunkDecisionReason::EarlyFirstChunk:
        return "early_first_chunk";
    }
    return "unknown";
}

std::vector<std::string> split_spoken_text_for_tts(const std::string& text,
                                                   bool split_sentence_boundaries,
                                                   size_t min_first_chunk_chars) {
    std::vector<std::string> chunks;
    if (!split_sentence_boundaries) {
        size_t begin = 0;
        while (begin < text.size() &&
               std::isspace(static_cast<unsigned char>(text[begin]))) {
            ++begin;
        }
        size_t end = text.size();
        while (end > begin &&
               std::isspace(static_cast<unsigned char>(text[end - 1]))) {
            --end;
        }
        if (end > begin) {
            chunks.push_back(text.substr(begin, end - begin));
        }
        return chunks;
    }

    std::string current;
    auto flush = [&]() {
        size_t begin = 0;
        while (begin < current.size() &&
               std::isspace(static_cast<unsigned char>(current[begin]))) {
            ++begin;
        }
        size_t end = current.size();
        while (end > begin &&
               std::isspace(static_cast<unsigned char>(current[end - 1]))) {
            --end;
        }
        if (end > begin) {
            chunks.push_back(current.substr(begin, end - begin));
        }
        current.clear();
    };

    for (char ch : text) {
        current.push_back(ch);
        if (ends_with_tts_chunk_boundary(current)) {
            flush();
        }
    }
    flush();

    while (min_first_chunk_chars > 0 &&
           chunks.size() > 1 &&
           chunks.front().size() < min_first_chunk_chars) {
        std::string merged = std::move(chunks[0]);
        const std::string& next = chunks[1];
        if (!merged.empty() && !next.empty()) {
            const unsigned char last =
                static_cast<unsigned char>(merged.back());
            const unsigned char first =
                static_cast<unsigned char>(next.front());
            const bool ascii_sentence_join =
                (last == '.' || last == '!' || last == '?' ||
                 last == ';' || last == ',' || last == ':') &&
                std::isalnum(first) != 0;
            const bool ascii_word_join =
                std::isalnum(last) != 0 && std::isalnum(first) != 0;
            if (ascii_sentence_join || ascii_word_join) {
                merged.push_back(' ');
            }
        }
        merged += next;
        chunks[0] = std::move(merged);
        chunks.erase(chunks.begin() + 1);
    }
    return chunks;
}

TtsFirstChunkEarlyFlushConfig read_tts_first_chunk_early_flush_config() {
    TtsFirstChunkEarlyFlushConfig config;
    config.enabled =
        aila::env::read_flag("AILA_TTS_FIRST_CHUNK_EARLY_FLUSH", true);
    config.token_delay = std::max(
        0, aila::env::read_int_raw("AILA_TTS_FIRST_CHUNK_EARLY_TOKEN_DELAY", 0));
    config.delay_ms = std::max(
        0, aila::env::read_int_raw("AILA_TTS_FIRST_CHUNK_EARLY_MS", 0));
    return config;
}

TtsFirstAudioPriorityConfig read_tts_first_audio_priority_config() {
    TtsFirstAudioPriorityConfig config;
    config.enabled =
        aila::env::read_flag("AILA_FOREGROUND_TTS_FIRST_AUDIO_PRIORITY", true);
    config.base_timeout_ms = std::max(
        0, aila::env::read_int_raw("AILA_FOREGROUND_TTS_FIRST_AUDIO_PRIORITY_TIMEOUT_MS", 250));
    config.active_extra_ms = std::max(
        0,
        aila::env::read_int_raw(
            "AILA_FOREGROUND_TTS_FIRST_AUDIO_PRIORITY_ACTIVE_EXTRA_MS", 120));
    return config;
}

TtsStreamActionTagGuardConfig read_tts_stream_action_tag_guard_config() {
    TtsStreamActionTagGuardConfig config;
    config.enabled =
        aila::env::read_flag("AILA_TTS_STREAM_ACTION_TAG_GUARD", false);
    return config;
}

TtsTextChunkResult take_ready_tts_text_chunks(
    std::string& buffer,
    const TtsTextChunkPolicy& policy,
    const TtsTextChunkRequest& request) {
    const int soft_min_chars = request.low_latency_first_chunk
        ? policy.first_soft_min_chars
        : policy.steady_soft_min_chars;
    const int soft_max_chars = request.low_latency_first_chunk
        ? policy.first_soft_max_chars
        : policy.steady_soft_max_chars;
    const int hard_min_chars = request.low_latency_first_chunk
        ? policy.first_hard_min_chars
        : policy.steady_hard_min_chars;

    TtsTextChunkResult result;
    result.pending_chars_before = buffer.size();
    result.reason = waiting_reason(buffer,
                                   hard_min_chars,
                                   soft_min_chars,
                                   request.low_latency_first_chunk,
                                   policy.first_soft_boundary_flush);

    size_t cutoff = std::string::npos;
    if (request.force) {
        cutoff = buffer.size();
        result.reason = TtsChunkDecisionReason::Force;
    } else {
        const size_t hard_cutoff = last_tts_chunk_boundary(buffer);
        if (hard_cutoff != std::string::npos) {
            if (static_cast<int>(hard_cutoff) >= hard_min_chars) {
                cutoff = hard_cutoff;
                result.reason = TtsChunkDecisionReason::HardBoundaryFlush;
            } else {
                result.reason = TtsChunkDecisionReason::BelowHardMin;
            }
        }
        if (cutoff == std::string::npos &&
            request.low_latency_first_chunk &&
            policy.first_soft_boundary_flush) {
            const size_t soft_cutoff = last_tts_soft_chunk_boundary(
                buffer,
                static_cast<size_t>(soft_min_chars),
                buffer.size());
            if (soft_cutoff != std::string::npos) {
                cutoff = soft_cutoff;
                result.reason = TtsChunkDecisionReason::SoftBoundaryFlush;
            } else if (result.reason != TtsChunkDecisionReason::BelowHardMin &&
                       static_cast<int>(buffer.size()) < soft_min_chars) {
                result.reason = TtsChunkDecisionReason::BelowSoftMin;
            }
        }
        if (cutoff == std::string::npos &&
            soft_max_chars > 0 &&
            static_cast<int>(buffer.size()) >= soft_max_chars) {
            const size_t soft_max_cutoff = last_tts_soft_chunk_boundary(
                buffer,
                static_cast<size_t>(std::min(soft_min_chars, soft_max_chars)),
                static_cast<size_t>(soft_max_chars));
            if (soft_max_cutoff != std::string::npos) {
                cutoff = soft_max_cutoff;
                result.reason = TtsChunkDecisionReason::SoftMax;
            }
        }
        if (cutoff == std::string::npos &&
            request.early_first_chunk &&
            request.low_latency_first_chunk &&
            static_cast<int>(buffer.size()) >= hard_min_chars) {
            cutoff = buffer.size();
            result.reason = TtsChunkDecisionReason::EarlyFirstChunk;
        }
    }

    if (cutoff == std::string::npos || cutoff == 0) {
        return result;
    }

    std::string ready = buffer.substr(0, cutoff);
    buffer.erase(0, cutoff);
    result.cutoff_chars = cutoff;
    const bool split_sentence_boundaries =
        !(policy.coalesce_steady_text_chunks && !request.low_latency_first_chunk);
    result.chunks = split_spoken_text_for_tts(
        ready,
        split_sentence_boundaries,
        request.low_latency_first_chunk ? static_cast<size_t>(hard_min_chars) : 0);
    return result;
}

}  // namespace aila::alia
