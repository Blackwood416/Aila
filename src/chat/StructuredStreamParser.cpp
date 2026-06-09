#include "chat/StructuredStreamParser.hpp"

#include <algorithm>
#include <cctype>
#include <string>

namespace aila::chat {
namespace {

constexpr const char* kThinkOpen = "<think>";
constexpr const char* kThinkClose = "</think>";
constexpr const char* kToolOpen = "<tool_call>";
constexpr const char* kToolClose = "</tool_call>";

std::string trim(const std::string& value) {
    size_t begin = 0;
    while (begin < value.size() && std::isspace(static_cast<unsigned char>(value[begin]))) {
        ++begin;
    }

    size_t end = value.size();
    while (end > begin && std::isspace(static_cast<unsigned char>(value[end - 1]))) {
        --end;
    }
    return value.substr(begin, end - begin);
}

void trim_left_in_place(std::string& value) {
    size_t begin = 0;
    while (begin < value.size() && std::isspace(static_cast<unsigned char>(value[begin]))) {
        ++begin;
    }
    value.erase(0, begin);
}

size_t longest_marker_prefix_suffix(const std::string& value) {
    const std::string markers[] = {kThinkOpen, kToolOpen};
    size_t best = 0;
    for (const auto& marker : markers) {
        const size_t max_len = std::min(value.size(), marker.size() - 1);
        for (size_t len = 1; len <= max_len; ++len) {
            if (value.compare(value.size() - len, len, marker, 0, len) == 0) {
                best = std::max(best, len);
            }
        }
    }
    return best;
}

} // namespace

void StructuredStreamParser::push(const std::string& text,
                                  std::vector<StructuredStreamEvent>& out_events) {
    buffer_ += text;
    process(out_events, false);
}

void StructuredStreamParser::finish(std::vector<StructuredStreamEvent>& out_events) {
    process(out_events, true);
    if (buffer_.empty()) {
        return;
    }

    if (state_ == State::Reasoning) {
        emit(StructuredStreamEventType::ReasoningDelta, trim(buffer_), out_events);
    } else if (state_ == State::ToolCall) {
        emit(StructuredStreamEventType::ToolCallDelta, buffer_, out_events);
    } else {
        emit(StructuredStreamEventType::ContentDelta, buffer_, out_events);
    }
    buffer_.clear();
}

void StructuredStreamParser::process(std::vector<StructuredStreamEvent>& out_events, bool final) {
    while (!buffer_.empty()) {
        if (state_ == State::Reasoning) {
            const size_t close = buffer_.find(kThinkClose);
            if (close == std::string::npos) {
                return;
            }

            emit(StructuredStreamEventType::ReasoningDelta, trim(buffer_.substr(0, close)), out_events);
            buffer_.erase(0, close + std::string(kThinkClose).size());
            trim_left_in_place(buffer_);
            state_ = State::Content;
            continue;
        }

        if (state_ == State::ToolCall) {
            const size_t close = buffer_.find(kToolClose);
            if (close == std::string::npos) {
                return;
            }

            const size_t block_end = close + std::string(kToolClose).size();
            emit(StructuredStreamEventType::ToolCallDelta, buffer_.substr(0, block_end), out_events);
            buffer_.erase(0, block_end);
            state_ = State::Content;
            continue;
        }

        const size_t think_pos = buffer_.find(kThinkOpen);
        const size_t tool_pos = buffer_.find(kToolOpen);
        size_t marker_pos = std::string::npos;
        bool marker_is_think = false;
        if (think_pos != std::string::npos && (tool_pos == std::string::npos || think_pos < tool_pos)) {
            marker_pos = think_pos;
            marker_is_think = true;
        } else if (tool_pos != std::string::npos) {
            marker_pos = tool_pos;
        }

        if (marker_pos == std::string::npos) {
            const size_t keep = final ? 0 : longest_marker_prefix_suffix(buffer_);
            if (buffer_.size() <= keep) {
                return;
            }
            emit(StructuredStreamEventType::ContentDelta,
                 buffer_.substr(0, buffer_.size() - keep),
                 out_events);
            buffer_.erase(0, buffer_.size() - keep);
            return;
        }

        if (marker_pos > 0) {
            emit(StructuredStreamEventType::ContentDelta, buffer_.substr(0, marker_pos), out_events);
            buffer_.erase(0, marker_pos);
            continue;
        }

        if (marker_is_think) {
            buffer_.erase(0, std::string(kThinkOpen).size());
            state_ = State::Reasoning;
        } else {
            state_ = State::ToolCall;
        }
    }
}

void StructuredStreamParser::emit(StructuredStreamEventType type,
                                  const std::string& text,
                                  std::vector<StructuredStreamEvent>& out_events) const {
    if (text.empty()) {
        return;
    }

    StructuredStreamEvent event;
    event.type = type;
    event.text = text;
    out_events.push_back(std::move(event));
}

} // namespace aila::chat
