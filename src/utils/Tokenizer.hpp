#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <utility>
#include "engine/Types.hpp"

// ============================================================
// BPE Tokenizer for Qwen3
// 从 vocab.json + merges.txt 加载
// ============================================================
class Tokenizer {
public:
    Tokenizer() = default;

    // 从模型目录加载 vocab.json + merges.txt
    bool load(const std::string& model_dir);

    // 编码: 文本 -> token ID 列表
    std::vector<int> encode(const std::string& text) const;

    // 解码: 单个 token ID -> 文本片段
    std::string decode(int token_id) const;

    // 解码: token ID 列表 -> 文本
    std::string decode(const std::vector<int>& token_ids) const;

    // 应用 ChatML 对话模板
    // 用户消息末尾追加 "/no_think" 时，会尝试关闭思考模式注入。
    std::vector<int> apply_chat_template(
        const std::string& system_prompt,
        const std::string& user_message) const;

    // Multi-turn: encode full conversation history
    std::vector<int> apply_chat_template(
        const std::string& system_prompt,
        const ChatHistory& history) const;

    // 判断是否为结束 token
    bool is_eos(int token_id) const;

    int vocab_size() const { return static_cast<int>(id_to_token_.size()); }
    int im_start_id() const { return im_start_id_; }
    int im_end_id() const { return im_end_id_; }
    int eos_id() const { return eos_id_; }

private:
    // 词表
    std::unordered_map<std::string, int> token_to_id_;
    std::vector<std::string> id_to_token_;

    // BPE 合并规则: (token_a, token_b) -> merged_token, 按优先级排序
    struct MergeRule {
        std::string a;
        std::string b;
        std::string merged;
    };
    std::vector<MergeRule> merges_;
    // 快速查找合并优先级
    std::unordered_map<std::string, int> merge_rank_;  // "a b" -> rank

    // 特殊 token ID
    int im_start_id_ = 151644;
    int im_end_id_ = 151645;
    int eos_id_ = 151645;
    int eot_id_ = 151643;

    // 特殊 token 映射
    std::unordered_map<std::string, int> special_tokens_;

    // BPE 编码核心: 将 byte 序列进行 BPE 合并
    std::vector<std::string> bpe_encode(const std::vector<std::string>& tokens) const;

    // 将 UTF-8 文本拆分为初始 token (byte-level BPE 的初始化)
    std::vector<std::string> pretokenize(const std::string& text) const;

    // 将单个 word 拆分为字符级别 token
    std::vector<std::string> split_to_chars(const std::string& word) const;

    // Byte-to-unicode 映射 (GPT-2 / Qwen 风格)
    static std::unordered_map<uint8_t, std::string> byte_to_unicode();
    static std::unordered_map<std::string, uint8_t> unicode_to_byte();
};
