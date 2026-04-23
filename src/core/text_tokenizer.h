#pragma once

#include "gguf.h"

#include <string>
#include <vector>
#include <unordered_map>
#include <map>

namespace qwen3_tts {

// Pre-tokenizer type
enum class pre_tokenize_type { none = 0, qwen2 };

// BPE tokenizer configuration
struct tokenizer_config {
    int32_t vocab_size = 151936;
    int32_t pad_token_id = 151643;
    int32_t eos_token_id = 151645;  // <|im_end|>
    int32_t bos_token_id = 151644;  // <|im_start|>
};

// Text tokenizer class (BPE-based, GPT-2 style with Qwen2 pre-tokenization)
class TextTokenizer {
public:
    TextTokenizer();
    ~TextTokenizer();
    
    // Load tokenizer from GGUF file
    bool load_from_gguf(struct gguf_context * ctx);
    
    // Pre-tokenize text using the configured pre-tokenizer (for testing/debugging)
    // Returns empty vector if no pre-tokenizer is configured
    std::vector<std::string> pre_tokenize(const std::string & text) const;

    // Unicode property helpers (public for testing)
    static bool is_unicode_letter(uint32_t cp);
    static bool is_unicode_number(uint32_t cp);
    static bool is_unicode_whitespace(uint32_t cp);
    static bool is_unicode_newline(uint32_t cp);
    static uint32_t decode_utf8_codepoint(const std::string & text, size_t pos, size_t & byte_len);

    // Encode text to token IDs
    std::vector<int32_t> encode(const std::string & text) const;
    
    // Encode assistant text turn for TTS generation:
    // <|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n
    std::vector<int32_t> encode_for_tts(const std::string & text,
                                        const std::string & instruct = "",
                                        const std::string & speaker = "",
                                        const std::string & reference_text = "") const;

    // Encode instruct turn for TTS generation:
    // <|im_start|>user\n{instruct}<|im_end|>\n
    std::vector<int32_t> encode_instruct_for_tts(const std::string & instruct) const;
    
    // Decode token IDs to text
    std::string decode(const std::vector<int32_t> & tokens) const;
    
    // Decode single token
    std::string decode_token(int32_t token_id) const;
    
    // Get configuration
    const tokenizer_config & get_config() const { return config_; }
    
    // Get error message
    const std::string & get_error() const { return error_msg_; }
    
    // Check if loaded
    bool is_loaded() const { return loaded_; }
    
    // Get special token IDs
    int32_t bos_token_id() const { return config_.bos_token_id; }
    int32_t eos_token_id() const { return config_.eos_token_id; }
    int32_t pad_token_id() const { return config_.pad_token_id; }
    
private:
    tokenizer_config config_;
    std::string error_msg_;
    bool loaded_ = false;
    pre_tokenize_type pre_tokenize_type_ = pre_tokenize_type::none;
    
    // Vocabulary: token string -> token ID
    std::unordered_map<std::string, int32_t> vocab_;
    
    // Reverse vocabulary: token ID -> token string
    std::vector<std::string> id_to_token_;
    
    // BPE merges: pair -> rank (lower rank = higher priority)
    std::map<std::pair<std::string, std::string>, int32_t> bpe_ranks_;
    
    // Special tokens for chat template roles and newline
    int32_t assistant_token_id_ = 77091;
    int32_t user_token_id_ = -1;
    int32_t newline_token_id_ = 198;  // '\n' encoded
    
    // Helper: convert bytes to unicode (GPT-2 style byte encoding)
    static std::string bytes_to_unicode(const std::string & text);
    static std::string unicode_to_bytes(const std::string & text);
    
    // Helper: get UTF-8 character length
    static size_t utf8_len(char c);
    
    // BPE encoding for a single word
    std::vector<std::string> bpe(const std::string & token) const;

    // Byte-level fallback when a merged token is not in vocab.
    std::vector<int32_t> encode_unknown_bpe_token_bytes(const std::string & token) const;
    
    // Find the pair with lowest rank in a sequence
    std::pair<std::string, std::string> get_min_pair(
        const std::vector<std::string> & word) const;

    // Qwen2 regex pre-tokenizer
    std::vector<std::string> pre_tokenize_qwen2(const std::string & text) const;
};

} // namespace qwen3_tts
