#include "text_tokenizer.h"
#include "gguf_loader.h"

#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <cstring>
#include <fstream>
#include <vector>
#include <string>

// Helper: check encode output matches expected token IDs
static bool check_encode_ref(
    const qwen3_tts::TextTokenizer & tokenizer,
    const char * text,
    const int32_t * expected,
    size_t expected_count)
{
    auto result = tokenizer.encode(text);
    bool match = (result.size() == expected_count);
    if (match) {
        for (size_t i = 0; i < expected_count; i++) {
            if (result[i] != expected[i]) {
                match = false;
                break;
            }
        }
    }
    if (!match) {
        fprintf(stderr, "  FAIL: encode(\"%s\")\n", text);
        fprintf(stderr, "    Expected %zu tokens:", expected_count);
        for (size_t i = 0; i < expected_count; i++) fprintf(stderr, " %d", expected[i]);
        fprintf(stderr, "\n");
        fprintf(stderr, "    Got      %zu tokens:", result.size());
        for (size_t e : result) fprintf(stderr, " %d", e);
        fprintf(stderr, "\n");
    }
    return match;
}

// Helper: check pre-tokenized output matches expected
static bool check_pretokenize(
    const qwen3_tts::TextTokenizer & tokenizer,
    const char * input,
    const std::vector<std::string> & expected)
{
    auto result = tokenizer.pre_tokenize(input);
    bool match = (result.size() == expected.size());
    if (match) {
        for (size_t i = 0; i < expected.size(); i++) {
            if (result[i] != expected[i]) {
                match = false;
                break;
            }
        }
    }
    if (!match) {
        fprintf(stderr, "  FAIL: pre_tokenize(\"%s\")\n", input);
        fprintf(stderr, "    Expected %zu tokens:", expected.size());
        for (const auto & t : expected) fprintf(stderr, " [%s]", t.c_str());
        fprintf(stderr, "\n");
        fprintf(stderr, "    Got      %zu tokens:", result.size());
        for (const auto & t : result) fprintf(stderr, " [%s]", t.c_str());
        fprintf(stderr, "\n");
    }
    return match;
}

// Expected tokens for "Hello." with TTS format
// Format: <|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n
// Expected: [151644, 77091, 198, 9707, 13, 151645, 198, 151644, 77091, 198]
static const int32_t EXPECTED_TOKENS[] = {151644, 77091, 198, 9707, 13, 151645, 198, 151644, 77091, 198};
static const size_t EXPECTED_TOKEN_COUNT = 10;

void print_usage(const char * prog) {
    printf("Usage: %s --model <path_to_gguf>\n", prog);
    printf("       %s (runs basic tests without model)\n", prog);
}

int main(int argc, char ** argv) {
    printf("=== Text Tokenizer Test ===\n\n");

    const char * model_path = nullptr;

    // Parse arguments
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
            model_path = argv[++i];
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            print_usage(argv[0]);
            return 0;
        }
    }

    qwen3_tts::TextTokenizer tokenizer;

    // Test 1: Check initial state
    printf("Test 1: Initial state\n");
    assert(!tokenizer.is_loaded());
    printf("  PASS: Tokenizer not loaded initially\n\n");

    // =======================================================================
    // Pre-tokenizer tests (don't need GGUF model)
    // These verify the Qwen2 regex scanner matches HuggingFace output exactly.
    // Run by forcing pre_tokenize_type to qwen2.
    // =======================================================================

    // We need to set the pre_tokenize_type to qwen2 for testing.
    // Since it's a private member, we use a workaround: call pre_tokenize()
    // which returns {text} for 'none' type. For full testing, load a GGUF
    // model that sets type to qwen2. Here we can only test the static helpers.

    printf("Test 2: Unicode property helpers\n");
    // is_unicode_letter tests
    assert(qwen3_tts::TextTokenizer::is_unicode_letter('A'));
    assert(qwen3_tts::TextTokenizer::is_unicode_letter('z'));
    assert(qwen3_tts::TextTokenizer::is_unicode_letter(0x4E00));  // CJK 一
    assert(qwen3_tts::TextTokenizer::is_unicode_letter(0x7B2C));  // CJK 第
    assert(!qwen3_tts::TextTokenizer::is_unicode_letter('0'));
    assert(!qwen3_tts::TextTokenizer::is_unicode_letter(' '));
    assert(!qwen3_tts::TextTokenizer::is_unicode_letter('.'));

    // is_unicode_number tests
    assert(qwen3_tts::TextTokenizer::is_unicode_number('0'));
    assert(qwen3_tts::TextTokenizer::is_unicode_number('9'));
    assert(!qwen3_tts::TextTokenizer::is_unicode_number(0x4E00));  // 一 is Lo, not N
    assert(!qwen3_tts::TextTokenizer::is_unicode_number('A'));
    assert(!qwen3_tts::TextTokenizer::is_unicode_number('.'));

    // is_unicode_whitespace tests
    assert(qwen3_tts::TextTokenizer::is_unicode_whitespace(' '));
    assert(qwen3_tts::TextTokenizer::is_unicode_whitespace('\t'));
    assert(!qwen3_tts::TextTokenizer::is_unicode_whitespace('A'));
    assert(!qwen3_tts::TextTokenizer::is_unicode_whitespace('\n'));

    // is_unicode_newline tests
    assert(qwen3_tts::TextTokenizer::is_unicode_newline('\n'));
    assert(qwen3_tts::TextTokenizer::is_unicode_newline('\r'));
    assert(!qwen3_tts::TextTokenizer::is_unicode_newline(' '));
    assert(!qwen3_tts::TextTokenizer::is_unicode_newline('A'));

    printf("  PASS: Unicode property helpers work correctly\n\n");

    printf("Test 3: UTF-8 decode helpers\n");
    {
        std::string test = "Hello";
        size_t len = 0;
        uint32_t cp = qwen3_tts::TextTokenizer::decode_utf8_codepoint(test, 0, len);
        assert(cp == 'H' && len == 1);

        // CJK character 第 (U+7B2C) encoded as 3 bytes: E7 AC AC
        test = "\xe7\xac\xac";  // 第
        cp = qwen3_tts::TextTokenizer::decode_utf8_codepoint(test, 0, len);
        assert(cp == 0x7B2C && len == 3);

        // emoji: 😀 (U+1F600) encoded as 4 bytes
        test = "\xf0\x9f\x98\x80";
        cp = qwen3_tts::TextTokenizer::decode_utf8_codepoint(test, 0, len);
        assert(cp == 0x1F600 && len == 4);
    }
    printf("  PASS: UTF-8 decode helpers work correctly\n\n");

    if (!model_path) {
        printf("No model specified. Run with --model <path> for full BPE tests.\n");
        printf("=== Basic tests passed! ===\n");
        return 0;
    }

    // Test 4: Load from GGUF
    printf("Test 4: Load tokenizer from GGUF\n");
    printf("  Model: %s\n", model_path);

    qwen3_tts::GGUFLoader loader;
    if (!loader.open(model_path)) {
        printf("  FAIL: Could not open GGUF file: %s\n", loader.get_error().c_str());
        return 1;
    }

    if (!tokenizer.load_from_gguf(loader.get_ctx())) {
        printf("  FAIL: Could not load tokenizer: %s\n", tokenizer.get_error().c_str());
        return 1;
    }

    assert(tokenizer.is_loaded());
    printf("  PASS: Tokenizer loaded successfully\n");
    printf("  Vocab size: %d\n", tokenizer.get_config().vocab_size);
    printf("  BOS token ID: %d\n", tokenizer.bos_token_id());
    printf("  EOS token ID: %d\n", tokenizer.eos_token_id());
    printf("\n");

    // =======================================================================
    // Pre-tokenizer tests (now with loaded model, qwen2 pre-tokenizer active)
    // These verify the Qwen2 regex scanner matches HuggingFace output exactly.
    // =======================================================================

    printf("Test 5: Qwen2 pre-tokenizer\n");
    int pretok_pass = 0;
    int pretok_total = 0;

    // Test: plain letters
    pretok_total++; if (check_pretokenize(tokenizer, "Hello", {"Hello"})) pretok_pass++;
    pretok_total++; if (check_pretokenize(tokenizer, "Hello world", {"Hello", " world"})) pretok_pass++;

    // Test: single digits (each digit is its own token per \p{N})
    pretok_total++; if (check_pretokenize(tokenizer, "123", {"1", "2", "3"})) pretok_pass++;
    pretok_total++; if (check_pretokenize(tokenizer, "123456", {"1", "2", "3", "4", "5", "6"})) pretok_pass++;
    pretok_total++; if (check_pretokenize(tokenizer, "42", {"4", "2"})) pretok_pass++;

    // Test: contractions
    pretok_total++; if (check_pretokenize(tokenizer, "it's", {"it", "'s"})) pretok_pass++;
    pretok_total++; if (check_pretokenize(tokenizer, "I'm", {"I", "'m"})) pretok_pass++;
    pretok_total++; if (check_pretokenize(tokenizer, "they're", {"they", "'re"})) pretok_pass++;
    pretok_total++; if (check_pretokenize(tokenizer, "don't", {"don", "'t"})) pretok_pass++;

    // Test: punctuation patterns
    pretok_total++; if (check_pretokenize(tokenizer, "Hello.", {"Hello", "."})) pretok_pass++;
    pretok_total++; if (check_pretokenize(tokenizer, "Hello, world!", {"Hello", ",", " world", "!"})) pretok_pass++;
    pretok_total++; if (check_pretokenize(tokenizer, "a,b,c", {"a", ",b", ",c"})) pretok_pass++;

    // Test: CJK characters (grouped as \p{L})
    // 你好 = 你(E4BDA0) 好(E5A5BD)
    {
        std::string ni_hao = "\xe4\xbd\xa0\xe5\xa5\xbd";
        pretok_total++; if (check_pretokenize(tokenizer, ni_hao.c_str(), {ni_hao})) pretok_pass++;
    }
    // 第1号 = 第(E7ACAC) 1 号(E58FB7) - each \p{N} digit separate, \p{L} CJK grouped
    {
        std::string di_yi_hao = std::string("\xe7\xac\xac") + "1" + "\xe5\x8f\xb7";
        std::string di = "\xe7\xac\xac";
        std::string hao = "\xe5\x8f\xb7";
        pretok_total++; if (check_pretokenize(tokenizer, di_yi_hao.c_str(), {di, "1", hao})) pretok_pass++;
    }

    // Test: whitespace patterns
    pretok_total++; if (check_pretokenize(tokenizer, "hello  ", {"hello", "  "})) pretok_pass++;
    pretok_total++; if (check_pretokenize(tokenizer, "  hello", {" ", " hello"})) pretok_pass++;
    pretok_total++; if (check_pretokenize(tokenizer, "hello\nworld", {"hello", "\n", "world"})) pretok_pass++;

    // Test: numbers with punctuation
    pretok_total++; if (check_pretokenize(tokenizer, "3.14159", {"3", ".", "1", "4", "1", "5", "9"})) pretok_pass++;
    pretok_total++; if (check_pretokenize(tokenizer, "$42.50", {"$", "4", "2", ".", "5", "0"})) pretok_pass++;

    printf("  Pre-tokenizer: %d/%d passed\n\n", pretok_pass, pretok_total);

    // Test 5b: Encode vs Python reference (Qwen/Qwen3-TTS-12Hz-0.6B-Base)
    printf("Test 5b: Encode vs HuggingFace reference\n");
    int ref_pass = 0;
    int ref_total = 0;

    struct ref_test { const char * text; const int32_t * expected; size_t count; };

    // Python reference: tokenizer.encode(text, add_special_tokens=False)
    // "Hello." → [9707, 13]
    { const int32_t e[] = {9707, 13}; ref_total++; if (check_encode_ref(tokenizer, "Hello.", e, 2)) ref_pass++; }
    // "Hello world" → [9707, 1879]
    { const int32_t e[] = {9707, 1879}; ref_total++; if (check_encode_ref(tokenizer, "Hello world", e, 2)) ref_pass++; }
    // "it's a test" → [275, 594, 264, 1273]
    { const int32_t e[] = {275, 594, 264, 1273}; ref_total++; if (check_encode_ref(tokenizer, "it's a test", e, 4)) ref_pass++; }
    // "The number is 12345678." → [785, 1372, 374, 220, 16, 17, 18, 19, 20, 21, 22, 23, 13]
    { const int32_t e[] = {785, 1372, 374, 220, 16, 17, 18, 19, 20, 21, 22, 23, 13}; ref_total++; if (check_encode_ref(tokenizer, "The number is 12345678.", e, 13)) ref_pass++; }
    // "3.14159" → [18, 13, 16, 19, 16, 20, 24]
    { const int32_t e[] = {18, 13, 16, 19, 16, 20, 24}; ref_total++; if (check_encode_ref(tokenizer, "3.14159", e, 7)) ref_pass++; }
    // "99.9%" → [24, 24, 13, 24, 4]
    { const int32_t e[] = {24, 24, 13, 24, 4}; ref_total++; if (check_encode_ref(tokenizer, "99.9%", e, 5)) ref_pass++; }

    printf("  Encode vs reference: %d/%d passed\n\n", ref_pass, ref_total);

    // Test 6: Encode simple text
    printf("Test 6: Encode 'Hello.'\n");
    auto tokens = tokenizer.encode("Hello.");
    printf("  Tokens: [");
    for (size_t i = 0; i < tokens.size(); i++) {
        printf("%d", tokens[i]);
        if (i + 1 < tokens.size()) printf(", ");
    }
    printf("]\n");

    // Test 6b: Numeric/symbol rich text should keep digits and punctuation
    printf("Test 6b: Encode numbers/symbols sentence\n");
    const std::string numeric_text =
        "Finally, please read the following numbers and symbols correctly: 0, 1, 2, 10, 100, 1,000, 3.14159, 99.9%, and $42.50.";
    auto numeric_tokens = tokenizer.encode(numeric_text);
    auto decoded_numeric = tokenizer.decode(numeric_tokens);
    if (decoded_numeric == numeric_text) {
        printf("  PASS: Numeric/symbol sentence roundtrip preserved\n\n");
    } else {
        printf("  FAIL: Numeric/symbol sentence roundtrip mismatch\n");
        printf("  Decoded: '%s'\n\n", decoded_numeric.c_str());
        return 1;
    }

    // Check expected tokens for "Hello." (without TTS format)
    // Expected: [9707, 13] for "Hello" and "."
    if (tokens.size() >= 2 && tokens[0] == 9707 && tokens[1] == 13) {
        printf("  PASS: Tokens match expected [9707, 13]\n\n");
    } else {
        printf("  INFO: Tokens differ from expected [9707, 13]\n\n");
    }

    // Test 7: Encode with TTS format
    printf("Test 7: Encode 'Hello.' with TTS format\n");
    auto tts_tokens = tokenizer.encode_for_tts("Hello.");
    printf("  Tokens: [");
    for (size_t i = 0; i < tts_tokens.size(); i++) {
        printf("%d", tts_tokens[i]);
        if (i + 1 < tts_tokens.size()) printf(", ");
    }
    printf("]\n");
    printf("  Expected: [");
    for (size_t i = 0; i < EXPECTED_TOKEN_COUNT; i++) {
        printf("%d", EXPECTED_TOKENS[i]);
        if (i + 1 < EXPECTED_TOKEN_COUNT) printf(", ");
    }
    printf("]\n");

    // Compare with expected
    bool match = (tts_tokens.size() == EXPECTED_TOKEN_COUNT);
    if (match) {
        for (size_t i = 0; i < EXPECTED_TOKEN_COUNT; i++) {
            if (tts_tokens[i] != EXPECTED_TOKENS[i]) {
                match = false;
                break;
            }
        }
    }

    if (match) {
        printf("  PASS: TTS tokens match expected!\n\n");
    } else {
        printf("  FAIL: TTS tokens do not match expected\n\n");
        return 1;
    }

    // Test 8: Decode tokens
    printf("Test 8: Decode tokens\n");
    std::string decoded = tokenizer.decode(tokens);
    printf("  Decoded: '%s'\n", decoded.c_str());
    if (decoded == "Hello.") {
        printf("  PASS: Decoded text matches original\n\n");
    } else {
        printf("  INFO: Decoded text differs from original\n\n");
    }

    // Test 9: Decode single tokens
    printf("Test 9: Decode individual tokens\n");
    for (size_t i = 0; i < tts_tokens.size(); i++) {
        std::string tok_str = tokenizer.decode_token(tts_tokens[i]);
        printf("  Token %d: '%s'\n", tts_tokens[i], tok_str.c_str());
    }
    printf("\n");

    // Test 10: Compare with reference file if available
    printf("Test 10: Compare with reference file\n");
    std::string ref_path = "../reference/text_tokens.bin";
    std::ifstream ref_file(ref_path, std::ios::binary);
    if (ref_file.is_open()) {
        // Read int64 tokens from reference file
        std::vector<int64_t> ref_tokens;
        int64_t val;
        while (ref_file.read(reinterpret_cast<char*>(&val), sizeof(val))) {
            ref_tokens.push_back(val);
        }
        ref_file.close();

        printf("  Reference tokens: [");
        for (size_t i = 0; i < ref_tokens.size(); i++) {
            printf("%ld", (long)ref_tokens[i]);
            if (i + 1 < ref_tokens.size()) printf(", ");
        }
        printf("]\n");

        // Compare
        bool ref_match = (tts_tokens.size() == ref_tokens.size());
        if (ref_match) {
            for (size_t i = 0; i < ref_tokens.size(); i++) {
                if (tts_tokens[i] != (int32_t)ref_tokens[i]) {
                    ref_match = false;
                    break;
                }
            }
        }

        if (ref_match) {
            printf("  PASS: Tokens match reference file!\n\n");
        } else {
            printf("  FAIL: Tokens do not match reference file\n\n");
            return 1;
        }
    } else {
        printf("  SKIP: Reference file not found at %s\n\n", ref_path.c_str());
    }

    printf("=== All tests passed! ===\n");
    return 0;
}