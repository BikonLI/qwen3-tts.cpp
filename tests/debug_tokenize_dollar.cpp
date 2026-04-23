#include "text_tokenizer.h"
#include "gguf_loader.h"
#include <cstdio>
#include <vector>
#include <string>

int main(int argc, char ** argv) {
    const char * model_path = argc > 1 ? argv[1] : "models/gguf/0.6b-base/qwen3-tts-12hz-0.6b-base-f16.gguf";

    qwen3_tts::GGUFLoader loader;
    if (!loader.open(model_path)) {
        fprintf(stderr, "Failed to open GGUF: %s\n", loader.get_error().c_str());
        return 1;
    }

    qwen3_tts::TextTokenizer tokenizer;
    if (!tokenizer.load_from_gguf(loader.get_ctx())) {
        fprintf(stderr, "Failed to load tokenizer: %s\n", tokenizer.get_error().c_str());
        return 1;
    }

    const char * text = "please read the following numbers and symbols correctly: 0, 1, 2, 10, 100, 1,000, 3.14159, 99.9%, and $42.50.";
    auto tokens = tokenizer.encode(text);

    printf("Input text: %s\n\n", text);
    printf("C++ Token IDs (%zu tokens):\n", tokens.size());
    for (size_t i = 0; i < tokens.size(); i++) {
        std::string decoded = tokenizer.decode_token(tokens[i]);
        printf("  [%2zu] ID=%5d  decoded='%s'\n", i, tokens[i], decoded.c_str());
    }

    printf("\nLast 10 tokens:\n");
    for (size_t i = (tokens.size() > 10 ? tokens.size() - 10 : 0); i < tokens.size(); i++) {
        std::string decoded = tokenizer.decode_token(tokens[i]);
        printf("  [%2zu] ID=%5d  decoded='%s'\n", i, tokens[i], decoded.c_str());
    }

    // Test substrings
    printf("\n=== Testing substrings ===\n");
    const char * substrings[] = {"$42.50", "$42", "$", "42.50", ".50", "$4", "42", nullptr};
    for (int i = 0; substrings[i]; i++) {
        auto toks = tokenizer.encode(substrings[i]);
        printf("%-15s -> IDs=[", substrings[i]);
        for (size_t j = 0; j < toks.size(); j++) {
            if (j > 0) printf(", ");
            printf("%d", toks[j]);
        }
        printf("] -> decoded=[");
        for (size_t j = 0; j < toks.size(); j++) {
            if (j > 0) printf(", ");
            std::string d = tokenizer.decode_token(toks[j]);
            printf("'%s'", d.c_str());
        }
        printf("]\n");
    }

    return 0;
}