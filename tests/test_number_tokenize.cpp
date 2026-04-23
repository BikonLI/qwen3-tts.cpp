#include "text_tokenizer.h"
#include "gguf_loader.h"
#include <cstdio>
#include <cstring>

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
    
    // Test cases matching Python reference
    const char * test_texts[] = {
        "The number is 12345678.",
        "Hello.",
        "Hello world",
        "it's a test",
        "Finally, please read: 0, 1, 2, 10, 100",
        "3.14159",
        "$42.50",
        "99.9%",
        nullptr
    };
    
    for (int i = 0; test_texts[i]; i++) {
        auto tokens = tokenizer.encode(test_texts[i]);
        printf("Text: \"%s\"\n", test_texts[i]);
        printf("Tokens: [");
        for (size_t j = 0; j < tokens.size(); j++) {
            printf("%d", tokens[j]);
            if (j + 1 < tokens.size()) printf(", ");
        }
        printf("]\n");
        
        // Decode each token
        printf("Decoded: [");
        for (size_t j = 0; j < tokens.size(); j++) {
            std::string decoded = tokenizer.decode_token(tokens[j]);
            printf("'%s'", decoded.c_str());
            if (j + 1 < tokens.size()) printf(", ");
        }
        printf("]\n\n");
    }
    
    return 0;
}