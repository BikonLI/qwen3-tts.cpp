#include "text_tokenizer.h"
#include "gguf_loader.h"
#include <cstdio>
#include <vector>
#include <string>
#include <fstream>
#include <cstring>

// Expected tokenization results from Python reference (scripts/tokenize_text8_ref.py)
// Generated 2026-04-22 using official qwen-tts tokenizer with add_special_tokens=False
static const int32_t expected_line1[] = {100644, 104307, 100832, 3837, 785, 12884, 374, 2797, 13};
static const int32_t expected_line2[] = {35946, 18830, 18, 18947, 22377, 3837, 100626, 17, 13, 20, 18947, 34164, 1773};
static const int32_t expected_line3[] = {14880, 18493, 17, 15, 17, 21, 12, 15, 19, 12, 17, 18, 220, 16, 20, 25, 18, 15, 71971, 100370, 1773};
static const int32_t expected_line4[] = {104273, 20412, 17, 22, 13, 20, 30937, 3837, 99208, 94299, 16, 17, 16017, 7530, 1773};
static const int32_t expected_line5[] = {99487, 97480, 20412, 3, 16, 24, 13, 24, 24, 3837, 101174, 104698, 9370, 1773};
static const int32_t expected_line6[] = {26218, 3703, 1110, 8687, 905, 1431, 13};
static const int32_t expected_line7[] = {93568, 20412, 1944, 3317, 16, 17, 18, 10375, 905, 1773};
static const int32_t expected_line8[] = {31615, 37029, 110910, 24, 24, 69982, 105440, 1773};
static const int32_t expected_line9[] = {42411, 99882, 24732, 3837, 99520, 20217, 1773};
static const int32_t expected_line10[] = {3925, 3837, 97639, 55286, 100003, 6313};
static const int32_t expected_line11[] = {99487, 32804, 20412, 69, 2075, 11730, 87, 61, 17, 10, 17, 87, 10, 16, 1773};
static const int32_t expected_line12[] = {87805, 20412, 10, 16, 12, 23, 15, 15, 12, 16, 17, 18, 12, 19, 20, 21, 22, 1773};
static const int32_t expected_line13[] = {100644, 20412, 102600, 75108, 9909, 34620, 74276};
static const int32_t expected_line14[] = {104318, 36987, 9707, 11, 1879, 17839, 101889, 104093, 1773};
static const int32_t expected_line15[] = {26898, 13072, 20412, 691, 2273, 17, 13, 16, 20676, 3909, 1773};
static const int32_t expected_line16[] = {15469, 104949, 29524, 38, 2828, 12, 20, 99165, 102553, 1773};
static const int32_t expected_line17[] = {102136, 20412, 16, 25, 18, 3837, 100148, 100460, 28726, 1773};
static const int32_t expected_line18[] = {14880, 101113, 29437, 18, 44928, 9909, 25020, 220, 18, 74276};
static const int32_t expected_line19[] = {99517, 105275, 44740, 220, 16, 20, 1298, 1773};
static const int32_t expected_line20[] = {9454, 14, 2753, 86119, 106882, 3837, 32664, 100003, 11319};

static const int32_t * expected_tokens[] = {
    expected_line1, expected_line2, expected_line3, expected_line4, expected_line5,
    expected_line6, expected_line7, expected_line8, expected_line9, expected_line10,
    expected_line11, expected_line12, expected_line13, expected_line14, expected_line15,
    expected_line16, expected_line17, expected_line18, expected_line19, expected_line20,
};
static const size_t expected_counts[] = {
    sizeof(expected_line1)/sizeof(expected_line1[0]),
    sizeof(expected_line2)/sizeof(expected_line2[0]),
    sizeof(expected_line3)/sizeof(expected_line3[0]),
    sizeof(expected_line4)/sizeof(expected_line4[0]),
    sizeof(expected_line5)/sizeof(expected_line5[0]),
    sizeof(expected_line6)/sizeof(expected_line6[0]),
    sizeof(expected_line7)/sizeof(expected_line7[0]),
    sizeof(expected_line8)/sizeof(expected_line8[0]),
    sizeof(expected_line9)/sizeof(expected_line9[0]),
    sizeof(expected_line10)/sizeof(expected_line10[0]),
    sizeof(expected_line11)/sizeof(expected_line11[0]),
    sizeof(expected_line12)/sizeof(expected_line12[0]),
    sizeof(expected_line13)/sizeof(expected_line13[0]),
    sizeof(expected_line14)/sizeof(expected_line14[0]),
    sizeof(expected_line15)/sizeof(expected_line15[0]),
    sizeof(expected_line16)/sizeof(expected_line16[0]),
    sizeof(expected_line17)/sizeof(expected_line17[0]),
    sizeof(expected_line18)/sizeof(expected_line18[0]),
    sizeof(expected_line19)/sizeof(expected_line19[0]),
    sizeof(expected_line20)/sizeof(expected_line20[0]),
};
static const size_t num_expected_lines = sizeof(expected_tokens)/sizeof(expected_tokens[0]);

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

    std::ifstream file("text/text8.txt");
    if (!file.is_open()) {
        fprintf(stderr, "Failed to open text/text8.txt\n");
        return 1;
    }

    printf("============================================================\n");
    printf("C++ Tokenizer Output (with automated verification)\n");
    printf("============================================================\n\n");

    std::string line;
    int line_num = 0;
    int failures = 0;
    while (std::getline(file, line)) {
        line_num++;
        auto tokens = tokenizer.encode(line.c_str());

        printf("Line %d: '%s'\n", line_num, line.c_str());
        printf("  Token count: %zu\n", tokens.size());
        printf("  Token IDs: [");
        for (size_t i = 0; i < tokens.size(); i++) {
            if (i > 0) printf(", ");
            printf("%d", tokens[i]);
        }
        printf("]\n");

        // Automated verification against Python reference
        if ((size_t)line_num <= num_expected_lines) {
            const size_t expected_count = expected_counts[line_num - 1];
            const int32_t * expected = expected_tokens[line_num - 1];
            bool match = (tokens.size() == expected_count);
            if (match) {
                for (size_t i = 0; i < expected_count; i++) {
                    if (tokens[i] != expected[i]) {
                        match = false;
                        break;
                    }
                }
            }
            if (match) {
                printf("  [PASS] Matches Python reference\n");
            } else {
                printf("  [FAIL] Mismatch with Python reference!\n");
                printf("  Expected: [");
                for (size_t i = 0; i < expected_count; i++) {
                    if (i > 0) printf(", ");
                    printf("%d", expected[i]);
                }
                printf("]\n");
                failures++;
            }
        } else {
            printf("  [SKIP] No reference data for this line\n");
        }
        printf("\n");
    }

    printf("============================================================\n");
    if (failures == 0) {
        printf("ALL %d LINES PASSED - Tokenization matches Python reference exactly.\n", line_num);
    } else {
        printf("FAILED: %d out of %d lines do not match Python reference.\n", failures, line_num);
    }
    printf("============================================================\n");

    return failures > 0 ? 1 : 0;
}