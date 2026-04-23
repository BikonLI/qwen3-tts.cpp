#include "text_tokenizer.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <sstream>
#include <cctype>

namespace qwen3_tts {

// ---------------------------------------------------------------------------
// Unicode property helpers for Qwen2 pre-tokenizer
// ---------------------------------------------------------------------------

uint32_t TextTokenizer::decode_utf8_codepoint(const std::string & text, size_t pos, size_t & byte_len) {
    if (pos >= text.size()) {
        byte_len = 0;
        return 0;
    }
    const unsigned char c0 = static_cast<unsigned char>(text[pos]);
    if ((c0 & 0x80) == 0) {
        byte_len = 1;
        return static_cast<uint32_t>(c0);
    }
    if ((c0 & 0xE0) == 0xC0) {
        byte_len = 2;
        if (pos + 1 >= text.size()) return static_cast<uint32_t>(c0);
        return (static_cast<uint32_t>(c0 & 0x1F) << 6) |
               (static_cast<uint32_t>(static_cast<unsigned char>(text[pos + 1]) & 0x3F));
    }
    if ((c0 & 0xF0) == 0xE0) {
        byte_len = 3;
        if (pos + 2 >= text.size()) return static_cast<uint32_t>(c0);
        return (static_cast<uint32_t>(c0 & 0x0F) << 12) |
               (static_cast<uint32_t>(static_cast<unsigned char>(text[pos + 1]) & 0x3F) << 6) |
               (static_cast<uint32_t>(static_cast<unsigned char>(text[pos + 2]) & 0x3F));
    }
    if ((c0 & 0xF8) == 0xF0) {
        byte_len = 4;
        if (pos + 3 >= text.size()) return static_cast<uint32_t>(c0);
        return (static_cast<uint32_t>(c0 & 0x07) << 18) |
               (static_cast<uint32_t>(static_cast<unsigned char>(text[pos + 1]) & 0x3F) << 12) |
               (static_cast<uint32_t>(static_cast<unsigned char>(text[pos + 2]) & 0x3F) << 6) |
               (static_cast<uint32_t>(static_cast<unsigned char>(text[pos + 3]) & 0x3F));
    }
    byte_len = 1;
    return static_cast<uint32_t>(c0);
}

bool TextTokenizer::is_unicode_letter(uint32_t cp) {
    // ASCII letters
    if ((cp >= 'A' && cp <= 'Z') || (cp >= 'a' && cp <= 'z')) return true;

    // Latin extensions
    if (cp >= 0x00C0 && cp <= 0x00FF) return true;   // Latin-1 Supplement letters
    if (cp >= 0x0100 && cp <= 0x024F) return true;   // Latin Extended-A/B
    if (cp >= 0x0250 && cp <= 0x02AF) return true;   // IPA Extensions (letters)
    if (cp >= 0x1E00 && cp <= 0x1EFF) return true;   // Latin Extended Additional

    // Greek
    if (cp >= 0x0370 && cp <= 0x03FF) return true;

    // Cyrillic
    if (cp >= 0x0400 && cp <= 0x04FF) return true;

    // CJK Unified Ideographs
    if (cp >= 0x4E00 && cp <= 0x9FFF) return true;

    // CJK Extension A
    if (cp >= 0x3400 && cp <= 0x4DBF) return true;

    // CJK Extension B (most of it)
    if (cp >= 0x20000 && cp <= 0x2A6DF) return true;

    // CJK Compatibility Ideographs
    if (cp >= 0xF900 && cp <= 0xFAFF) return true;

    // Hiragana
    if (cp >= 0x3040 && cp <= 0x309F) return true;

    // Katakana
    if (cp >= 0x30A0 && cp <= 0x30FF) return true;

    // Katakana Phonetic Extensions
    if (cp >= 0x31F0 && cp <= 0x31FF) return true;

    // Hangul Syllables
    if (cp >= 0xAC00 && cp <= 0xD7AF) return true;

    // Hangul Jamo
    if (cp >= 0x1100 && cp <= 0x11FF) return true;

    // Hangul Compatibility Jamo
    if (cp >= 0x3130 && cp <= 0x318F) return true;

    // Arabic
    if (cp >= 0x0600 && cp <= 0x06FF) return true;
    if (cp >= 0x0750 && cp <= 0x077F) return true;
    if (cp >= 0xFB50 && cp <= 0xFDFF) return true;
    if (cp >= 0xFE70 && cp <= 0xFEFF) return true;

    // Hebrew
    if (cp >= 0x0590 && cp <= 0x05FF) return true;

    // Devanagari
    if (cp >= 0x0900 && cp <= 0x097F) return true;

    // Thai
    if (cp >= 0x0E00 && cp <= 0x0E7F) return true;

    // Various other script ranges that Unicode classifies as Letter
    // Armenian
    if (cp >= 0x0530 && cp <= 0x058F) return true;
    // Bengali
    if (cp >= 0x0980 && cp <= 0x09FF) return true;
    // Georgian
    if (cp >= 0x10A0 && cp <= 0x10FF) return true;
    // Ethiopic
    if (cp >= 0x1200 && cp <= 0x137F) return true;
    // Tibetan
    if (cp >= 0x0F00 && cp <= 0x0FFF) return true;

    // Special: underscore is classified as Letter (\p{L}) in some regex flavors
    // but NOT in Unicode. Qwen2 pre-tokenizer uses \p{L} which is Unicode
    // Letter category. Underscore should NOT match.
    // The GPT-2 pre-tokenizer treats '_' as punctuation, not letter.

    // Full-width Latin letters
    if (cp >= 0xFF21 && cp <= 0xFF3A) return true;  // Ａ-Ｚ
    if (cp >= 0xFF41 && cp <= 0xFF5A) return true;  // ａ-ｚ

    return false;
}

bool TextTokenizer::is_unicode_number(uint32_t cp) {
    // \p{N} in Unicode means: Nd (Decimal_Number) + Nl (Letter_Number) + No (Other_Number)
    // ASCII digits are Nd
    if (cp >= '0' && cp <= '9') return true;

    // Full-width digits (Nd)
    if (cp >= 0xFF10 && cp <= 0xFF19) return true;

    // Arabic-Indic digits (Nd)
    if (cp >= 0x0660 && cp <= 0x0669) return true;
    // Extended Arabic-Indic digits (Nd)
    if (cp >= 0x06F0 && cp <= 0x06F9) return true;
    // Devanagari digits (Nd)
    if (cp >= 0x0966 && cp <= 0x096F) return true;

    // Nl (Letter_Number) category - Roman numerals, circled numbers, etc.
    if (cp >= 0x2160 && cp <= 0x2188) return true;  // Roman numerals

    // No (Other_Number) category - fractions, superscripts, subscripts
    if (cp >= 0x00B2 && cp <= 0x00B3) return true;  // superscript 2, 3
    if (cp == 0x00B9) return true;  // superscript 1
    if (cp >= 0x00BC && cp <= 0x00BE) return true;  // fractions ¼ ½ ¾
    if (cp >= 0x2070 && cp <= 0x2079) return true;  // superscript digits
    if (cp >= 0x2080 && cp <= 0x2089) return true;  // subscript digits

    // CJK ideographs like 一二三四五六七八九十百千万亿零 are \p{Lo} (Letter, Other)
    // NOT \p{N}. They should match \p{L} (letter pattern), not \p{N} (number pattern).
    // This is how the official Qwen2 tokenizer works - these characters would be
    // grouped with other letters in the [^\r\n\p{L}\p{N}]?\p{L}+ pattern.

    return false;
}

bool TextTokenizer::is_unicode_whitespace(uint32_t cp) {
    // Standard ASCII whitespace
    if (cp == ' ' || cp == '\t' || cp == '\n' || cp == '\r' ||
        cp == '\f' || cp == '\v') return true;

    // Unicode whitespace characters (Unicode Zs category + BOM)
    if (cp == 0x00A0) return true;  // NO-BREAK SPACE
    if (cp == 0x1680) return true;  // OGHAM SPACE MARK
    if (cp == 0x2000) return true;  // EN QUAD
    if (cp == 0x2001) return true;  // EM QUAD
    if (cp == 0x2002) return true;  // EN SPACE
    if (cp == 0x2003) return true;  // EM SPACE
    if (cp == 0x2004) return true;  // THREE-PER-EM SPACE
    if (cp == 0x2005) return true;  // FOUR-PER-EM SPACE
    if (cp == 0x2006) return true;  // SIX-PER-EM SPACE
    if (cp == 0x2007) return true;  // FIGURE SPACE
    if (cp == 0x2008) return true;  // PUNCTUATION SPACE
    if (cp == 0x2009) return true;  // THIN SPACE
    if (cp == 0x200A) return true;  // HAIR SPACE
    if (cp == 0x202F) return true;  // NARROW NO-BREAK SPACE
    if (cp == 0x205F) return true;  // MEDIUM MATHEMATICAL SPACE
    if (cp == 0x3000) return true;  // IDEOGRAPHIC SPACE
    if (cp == 0xFEFF) return true;  // ZERO WIDTH NO-BREAK SPACE (BOM)

    return false;
}

bool TextTokenizer::is_unicode_newline(uint32_t cp) {
    return cp == '\n' || cp == '\r' ||
           cp == 0x0085 ||  // NEXT LINE (NEL)
           cp == 0x2028 ||  // LINE SEPARATOR
           cp == 0x2029;    // PARAGRAPH SEPARATOR
}

// ---------------------------------------------------------------------------
// Qwen2 regex pre-tokenizer
// ---------------------------------------------------------------------------

std::vector<std::string> TextTokenizer::pre_tokenize_qwen2(const std::string & text) const {
    // Qwen2 pre-tokenizer regex pattern (extracted from HuggingFace tokenizer config):
    //   (?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
    //
    // Key differences from common assumptions:
    //   - \p{N} matches EXACTLY ONE digit (not {1,3})
    //   - \s*[\r\n]+ requires ONE OR MORE newlines (not zero or more)
    //
    // We implement this as a scanner that tries each alternative at each position
    // in priority order, advancing through the input.

    std::vector<std::string> tokens;
    if (text.empty()) return tokens;

    size_t pos = 0;
    const size_t len = text.size();

    while (pos < len) {
        size_t cp_len = 0;
        uint32_t cp = decode_utf8_codepoint(text, pos, cp_len);

        // ---- Attempt 1: Contractions (?i:'s|'t|'re|'ve|'m|'ll|'d) ----
        // Match apostrophe (ASCII ' or Unicode right single quotation mark) followed by
        // one of: s, t, re, ve, m, ll, d (case-insensitive)
        if (cp == '\'' || cp == 0x2019) {
            const size_t apos_end = pos + cp_len;
            if (apos_end < len) {
                size_t next_len = 0;
                uint32_t next_cp = decode_utf8_codepoint(text, apos_end, next_len);
                const uint32_t n1 = (next_cp >= 'A' && next_cp <= 'Z') ? (next_cp + 32) : next_cp;
                // Two-char contractions: 's 't 'm 'd
                size_t match_len = 0;
                if (n1 == 's' || n1 == 't' || n1 == 'm' || n1 == 'd') {
                    match_len = cp_len + next_len;
                }
                // Three-char contractions: 're 've
                if (match_len == 0 && (n1 == 'r' || n1 == 'v')) {
                    const size_t after2 = apos_end + next_len;
                    if (after2 < len) {
                        size_t cp2_len = 0;
                        uint32_t cp2 = decode_utf8_codepoint(text, after2, cp2_len);
                        const uint32_t n2 = (cp2 >= 'A' && cp2 <= 'Z') ? (cp2 + 32) : cp2;
                        if ((n1 == 'r' && n2 == 'e') || (n1 == 'v' && n2 == 'e')) {
                            match_len = cp_len + next_len + cp2_len;
                        }
                    }
                }
                // Three-char contraction: 'll (two Ls)
                if (match_len == 0 && n1 == 'l') {
                    const size_t after2 = apos_end + next_len;
                    if (after2 < len) {
                        size_t cp2_len = 0;
                        uint32_t cp2 = decode_utf8_codepoint(text, after2, cp2_len);
                        const uint32_t n2 = (cp2 >= 'A' && cp2 <= 'Z') ? (cp2 + 32) : cp2;
                        if (n2 == 'l') {
                            match_len = cp_len + next_len + cp2_len;
                        }
                    }
                }
                if (match_len > 0) {
                    tokens.push_back(text.substr(pos, match_len));
                    pos += match_len;
                    continue;
                }
            }
        }

        // ---- Attempt 2: [^\r\n\p{L}\p{N}]?\p{L}+ ----
        // Optional non-letter/non-number/non-newline prefix, then 1+ letters
        {
            size_t try_pos = pos;

            if (cp_len > 0 && !is_unicode_newline(cp) && !is_unicode_letter(cp) && !is_unicode_number(cp)) {
                // This character could be the optional prefix, but only if
                // the next character is a letter
                size_t next_len = 0;
                uint32_t next_cp = decode_utf8_codepoint(text, pos + cp_len, next_len);
                if (is_unicode_letter(next_cp)) {
                    try_pos += cp_len;  // consume the optional prefix
                }
            }

            size_t check_pos = try_pos;
            size_t letter_len = 0;
            uint32_t letter_cp = decode_utf8_codepoint(text, check_pos, letter_len);

            if (is_unicode_letter(letter_cp)) {
                size_t end_pos = check_pos + letter_len;
                while (end_pos < len) {
                    size_t next_len = 0;
                    uint32_t next_cp = decode_utf8_codepoint(text, end_pos, next_len);
                    if (!is_unicode_letter(next_cp)) break;
                    end_pos += next_len;
                }
                tokens.push_back(text.substr(pos, end_pos - pos));
                pos = end_pos;
                continue;
            }
        }

        // ---- Attempt 3: \p{N} ----
        // Single number character (official Qwen2 regex uses \p{N}, not \p{N}{1,3})
        {
            if (is_unicode_number(cp)) {
                tokens.push_back(text.substr(pos, cp_len));
                pos += cp_len;
                continue;
            }
        }

        // ---- Attempt 4:  ?[^\s\p{L}\p{N}]+[\r\n]* ----
        // Optional space, then 1+ punctuation/symbol chars, then optional newlines
        {
            size_t try_pos = pos;

            if (cp == ' ') {
                try_pos += cp_len;
            }

            // Must find at least one [^\s\p{L}\p{N}] at try_pos
            size_t p_len = 0;
            uint32_t p_cp = decode_utf8_codepoint(text, try_pos, p_len);

            if (p_len > 0 && !is_unicode_whitespace(p_cp) && !is_unicode_letter(p_cp) && !is_unicode_number(p_cp)) {
                // Found punctuation/symbol start - consume all [^\s\p{L}\p{N}]
                size_t end_pos = try_pos + p_len;
                while (end_pos < len) {
                    size_t next_len = 0;
                    uint32_t next_cp = decode_utf8_codepoint(text, end_pos, next_len);
                    if (is_unicode_whitespace(next_cp) || is_unicode_letter(next_cp) || is_unicode_number(next_cp)) break;
                    end_pos += next_len;
                }

                // Consume optional trailing newlines [\r\n]*
                while (end_pos < len) {
                    size_t nl_len = 0;
                    uint32_t nl_cp = decode_utf8_codepoint(text, end_pos, nl_len);
                    if (nl_cp == '\n' || nl_cp == '\r') {
                        end_pos += nl_len;
                    } else {
                        break;
                    }
                }

                tokens.push_back(text.substr(pos, end_pos - pos));
                pos = end_pos;
                continue;
            }
        }

        // ---- Attempt 5: \s*[\r\n]+ ----
        // Optional whitespace (non-newline) then one or more newlines
        // The official Qwen2 regex uses \s*[\r\n]+ (not \s*[\r\n])
        {
            size_t try_pos = pos;
            // Consume optional whitespace (not newlines)
            while (try_pos < len) {
                size_t ws_len = 0;
                uint32_t ws_cp = decode_utf8_codepoint(text, try_pos, ws_len);
                if (is_unicode_whitespace(ws_cp) && !is_unicode_newline(ws_cp)) {
                    try_pos += ws_len;
                } else {
                    break;
                }
            }
            // Must have at least one newline [\r\n]
            size_t nl_len = 0;
            uint32_t nl_cp = decode_utf8_codepoint(text, try_pos, nl_len);
            if (is_unicode_newline(nl_cp) && (nl_cp == '\r' || nl_cp == '\n')) {
                try_pos += nl_len;
                // Handle CRLF
                if (nl_cp == '\r' && try_pos < len) {
                    size_t lf_len = 0;
                    uint32_t lf_cp = decode_utf8_codepoint(text, try_pos, lf_len);
                    if (lf_cp == '\n') {
                        try_pos += lf_len;
                    }
                }
                // Consume additional newlines ([\r\n]+ means one or more)
                while (try_pos < len) {
                    size_t extra_nl_len = 0;
                    uint32_t extra_nl_cp = decode_utf8_codepoint(text, try_pos, extra_nl_len);
                    if (extra_nl_cp == '\r' || extra_nl_cp == '\n') {
                        try_pos += extra_nl_len;
                        // Handle CRLF for additional newlines too
                        if (extra_nl_cp == '\r' && try_pos < len) {
                            size_t extra_lf_len = 0;
                            uint32_t extra_lf_cp = decode_utf8_codepoint(text, try_pos, extra_lf_len);
                            if (extra_lf_cp == '\n') {
                                try_pos += extra_lf_len;
                            }
                        }
                    } else {
                        break;
                    }
                }
                tokens.push_back(text.substr(pos, try_pos - pos));
                pos = try_pos;
                continue;
            }
        }

// ---- Attempt 6: \s+(?!\S) ----
        // Match whitespace NOT directly followed by non-whitespace.
        // This handles trailing whitespace and whitespace between other whitespace.
        // We need regex-style backtracking: greedily consume all whitespace,
        // then shrink from the end until (?!\S) succeeds.
        {
            if (is_unicode_whitespace(cp)) {
                // Greedily consume all whitespace
                size_t end_pos = pos + cp_len;
                while (end_pos < len) {
                    size_t ws_len = 0;
                    uint32_t ws_cp = decode_utf8_codepoint(text, end_pos, ws_len);
                    if (!is_unicode_whitespace(ws_cp)) break;
                    end_pos += ws_len;
                }

                // Try from longest to shortest, finding the longest span where
                // (?!\S) succeeds (position after match is end-of-string or whitespace)
                size_t try_end = end_pos;
                while (try_end > pos) {
                    if (try_end >= len) {
                        // End of string: (?!\S) always succeeds
                        tokens.push_back(text.substr(pos, try_end - pos));
                        pos = try_end;
                        goto next_position;
                    }
                    size_t after_len = 0;
                    uint32_t after_cp = decode_utf8_codepoint(text, try_end, after_len);
                    if (is_unicode_whitespace(after_cp)) {
                        // Next char is whitespace: (?!\S) succeeds
                        tokens.push_back(text.substr(pos, try_end - pos));
                        pos = try_end;
                        goto next_position;
                    }
                    // Next char is \S: (?!\S) fails.
                    // Shrink by one whitespace character from the end.
                    // Walk back from try_end to find the start of the last codepoint.
                    size_t back = try_end - 1;
                    while (back > pos && ((unsigned char)text[back] & 0xC0) == 0x80) {
                        back--;
                    }
                    try_end = back;
                }
// Could not find a match for \s+(?!\S), fall through to attempt 7.
            }
        }

        // ---- Attempt 7: \s+ ----
        // One or more whitespace characters
        {
            if (is_unicode_whitespace(cp)) {
                size_t end_pos = pos + cp_len;
                while (end_pos < len) {
                    size_t ws_len = 0;
                    uint32_t ws_cp = decode_utf8_codepoint(text, end_pos, ws_len);
                    if (!is_unicode_whitespace(ws_cp)) break;
                    end_pos += ws_len;
                }
                tokens.push_back(text.substr(pos, end_pos - pos));
                pos = end_pos;
                continue;
            }
        }

        // ---- Fallback: single character ----
        // Handles any unrecognized character. This should not normally trigger
        // if the regex is complete, but serves as a safety net.
        {
            tokens.push_back(text.substr(pos, cp_len > 0 ? cp_len : 1));
            pos += (cp_len > 0 ? cp_len : 1);
        }

        next_position:;
    }

    return tokens;
}

// GPT-2 byte-to-unicode mapping
// Maps bytes 0-255 to unicode characters to avoid control characters
static const char * BYTE_TO_UNICODE[256] = {
    "Ā", "ā", "Ă", "ă", "Ą", "ą", "Ć", "ć", "Ĉ", "ĉ", "Ċ", "ċ", "Č", "č", "Ď", "ď",
    "Đ", "đ", "Ē", "ē", "Ĕ", "ĕ", "Ė", "ė", "Ę", "ę", "Ě", "ě", "Ĝ", "ĝ", "Ğ", "ğ",
    "Ġ", "!", "\"", "#", "$", "%", "&", "'", "(", ")", "*", "+", ",", "-", ".", "/",
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", ":", ";", "<", "=", ">", "?",
    "@", "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O",
    "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "[", "\\", "]", "^", "_",
    "`", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o",
    "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z", "{", "|", "}", "~", "ġ",
    "Ģ", "ģ", "Ĥ", "ĥ", "Ħ", "ħ", "Ĩ", "ĩ", "Ī", "ī", "Ĭ", "ĭ", "Į", "į", "İ", "ı",
    "Ĳ", "ĳ", "Ĵ", "ĵ", "Ķ", "ķ", "ĸ", "Ĺ", "ĺ", "Ļ", "ļ", "Ľ", "ľ", "Ŀ", "ŀ", "Ł",
    "ł", "¡", "¢", "£", "¤", "¥", "¦", "§", "¨", "©", "ª", "«", "¬", "Ń", "®", "¯",
    "°", "±", "²", "³", "´", "µ", "¶", "·", "¸", "¹", "º", "»", "¼", "½", "¾", "¿",
    "À", "Á", "Â", "Ã", "Ä", "Å", "Æ", "Ç", "È", "É", "Ê", "Ë", "Ì", "Í", "Î", "Ï",
    "Ð", "Ñ", "Ò", "Ó", "Ô", "Õ", "Ö", "×", "Ø", "Ù", "Ú", "Û", "Ü", "Ý", "Þ", "ß",
    "à", "á", "â", "ã", "ä", "å", "æ", "ç", "è", "é", "ê", "ë", "ì", "í", "î", "ï",
    "ð", "ñ", "ò", "ó", "ô", "õ", "ö", "÷", "ø", "ù", "ú", "û", "ü", "ý", "þ", "ÿ"
};

// Build reverse mapping at runtime
static std::unordered_map<std::string, uint8_t> build_unicode_to_byte() {
    std::unordered_map<std::string, uint8_t> result;
    for (int i = 0; i < 256; i++) {
        result[BYTE_TO_UNICODE[i]] = (uint8_t)i;
    }
    return result;
}

static const std::unordered_map<std::string, uint8_t> UNICODE_TO_BYTE = build_unicode_to_byte();

static size_t utf8_symbol_len(char c) {
    const unsigned char uc = (unsigned char)c;
    if ((uc & 0x80u) == 0) return 1;
    if ((uc & 0xE0u) == 0xC0u) return 2;
    if ((uc & 0xF0u) == 0xE0u) return 3;
    if ((uc & 0xF8u) == 0xF0u) return 4;
    return 1;
}

static bool unicode_symbol_to_byte(const std::string & symbol, uint8_t & out) {
    auto it = UNICODE_TO_BYTE.find(symbol);
    if (it == UNICODE_TO_BYTE.end()) {
        return false;
    }
    out = it->second;
    return true;
}

static std::vector<std::string> split_unicode_symbols(const std::string & text) {
    std::vector<std::string> symbols;
    symbols.reserve(text.size());

    size_t i = 0;
    while (i < text.size()) {
        const size_t len = utf8_symbol_len(text[i]);
        symbols.push_back(text.substr(i, len));
        i += len;
    }

    return symbols;
}

TextTokenizer::TextTokenizer() = default;

TextTokenizer::~TextTokenizer() = default;

size_t TextTokenizer::utf8_len(char c) {
    if ((c & 0x80) == 0) return 1;
    if ((c & 0xE0) == 0xC0) return 2;
    if ((c & 0xF0) == 0xE0) return 3;
    if ((c & 0xF8) == 0xF0) return 4;
    return 1; // Invalid UTF-8, treat as single byte
}

std::string TextTokenizer::bytes_to_unicode(const std::string & text) {
    std::string result;
    for (unsigned char c : text) {
        result += BYTE_TO_UNICODE[c];
    }
    return result;
}

std::string TextTokenizer::unicode_to_bytes(const std::string & text) {
    std::string result;
    size_t i = 0;
    while (i < text.size()) {
        size_t len = utf8_len(text[i]);
        std::string ch = text.substr(i, len);
        auto it = UNICODE_TO_BYTE.find(ch);
        if (it != UNICODE_TO_BYTE.end()) {
            result += (char)it->second;
        } else {
            // Not in mapping, keep as-is (shouldn't happen for valid tokens)
            result += ch;
        }
        i += len;
    }
    return result;
}

bool TextTokenizer::load_from_gguf(struct gguf_context * ctx) {
    if (!ctx) {
        error_msg_ = "GGUF context is null";
        return false;
    }
    
    // Get vocabulary
    int64_t tokens_key = gguf_find_key(ctx, "tokenizer.ggml.tokens");
    if (tokens_key < 0) {
        error_msg_ = "tokenizer.ggml.tokens not found in GGUF";
        return false;
    }
    
    size_t n_vocab = gguf_get_arr_n(ctx, tokens_key);
    if (n_vocab == 0) {
        error_msg_ = "Empty vocabulary";
        return false;
    }
    
    config_.vocab_size = (int32_t)n_vocab;
    id_to_token_.resize(n_vocab);
    
    for (size_t i = 0; i < n_vocab; i++) {
        const char * token = gguf_get_arr_str(ctx, tokens_key, i);
        if (token) {
            id_to_token_[i] = token;
            vocab_[token] = (int32_t)i;
        }
    }
    
    // Get merges
    int64_t merges_key = gguf_find_key(ctx, "tokenizer.ggml.merges");
    if (merges_key >= 0) {
        size_t n_merges = gguf_get_arr_n(ctx, merges_key);
        for (size_t i = 0; i < n_merges; i++) {
            const char * merge = gguf_get_arr_str(ctx, merges_key, i);
            if (merge) {
                std::string merge_str(merge);
                // Parse "token1 token2" format
                size_t space_pos = merge_str.find(' ');
                if (space_pos != std::string::npos) {
                    std::string first = merge_str.substr(0, space_pos);
                    std::string second = merge_str.substr(space_pos + 1);
                    bpe_ranks_[{first, second}] = (int32_t)i;
                }
            }
        }
    }
    
    // Get special token IDs (optional, use defaults if not found)
    int64_t bos_key = gguf_find_key(ctx, "tokenizer.ggml.bos_token_id");
    if (bos_key >= 0) {
        config_.bos_token_id = (int32_t)gguf_get_val_u32(ctx, bos_key);
    }
    
    int64_t eos_key = gguf_find_key(ctx, "tokenizer.ggml.eos_token_id");
    if (eos_key >= 0) {
        config_.eos_token_id = (int32_t)gguf_get_val_u32(ctx, eos_key);
    }
    
    int64_t pad_key = gguf_find_key(ctx, "tokenizer.ggml.padding_token_id");
    if (pad_key >= 0) {
        config_.pad_token_id = (int32_t)gguf_get_val_u32(ctx, pad_key);
    }
    
    // Find special tokens by content
    auto find_token = [this](const std::string & text) -> int32_t {
        auto it = vocab_.find(text);
        return (it != vocab_.end()) ? it->second : -1;
    };
    
    assistant_token_id_ = find_token("assistant");
    if (assistant_token_id_ < 0) {
        // Try with space prefix (GPT-2 style)
        assistant_token_id_ = find_token("Ġassistant");
    }

    user_token_id_ = find_token("user");
    if (user_token_id_ < 0) {
        user_token_id_ = find_token("Ġuser");
    }
    
    // Newline token
    newline_token_id_ = find_token("Ċ");  // GPT-2 encoding for '\n'
    if (newline_token_id_ < 0) {
        newline_token_id_ = find_token("\n");
    }
    
    // Read pre-tokenizer type from GGUF metadata
    int64_t pre_key = gguf_find_key(ctx, "tokenizer.ggml.pre");
    if (pre_key >= 0) {
        const char * pre_type = gguf_get_val_str(ctx, pre_key);
        if (pre_type && std::strcmp(pre_type, "qwen2") == 0) {
            pre_tokenize_type_ = pre_tokenize_type::qwen2;
            fprintf(stderr, "  Text tokenizer: using Qwen2 pre-tokenizer\n");
        } else if (pre_type) {
            fprintf(stderr, "  Text tokenizer: unknown pre-tokenizer type '%s', using none\n", pre_type);
        }
    } else {
        fprintf(stderr, "  Text tokenizer: no pre-tokenizer metadata found, using none\n");
    }
    
    loaded_ = true;
    return true;
}

std::pair<std::string, std::string> TextTokenizer::get_min_pair(
    const std::vector<std::string> & word) const {
    
    std::pair<std::string, std::string> min_pair;
    int32_t min_rank = std::numeric_limits<int32_t>::max();
    
    for (size_t i = 0; i + 1 < word.size(); i++) {
        auto pair = std::make_pair(word[i], word[i + 1]);
        auto it = bpe_ranks_.find(pair);
        if (it != bpe_ranks_.end() && it->second < min_rank) {
            min_rank = it->second;
            min_pair = pair;
        }
    }
    
    return min_pair;
}

std::vector<std::string> TextTokenizer::bpe(const std::string & token) const {
    if (token.empty()) {
        return {};
    }
    
    // Split into unicode characters
    std::vector<std::string> word;
    size_t i = 0;
    while (i < token.size()) {
        size_t len = utf8_len(token[i]);
        word.push_back(token.substr(i, len));
        i += len;
    }
    
    if (word.size() == 1) {
        return word;
    }
    
    // Iteratively merge pairs
    while (true) {
        auto min_pair = get_min_pair(word);
        if (min_pair.first.empty()) {
            break;  // No more merges possible
        }
        
        // Merge all occurrences of the pair
        std::vector<std::string> new_word;
        size_t j = 0;
        while (j < word.size()) {
            if (j + 1 < word.size() && 
                word[j] == min_pair.first && 
                word[j + 1] == min_pair.second) {
                new_word.push_back(min_pair.first + min_pair.second);
                j += 2;
            } else {
                new_word.push_back(word[j]);
                j += 1;
            }
        }
        word = std::move(new_word);
        
        if (word.size() == 1) {
            break;
        }
    }
    
    return word;
}

std::vector<int32_t> TextTokenizer::encode_unknown_bpe_token_bytes(const std::string & token) const {
    std::vector<int32_t> out;
    if (token.empty()) {
        return out;
    }

    const std::vector<std::string> symbols = split_unicode_symbols(token);
    out.reserve(symbols.size());

    for (const std::string & symbol : symbols) {
        uint8_t byte = 0;
        if (!unicode_symbol_to_byte(symbol, byte)) {
            continue;
        }
        const std::string byte_tok = BYTE_TO_UNICODE[byte];
        const auto it = vocab_.find(byte_tok);
        if (it != vocab_.end()) {
            out.push_back(it->second);
        }
    }

    return out;
}

std::vector<std::string> TextTokenizer::pre_tokenize(const std::string & text) const {
    if (pre_tokenize_type_ == pre_tokenize_type::qwen2) {
        return pre_tokenize_qwen2(text);
    }
    // No pre-tokenizer configured; return the whole text as a single token
    return {text};
}

std::vector<int32_t> TextTokenizer::encode(const std::string & text) const {
    if (!loaded_) {
        return {};
    }
    
    std::vector<int32_t> tokens;

    if (pre_tokenize_type_ == pre_tokenize_type::qwen2) {
        // Qwen2 pre-tokenizer: split raw UTF-8 text using regex-based pre-tokenization,
        // then apply bytes_to_unicode and BPE to each pre-token separately.
        // This matches the HuggingFace Qwen2TokenizerFast behavior exactly.
        std::vector<std::string> pre_tokens = pre_tokenize_qwen2(text);

        for (const auto & pre_token : pre_tokens) {
            // Convert each pre-token to GPT-2 unicode representation
            std::string unicode_token = bytes_to_unicode(pre_token);

            // BPE encode
            auto bpe_tokens = bpe(unicode_token);
            for (const auto & tok : bpe_tokens) {
                auto it = vocab_.find(tok);
                if (it != vocab_.end()) {
                    tokens.push_back(it->second);
                } else {
                    const std::vector<int32_t> fallback = encode_unknown_bpe_token_bytes(tok);
                    tokens.insert(tokens.end(), fallback.begin(), fallback.end());
                }
            }
        }
    } else {
        // Fall back to simple space-split (original behavior)
        std::string unicode_text = bytes_to_unicode(text);

        std::vector<std::string> words;
        std::string current_word;

        size_t i = 0;
        while (i < unicode_text.size()) {
            size_t len = utf8_len(unicode_text[i]);
            std::string ch = unicode_text.substr(i, len);

            if (ch == "Ġ") {
                if (!current_word.empty()) {
                    words.push_back(current_word);
                    current_word.clear();
                }
                current_word = ch;
            } else {
                current_word += ch;
            }
            i += len;
        }
        if (!current_word.empty()) {
            words.push_back(current_word);
        }

        for (const auto & word : words) {
            auto bpe_tokens = bpe(word);
            for (const auto & tok : bpe_tokens) {
                auto it = vocab_.find(tok);
                if (it != vocab_.end()) {
                    tokens.push_back(it->second);
                } else {
                    const std::vector<int32_t> fallback = encode_unknown_bpe_token_bytes(tok);
                    tokens.insert(tokens.end(), fallback.begin(), fallback.end());
                }
            }
        }
    }

    return tokens;
}

std::vector<int32_t> TextTokenizer::encode_for_tts(const std::string & text,
                                                   const std::string & instruct,
                                                   const std::string & speaker,
                                                   const std::string & reference_text) const {
    if (!loaded_) {
        return {};
    }
    
    // Keep assistant text prompt separate from instruct prompt.
    // In official pipeline, instruct is passed as a separate input branch.
    std::vector<int32_t> tokens;
    (void) instruct;
    (void) speaker;
    (void) reference_text;
    
    // <|im_start|>
    tokens.push_back(config_.bos_token_id);
    
    // assistant
    tokens.push_back(assistant_token_id_);
    
    // \n
    tokens.push_back(newline_token_id_);
    
    // Encode the transcript section only
    auto text_tokens = encode(text);
    tokens.insert(tokens.end(), text_tokens.begin(), text_tokens.end());
    
    // <|im_end|>
    tokens.push_back(config_.eos_token_id);
    
    // \n
    tokens.push_back(newline_token_id_);
    
    // <|im_start|>
    tokens.push_back(config_.bos_token_id);
    
    // assistant
    tokens.push_back(assistant_token_id_);
    
    // \n
    tokens.push_back(newline_token_id_);
    
    return tokens;
}

std::vector<int32_t> TextTokenizer::encode_instruct_for_tts(const std::string & instruct) const {
    if (!loaded_ || instruct.empty() || user_token_id_ < 0) {
        return {};
    }

    // Format: <|im_start|>user\n{instruct}<|im_end|>\n
    std::vector<int32_t> tokens;
    tokens.push_back(config_.bos_token_id);
    tokens.push_back(user_token_id_);
    tokens.push_back(newline_token_id_);

    auto instruct_tokens = encode(instruct);
    tokens.insert(tokens.end(), instruct_tokens.begin(), instruct_tokens.end());

    tokens.push_back(config_.eos_token_id);
    tokens.push_back(newline_token_id_);
    return tokens;
}

std::string TextTokenizer::decode(const std::vector<int32_t> & tokens) const {
    std::string result;
    for (int32_t token : tokens) {
        result += decode_token(token);
    }
    return result;
}

std::string TextTokenizer::decode_token(int32_t token_id) const {
    if (token_id < 0 || token_id >= (int32_t)id_to_token_.size()) {
        return "";
    }
    
    const std::string & token = id_to_token_[token_id];
    
    // Convert from GPT-2 unicode back to bytes
    return unicode_to_bytes(token);
}

} // namespace qwen3_tts
