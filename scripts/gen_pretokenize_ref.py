#!/usr/bin/env python3
"""
Generate reference pre-tokenization output for C++ verification.

Outputs a JSON file with test cases and expected pre-tokenized chunks
(where each chunk is the ORIGINAL text span, not byte-encoded).

Usage:
    python scripts/gen_pretokenize_ref.py > reference/pretokenize_reference.json
"""
import sys
import io
import json

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-TTS-12Hz-0.6B-Base", trust_remote_code=True)
pre_tokenizer = tokenizer.backend_tokenizer.pre_tokenizer

# Comprehensive test cases covering all regex branches
test_cases = [
    # Basic text (letter pattern: [^\r\n\p{L}\p{N}]?\p{L}+)
    "Hello",
    "Hello world",
    
    # Single digits (\p{N} - exactly one digit)
    "123",
    "123456",
    "12345678",
    
    # Numbers with punctuation
    "3.14159",
    "0, 1, 2, 10, 100, 1,000",
    "99.9%",
    "$42.50",
    "2024-01-15",
    
    # Contractions (?i:'s|'t|'re|'ve|'m|'ll|'d)
    "it's",
    "I'm",
    "they're",
    "we've",
    "you'll",
    "he'd",
    "IT'S",
    "I'VE",
    "that's good",
    "don't",
    
    # Punctuation ( ?[^\s\p{L}\p{N}]+[\r\n]*)
    "Hello.",
    "Hello, world!",
    "a,b,c",
    
    # CJK characters (should match \p{L}+)
    "你好",
    "这是一个测试",
    "中文English混合",
    "一二三四五六七八九十",
    
    # Mixed CJK and numbers (第123号 pattern)
    "第123号",
    "Area 51",
    
    # Whitespace patterns
    "hello   world",  # multiple spaces between words
    "hello\nworld",    # newline
    "hello\r\nworld",  # CRLF
    "hello  ",         # trailing spaces (\s+(?!\S))
    "  hello",         # leading spaces
    "a  b  c",         # spaces between letters
    
    # Complex mixed text (the original bug case)
    "Finally, please read the following numbers: 0, 1, 2, 10, 100, 3.14159, 99.9%, and $42.50.",
    "The price is $42.50, and 30% off!",
    
    # Unicode letters
    "café",
    "naïve",
    "résumé",
    
    # Number-edge cases
    "1,000,000",
    "0",
    "42",
    "100",
    
    # Pure whitespace
    "   ",
    "\n",
    "\r\n",
    "\n\n\n",
    
    # Empty and single char
    "",
    "a",
    "1",
    ".",
]

results = []
for text in test_cases:
    if not text:
        # Empty string
        results.append({"input": text, "chunks": [], "offsets": []})
        continue
    
    pre_result = pre_tokenizer.pre_tokenize_str(text)
    chunks = [text[start:end] for token, (start, end) in pre_result]
    offsets = [(start, end) for token, (start, end) in pre_result]
    
    results.append({
        "input": text,
        "chunks": chunks,
        "offsets": offsets,
    })

print(json.dumps(results, ensure_ascii=False, indent=2))