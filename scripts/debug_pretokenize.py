#!/usr/bin/env python3
"""
Quick script to examine pre-tokenizer offsets (which reveal the actual regex-matched spans).
"""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-TTS-12Hz-0.6B-Base", trust_remote_code=True)

# Get the raw pre-tokenizer output with offsets
pre_tokenizer = tokenizer.backend_tokenizer.pre_tokenizer

test_cases = [
    "123",
    "123456",
    "12345678",
    "Hello, world!",
    "it's",
    "0, 1, 2, 10, 100",
    "hello   world",
    "hello  ",
    "  hello",
    "你好",
    "第123号",
    "Finally, please read: 0, 1, 2, 10, 100, 3.14159, 99.9%",
]

for text in test_cases:
    result = pre_tokenizer.pre_tokenize_str(text)
    print(f"\nInput: {repr(text)}")
    for token, (start, end) in result:
        # Show the original text span
        original_span = text[start:end]
        print(f"  [{start:3d}:{end:3d}] byte_encoded={repr(token):30s} original={repr(original_span)}")