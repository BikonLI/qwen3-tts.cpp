#!/usr/bin/env python3
"""Compare C++ tokenizer output with Python reference for the exact user text."""
import sys
import subprocess
import json

from transformers import AutoTokenizer

# The exact text from user's request
text = "please read the following numbers and symbols correctly: 0, 1, 2, 10, 100, 1,000, 3.14159, 99.9%, and $42.50."

# Get Python reference
py_tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-TTS-12Hz-0.6B-Base", trust_remote_code=True)
py_tokens = py_tok.encode(text, add_special_tokens=False)
py_decoded = [py_tok.decode([t]) for t in py_tokens]

print("=== Python Reference ===")
print(f"Tokens ({len(py_tokens)}):")
for i, (tid, dec) in enumerate(zip(py_tokens, py_decoded)):
    print(f"  [{i:2d}] {tid:5d}  '{dec}'")

print()

# Run C++ tokenizer via test binary
model_path = "models/gguf/0.6b-base/qwen3-tts-12hz-0.6b-base-f16.gguf"
result = subprocess.run(
    ["build/Release/test_tokenizer.exe", "--model", model_path],
    capture_output=True, text=True
)

# Parse C++ output
# The test binary prints tokens for "Hello." and a numeric sentence
# We need to modify it to print our specific text, or use CLI
# For now, let's just compare the last few tokens of the numeric_text test

# Actually, let me write a quick C++ program to get exact output
# For now, compare using the encode function directly through a simple C++ program

print("=== Need C++ output for comparison ===")
print("Python last 10 tokens (the $42.50 part):")
for i in range(len(py_tokens)-10, len(py_tokens)):
    print(f"  [{i:2d}] {py_tokens[i]:5d}  '{py_decoded[i]}'")

print()

# Verify the specific problematic substring
print("=== Substring '$42.50' ===")
sub_tokens = py_tok.encode("$42.50", add_special_tokens=False)
sub_decoded = [py_tok.decode([t]) for t in sub_tokens]
print(f"Python: IDs={sub_tokens}  decoded={sub_decoded}")

# Check if there's a special token for '$'  
print()
print("=== Checking vocab for $ ===")
# Qwen2 tokenizer vocab check
tid_dollar = py_tok.convert_tokens_to_ids("$")
print(f"Token '$' -> ID {tid_dollar}")
tid_dollar_space = py_tok.convert_tokens_to_ids(" $")
print(f"Token ' $' -> ID {tid_dollar_space}")

# The BPE merge might combine certain patterns
print()
print("=== BPE-level analysis ===")
# Use the pre-tokenizer directly
from tokenizers import pre_tokenizers
pt = py_tok.backend_tokenizer.pre_tokenizer
result = pt.pre_tokenize_str("$42.50")
print("Pre-tokenized chunks (with offsets):")
for token, (start, end) in result:
    print(f"  {repr(token):20s}  [{start:2d}:{end:2d}]  original={repr(text[start:end]) if start < len(text) else 'N/A'}")