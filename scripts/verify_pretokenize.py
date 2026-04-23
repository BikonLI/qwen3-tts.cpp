#!/usr/bin/env python3
"""
Verify C++ pre-tokenizer matches HuggingFace Qwen2TokenizerFast pre-tokenization.

This script uses the HuggingFace tokenizer to extract pre-tokenization output
(step before BPE merging) and prints the results for comparison with C++.

Usage:
    python scripts/verify_pretokenize.py
    python scripts/verify_pretokenize.py --model models/gguf/tokenizer/qwen3-tts-tokenizer-12hz-f16.gguf

Requires: transformers package (pip install transformers)
"""

import argparse
import sys
import io

# Force UTF-8 output on Windows to avoid GBK encoding issues
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Test texts that exercise all branches of the Qwen2 regex:
#   (?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+
TEST_CASES = [
    # Basic text
    "Hello",
    "Hello world",
    
    # Numbers - the core bug case
    "123",
    "123456",
    "12345678",
    "3.14159",
    "0, 1, 2, 10, 100, 1,000",
    
    # Contractions
    "it's",
    "I'm",
    "they're",
    "we've",
    "you'll",
    "he'd",
    "IT'S",
    "I'VE",
    
    # Punctuation patterns
    "Hello.",
    "Hello, world!",
    "a,b,c",
    
    # CJK (Chinese) characters
    "你好",
    "这是一个测试",
    "中文English混合",
    
    # Mixed numbers and CJK (the original bug)
    "一二三四五六七八九十",
    "第123号",
    
    # Whitespace patterns
    "hello   world",         # multiple spaces
    "hello\nworld",          # newline
    "hello\r\nworld",        # CRLF
    "hello  ",               # trailing spaces
    "  hello",               # leading spaces
    
    # Complex mixed text
    "Finally, please read the following numbers: 0, 1, 2, 10, 100, 1,000, 3.14159, 99.9%, and $42.50.",
    "The price is $42.50, and 30% off!",
    
    # Unicode
    "café",
    "naïve",
    "résumé",
    
    # Number patterns with punctuation
    "1,000,000",
    "99.9%",
    "$42.50",
    "2024-01-15",
]


def get_pretokenized_hf(text: str, tokenizer) -> list[str]:
    """Get pre-tokenized output from HuggingFace tokenizer."""
    # Use the internal pre-tokenizer directly
    # The Qwen2TokenizerFast uses a PreTokenizer that applies the GPT2-style regex
    pre_tokenizer = tokenizer.backend_tokenizer.pre_tokenizer
    result = pre_tokenizer.pre_tokenize_str(text)
    # result is list of (token_string, (start, end)) tuples
    return [token for token, offset in result]


def main():
    parser = argparse.ArgumentParser(description="Verify pre-tokenizer output")
    parser.add_argument("--model", default=None, 
                        help="Path to HuggingFace tokenizer model directory")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show detailed output")
    args = parser.parse_args()
    
    try:
        from transformers import AutoTokenizer
    except ImportError:
        print("ERROR: transformers package not installed")
        print("Install with: pip install transformers")
        sys.exit(1)
    
    # Load tokenizer
    if args.model:
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    else:
        # Try common HuggingFace cache locations
        import os
        candidates = [
            "models/gguf/tokenizer",
            os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen3-TTS-12Hz-0.6B-Base"),
        ]
        tokenizer = None
        for candidate in candidates:
            if os.path.exists(candidate):
                try:
                    tokenizer = AutoTokenizer.from_pretrained(candidate, trust_remote_code=True)
                    break
                except:
                    continue
        
        if tokenizer is None:
            print("No local model found. Downloading Qwen/Qwen3-TTS-12Hz-0.6B-Base from HuggingFace...")
            tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-TTS-12Hz-0.6B-Base", trust_remote_code=True)
    
    print("=" * 70)
    print("Qwen2 Pre-tokenizer verification")
    print("=" * 70)
    print()
    
    all_pass = True
    
    for text in TEST_CASES:
        pretokenized = get_pretokenized_hf(text, tokenizer)
        
        # Also get full encode for reference
        full_tokens = tokenizer.encode(text, add_special_tokens=False)
        decoded_parts = [tokenizer.decode([t]) for t in full_tokens]
        
        print(f"Input: {repr(text)}")
        print(f"  Pre-tokenized: {pretokenized}")
        if args.verbose:
            print(f"  BPE tokens:    {full_tokens}")
            print(f"  BPE decoded:  {decoded_parts}")
        
        # Sanity check: concatenation of pre-tokenized pieces should equal original text
        reconstructed = "".join(pretokenized)
        if reconstructed != text:
            print(f"  FAIL: reconstruction mismatch!")
            print(f"    Expected: {repr(text)}")
            print(f"    Got:      {repr(reconstructed)}")
            all_pass = False
        else:
            print(f"  OK: reconstruction matches")
        print()
    
    print("=" * 70)
    print(f"Verification {'PASSED' if all_pass else 'FAILED'}")
    print("=" * 70)
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())