#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tokenize text/text8.txt using the official Qwen3-TTS tokenizer.
Outputs exact token IDs for comparison with C++ implementation.
"""

import sys
import io

# Force UTF-8 for stdout/stderr on Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

def main():
    text_file = "text/text8.txt"
    
    # Try loading the official Qwen3-TTS tokenizer
    tokenizer = None
    model_name = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
    
    try:
        from transformers import AutoTokenizer
        print(f"Loading tokenizer from: {model_name}", file=sys.stderr)
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        print(f"Successfully loaded tokenizer from {model_name}", file=sys.stderr)
    except Exception as e:
        print(f"Failed to load {model_name}: {e}", file=sys.stderr)
        try:
            from transformers import AutoTokenizer
            print("Falling back to generic AutoTokenizer...", file=sys.stderr)
            tokenizer = AutoTokenizer.from_pretrained("gpt2")
            print("WARNING: Using gpt2 tokenizer as fallback — token IDs will NOT match Qwen3-TTS!", file=sys.stderr)
        except Exception as e2:
            print(f"Failed to load any tokenizer: {e2}", file=sys.stderr)
            sys.exit(1)
    
    # Read the text file
    try:
        with open(text_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"Error: {text_file} not found.", file=sys.stderr)
        sys.exit(1)
    
    # Process each line
    print("=" * 60)
    print("Qwen3-TTS Tokenizer Reference Output")
    print("=" * 60)
    
    for i, line in enumerate(lines, 1):
        # Strip trailing newline only, preserve other whitespace
        text = line.rstrip('\n').rstrip('\r')
        if not text:
            continue
        
        # Encode without special tokens
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        
        print(f"\nLine {i}: {text!r}")
        print(f"  Token count: {len(token_ids)}")
        print(f"  Token IDs: {token_ids}")
        
        # Also print tokens for readability
        tokens = tokenizer.convert_ids_to_tokens(token_ids)
        print(f"  Tokens: {tokens}")
    
    print("\n" + "=" * 60)
    print("Done.")
    print("=" * 60)

if __name__ == "__main__":
    main()
