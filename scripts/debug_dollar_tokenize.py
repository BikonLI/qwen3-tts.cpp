from transformers import AutoTokenizer
import json

t = AutoTokenizer.from_pretrained('Qwen/Qwen3-TTS-12Hz-0.6B-Base', trust_remote_code=True)

text = 'please read the following numbers and symbols correctly: 0, 1, 2, 10, 100, 1,000, 3.14159, 99.9%, and $42.50.'
tokens = t.encode(text, add_special_tokens=False)
decoded = [t.decode([x]) for x in tokens]

print("Input text:")
print(repr(text))
print()
print("Token IDs and decoded:")
for i, (tid, dec) in enumerate(zip(tokens, decoded)):
    print(f"  [{i:2d}] ID={tid:5d}  decoded={repr(dec)}")

# Focus on the last part
print()
print("Last 10 tokens:")
for i in range(max(0, len(tokens)-10), len(tokens)):
    print(f"  [{i:2d}] ID={tokens[i]:5d}  decoded={repr(decoded[i])}")

# Test just the problematic substring
print()
print("=== Testing substrings ===")
for substr in ['$42.50', '$42', '$', '42.50', '.50', '$4', '42']:
    toks = t.encode(substr, add_special_tokens=False)
    decs = [t.decode([x]) for x in toks]
    print(f"{repr(substr):15s} -> IDs={toks} -> decoded={decs}")