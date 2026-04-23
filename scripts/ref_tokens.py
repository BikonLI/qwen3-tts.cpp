from transformers import AutoTokenizer
t = AutoTokenizer.from_pretrained("Qwen/Qwen3-TTS-12Hz-0.6B-Base", trust_remote_code=True)

texts = [
    "The number is 12345678.",
    "Hello.",
    "Hello world",
    "it's a test",
    "Finally, please read: 0, 1, 2, 10, 100",
    "3.14159",
    "$42.50",
    "99.9%",
]

for text in texts:
    tokens = t.encode(text, add_special_tokens=False)
    print(f"{text}")
    print(f"  Python: {tokens}")
    print()