#!/usr/bin/env python3
"""
V2 converter for Qwen3-TTS GGUF.

Adds code predictor projection tensors and full code predictor metadata so
runtime can safely load both 0.6B and 1.7B model families without dimension
mismatch.

Usage:
    python scripts/convert_tts_to_gguf_v2.py \
        --input models/Qwen3-TTS-12Hz-1.7B-CustomVoice \
        --output models/qwen3-tts-1.7b-f16.gguf \
        --type f16
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from convert_tts_to_gguf import Qwen3TTSConverter as BaseConverter


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


class Qwen3TTSConverterV2(BaseConverter):
    """Backward-compatible converter with extra metadata for mixed-dimension code predictor."""

    TENSOR_MAP = dict(BaseConverter.TENSOR_MAP)
    TENSOR_MAP.update({
        "talker.code_predictor.small_to_mtp_projection.weight": "code_pred.small_to_mtp_proj.weight",
        "talker.code_predictor.small_to_mtp_projection.bias": "code_pred.small_to_mtp_proj.bias",
    })

    def _extract_params(self) -> None:
        super()._extract_params()

        talker_config = self.config.get("talker_config", {})
        code_predictor_config = talker_config.get("code_predictor_config", {})

        self.code_predictor_hidden_size = code_predictor_config.get("hidden_size", 1024)
        self.code_predictor_intermediate_size = code_predictor_config.get("intermediate_size", 3072)
        self.code_predictor_attention_heads = code_predictor_config.get("num_attention_heads", self.num_attention_heads)
        self.code_predictor_kv_heads = code_predictor_config.get("num_key_value_heads", self.num_kv_heads)
        self.code_predictor_head_dim = code_predictor_config.get("head_dim", self.head_dim)

        # code predictor receives talker-space hidden/code embeddings as input.
        self.code_predictor_input_size = self.hidden_size

        self.model_name = self.config.get("_name_or_path", self.input_dir.name)

    def _add_metadata(self, writer) -> None:
        super()._add_metadata(writer)

        arch = "qwen3-tts"

        # Preferred metadata keys
        writer.add_uint32(f"{arch}.code_predictor.embedding_length", self.code_predictor_hidden_size)
        writer.add_uint32(f"{arch}.code_predictor.feed_forward_length", self.code_predictor_intermediate_size)
        writer.add_uint32(f"{arch}.code_predictor.attention.head_count", self.code_predictor_attention_heads)
        writer.add_uint32(f"{arch}.code_predictor.attention.head_count_kv", self.code_predictor_kv_heads)
        writer.add_uint32(f"{arch}.code_predictor.attention.key_length", self.code_predictor_head_dim)
        writer.add_uint32(f"{arch}.code_predictor.input_embedding_length", self.code_predictor_input_size)

        # Backward-compatible aliases consumed by runtime fallback logic
        writer.add_uint32(f"{arch}.code_pred.embedding_length", self.code_predictor_hidden_size)
        writer.add_uint32(f"{arch}.code_pred.feed_forward_length", self.code_predictor_intermediate_size)
        writer.add_uint32(f"{arch}.code_pred.attention.head_count", self.code_predictor_attention_heads)
        writer.add_uint32(f"{arch}.code_pred.attention.head_count_kv", self.code_predictor_kv_heads)
        writer.add_uint32(f"{arch}.code_pred.attention.key_length", self.code_predictor_head_dim)
        writer.add_uint32(f"{arch}.code_pred.input_embedding_length", self.code_predictor_input_size)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert Qwen3-TTS model to GGUF format (v2 with code predictor metadata/projection)"
    )
    parser.add_argument("--input", "-i", type=Path, required=True, help="Path to HuggingFace model directory")
    parser.add_argument("--output", "-o", type=Path, required=True, help="Output GGUF file path")
    parser.add_argument(
        "--type",
        "-t",
        choices=["f16", "f32", "q8_0", "q4_k"],
        default="f16",
        help="Output data type (default: f16)",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    converter = Qwen3TTSConverterV2(
        input_dir=args.input,
        output_path=args.output,
        output_type=args.type,
    )
    converter.convert()


if __name__ == "__main__":
    main()
