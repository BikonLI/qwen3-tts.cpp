#!/usr/bin/env python3
"""
Download and convert Qwen3-TTS model family for qwen3-tts.cpp.

Defaults:
- Download ALL Qwen3-TTS 12Hz model variants
- Convert to F16 GGUF
- Root output directory: ./models
- Always ensure tokenizer GGUF exists (download/convert unless already present)

Examples:
  python scripts/setup_pipeline_models.py
  python scripts/setup_pipeline_models.py --models 0.6b-base 1.7b-custom-voice --quant q8_0
  python scripts/setup_pipeline_models.py --models all --quant q4_k --force
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"


@dataclass(frozen=True)
class ModelSpec:
    key: str
    repo_ids: tuple[str, ...]
    source_dir_name: str
    gguf_subdir: str
    gguf_stem: str


MODEL_SPECS: dict[str, ModelSpec] = {
    "0.6b-base": ModelSpec(
        key="0.6b-base",
        repo_ids=(
            "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
            "Qwen/Qwen3-TTS-0.6B-Base",
        ),
        source_dir_name="Qwen3-TTS-12Hz-0.6B-Base",
        gguf_subdir="gguf/0.6b-base",
        gguf_stem="qwen3-tts-12hz-0.6b-base",
    ),
    "0.6b-custom-voice": ModelSpec(
        key="0.6b-custom-voice",
        repo_ids=(
            "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
        ),
        source_dir_name="Qwen3-TTS-12Hz-0.6B-CustomVoice",
        gguf_subdir="gguf/0.6b-custom-voice",
        gguf_stem="qwen3-tts-12hz-0.6b-custom-voice",
    ),
    "1.7b-base": ModelSpec(
        key="1.7b-base",
        repo_ids=(
            "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        ),
        source_dir_name="Qwen3-TTS-12Hz-1.7B-Base",
        gguf_subdir="gguf/1.7b-base",
        gguf_stem="qwen3-tts-12hz-1.7b-base",
    ),
    "1.7b-custom-voice": ModelSpec(
        key="1.7b-custom-voice",
        repo_ids=(
            "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
        ),
        source_dir_name="Qwen3-TTS-12Hz-1.7B-CustomVoice",
        gguf_subdir="gguf/1.7b-custom-voice",
        gguf_stem="qwen3-tts-12hz-1.7b-custom-voice",
    ),
    "1.7b-voice-design": ModelSpec(
        key="1.7b-voice-design",
        repo_ids=(
            "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
        ),
        source_dir_name="Qwen3-TTS-12Hz-1.7B-VoiceDesign",
        gguf_subdir="gguf/1.7b-voice-design",
        gguf_stem="qwen3-tts-12hz-1.7b-voice-design",
    ),
}

TOKENIZER_REPO_IDS = (
    "Qwen/Qwen3-TTS-Tokenizer-12Hz",
)

TOKENIZER_SOURCE_DIR = "Qwen3-TTS-Tokenizer-12Hz"
TOKENIZER_GGUF_SUBDIR = "gguf/tokenizer"
TOKENIZER_GGUF_STEM = "qwen3-tts-tokenizer-12hz"

TTS_ALLOW_PATTERNS = [
    "config.json",
    "generation_config.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "special_tokens_map.json",
    "*.safetensors",
    "*.safetensors.index.json",
]

TOKENIZER_ALLOW_PATTERNS = [
    "config.json",
    "configuration.json",
    "preprocessor_config.json",
    "*.safetensors",
    "*.safetensors.index.json",
    "speech_tokenizer/*.safetensors",
    "speech_tokenizer/*.json",
]


def eprint(msg: str) -> None:
    print(msg, file=sys.stderr)


def has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def require_modules(modules: Iterable[tuple[str, str]]) -> None:
    missing = [f"{mod} ({pip_name})" for mod, pip_name in modules if not has_module(mod)]
    if not missing:
        return
    pip_list = " ".join(pip_name for _, pip_name in modules)
    raise RuntimeError(
        "Missing Python modules: " + ", ".join(missing) + "\n"
        + "Install with:\n"
        + f"  {sys.executable} -m pip install {pip_list}"
    )


def run_cmd(cmd: list[str], cwd: Path) -> None:
    eprint("[run] " + " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd), check=True)


def download_repo(
    repo_ids: tuple[str, ...],
    local_dir: Path,
    hf_token: Optional[str],
    allow_patterns: Optional[list[str]],
) -> None:
    from huggingface_hub import snapshot_download

    local_dir.mkdir(parents=True, exist_ok=True)
    last_err: Optional[Exception] = None

    for repo_id in repo_ids:
        try:
            eprint(f"[download] {repo_id} -> {local_dir}")
            snapshot_download(
                repo_id=repo_id,
                local_dir=str(local_dir),
                token=hf_token,
                allow_patterns=allow_patterns,
                resume_download=True,
            )
            return
        except Exception as err:  # pragma: no cover
            last_err = err
            eprint(f"[warn] failed {repo_id}: {err}")

    if last_err is not None:
        raise last_err
    raise RuntimeError("No valid repository ID was provided")


def ensure_source_dir(
    label: str,
    repo_ids: tuple[str, ...],
    source_dir: Path,
    allow_patterns: list[str],
    skip_download: bool,
    force_download: bool,
    hf_token: Optional[str],
) -> None:
    if force_download and source_dir.exists():
        eprint(f"[clean] remove {source_dir}")
        shutil.rmtree(source_dir)

    if source_dir.exists() and any(source_dir.glob("*.safetensors")):
        eprint(f"[ok] {label} source exists: {source_dir}")
        return

    if skip_download:
        raise RuntimeError(
            f"{label} source is missing and --skip-download is set: {source_dir}"
        )

    download_repo(repo_ids, source_dir, hf_token, allow_patterns)


def gguf_output_path(models_dir: Path, subdir: str, stem: str, quant: str) -> Path:
    return models_dir / subdir / f"{stem}-{quant}.gguf"


def ensure_tts_gguf(
    spec: ModelSpec,
    source_dir: Path,
    output_path: Path,
    quant: str,
    force_convert: bool,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if force_convert and output_path.exists():
        eprint(f"[clean] remove {output_path}")
        output_path.unlink()

    if output_path.exists():
        eprint(f"[ok] exists: {output_path}")
        return

    run_cmd(
        [
            sys.executable,
            str(SCRIPTS_DIR / "convert_tts_to_gguf.py"),
            "--input",
            str(source_dir),
            "--output",
            str(output_path),
            "--type",
            quant,
        ],
        cwd=REPO_ROOT,
    )


def ensure_tokenizer_gguf(
    source_dir: Path,
    output_path: Path,
    quant: str,
    force_convert: bool,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if force_convert and output_path.exists():
        eprint(f"[clean] remove {output_path}")
        output_path.unlink()

    if output_path.exists():
        eprint(f"[ok] exists: {output_path}")
        return

    run_cmd(
        [
            sys.executable,
            str(SCRIPTS_DIR / "convert_tokenizer_to_gguf.py"),
            "--input",
            str(source_dir),
            "--output",
            str(output_path),
            "--type",
            quant,
        ],
        cwd=REPO_ROOT,
    )


def parse_selected_models(items: list[str]) -> list[ModelSpec]:
    values: list[str] = []
    for item in items:
        values.extend(v.strip().lower() for v in item.split(",") if v.strip())

    if not values or "all" in values:
        return [MODEL_SPECS[k] for k in MODEL_SPECS]

    out: list[ModelSpec] = []
    unknown: list[str] = []
    seen: set[str] = set()
    for key in values:
        if key in seen:
            continue
        seen.add(key)
        spec = MODEL_SPECS.get(key)
        if spec is None:
            unknown.append(key)
        else:
            out.append(spec)

    if unknown:
        valid = ", ".join(["all", *MODEL_SPECS.keys()])
        raise RuntimeError(f"Unknown --models value: {', '.join(unknown)}. Valid: {valid}")

    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download Qwen3-TTS model family and convert tensors to GGUF"
    )
    parser.add_argument(
        "--models-dir",
        default=str(REPO_ROOT / "models"),
        help="Root models directory (default: ./models)",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["all"],
        help="Models to setup: all, 0.6b-base, 0.6b-custom-voice, 1.7b-base, 1.7b-custom-voice, 1.7b-voice-design",
    )
    parser.add_argument(
        "--quant",
        choices=["f16", "f32", "q8_0", "q4_k"],
        default="f16",
        help="GGUF quantization/output type",
    )
    parser.add_argument(
        "--hf-token",
        default=os.environ.get("HF_TOKEN", ""),
        help="Hugging Face token (or set HF_TOKEN)",
    )
    parser.add_argument("--skip-download", action="store_true", help="Do not download from Hugging Face")
    parser.add_argument("--skip-convert", action="store_true", help="Do not run GGUF conversion")
    parser.add_argument("--force-download", action="store_true", help="Force re-download sources")
    parser.add_argument("--force-convert", action="store_true", help="Force regenerate GGUF files")
    parser.add_argument("--force", action="store_true", help="Equivalent to --force-download --force-convert")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.force:
        args.force_download = True
        args.force_convert = True

    models_dir = Path(args.models_dir).resolve()
    models_dir.mkdir(parents=True, exist_ok=True)

    selected_specs = parse_selected_models(args.models)
    hf_token = args.hf_token.strip() or None

    eprint("[config]")
    eprint(f"  models_dir: {models_dir}")
    eprint(f"  quant:      {args.quant}")
    eprint(f"  models:     {', '.join(spec.key for spec in selected_specs)}")

    # Tokenizer is mandatory for runtime; always ensure it exists.
    tokenizer_source = models_dir / TOKENIZER_SOURCE_DIR
    tokenizer_gguf = gguf_output_path(models_dir, TOKENIZER_GGUF_SUBDIR, TOKENIZER_GGUF_STEM, args.quant)

    if not args.skip_download:
        require_modules([("huggingface_hub", "huggingface_hub")])

    if not args.skip_download or not tokenizer_source.exists():
        ensure_source_dir(
            label="Tokenizer",
            repo_ids=TOKENIZER_REPO_IDS,
            source_dir=tokenizer_source,
            allow_patterns=TOKENIZER_ALLOW_PATTERNS,
            skip_download=args.skip_download,
            force_download=args.force_download,
            hf_token=hf_token,
        )

    # Download selected TTS model sources.
    tts_sources: dict[str, Path] = {}
    for spec in selected_specs:
        source_dir = models_dir / spec.source_dir_name
        ensure_source_dir(
            label=spec.key,
            repo_ids=spec.repo_ids,
            source_dir=source_dir,
            allow_patterns=TTS_ALLOW_PATTERNS,
            skip_download=args.skip_download,
            force_download=args.force_download,
            hf_token=hf_token,
        )
        tts_sources[spec.key] = source_dir

    if not args.skip_convert:
        require_modules(
            [
                ("gguf", "gguf"),
                ("torch", "torch"),
                ("safetensors", "safetensors"),
                ("numpy", "numpy"),
                ("tqdm", "tqdm"),
            ]
        )

        # Tokenizer conversion is mandatory unless already available locally.
        ensure_tokenizer_gguf(
            source_dir=tokenizer_source,
            output_path=tokenizer_gguf,
            quant=args.quant,
            force_convert=args.force_convert,
        )

        for spec in selected_specs:
            out_path = gguf_output_path(models_dir, spec.gguf_subdir, spec.gguf_stem, args.quant)
            ensure_tts_gguf(
                spec=spec,
                source_dir=tts_sources[spec.key],
                output_path=out_path,
                quant=args.quant,
                force_convert=args.force_convert,
            )
    else:
        eprint("[skip] conversion disabled by --skip-convert")

    eprint("\n[done] setup complete")
    eprint(f"  tokenizer: {tokenizer_gguf}")
    for spec in selected_specs:
        out_path = gguf_output_path(models_dir, spec.gguf_subdir, spec.gguf_stem, args.quant)
        eprint(f"  {spec.key}: {out_path}")

    eprint("\nExample load (new multi-model CLI):")
    eprint(
        "  build\\Release\\qwen3-tts-cli.exe "
        + f"--tokenizer-12hz {tokenizer_gguf} "
        + f"--load-17b-custom-voice {models_dir / 'gguf/1.7b-custom-voice'} "
        + "--model 1.7b_custom_voice --mode custom_instruct "
        + "--speaker ryan --instruct \"happy\" -t \"Hello\" -o out.wav"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
