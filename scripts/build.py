#!/usr/bin/env python3
"""
Build script for qwen3-tts.cpp — automates GGML and project compilation.

Platform behavior:
  - Windows: builds shared library (DLL), CLI, and GUI executables
  - Linux/macOS/other: builds shared library and CLI only (no GUI)

Usage examples:
  python scripts/build.py
  python scripts/build.py --sdl3-dir /path/to/SDL3
  python scripts/build.py --backend cuda
  python scripts/build.py --backend metal --build-type Debug
  python scripts/build.py --backend cpu --clean
"""

from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional


REPO_ROOT = Path(__file__).resolve().parents[1]
GGML_DIR = REPO_ROOT / "ggml"
GGML_BUILD_DIR = GGML_DIR / "build"
PROJECT_BUILD_DIR = REPO_ROOT / "build"


# ── Backend definitions ──────────────────────────────────────────────
# Each backend maps to one or more CMake -D flags for the GGML build.
BACKENDS: dict[str, dict[str, str | bool]] = {
    "cpu": {
        "description": "CPU only (default)",
        "ggml_flags": {},
        "is_default": True,
    },
    "cuda": {
        "description": "NVIDIA CUDA GPU",
        "ggml_flags": {"GGML_CUDA": "ON"},
    },
    "hip": {
        "description": "AMD HIP/ROCm GPU",
        "ggml_flags": {"GGML_HIP": "ON"},
    },
    "vulkan": {
        "description": "Vulkan GPU",
        "ggml_flags": {"GGML_VULKAN": "ON"},
    },
    "metal": {
        "description": "Apple Metal GPU (macOS only)",
        "ggml_flags": {"GGML_METAL": "ON"},
    },
    "sycl": {
        "description": "Intel SYCL GPU",
        "ggml_flags": {"GGML_SYCL": "ON"},
    },
    "blas": {
        "description": "BLAS acceleration",
        "ggml_flags": {"GGML_BLAS": "ON"},
    },
}


def detect_generator() -> str:
    """Pick a CMake generator appropriate for the current platform."""
    system = platform.system()
    if system == "Windows":
        # Prefer VS if available; fall back to Ninja
        vs_where = shutil.which("cmake")
        if vs_where:
            # Check if Visual Studio CMake generators are available by inspecting
            # cmake --help. For simplicity, use the env var or default.
            # Most Windows dev setups have "Visual Studio 17 2022" or similar.
            # Let's detect what's available.
            result = subprocess.run(
                ["cmake", "--help"],
                capture_output=True,
                text=True,
            )
            if "Visual Studio 17 2022" in result.stdout:
                return "Visual Studio 17 2022"
            if "Visual Studio 16 2019" in result.stdout:
                return "Visual Studio 16 2019"
        return "Ninja"
    else:
        return "Ninja"


def run_cmd(
    cmd: list[str | Path],
    *,
    cwd: Optional[Path] = None,
    desc: Optional[str] = None,
) -> int:
    """Run a subprocess command, printing it and streaming output."""
    if desc:
        print(f"\n{'='*60}")
        print(f"  {desc}")
        print(f"{'='*60}\n")
    cmd_strs = [str(c) for c in cmd]
    print(f"  $ {' '.join(cmd_strs)}")
    result = subprocess.run(cmd_strs, cwd=cwd)
    return result.returncode


def build_ggml(
    *,
    generator: str,
    backend: str,
    build_type: str,
    clean: bool,
    pushd: bool,
) -> int:
    """Build the vendored GGML library first (required before main project)."""
    if clean and GGML_BUILD_DIR.exists():
        print(f"Cleaning GGML build directory: {GGML_BUILD_DIR}")
        shutil.rmtree(GGML_BUILD_DIR, ignore_errors=True)

    GGML_BUILD_DIR.mkdir(parents=True, exist_ok=True)

    cmake_configure = [
        "cmake",
        "-S", str(GGML_DIR),
        "-B", str(GGML_BUILD_DIR),
        "-G", generator,
    ]

    # Build type (for multi-config generators like VS, set via --build --config)
    if generator.startswith("Visual Studio"):
        cmake_configure.append(f"-DCMAKE_BUILD_TYPE={build_type}")
    else:
        cmake_configure.append(f"-DCMAKE_BUILD_TYPE={build_type}")

    # Apply backend-specific flags
    backend_spec = BACKENDS[backend]
    for flag_name, flag_value in backend_spec["ggml_flags"].items():
        if flag_name.startswith("GGML_"):
            cmake_configure.append(f"-D{flag_name}={flag_value}")

    # HIP requires special compilers
    if backend == "hip":
        hipconfig = shutil.which("hipconfig")
        if hipconfig:
            hip_info = subprocess.run(
                [hipconfig, "--path"],
                capture_output=True,
                text=True,
            )
            hip_path = hip_info.stdout.strip()
            cmake_configure.extend([
                f"-DCMAKE_C_COMPILER={hip_path}/clang",
                f"-DCMAKE_CXX_COMPILER={hip_path}/clang++",
            ])

    rc = run_cmd(cmake_configure, cwd=GGML_BUILD_DIR, desc="Configuring GGML")
    if rc != 0:
        print(f"ERROR: GGML cmake configure failed (exit {rc})")
        return rc

    build_cmd = ["cmake", "--build", str(GGML_BUILD_DIR), "--config", build_type, "-j"]
    rc = run_cmd(build_cmd, desc=f"Building GGML ({build_type})")
    if rc != 0:
        print(f"ERROR: GGML build failed (exit {rc})")
        return rc

    return 0


def build_project(
    *,
    generator: str,
    sdl3_dir: Optional[Path],
    backend: str,
    build_type: str,
    timing: bool,
    clean: bool,
    coreml: bool,
    install: bool,
    install_prefix: Optional[Path],
    pushd: bool,
) -> int:
    """Build the main qwen3-tts.cpp project."""
    if clean and PROJECT_BUILD_DIR.exists():
        print(f"Cleaning project build directory: {PROJECT_BUILD_DIR}")
        shutil.rmtree(PROJECT_BUILD_DIR, ignore_errors=True)

    PROJECT_BUILD_DIR.mkdir(parents=True, exist_ok=True)

    cmake_configure = [
        "cmake",
        "-S", str(REPO_ROOT),
        "-B", str(PROJECT_BUILD_DIR),
        "-G", generator,
    ]

    # Build type
    if generator.startswith("Visual Studio"):
        cmake_configure.append(f"-DCMAKE_BUILD_TYPE={build_type}")
    else:
        cmake_configure.append(f"-DCMAKE_BUILD_TYPE={build_type}")

    # SDL3_DIR - can be provided via CLI or environment variable
    sdl3_path = None
    if sdl3_dir:
        sdl3_path = sdl3_dir
    elif os.environ.get("SDL3_DIR"):
        sdl3_path = Path(os.environ["SDL3_DIR"])

    if sdl3_path:
        cmake_configure.append(f"-DSDL3_DIR={sdl3_path}")

    # Optional flags
    if timing:
        cmake_configure.append("-DQWEN3_TTS_TIMING=ON")

    if coreml and platform.system() == "Darwin":
        cmake_configure.append("-DQWEN3_TTS_COREML=ON")
    else:
        cmake_configure.append("-DQWEN3_TTS_COREML=OFF")

    # Backend-specific project-level flags
    backend_spec = BACKENDS[backend]
    for flag_name, flag_value in backend_spec["ggml_flags"].items():
        cmake_configure.append(f"-D{flag_name}={flag_value}")

    # HIP compiler override at project level too
    if backend == "hip":
        hipconfig = shutil.which("hipconfig")
        if hipconfig:
            hip_info = subprocess.run(
                [hipconfig, "--path"],
                capture_output=True,
                text=True,
            )
            hip_path = hip_info.stdout.strip()
            cmake_configure.extend([
                f"-DCMAKE_C_COMPILER={hip_path}/clang",
                f"-DCMAKE_CXX_COMPILER={hip_path}/clang++",
            ])

    rc = run_cmd(cmake_configure, cwd=PROJECT_BUILD_DIR, desc="Configuring qwen3-tts.cpp")
    if rc != 0:
        print(f"ERROR: Project cmake configure failed (exit {rc})")
        return rc

    # Build targets based on platform
    targets = ["qwen3tts_shared", "qwen3-tts-cli"]
    if platform.system() == "Windows":
        targets.append("qwen3-tts-gui")

    build_cmd = [
        "cmake", "--build", str(PROJECT_BUILD_DIR),
        "--config", build_type,
        "-j",
    ]

    # Add explicit targets for faster builds
    for target in targets:
        build_cmd.extend(["--target", target])

    desc_target = ", ".join(targets)
    rc = run_cmd(build_cmd, desc=f"Building {desc_target} ({build_type})")
    if rc != 0:
        print(f"ERROR: Project build failed (exit {rc})")
        return rc

    # Install step
    if install:
        install_cmd = ["cmake", "--install", str(PROJECT_BUILD_DIR), "--config", build_type]
        if install_prefix:
            install_cmd.extend(["--prefix", str(install_prefix)])
        rc = run_cmd(install_cmd, desc="Installing to dist/")
        if rc != 0:
            print(f"ERROR: Install failed (exit {rc})")
            return rc

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build qwen3-tts.cpp — automates GGML + project compilation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Available backends:
  cpu     CPU only (default)
  cuda    NVIDIA CUDA GPU acceleration
  hip     AMD HIP/ROCm GPU acceleration
  vulkan  Vulkan GPU acceleration
  metal   Apple Metal GPU (macOS only)
  sycl    Intel SYCL GPU acceleration
  blas    BLAS-accelerated CPU

Examples:
  python scripts/build.py
  python scripts/build.py --sdl3-dir /path/to/SDL3/cmake
  python scripts/build.py --backend cuda
  python scripts/build.py --backend metal --build-type Debug
  python scripts/build.py --backend cpu --clean --install
  python scripts/build.py --backend cuda --install --prefix ./dist
""",
    )

    parser.add_argument(
        "--sdl3-dir",
        type=Path,
        default=None,
        help="Path to SDL3 cmake config directory (or set SDL3_DIR env var)",
    )
    parser.add_argument(
        "--backend",
        choices=list(BACKENDS.keys()),
        default="cpu",
        help="GGML compute backend (default: cpu)",
    )
    parser.add_argument(
        "--build-type",
        choices=["Release", "Debug", "RelWithDebInfo", "MinSizeRel"],
        default="Release",
        help="CMake build type (default: Release)",
    )
    parser.add_argument(
        "--generator",
        default=None,
        help="CMake generator override (default: auto-detect)",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Clean build directories before building",
    )
    parser.add_argument(
        "--timing",
        action="store_true",
        help="Enable QWEN3_TTS_TIMING instrumentation",
    )
    parser.add_argument(
        "--coreml",
        action="store_true",
        help="Enable CoreML code predictor (macOS only)",
    )
    parser.add_argument(
        "--install",
        action="store_true",
        help="Run cmake --install after building",
    )
    parser.add_argument(
        "--prefix",
        type=Path,
        default=None,
        help="Install prefix directory (default: ./dist)",
    )
    parser.add_argument(
        "--ggml-only",
        action="store_true",
        help="Only build the vendored GGML library, skip the main project",
    )
    parser.add_argument(
        "--project-only",
        action="store_true",
        help="Only build the main project (assumes GGML is already built)",
    )
    parser.add_argument(
        "--skip-ggml",
        action="store_true",
        help="Skip GGML build step (same as --project-only)",
    )

    args = parser.parse_args()

    # Validate backend/platform combinations
    if args.backend == "metal" and platform.system() != "Darwin":
        print("ERROR: Metal backend is only available on macOS")
        return 1

    # Determine generator
    generator = args.generator or detect_generator()

    # Compute flags
    pushd = False
    install_prefix = args.prefix or (REPO_ROOT / "dist")

    print(f"Build Configuration:")
    print(f"  Platform:      {platform.system()} ({platform.machine()})")
    print(f"  Generator:     {generator}")
    print(f"  Backend:       {args.backend} ({BACKENDS[args.backend]['description']})")
    print(f"  Build type:    {args.build_type}")
    print(f"  SDL3_DIR:      {args.sdl3_dir or os.environ.get('SDL3_DIR', '(not set)')}")
    print(f"  Clean:         {args.clean}")
    print(f"  Timing:        {args.timing}")
    print(f"  Install:       {args.install}")
    if args.install:
        print(f"  Install prefix: {install_prefix}")
    print()

    skip_ggml = args.skip_ggml or args.project_only

    # Step 1: Build GGML (unless skipped)
    if not skip_ggml:
        rc = build_ggml(
            generator=generator,
            backend=args.backend,
            build_type=args.build_type,
            clean=args.clean,
            pushd=pushd,
        )
        if rc != 0:
            return rc
        print()
    else:
        print("Skipping GGML build (--skip-ggml / --project-only)")
        print()

    # Step 2: Build main project (unless --ggml-only)
    if not args.ggml_only:
        rc = build_project(
            generator=generator,
            sdl3_dir=args.sdl3_dir,
            backend=args.backend,
            build_type=args.build_type,
            timing=args.timing,
            clean=args.clean,
            coreml=args.coreml,
            install=args.install,
            install_prefix=install_prefix,
            pushd=pushd,
        )
        if rc != 0:
            return rc
        print()
    else:
        print("Skipping project build (--ggml-only)")
        print()

    print("=" * 60)
    print("  Build completed successfully!")
    print("=" * 60)
    print()
    print(f"  GGML build:    {GGML_BUILD_DIR}")
    print(f"  Project build:  {PROJECT_BUILD_DIR}")
    if args.install:
        print(f"  Installed to:   {install_prefix}")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())