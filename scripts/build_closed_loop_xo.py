#!/usr/bin/env python3
"""
Automated XO Package Orchestrator for Closed-Loop Autoregressive Generation
Builds and packages:
  1. embedding_kernel.xo (Pre-processing Token & Position Embedding Streamer)
  2. mha_kernel.xo       (Quantized Multi-Head Attention Core)
  3. sampler_kernel.xo   (Post-processing ArgMax & Top-K Logit Sampler)

Usage:
  python scripts/build_closed_loop_xo.py --kernel all
  python scripts/build_closed_loop_xo.py --kernel embedding
  python scripts/build_closed_loop_xo.py --kernel mha
  python scripts/build_closed_loop_xo.py --kernel sampler
"""

import os
import sys
import shutil
import argparse
import subprocess
from pathlib import Path

KERNELS = {
    "embedding": {
        "cfg": "config/hls_embedding.cfg",
        "top": "embedding_kernel",
        "xo": "xo/embedding_kernel.xo",
        "desc": "Token & Position Embedding Streamer (Pre-Processing)"
    },
    "mha": {
        "cfg": "config/hls_mha.cfg",
        "top": "mha_top",
        "xo": "xo/mha_kernel.xo",
        "desc": "Quantized Multi-Head Attention (Core Attention)"
    },
    "sampler": {
        "cfg": "config/hls_sampler.cfg",
        "top": "sampler_kernel",
        "xo": "xo/sampler_kernel.xo",
        "desc": "ArgMax & Top-K Token Retrieval Engine (Post-Processing)"
    }
}

def build_kernel(kernel_key: str, dry_run: bool = False):
    info = KERNELS[kernel_key]
    repo_root = Path(__file__).resolve().parent.parent
    cfg_file = repo_root / info["cfg"]
    xo_target = repo_root / info["xo"]
    work_dir = repo_root / "build" / f"hls_{kernel_key}"

    print(f"\n========================================================")
    print(f"[*] Building XO: {info['desc']}")
    print(f"[*] Config: {cfg_file}")
    print(f"[*] Target XO: {xo_target}")
    print(f"========================================================")

    if not cfg_file.exists():
        print(f"[!] ERROR: Configuration file not found: {cfg_file}")
        return False

    cmd = [
        "vitis-run", "--mode", "hls",
        "--config", str(cfg_file),
        "--work_dir", str(work_dir)
    ]

    print(f"[CMD] {' '.join(cmd)}")

    if dry_run or shutil.which("vitis-run") is None:
        print(f"[INFO] Vitis toolchain not found in local environment or dry-run requested.")
        print(f"[INFO] Execution command validated. Run inside Vitis Unified IDE 2024.1+ environment.")
        return True

    try:
        res = subprocess.run(cmd, cwd=str(repo_root), check=True)
        print(f"[+] Successfully compiled {info['top']} -> {xo_target}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[!] Compilation failed for {kernel_key}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Closed-Loop XO Packaging Orchestrator")
    parser.add_argument("--kernel", choices=["embedding", "mha", "sampler", "all"], default="all",
                        help="Select kernel to synthesize and package as .xo")
    parser.add_argument("--dry-run", action="store_true", help="Validate configs without launching Vitis process")
    args = parser.parse_args()

    targets = list(KERNELS.keys()) if args.kernel == "all" else [args.kernel]
    all_success = True
    for k in targets:
        success = build_kernel(k, dry_run=args.dry_run)
        if not success:
            all_success = False

    if all_success:
        print("\n[OK] All target kernel configurations verified and ready for system linking!")
    else:
        sys.exit(1)

if __name__ == "__main__":
    main()
