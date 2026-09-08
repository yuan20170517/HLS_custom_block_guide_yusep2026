#!/usr/bin/env python3
"""
================================================================================
autoscript.py - Automated Multi-Stage Vitis HLS Build Pipeline
Target Platform: AMD Versal AI Edge Gen 2 VEK385 (xc2ve3858-ssva2112-2MP-e-S)
China Telecom Singapore Innovation and Research Institute (CTSIRI)
================================================================================
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = REPO_ROOT / "config"
BUILD_DIR = REPO_ROOT / "build"
SRC_DIR = REPO_ROOT / "src"

def run_stage(command, desc):
    print(f"\n=======================================================")
    print(f"[*] Running Stage: {desc}")
    print(f"[*] Command: {' '.join(command) if isinstance(command, list) else command}")
    print(f"=======================================================")
    res = subprocess.run(command, shell=True)
    if res.returncode != 0:
        print(f"[!] Error: Stage '{desc}' failed with exit code {res.returncode}")
        return False
    print(f"[+] Success: Stage '{desc}' completed.")
    return True

def main():
    parser = argparse.ArgumentParser(description="Automated Vitis HLS Build Orchestrator for VEK385")
    parser.add_argument("--csim", action="store_true", help="Run C-Simulation")
    parser.add_argument("--syn", action="store_true", help="Run C-Synthesis (RTL Generation)")
    parser.add_argument("--cosim", action="store_true", help="Run C/RTL Co-Simulation")
    parser.add_argument("--package", action="store_true", help="Package deployable .xo container")
    parser.add_argument("--all", action="store_true", help="Run all build stages end-to-end")
    parser.add_argument("--kernel", type=str, default="gelu", help="Target kernel operator name")
    args = parser.parse_args()

    cfg_file = CONFIG_DIR / "hls_config.cfg"
    if not cfg_file.exists():
        cfg_file = CONFIG_DIR / "hls_config.cfg.template"

    print(f"[*] Repo Root: {REPO_ROOT}")
    print(f"[*] Target Kernel: {args.kernel}")
    print(f"[*] Config Template: {cfg_file}")

    stages_to_run = []
    if args.all or args.csim:
        stages_to_run.append((f"vitis-run --mode hls --config {cfg_file} --work_dir build/{args.kernel}_sim --csim", "1. C-Simulation"))
    if args.all or args.syn:
        stages_to_run.append((f"vitis-run --mode hls --config {cfg_file} --work_dir build/{args.kernel}_syn --syn", "2. C-Synthesis"))
    if args.all or args.cosim:
        stages_to_run.append((f"vitis-run --mode hls --config {cfg_file} --work_dir build/{args.kernel}_cosim --cosim", "3. C/RTL Co-Simulation"))
    if args.all or args.package:
        is_win = sys.platform.startswith("win")
        runner = "vitis-run.bat" if is_win else "vitis-run"
        stages_to_run.append((f"{runner} --mode hls --config {cfg_file} --work_dir build/{args.kernel}_export --package", "4. Package Kernel (.xo)"))

    if not stages_to_run:
        print("[!] No build stages selected. Use --all, --csim, --syn, --cosim, or --package.")
        sys.exit(0)

    for cmd, desc in stages_to_run:
        ok = run_stage(cmd, desc)
        if not ok:
            sys.exit(1)

    print("\n[+] All requested Vitis HLS build stages completed successfully!")

if __name__ == "__main__":
    main()
