#!/usr/bin/env python3
"""
HG-PIPE Step 1: Automated Vitis HLS Compilation & Synthesis
Generates hls_config.cfg and executes vitis-run --mode hls --csynth.
"""

import os
import subprocess
import argparse
from pathlib import Path

def run_hls(kernel_name: str, part: str = "xc2ve3858-ssva2112-2MP-e-S", clock: str = "3.33ns"):
    print(f"[HG-PIPE] Step 1: Building HLS Kernel '{kernel_name}'...")
    kernel_dir = Path("src/hls") / kernel_name
    if not kernel_dir.exists():
        print(f"[!] Kernel directory not found: {kernel_dir}")
        return

    cfg_path = kernel_dir / "hls_config.cfg"
    cfg_content = f"""# AMD Vitis HLS Automated Build Config
part={part}

[hls]
syn.top={kernel_name}_kernel
syn.file={kernel_name}_kernel.cpp
clock={clock}
clock_uncertainty=12%
syn.compile.pipeline_loops=5
package.output.format=xo
"""
    cfg_path.write_text(cfg_content, encoding="utf-8")
    print(f"  [+] Created config: {cfg_path}")

    cmd = ["vitis-run", "--mode", "hls", "--csynth", "--config", str(cfg_path.name), "--work_dir", "build"]
    print(f"  [>] Command: {' '.join(cmd)}")
    print("  [!] (Dry run / ready for Vitis environment execution)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HG-PIPE Automated HLS Builder")
    parser.add_argument("--kernel", default="layernorm", choices=["layernorm", "softmax", "gelu"])
    parser.add_argument("--part", default="xc2ve3858-ssva2112-2MP-e-S")
    parser.add_argument("--clock", default="3.33ns")
    args = parser.parse_args()
    run_hls(args.kernel, args.part, args.clock)
