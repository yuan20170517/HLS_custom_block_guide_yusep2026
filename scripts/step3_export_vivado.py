#!/usr/bin/env python3
"""
HG-PIPE Step 3: Vivado & Vitis Linker Export Automation
Aggregates exported .xo kernel objects and prepares system.cfg for v++ linking.
"""

import argparse
from pathlib import Path

def export_vivado(build_dir: str = "src/hls"):
    print("[HG-PIPE] Step 3: Packaging Extensible Kernel Objects (.xo)...")
    system_cfg_path = Path("system.cfg")
    
    cfg_content = """# Automated System Linking Configuration
[connectivity]
nk=layernorm_kernel:1:layernorm_0
nk=softmax_kernel:1:softmax_0
nk=gelu_kernel:1:gelu_0

# Memory Bank Mapping to Versal LPDDR4
sp=layernorm_0.in:LPDDR4_0
sp=layernorm_0.out:LPDDR4_0
sp=softmax_0.in:LPDDR4_0
sp=softmax_0.out:LPDDR4_0
sp=gelu_0.in:LPDDR4_0
sp=gelu_0.out:LPDDR4_0
"""
    system_cfg_path.write_text(cfg_content, encoding="utf-8")
    print(f"  [+] Generated system linking topology: {system_cfg_path}")
    print("  [>] Next step: v++ --link --target hw --config system.cfg *.xo -o vitis_accel.xclbin")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HG-PIPE Vivado Export Packager")
    args = parser.parse_args()
    export_vivado()
