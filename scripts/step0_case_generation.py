#!/usr/bin/env python3
"""
HG-PIPE Step 0: Parametric Case Generation
Reads model configuration statistics and populates C++ HLS kernel templates.
"""

import json
import argparse
from pathlib import Path

def generate_cases(config_path: str, output_dir: str):
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_file, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    repo_root = config_file.parent.parent
    case_dir = repo_root / "case"
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[HG-PIPE] Loading configuration for: {cfg.get('model_name')}")
    print(f"         Embed Dim : {cfg.get('embed_dim')}")
    print(f"         Seq Length: {cfg.get('sequence_length', 1024)}")
    print(f"         Target    : {cfg.get('target_device')}")

    replacements = {
        "{{MODEL_NAME}}": str(cfg.get("model_name")),
        "{{EMBED_DIM}}": str(cfg.get("embed_dim")),
        "{{SEQ_LEN}}": str(cfg.get("sequence_length", 1024)),
        "{{NACC}}": "16",
        "{{MAX_DEPTH}}": "8192",
        "{{DATA_TYPE}}": "float"
    }

    # Generate LayerNorm
    ln_template = case_dir / "layernorm.cpp.template"
    if ln_template.exists():
        content = ln_template.read_text(encoding="utf-8")
        for k, v in replacements.items():
            content = content.replace(k, v)
        target = out_dir / "layernorm_kernel_gen.cpp"
        target.write_text(content, encoding="utf-8")
        print(f"  [+] Generated: {target}")

    # Generate Softmax
    sm_template = case_dir / "softmax.cpp.template"
    if sm_template.exists():
        content = sm_template.read_text(encoding="utf-8")
        for k, v in replacements.items():
            content = content.replace(k, v)
        target = out_dir / "softmax_kernel_gen.cpp"
        target.write_text(content, encoding="utf-8")
        print(f"  [+] Generated: {target}")

    # Generate GELU
    gelu_template = case_dir / "gelu.cpp.template"
    if gelu_template.exists():
        content = gelu_template.read_text(encoding="utf-8")
        for k, v in replacements.items():
            content = content.replace(k, v)
        target = out_dir / "gelu_kernel_gen.cpp"
        target.write_text(content, encoding="utf-8")
        print(f"  [+] Generated: {target}")

    print("[HG-PIPE] Step 0 complete: All case templates populated.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HG-PIPE Parametric Case Generator")
    parser.add_argument("--config", default="statistics/vit_base_config.json", help="Path to model config")
    parser.add_argument("--out", default="src/hls/instances", help="Output directory for generated cases")
    args = parser.parse_args()
    generate_cases(args.config, args.out)
