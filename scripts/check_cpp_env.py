#!/usr/bin/env python3
"""
Environment Diagnostic Tool: Check C++ and Vitis HLS Toolchains on Windows
"""

import os
import shutil
import subprocess
from pathlib import Path

def diagnose():
    print("=================================================================")
    print(" Diagnostic: C++ Compilers & AMD Vitis Toolchain Availability    ")
    print("=================================================================")

    # 1. Check commands in PATH
    tools = ["vitis-run", "vitis_hls", "v++", "vivado", "g++", "clang++", "cl", "wsl"]
    found_in_path = {}
    for t in tools:
        p = shutil.which(t)
        found_in_path[t] = p
        status = f"FOUND: {p}" if p else "NOT FOUND in PATH"
        print(f" {t:<12} : {status}")

    # 2. Check standard Xilinx installation roots
    print("\n--- Scanning Default Installation Directories ---")
    xilinx_paths = [
        Path("C:/Xilinx"),
        Path("D:/Xilinx"),
        Path("C:/Tools/Xilinx"),
        Path("D:/Tools/Xilinx"),
        Path("C:/Program Files/Microsoft Visual Studio"),
        Path("C:/Program Files (x86)/Microsoft Visual Studio"),
        Path("C:/msys64/mingw64/bin"),
        Path("C:/MinGW/bin")
    ]

    detected_paths = []
    for p in xilinx_paths:
        if p.exists():
            print(f" [OK] Detected directory: {p}")
            detected_paths.append(p)
        else:
            print(f" [ ] Not found: {p}")

    # 3. Check WSL if available
    wsl_has_gpp = False
    if found_in_path.get("wsl"):
        try:
            res = subprocess.run(["wsl", "which", "g++"], capture_output=True, text=True, timeout=5)
            if res.returncode == 0 and res.stdout.strip():
                print(f"\n [OK] Linux g++ available inside WSL: {res.stdout.strip()}")
                wsl_has_gpp = True
            else:
                print("\n [ ] WSL detected, but g++ is not installed inside WSL.")
        except Exception as e:
            print(f"\n [!] WSL check encountered error: {e}")

    # 4. Search for settings64.bat in detected Xilinx directories
    settings_scripts = []
    for d in detected_paths:
        if "Xilinx" in str(d):
            for pattern in ["**/settings64.bat", "**/vitis-run.bat"]:
                try:
                    for f in d.glob(pattern):
                        settings_scripts.append(f)
                        print(f" [FOUND SCRIPT]: {f}")
                except Exception:
                    pass

    return {
        "found_in_path": found_in_path,
        "detected_paths": [str(x) for x in detected_paths],
        "settings_scripts": [str(x) for x in settings_scripts],
        "wsl_has_gpp": wsl_has_gpp
    }

if __name__ == "__main__":
    diagnose()
