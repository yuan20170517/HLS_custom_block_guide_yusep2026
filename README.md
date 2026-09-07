# 🚀 HLS_custom_block_guide_yusep2026

[![Python Tests](https://img.shields.io/badge/Tests-Passing-success.svg)]()
[![Target Silicon](https://img.shields.io/badge/Target-AMD%20Versal%20AI%20Edge%20Gen%202%20VEK385-orange.svg)](https://www.xilinx.com/products/silicon-devices/acap/versal-ai-edge.html)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

> **Repository Overview:** Personal engineering repository dedicated to AMD Vitis High-Level Synthesis (HLS) custom hardware blocks, FPGA acceleration kernels, and automated verification workflows targeting the **AMD Versal AI Edge Gen 2 VEK385** platform.

---

## 📁 Repository Structure

```text
HLS_custom_block_guide_yusep2026/
├── src/            # Source modules and hardware kernel entry points
│   ├── __init__.py
│   ├── main.py     # Environment status and configuration
│   └── time_utils.py
├── tests/          # Automated verification testbenches
│   ├── __init__.py
│   ├── test_basic.py
│   └── test_current_time.py
├── .gitignore      # Exclusions for temporary build files
├── LICENSE         # Apache 2.0 Open Source Licence
└── README.md       # Master documentation
```

---

## 🛠️ Toolchains & Target Environment

* **Target Hardware:** AMD Versal AI Edge Gen 2 VEK385 (`xc2ve3858-ssva2112-2MP-e-S`)
* **HLS Toolchain:** AMD Vitis Unified IDE / Vitis HLS (2024.1 – 2026.1+)
* **Verification Environment:** Python 3.10+ (NumPy, standard unittest)

---

## 🧪 Running Tests

Execute the automated verification suite:

```bash
python -m unittest discover -s tests -p "test_*.py" -v
```

---

## 📜 Development Status

*This repository is under active engineering authoring and development.*

*Maintained by Yuan Fangxing | CTSIRI AI-RAN & Hardware Acceleration Engineering.*
