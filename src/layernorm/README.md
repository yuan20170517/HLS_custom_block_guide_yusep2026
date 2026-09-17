# Layer Normalization (`layernorm_kernel.xo`)

## 1. What It Does (Mathematical & Algorithmic Role)

Layer Normalization normalises feature activations within a single token vector independently of the batch size:

$$\mu = \frac{1}{D} \sum_{i=1}^D x_i, \quad \sigma^2 = \frac{1}{D} \sum_{i=1}^D (x_i - \mu)^2$$
$$y_i = \gamma_i \cdot \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta_i$$

Where $D = 768$ is the hidden feature dimension, $\gamma$ and $\beta$ are learnable scale and bias vectors, and $\epsilon = 10^{-5}$ prevents division by zero.

### Algorithmic & Hardware Implementation
* **16-Way Accumulator Bank ($II = 1$):** Floating-point addition on DSP58 blocks requires a 4-to-5 cycle pipeline latency. Naive loop accumulation creates loop-carried dependencies (`sum += x[i]`). The kernel implements a 16-way round-robin partial accumulator array (`NACC = 16`), allowing 16 clock cycles before register reuse and guaranteeing an Initiation Interval of **$II = 1$**.
* **Hardware Interfaces:**
  * `in`: AXI4-Master (`m_axi`, bundle `gmem0`, depth = 768), input feature activations.
  * `gamma`, `beta`: AXI4-Master (`m_axi`, bundles `gmem1`, `gmem2`), scale and bias vectors.
  * `out`: AXI4-Master (`m_axi`, bundle `gmem3`, depth = 768), normalised output features.
  * `total_tokens`: AXI-Lite (`s_axilite`, bundle `control`), sequence length.

---

## 2. Verification & Benchmark Execution

### Performance Benchmark & Hardware Telemetry
```bash
python src/layernorm/layernorm_golden.py
```
* **Host CPU Latency:** $151.5\,\mu\text{s}$ per token ($D = 768$).
* **FPGA PL Latency (VEK385 @ 300 MHz):** **$2.60\,\mu\text{s}$** (780 clock cycles, $II = 1$).
* **Hardware Speedup:** **$58.3\times$** over host CPU.

### Functional Testbench Execution
* **Native C++ Testbench:**
  ```bash
  clang++ -std=c++17 -O3 -Isrc/common src/layernorm/tb_layernorm_kernel.cpp src/layernorm/layernorm_kernel.cpp -o tb_layernorm.exe
  ./tb_layernorm.exe
  ```
* **Master Benchmark Suite:**
  ```bash
  python scripts/run_benchmarks.py
  ```
