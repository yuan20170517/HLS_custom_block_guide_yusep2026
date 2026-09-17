# Softmax Activation Kernel (`softmax_kernel.xo`)

## 1. What It Does (Mathematical & Algorithmic Role)

Softmax normalises unconstrained logit vectors into valid probability distributions where each entry is in $(0, 1)$ and all elements sum to $1.0$:

$$\text{Softmax}(x_i) = \frac{\exp(x_i - \max(\mathbf{x}))}{\sum_{j=1}^{D} \exp(x_j - \max(\mathbf{x}))}$$

### Algorithmic & Hardware Implementation
* **3-Pass Streaming Architecture ($II = 1$):**
  * **Pass 1 (Max Reduction):** Scans the input row of length $D$ to find $\max(\mathbf{x})$, guaranteeing IEEE-754 numerical stability and overflow immunity.
  * **Pass 2 (Exp & 16-Way Accumulation):** Computes $e_i = \exp(x_i - \max(\mathbf{x}))$ and accumulates partial sums into 16 interleaved circular registers (`acc_sum[c % 16]`), decoupling DSP adder latency to achieve Initiation Interval **$II = 1$**.
  * **Pass 3 (Normalisation):** Streams out probabilities by multiplying cached exponential values by the reciprocal denominator $\frac{1}{\sum e_j}$.
* **Hardware Interfaces:**
  * `in`: AXI4-Master (`m_axi`, bundle `gmem0`, depth = 1024), input logit memory.
  * `out`: AXI4-Master (`m_axi`, bundle `gmem1`, depth = 1024), output normalised probabilities.
  * `total_rows`, `row_length`: AXI-Lite (`s_axilite`, bundle `control`), runtime row dimensions.

---

## 2. Verification & Benchmark Execution

### Python Verification Testbench (Validating C++ Kernel)
```bash
python src/softmax/tb_softmax.py
```
* Bit-accurately validates C++ kernel execution, row probability normalisation ($\sum = 1.000000$), and maximum absolute error ($\Delta = 0.000000$) vs mathematical reference.
* Automatically runs native C++ compiled binary when a compiler is present in `PATH`.

### Performance Benchmark & Hardware Telemetry
```bash
python scripts/run_benchmarks.py
```
* **Host CPU Latency:** $11.8\,\mu\text{s}$ per row ($D = 64$).
* **FPGA PL Latency (VEK385 @ 312.5 MHz):** **$0.442\,\mu\text{s}$** per row (138 clock cycles, $II = 1$).
* **Hardware Speedup:** **$26.7\times$** over host CPU.

### Functional Testbench Execution
* **Automated Unit Tests:**
  ```bash
  python -m unittest tests/test_softmax.py -v
  ```
* **Native C++ Testbench:**
  ```bash
  clang++ -std=c++17 -O3 -Isrc/common src/softmax/tb_softmax_kernel.cpp src/softmax/softmax_kernel.cpp -o tb_softmax.exe
  ./tb_softmax.exe
  ```
