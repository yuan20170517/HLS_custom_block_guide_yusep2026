# Quantized Multi-Head Attention Core (`mha_kernel.xo`)

## 1. What It Does (Mathematical & Algorithmic Role)

Multi-Head Attention (MHA) allows Transformer representations to jointly attend to information from distinct representation subspaces across sequence tokens:

$$\text{head}_h = \text{Softmax}\left(\frac{Q_h K_h^T}{\sqrt{d_k}} + M\right) V_h$$
$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \dots, \text{head}_H) W^O$$

Where $M$ is the causal triangular mask preventing query positions from attending to subsequent key tokens.

### Algorithmic & Hardware Implementation
* **Pipelined INT8 Attention Core:** Implements parallel Q/K/V projections, fixed-point causal masked dot products scaled via bit shifts, a 16-entry piecewise LUT Softmax, and weighted value accumulation.
* **Deterministic Execution:** While custom attention variants (causal masking, FlashAttention chunking) frequently trigger partition fallback to host CPU in Vitis AI, `mha_kernel.xo` runs deterministically in the Versal PL fabric in exactly 2,560 clock cycles (**$8.19\,\mu\text{s}$**).
* **Hardware Interfaces:**
  * `in_tokens`: AXI4-Master (`m_axi`, bundle `gmem0`, depth = 256), input token activations.
  * `weight_q`, `weight_k`, `weight_v`: AXI4-Master (`m_axi`, bundle `gmem1`, depth = 256), INT8 projection weights.
  * `out_tokens`: AXI4-Master (`m_axi`, bundle `gmem2`, depth = 256), output attention features.
  * `num_sequences`: AXI-Lite (`s_axilite`, bundle `control`), sequence length.

---

## 2. Verification & Benchmark Execution

### Performance Benchmark & Hardware Telemetry
```bash
python src/mha_kernel/mha_golden.py
```
* **Host CPU Latency:** $2,373.54\,\mu\text{s}$ ($2.37\text{ ms}$).
* **FPGA PL Latency (VEK385 @ 312.5 MHz):** **$8.19\,\mu\text{s}$** (2,560 clock cycles).
* **Hardware Speedup:** **$289.7\times$** over host CPU ($295.0\times$ in master suite).

### Functional Testbench Execution
* **Closed-Loop Pipeline Test:**
  ```bash
  python -m unittest tests/test_closed_loop_pipeline.py -v
  ```
* **Native C++ Testbench:**
  ```bash
  clang++ -std=c++17 -O3 -Isrc/common src/mha_kernel/tb_mha_kernel.cpp src/mha_kernel/mha_kernel.cpp src/mha_kernel/mha_top.cpp -o tb_mha.exe
  ./tb_mha.exe
  ```
