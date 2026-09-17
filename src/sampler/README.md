# Output Token Retrieval & Sampler (`sampler_kernel.xo`)

## 1. What It Does (Mathematical & Algorithmic Role)

At the final stage of autoregressive token generation, the Transformer language model head projects the hidden state $\mathbf{h} \in \mathbb{R}^{d_{\text{model}}}$ into unnormalised logits across the vocabulary:

$$\mathbf{z} = \mathbf{h} \cdot \mathbf{W}_{\text{head}}^T \in \mathbb{R}^{V} \quad (V = 50,257)$$

To select the next discrete token ID:
1. **Temperature Scaling:**
   $$\tilde{z}_i = \frac{z_i}{T} \quad (T > 0)$$
2. **Greedy ArgMax Retrieval:**
   $$y_{\text{next}} = \arg\max_{i \in \{0, \dots, V-1\}} \tilde{z}_i$$
3. **Top-K Candidate Extraction:**
   Maintains the top $K$ ($K \le 8$) highest-probability candidate tokens and their logit values for stochastic or speculative sampling.

### Algorithmic & Hardware Implementation
* **PCIe Transfer Elimination (200 KB $\rightarrow$ 4 Bytes):** Host-based token selection requires transferring all 50,257 floating-point logits ($201\text{ KB}$) over PCIe for every generated token ($80\text{--}250\,\mu\text{s}$ DMA penalty). `sampler_kernel.xo` resides in the Programmable Logic (PL) directly attached to LPDDR4, streaming logits at **$II = 1$** and returning only the 4-byte `best_token_id` over AXI-Lite.
* **NPU Core Preservation:** Offloading the reduction scan prevents tying up AMD AIE-ML v2 matrix processing tiles with non-GEMM serial logic.
* **Hardware Interfaces:**
  * `logits`: AXI4-Master (`m_axi`, bundle `gmem0`, depth = 50,257), input logit memory.
  * `best_token_id`: AXI-Lite (`s_axilite`, bundle `control`), output winning token ID (4 bytes).
  * `best_logit`: AXI-Lite (`s_axilite`, bundle `control`), maximum scaled logit value.
  * `top_k_indices` / `top_k_logits`: AXI4-Master (`m_axi`, bundle `gmem1`, depth = 8), top candidate tokens.
  * `vocab_size`, `temperature`: AXI-Lite (`s_axilite`, bundle `control`), runtime configuration.

---

## 2. Verification & Benchmark Execution

### Performance Benchmark & Hardware Telemetry
```bash
python src/sampler/sampler_golden.py
```
* **Host CPU Latency:** $16,282.07\,\mu\text{s}$ per token ($61\text{ tokens/s}$).
* **FPGA PL Latency (VEK385 @ 312.5 MHz):** **$160.82\,\mu\text{s}$** per token ($6,218\text{ tokens/s}$).
* **Hardware Speedup:** **$101.2\times$** over host CPU ($122.3\times$ in master suite).

### Functional Testbench Execution
* **Closed-Loop Pipeline Integration Test:**
  ```bash
  python -m unittest tests/test_closed_loop_pipeline.py -v
  ```
* **Native C++ Testbench:**
  ```bash
  clang++ -std=c++17 -O3 -Isrc/common src/sampler/tb_sampler_kernel.cpp src/sampler/sampler_kernel.cpp -o tb_sampler.exe
  ./tb_sampler.exe
  ```
