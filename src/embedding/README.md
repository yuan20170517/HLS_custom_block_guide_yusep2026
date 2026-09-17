# Token & Position Embedding Streamer (`embedding_kernel.xo`)

## 1. What It Does (Mathematical & Algorithmic Role)

In autoregressive Transformer models (NanoGPT, GPT-2), input tokens are discrete integer IDs converted into continuous vector representations prior to Layer 0:

$$\mathbf{x}_0 = \text{WTE}[\text{token\_id}] + \text{WPE}[\text{pos\_id}] \in \mathbb{R}^{d_{\text{model}}} \quad (d_{\text{model}} = 768)$$

* **WTE (Word Token Embedding):** Matrix $[50,257 \times 768]$ representing vocabulary tokens.
* **WPE (Word Position Embedding):** Matrix $[1,024 \times 768]$ representing sequence positions.
* **Vector Sum $\mathbf{x}_0$:** Streamed directly into the Transformer attention pipeline.

### Algorithmic & Hardware Implementation
* **NPU Scatter-Gather Bypass:** Non-linear address translation and table indexing on NPU vector engines (AMD AIE-ML v2) cause pipeline bubbles. Implementing embedding lookup in the Programmable Logic (PL) avoids stall cycles.
* **Wire-Speed PLIO Streaming ($II = 1$):** Fetches both table rows via AXI4-Master burst reads from LPDDR4 and streams the summed vectors directly into the AIE-ML array over **AXI4-Stream (PLIO)**, completely eliminating CPU-to-accelerator PCIe DMA latency ($5\text{--}15\,\mu\text{s}$).
* **Hardware Interfaces:**
  * `wte_table`: AXI4-Master (`m_axi`, bundle `gmem0`), token embedding matrix in LPDDR4.
  * `wpe_table`: AXI4-Master (`m_axi`, bundle `gmem1`), position embedding matrix in LPDDR4.
  * `stream_to_aie`: AXI4-Stream (`axis`), direct low-latency PLIO stream to Versal AIE-ML array.
  * `token_id`, `pos_id`, `embed_dim`: AXI-Lite (`s_axilite`, bundle `control`), runtime parameters.

---

## 2. Verification & Benchmark Execution

### Performance Benchmark & Hardware Telemetry
```bash
python src/embedding/embedding_golden.py
```
* **Host CPU Latency:** $35.48\,\mu\text{s}$ per token ($28,181\text{ tokens/s}$).
* **FPGA PL Latency (VEK385 @ 312.5 MHz):** **$2.458\,\mu\text{s}$** per token ($406,901\text{ tokens/s}$).
* **Speedup vs CPU:** **$14.4\times$** ($16.9\times$ in master suite).
* **Speedup vs GPU:** **$4.9\times$** (eliminates discrete GPU PCIe kernel launch latency).

### Functional Testbench Execution
* **Algorithmic Unit Tests:**
  ```bash
  python -m unittest tests/test_embedding.py -v
  ```
* **Closed-Loop Pipeline Test:**
  ```bash
  python -m unittest tests/test_closed_loop_pipeline.py -v
  ```
* **Native C++ Testbench:**
  ```bash
  clang++ -std=c++17 -O3 -Isrc/common src/embedding/tb_embedding_kernel.cpp src/embedding/embedding_kernel.cpp -o tb_embedding.exe
  ./tb_embedding.exe
  ```
