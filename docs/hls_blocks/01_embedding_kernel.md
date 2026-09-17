# 🧩 Token & Position Embedding Streamer Kernel (`embedding_kernel.xo`)

## 1. Overview & Architectural Role

In Transformer-based architectures (such as GPT-2, NanoGPT, and Physical AI VLA models), the input sequence must be converted from discrete integer tokens into continuous high-dimensional vector representations:

$$\mathbf{x}_0 = \text{WTE}[\text{token\_id}] + \text{WPE}[\text{pos\_id}] \in \mathbb{R}^{d_{\text{model}}}$$

Where:
* **WTE (Word Token Embedding):** Matrix of size $[V \times d_{\text{model}}]$ (e.g. $[50,257 \times 768]$ for NanoGPT).
* **WPE (Word Position Embedding):** Matrix of size $[L_{\text{max}} \times d_{\text{model}}]$ (e.g. $[1,024 \times 768]$).
* **$\mathbf{x}_0$:** Initial hidden state vector injected into Layer 0 of the Transformer.

`embedding_kernel.xo` executes this operation entirely in Programmable Logic (PL) on the **AMD Versal AI Edge VEK385**, burst-reading the two rows from LPDDR4/LPDDR5X and streaming the summed vector directly into the AIE-ML NPU via AXI4-Stream (PLIO) at wire speed.

---

## 2. Hardware Architecture & Interfaces

```text
  Off-Chip Memory (LPDDR4 / LPDDR5X)
    ┌──────────────────────┐   ┌──────────────────────┐
    │  WTE Table in DDR    │   │  WPE Table in DDR    │
    │  (50,257 x 768 floats│   │  (1,024 x 768 floats)│
    └──────────┬───────────┘   └──────────┬───────────┘
               │                          │
      AXI4-M (gmem0)             AXI4-M (gmem1)
      Burst Read 64              Burst Read 64
               ▼                          ▼
    ┌─────────────────────────────────────────────────┐
    │ embedding_kernel.cpp (PL Hardware Pipeline)     │
    │                                                 │
    │   [Row Fetch] -> [buf_wte]   [buf_wpe]          │
    │                       │          │              │
    │                       ▼          ▼              │
    │                 Vector Sum: WTE + WPE           │
    │                 Pipeline II = 1                 │
    └────────────────────────┬────────────────────────┘
                             │
                             │ AXI4-Stream (axis)
                             │ 768 floats, TLAST on dim 767
                             ▼
                 AIE-ML NPU Core (PLIO Stream)
```

### Interface Table

| Port Name | Protocol | Bundle | Width | Description |
| :--- | :--- | :--- | :---: | :--- |
| `wte_table` | `m_axi` | `gmem0` | 32-bit float | Pointer to Word Token Embedding matrix in LPDDR |
| `wpe_table` | `m_axi` | `gmem1` | 32-bit float | Pointer to Word Position Embedding matrix in LPDDR |
| `out_embed` | `m_axi` | `gmem2` | 32-bit float | Optional writeback memory buffer |
| `stream_to_aie` | `axis` | - | 32-bit float | Direct streaming channel to AIE NPU |
| `token_id` | `s_axilite` | `control` | 32-bit int | Token ID (0 to 50,256) |
| `pos_id` | `s_axilite` | `control` | 32-bit int | Position index (0 to 1,023) |
| `embed_dim` | `s_axilite` | `control` | 32-bit int | Feature dimension (default 768) |
| `write_to_mem` | `s_axilite` | `control` | 32-bit int | Enable flag for memory writeback |

---

## 3. Latency & Performance Telemetry

* **Clock Frequency:** $312.5\text{ MHz}$ (Period = $3.2\text{ ns}$).
* **Pipeline Throughput:** Initiation Interval $II = 1$.
* **Execution Cycles:** $\approx 768\text{ clock cycles}$.
* **Total Latency:** $\mathbf{2.46\mu\text{s}}$ per token vector.
* **Bandwidth Efficiency:** Zero host CPU copying; eliminates Linux kernel context switches and PCIe DMA setup times.

---

## 4. Compilation & Packaging

To compile and package into `xo/embedding_kernel.xo`:
```bash
python scripts/build_closed_loop_xo.py --kernel embedding
```
Or directly using Vitis HLS:
```bash
vitis-run --mode hls --config config/hls_embedding.cfg --work_dir build/hls_embedding
```
