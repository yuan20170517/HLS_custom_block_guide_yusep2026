# 🧩 Quantized Multi-Head Attention Kernel (`mha_kernel.xo`)

## 1. Overview & Architectural Role

Multi-Head Attention (MHA) is the computational centerpiece of modern Transformer architectures:

$$\text{Attention}(Q, K, V) = \text{Softmax}\left(\frac{Q K^T}{\sqrt{d_k}} + M\right) V$$

Where:
* $Q, K, V$: Query, Key, and Value projection matrices.
* $M$: Causal lower-triangular mask preventing attention to future token positions ($j > i$).
* $\text{Softmax}$: Row-wise exponential normalization.

`mha_kernel.xo` provides a synthesizable, hardware-quantized implementation executing on-chip within the Programmable Logic (PL) of the **AMD Versal AI Edge VEK385**, or functioning as a hardware fallback for models where NPU graphs are not natively mapped.

---

## 2. Hardware Architecture & Interfaces

```text
  Input Tokens [SEQ x DIM]
             │
      ┌──────┴──────┐
      │  mha_top.cpp│
      └──────┬──────┘
             │
             ▼
  ┌─────────────────────────────────────────────────────────────┐
  │ Stage 1: Parallel Q, K, V Linear Projections (II = 1)       │
  ├─────────────────────────────────────────────────────────────┤
  │ Stage 2: Multi-Head Dot Product (Q * K^T) + Causal Masking  │
  ├─────────────────────────────────────────────────────────────┤
  │ Stage 3: Look-Up Table (LUT) Quantized Softmax Evaluation   │
  ├─────────────────────────────────────────────────────────────┤
  │ Stage 4: Attended Value Reduction (Weights * V)             │
  └──────────────────────────────┬──────────────────────────────┘
                                 │
                                 ▼
                     Attended Tokens [SEQ x DIM]
```

### Interface Table

| Port Name | Protocol | Bundle | Width | Description |
| :--- | :--- | :--- | :---: | :--- |
| `in_tokens` | `m_axi` | `gmem0` | 8-bit int | Input token feature matrix |
| `weight_q` | `m_axi` | `gmem1` | 8-bit int | Query projection weight matrix |
| `weight_k` | `m_axi` | `gmem1` | 8-bit int | Key projection weight matrix |
| `weight_v` | `m_axi` | `gmem1` | 8-bit int | Value projection weight matrix |
| `out_tokens` | `m_axi` | `gmem2` | 16-bit int | Output attended token matrix |
| `num_sequences` | `s_axilite` | `control` | 32-bit int | Sequence batch count |

---

## 3. Compilation & Packaging

To compile and package into `xo/mha_kernel.xo`:
```bash
python scripts/build_closed_loop_xo.py --kernel mha
```
Or directly using Vitis HLS:
```bash
vitis-run --mode hls --config config/hls_mha.cfg --work_dir build/hls_mha
```
