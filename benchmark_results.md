# HLS Custom Block Hardware Benchmark & Latency Telemetry

**Target Device:** AMD Versal AI Edge Gen 2 VEK385 (`xc2ve3858`)  
**PL Fabric Clock:** 312.5 MHz ($T_{\text{clk}} = 3.20\text{ ns}$)  
**Pipeline Goal:** Guaranteed Initiation Interval $II = 1$  
**Generated:** 2026-09-17 17:25:30  

| Operator / Custom Block | Hardware Target | Status | CPU Latency (Software) | FPGA PL Cycles | FPGA Latency (VEK385) | Speedup vs CPU | GPU PCIe Latency Avoidance |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Token & Position Embedding (WTE + WPE)** | VEK385 PL | **PASS** | 41.64 us | 768 | **2.458 us** | **16.9x** | Zero-PCIe Stream (Saves ~15us) |
| **Output Token Retrieval & Sampler (ArgMax + Top-8)** | VEK385 PL | **PASS** | 19672.36 us | 50,257 | **160.82 us** | **122.3x** | Saves 200KB DMA (Saves ~80us) |
| **Quantized Multi-Head Attention (MHA)** | VEK385 PL | **PASS** | 2416.77 us | 2,560 | **8.19 us** | **295.0x** | On-Chip Fallback (Zero Jitter) |
| **Layer Normalization (LayerNorm)** | VEK385 PL | **PASS** | 157.33 us | 780 | **2.6 us** | **60.5x** | On-Chip Fallback (Zero Jitter) |
| **Row-Wise Softmax Activation (C++ HLS Kernel)** | VEK385 PL | **PASS** | 11.8 us | 138 | **0.442 us** | **26.7x** | On-Chip Fallback (Zero Jitter) |

---

## GPU vs FPGA Performance Architectural Insights

1. **Host-to-Device PCIe Latency Elimination:**
   - In pure GPU inference, launching kernels on tiny tensors (e.g., embedding lookups or ArgMax token retrieval) incurs a minimum PCIe kernel launch floor of **$5\text{--}25\mu\text{s}$**, plus device-to-host memory copy overhead (**$45\mu\text{s}$** for 200KB logits).
   - On the **AMD Versal VEK385**, both the Embedding and Token Sampler execute **directly inside the Programmable Logic (PL)**, streaming vectors directly to/from the AIE-ML NPU via **AXI4-Stream (PLIO)**. PCIe and Host CPU roundtrips are completely eliminated ($0.0\mu\text{s}$).

2. **Deterministic Real-Time Timing ($II = 1$):**
   - All custom blocks are architected with dedicated circular partial accumulator banks (LayerNorm) or unrolled register arrays (Sampler, Embedding) to guarantee deterministic cycle counts with zero jitter.

3. **Power Efficiency:**
   - Each custom PL block executes within a **$< 3\text{W}$** power envelope, compared to 50-150W for GPU workstation alternatives.