# GELU Non-Linear Activation & Embedding Kernel (`gelu_embed_kernel`)

## 1. What It Does (Mathematical & Algorithmic Role)

The Gaussian Error Linear Unit (GELU) provides smooth non-linear activation across Transformer representations:

$$\text{GELU}(x) = x \cdot \Phi(x) \approx 0.5x \left(1 + \tanh\left(\sqrt{\frac{2}{\pi}} \left(x + 0.044715 x^3\right)\right)\right)$$

### Algorithmic & Hardware Implementation
* **Single-Cycle Quantized LUT ($II = 1$):** Instead of dedicating floating-point DSP multipliers and dividers to transcendental polynomial evaluations, 8-bit quantized inputs index directly into a precomputed 256-entry Look-Up Table (`gelu_lut[256]`), executing at wire speed with zero DSP utilization.
* **Fused Embedding Retrieval:** Concurrently indexes a 2D weight matrix (`embed_lut[32][8]`) using incoming discrete token IDs.
* **Hardware Interfaces:**
  * `X`: Array of input feature activations (`int8_t_hls[64]`).
  * `token_ids`: Array of input token sequence IDs (`uint8_t_hls[8]`).
  * `gelu_lut`: Precomputed 256-entry Look-Up Table (`int8_t_hls[256]`).
  * `embed_lut`: 2D embedding weight matrix (`int8_t_hls[32][8]`).
  * `gelu_out`: Output activated activations (`int8_t_hls[64]`).
  * `embed_out`: Output retrieved embedding vector (`int8_t_hls[8]`).

---

## 2. Verification & Benchmark Execution

### Functional Testbench Execution
* **Automated Unit Tests:**
  ```bash
  python -m unittest tests/test_gelu_embed.py -v
  ```
  *Validates 64 output activation values and 8 retrieved embedding vectors against golden mathematical baselines with 0 bit errors.*
* **Native C++ Testbench:**
  ```bash
  clang++ -std=c++17 -O3 -Isrc/common src/GELU/tb_gelu_embed_kernel.cpp src/GELU/gelu_embed_kernel.cpp -o tb_gelu.exe
  ./tb_gelu.exe
  ```
