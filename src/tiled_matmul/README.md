# Output-Stationary Tiled Matrix Multiplication (`tiled_matmul_kernel`)

## 1. What It Does (Mathematical & Algorithmic Role)

General Matrix Multiplication (GEMM) forms the core mathematical operator for linear projections and feed-forward networks:

$$\mathbf{C} = \mathbf{A} \times \mathbf{B}, \quad C_{i, j} = \sum_{k=0}^{K-1} A_{i, k} \cdot B_{k, j}$$

Where $\mathbf{A} \in \mathbb{R}^{M \times K}$, $\mathbf{B} \in \mathbb{R}^{K \times N}$, and $\mathbf{C} \in \mathbb{R}^{M \times N}$ ($M = 16, N = 16, K = 16$).

### Algorithmic & Hardware Implementation
* **2D Output-Stationary Systolic Tile ($II = 1$):** Subdivides matrix dimensions into $4 \times 4$ spatial sub-blocks (`TM_TILE = 4`). Accumulator registers are completely partitioned to evaluate 16 simultaneous DSP MAC operations per cycle.
* **Minimized Memory Bandwidth:** Partial sums remain stationary in local registers until the inner contraction dimension $K$ is fully accumulated, eliminating redundant read/write traffic.
* **Hardware Matrix Interfaces:**
  * `A`: Input matrix $[16][16]$ (8-bit signed quantized, `int8_t_hls`).
  * `B`: Input matrix $[16][16]$ (8-bit signed quantized, `int8_t_hls`).
  * `C`: Output accumulated matrix $[16][16]$ (32-bit signed integer, `int32_t_hls`).

---

## 2. Verification & Benchmark Execution

### Native C++ Testbench Execution
```bash
clang++ -std=c++17 -O3 -Isrc/common src/tiled_matmul/tb_tiled_matmul.cpp src/tiled_matmul/tiled_matmul.cpp -o tb_matmul.exe
./tb_matmul.exe
```
* Multiplies $16 \times 16$ test matrices in the C++ hardware kernel.
* Evaluates element-by-element equality against CPU matrix multiplication for all 256 matrix cells.
* Emits `[PASS] Tiled MatMul C-Simulation Successful! (0 mismatches)`.

### Functional Testbench Execution
* **Repository Integrity Unit Tests:**
  ```bash
  python -m unittest tests/test_basic.py -v
  ```
