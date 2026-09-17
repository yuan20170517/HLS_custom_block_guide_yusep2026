# 2D 8x8 Discrete Cosine Transform (`dct`)

## 1. What It Does (Mathematical & Algorithmic Role)

The 2-Dimensional Discrete Cosine Transform (2D-DCT) transforms spatial-domain sample blocks into orthogonal frequency components:

$$X_{k_1, k_2} = \sum_{n_1=0}^{7} \sum_{n_2=0}^{7} x_{n_1, n_2} \cos\left[\frac{\pi}{8}\left(n_1+\frac{1}{2}\right)k_1\right] \cos\left[\frac{\pi}{8}\left(n_2+\frac{1}{2}\right)k_2\right]$$

### Algorithmic & Hardware Implementation
* **Separable 2D Decomposition:** Decomposes the $\mathcal{O}(N^4)$ transform into two consecutive 1D-DCT stages: 8 row-wise 1D transforms, an intermediate dual-port BRAM transpose memory, and 8 column-wise 1D transforms (16 total 1D passes).
* **Pipelined Fixed-Point Math:** Utilises 16-bit signed integer words (`DW = 16`, 64 elements per block) and a precomputed cosine coefficient array (`dct_coeff_table.txt`), achieving pipelined execution at **$II = 1$**.

---

## 2. Verification & Benchmark Execution

### C-Simulation against Golden Test Vector
```bash
clang++ -std=c++17 -O3 -Isrc/dct src/dct/dct_test.cpp src/dct/dct.cpp -o dct_test.exe
./dct_test.exe
```
* Ingests 64 canonical test samples from `in.dat`.
* Performs sample-by-sample diff comparison against golden hardware output `out.golden.dat`.
* Emits zero mismatches upon completion.

### Functional Testbench Execution
* **Repository Integrity Unit Tests:**
  ```bash
  python -m unittest tests/test_basic.py -v
  ```
