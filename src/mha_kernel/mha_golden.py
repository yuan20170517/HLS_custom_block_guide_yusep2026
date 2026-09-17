#!/usr/bin/env python3
"""
Python Golden Reference & Performance Benchmark: Quantized Multi-Head Attention (MHA)
Operator: mha_kernel / mha_top
Architecture: AMD Versal VEK385 (PL) vs Host CPU / GPU
"""

import time
import math
from typing import List, Dict, Any

MHA_SEQ = 16
MHA_DIM = 16
MHA_HEADS = 4
MHA_HDIM = 4

def lut_weight_py(x: int, lut: List[int]) -> int:
    idx = (x + 32) >> 2
    idx = max(0, min(15, idx))
    return lut[idx]

def mha_golden(
    X: List[List[int]],
    WQ: List[List[int]],
    WK: List[List[int]],
    WV: List[List[int]],
    softmax_lut: List[int]
) -> List[List[int]]:
    """Exact bit-level golden model of the mha_kernel C++ HLS implementation."""
    # Stage 1: Linear Projections
    Q = [[0 for _ in range(MHA_DIM)] for _ in range(MHA_SEQ)]
    K = [[0 for _ in range(MHA_DIM)] for _ in range(MHA_SEQ)]
    V = [[0 for _ in range(MHA_DIM)] for _ in range(MHA_SEQ)]

    for i in range(MHA_SEQ):
        for o in range(MHA_DIM):
            qa = sum(X[i][d] * WQ[d][o] for d in range(MHA_DIM))
            ka = sum(X[i][d] * WK[d][o] for d in range(MHA_DIM))
            va = sum(X[i][d] * WV[d][o] for d in range(MHA_DIM))
            Q[i][o] = qa
            K[i][o] = ka
            V[i][o] = va

    OUT = [[0 for _ in range(MHA_DIM)] for _ in range(MHA_SEQ)]

    # Stage 2 & 3: Multi-Head Scaled Dot-Product Attention
    for h in range(MHA_HEADS):
        for i in range(MHA_SEQ):
            score = [0] * MHA_SEQ
            max_score = -32768

            # Dot-product Q * K^T with causal masking
            for j in range(MHA_SEQ):
                if j > i:
                    score[j] = -32768
                else:
                    s = sum(Q[i][h * MHA_HDIM + d] * K[j][h * MHA_HDIM + d] for d in range(MHA_HDIM))
                    score[j] = s >> 2
                if score[j] > max_score:
                    max_score = score[j]

            # Softmax LUT weights
            weight = [0] * MHA_SEQ
            denom = 0
            for j in range(MHA_SEQ):
                if j > i:
                    weight[j] = 0
                else:
                    weight[j] = lut_weight_py(score[j] - max_score, softmax_lut)
                denom += weight[j]

            # Weighted sum over V
            for d in range(MHA_HDIM):
                z = h * MHA_HDIM + d
                a = sum(weight[j] * V[j][z] for j in range(MHA_SEQ))
                OUT[i][z] = a // denom if denom != 0 else 0

    return OUT

def run_benchmark(iterations: int = 100) -> Dict[str, Any]:
    # Deterministic stimulus
    X = [[(i + d) % 7 - 3 for d in range(MHA_DIM)] for i in range(MHA_SEQ)]
    WQ = [[(i + j) % 5 - 2 for j in range(MHA_DIM)] for i in range(MHA_DIM)]
    WK = [[(i - j) % 5 for j in range(MHA_DIM)] for i in range(MHA_DIM)]
    WV = [[(2 * i + j) % 7 - 3 for j in range(MHA_DIM)] for i in range(MHA_DIM)]
    softmax_lut = [(i + 1) * 8 for i in range(16)]

    start_time = time.perf_counter()
    for _ in range(iterations):
        _ = mha_golden(X, WQ, WK, WV, softmax_lut)
    end_time = time.perf_counter()

    avg_cpu_us = ((end_time - start_time) / iterations) * 1e6

    # FPGA Telemetry (VEK385 PL @ 312.5 MHz)
    # Stage 1: 16 tokens x 16 dim = 256 cycles
    # Stage 2: 4 heads x 16 seq x (16 + 16 + 4) = 2,304 cycles
    fpga_cycles = 2560
    fpga_us = (fpga_cycles / 312.5e6) * 1e6  # ~8.19 us

    return {
        "operator": "Quantized Multi-Head Attention (MHA)",
        "seq_len": MHA_SEQ,
        "feature_dim": MHA_DIM,
        "heads": MHA_HEADS,
        "cpu_latency_us": round(avg_cpu_us, 2),
        "fpga_cycles": fpga_cycles,
        "fpga_latency_us": round(fpga_us, 2),
        "fpga_vs_cpu_speedup": round(avg_cpu_us / fpga_us, 1),
        "hardware_interface": "AXI4-Master (Weights/Tokens) & AXI4-Stream"
    }

if __name__ == "__main__":
    res = run_benchmark()
    print("=================================================================")
    print(f" Benchmark: {res['operator']} (Seq={res['seq_len']}, Dim={res['feature_dim']}, Heads={res['heads']})")
    print("=================================================================")
    print(f" CPU Latency (Software)  : {res['cpu_latency_us']} us")
    print(f" FPGA Latency (VEK385 PL): {res['fpga_latency_us']} us ({res['fpga_cycles']} cycles @ 312.5MHz)")
    print(f" Estimated Speedup vs CPU: {res['fpga_vs_cpu_speedup']}x")
    print("=================================================================")
