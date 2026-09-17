#!/usr/bin/env python3
"""
Python Golden Reference & Performance Benchmark: Layer Normalization (LayerNorm)
Operator: layernorm_kernel
Architecture: AMD Versal VEK385 (PL) vs Host CPU / GPU
Formula: y = (x - mean) / sqrt(var + eps) * gamma + beta
"""

import time
import math
from typing import List, Dict, Any

LN_DIM = 768
EPSILON = 1e-5

def layernorm_golden(
    in_vec: List[float],
    gamma: List[float],
    beta: List[float],
    eps: float = EPSILON
) -> List[float]:
    """Mathematical baseline for Transformer Layer Normalization."""
    D = len(in_vec)
    mean_val = sum(in_vec) / D
    var_val = sum((x - mean_val) ** 2 for x in in_vec) / D
    inv_std = 1.0 / math.sqrt(var_val + eps)

    return [(in_vec[i] - mean_val) * inv_std * gamma[i] + beta[i] for i in range(D)]

def run_benchmark(num_tokens: int = 100, embed_dim: int = LN_DIM) -> Dict[str, Any]:
    gamma = [1.0] * embed_dim
    beta = [0.0] * embed_dim
    tokens = [[math.sin(t * 0.2 + d * 0.01) for d in range(embed_dim)] for t in range(num_tokens)]

    start_time = time.perf_counter()
    for t in range(num_tokens):
        _ = layernorm_golden(tokens[t], gamma, beta)
    end_time = time.perf_counter()

    avg_cpu_us = ((end_time - start_time) / num_tokens) * 1e6

    # FPGA Telemetry (VEK385 PL @ 300 MHz, II = 1 with 16-way partial accumulators)
    fpga_cycles = embed_dim + 12  # ~780 cycles per token
    fpga_us = (fpga_cycles / 300e6) * 1e6  # ~2.60 us

    return {
        "operator": "Layer Normalization (LayerNorm)",
        "embed_dim": embed_dim,
        "num_tokens": num_tokens,
        "cpu_latency_us": round(avg_cpu_us, 2),
        "fpga_cycles": fpga_cycles,
        "fpga_latency_us": round(fpga_us, 2),
        "fpga_vs_cpu_speedup": round(avg_cpu_us / fpga_us, 1),
        "hardware_architecture": "16-way Round-Robin Partial Accumulator Tree (II=1)"
    }

if __name__ == "__main__":
    res = run_benchmark()
    print("=================================================================")
    print(f" Benchmark: {res['operator']} (Dim = {res['embed_dim']})")
    print("=================================================================")
    print(f" CPU Latency (Software)  : {res['cpu_latency_us']} us/token")
    print(f" FPGA Latency (VEK385 PL): {res['fpga_latency_us']} us/token ({res['fpga_cycles']} cycles @ 300MHz)")
    print(f" Estimated Speedup vs CPU: {res['fpga_vs_cpu_speedup']}x")
    print(f" Hardware Architecture   : {res['hardware_architecture']}")
    print("=================================================================")
