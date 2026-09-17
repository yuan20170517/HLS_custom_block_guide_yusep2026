#!/usr/bin/env python3
"""
Python Golden Reference & Performance Benchmark: Output Token Retrieval & Logit Sampler
Operator: sampler_kernel
Architecture: AMD Versal VEK385 (PL) vs Host CPU / GPU
Formula: best_token_id = argmax(logits / temperature), Top-K extraction
"""

import time
import math
import random
from typing import List, Tuple, Dict, Any

DEFAULT_VOCAB_SIZE = 50257
DEFAULT_TOP_K = 8

def sampler_golden(
    logits: List[float],
    k_val: int = DEFAULT_TOP_K,
    temperature: float = 1.0
) -> Tuple[int, float, List[int], List[float]]:
    """Golden mathematical reference for temperature-scaled ArgMax & Top-K sampling.
    
    Args:
        logits: Unnormalized float logit distribution across vocabulary [50,257].
        k_val: Number of top candidate tokens to extract (1 <= k_val <= 8).
        temperature: Sampling temperature factor (> 0.0).
        
    Returns:
        (best_token_id, best_logit, top_k_indices, top_k_logits)
    """
    inv_temp = 1.0 / temperature if temperature > 1e-4 else 1.0
    scaled_logits = [l * inv_temp for l in logits]

    # Greedy ArgMax
    best_token_id = -1
    best_logit = -1e30
    for idx, val in enumerate(scaled_logits):
        if val > best_logit:
            best_logit = val
            best_token_id = idx

    # Top-K sorting
    k_actual = min(k_val, len(logits))
    sorted_indices = sorted(range(len(scaled_logits)), key=lambda i: scaled_logits[i], reverse=True)[:k_actual]
    top_k_logits = [scaled_logits[i] for i in sorted_indices]

    return best_token_id, best_logit, sorted_indices, top_k_logits

def run_benchmark(vocab_size: int = DEFAULT_VOCAB_SIZE, iterations: int = 50) -> Dict[str, Any]:
    """Measures Python/CPU software sorting latency vs FPGA single-pass pipeline telemetry."""
    # Synthetic logit distribution
    random.seed(42)
    logits = [random.gauss(0, 5.0) for _ in range(vocab_size)]
    logits[9812] = 25.4  # Inject winning token

    start_time = time.perf_counter()
    for _ in range(iterations):
        _ = sampler_golden(logits, k_val=8, temperature=0.8)
    end_time = time.perf_counter()

    total_cpu_time_sec = end_time - start_time
    avg_cpu_latency_us = (total_cpu_time_sec / iterations) * 1e6
    cpu_throughput_tokens_sec = iterations / total_cpu_time_sec

    # FPGA Hardware Telemetry (Versal VEK385 @ 312.5 MHz, II = 1)
    fpga_clock_mhz = 312.5
    fpga_cycles = vocab_size  # 50,257 clock cycles
    fpga_latency_us = (fpga_cycles / (fpga_clock_mhz * 1e6)) * 1e6  # ~160.8 us
    fpga_throughput_tokens_sec = (fpga_clock_mhz * 1e6) / fpga_cycles

    # GPU PCIe Bottleneck Estimation (Transfer 50,257 floats = 201KB over PCIe + Kernel Launch)
    gpu_pcie_transfer_us = 45.0
    gpu_kernel_launch_us = 20.0
    gpu_total_us = gpu_pcie_transfer_us + gpu_kernel_launch_us + 15.0  # ~80 us

    return {
        "operator": "Output Token Retrieval & Sampler (ArgMax + Top-8)",
        "vocab_size": vocab_size,
        "iterations": iterations,
        "cpu_latency_us": round(avg_cpu_latency_us, 2),
        "cpu_throughput_tokens_sec": int(cpu_throughput_tokens_sec),
        "fpga_cycles": fpga_cycles,
        "fpga_latency_us": round(fpga_latency_us, 2),
        "fpga_throughput_tokens_sec": int(fpga_throughput_tokens_sec),
        "fpga_vs_cpu_speedup": round(avg_cpu_latency_us / fpga_latency_us, 1),
        "hardware_interface": "AXI4-Master (50,257 Logits) -> AXI-Lite (Winning Token ID)"
    }

if __name__ == "__main__":
    res = run_benchmark(iterations=20)
    print("=================================================================")
    print(f" Benchmark: {res['operator']} (Vocab = {res['vocab_size']:,})")
    print("=================================================================")
    print(f" CPU Latency (Software)  : {res['cpu_latency_us']:,} us/token ({res['cpu_throughput_tokens_sec']} tokens/s)")
    print(f" FPGA Latency (VEK385 PL): {res['fpga_latency_us']} us/token ({res['fpga_throughput_tokens_sec']:,} tokens/s)")
    print(f" Speedup over CPU Python : {res['fpga_vs_cpu_speedup']}x")
    print(f" Hardware Interface      : {res['hardware_interface']}")
    print("=================================================================")
