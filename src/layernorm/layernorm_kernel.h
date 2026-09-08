/**
 * @file layernorm_kernel.h
 * @brief Floating-Point Layer Normalization (LayerNorm) Hardware Kernel Header for Vitis HLS.
 *
 * Implements standard Transformer LayerNorm:
 *   y = ((x - mean) / sqrt(var + eps)) * gamma + beta
 *
 * Architecture Highlights:
 *   - AXI4-Master memory interfaces (m_axi) for burst memory transfer to HBM/LPDDR.
 *   - 16-way Round-Robin Partial Accumulators (NACC = 16) to close timing with II = 1.
 *   - Cyclic array partitioning on local buffers to support parallel read/write.
 */

#ifndef LAYERNORM_KERNEL_H
#define LAYERNORM_KERNEL_H

#include "../common/types.h"

#define LN_DIM 768  ///< Transformer hidden dimension (e.g., BERT-Base, NanoGPT)

extern "C" {
/**
 * @brief Top-Level Hardware Kernel for Layer Normalization.
 *
 * @param[in]  in           Input activation tensor in off-chip memory [MAX_DEPTH].
 * @param[in]  gamma        Per-channel affine scale parameter [LN_DIM].
 * @param[in]  beta         Per-channel affine shift parameter [LN_DIM].
 * @param[out] out          Normalized and scaled output tensor in off-chip memory [MAX_DEPTH].
 * @param[in]  total_tokens Number of sequential token feature vectors to process.
 */
void layernorm_kernel(
    const fp32_t in[MAX_DEPTH],
    const fp32_t gamma[LN_DIM],
    const fp32_t beta[LN_DIM],
    fp32_t out[MAX_DEPTH],
    int total_tokens
);
}

#endif // LAYERNORM_KERNEL_H
