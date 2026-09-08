/**
 * @file softmax_kernel.h
 * @brief Numerically Stable Row-Wise Softmax Hardware Kernel Header for Vitis HLS.
 *
 * Implements row-wise Softmax activation:
 *   S[i] = exp(x[i] - max(x)) / sum(exp(x[j] - max(x)))
 *
 * Architecture Highlights:
 *   - Online max-value subtraction for 100% numerical stability against floating-point overflow.
 *   - 16-way Round-Robin Partial Accumulators (NACC = 16) to close timing with II = 1.
 *   - AXI4-Master memory interfaces (m_axi) with AXI4-Lite slave control.
 */

#ifndef SOFTMAX_KERNEL_H
#define SOFTMAX_KERNEL_H

#include "../common/types.h"

#define SOFTMAX_MAX_LEN 1024 ///< Maximum supported sequence length per row

extern "C" {
/**
 * @brief Top-Level Hardware Kernel for Row-Wise Softmax Activation.
 *
 * @param[in]  in          Input unnormalized logit tensor in off-chip memory [MAX_DEPTH].
 * @param[out] out         Output normalized probability distribution [MAX_DEPTH].
 * @param[in]  total_rows  Total number of independent sequence rows to normalize.
 * @param[in]  row_length  Number of logit elements per sequence row (<= SOFTMAX_MAX_LEN).
 */
void softmax_kernel(
    const fp32_t in[MAX_DEPTH],
    fp32_t out[MAX_DEPTH],
    int total_rows,
    int row_length
);
}

#endif // SOFTMAX_KERNEL_H
