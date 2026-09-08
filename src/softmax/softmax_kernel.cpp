/**
 * @file softmax_kernel.cpp
 * @brief High-Throughput Numerically Stable Softmax Kernel for AMD Versal / UltraScale+.
 *
 * Implements a 3-pass row-wise Softmax activation algorithm:
 *   1. Find max(x) across the row vector for numerical stability (prevents exp overflow).
 *   2. Compute exp(x - max(x)) and accumulate into 16 partial accumulator registers.
 *   3. Multiply each exponentiated element by the inverted total sum to yield probabilities.
 */

#include "softmax_kernel.h"
#include <cmath>

extern "C" {
void softmax_kernel(
    const fp32_t in[MAX_DEPTH],
    fp32_t out[MAX_DEPTH],
    int total_rows,
    int row_length
) {
    // --------------------------------------------------------------------------
    // AXI Interface Synthesis Pragmas
    // --------------------------------------------------------------------------
    #pragma HLS INTERFACE m_axi port=in  bundle=gmem0 depth=MAX_DEPTH offset=slave max_read_burst_length=64
    #pragma HLS INTERFACE m_axi port=out bundle=gmem1 depth=MAX_DEPTH offset=slave max_write_burst_length=64
    #pragma HLS INTERFACE s_axilite port=total_rows bundle=control
    #pragma HLS INTERFACE s_axilite port=row_length bundle=control
    #pragma HLS INTERFACE s_axilite port=return     bundle=control

    // --------------------------------------------------------------------------
    // On-Chip Buffers & Hardware Partitioning
    // --------------------------------------------------------------------------
    // Local row buffer partitioned cyclicly by factor 16 for high-bandwidth parallel access
    fp32_t local_row[SOFTMAX_MAX_LEN];
    #pragma HLS ARRAY_PARTITION variable=local_row cyclic factor=16

    // 16-way round-robin partial accumulators to decouple loop-carried fadd latency
    fp32_t acc_sum[NACC];
    #pragma HLS ARRAY_PARTITION variable=acc_sum complete

    // Iterate through all independent sequence rows
    Row_Loop: for (int r = 0; r < total_rows; r++) {
        int base = r * row_length;

        // ======================================================================
        // Stage 1: Online Maximum Search (Numerical Stability)
        // ======================================================================
        fp32_t max_val = -1e30f;
        Find_Max: for (int c = 0; c < row_length; c++) {
            #pragma HLS PIPELINE II=1
            fp32_t v = in[base + c];
            local_row[c] = v;
            if (v > max_val) {
                max_val = v;
            }
        }

        // ======================================================================
        // Stage 2: Exponential Transformation & 16-Way Accumulation
        // ======================================================================
        // Reset partial accumulators to zero
        Init_Acc: for (int a = 0; a < NACC; a++) {
            #pragma HLS UNROLL
            acc_sum[a] = 0.0f;
        }

        Exp_Sum: for (int c = 0; c < row_length; c++) {
            #pragma HLS PIPELINE II=1
            fp32_t e = expf(local_row[c] - max_val);
            local_row[c] = e;
            acc_sum[c % NACC] += e; // Interleaved accumulation breaks latency bottleneck
        }

        // Parallel reduction of 16 partial sums into scalar denominator
        fp32_t total_sum = 0.0f;
        Reduce_Sum: for (int a = 0; a < NACC; a++) {
            #pragma HLS UNROLL
            total_sum += acc_sum[a];
        }
        fp32_t inv_sum = 1.0f / total_sum;

        // ======================================================================
        // Stage 3: Probability Normalisation
        // ======================================================================
        Normalise: for (int c = 0; c < row_length; c++) {
            #pragma HLS PIPELINE II=1
            out[base + c] = local_row[c] * inv_sum;
        }
    }
}
}
