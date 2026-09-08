/**
 * @file layernorm_kernel.cpp
 * @brief High-Throughput Floating-Point Layer Normalization Kernel for AMD Versal / UltraScale+.
 *
 * This kernel executes standard Transformer LayerNorm over continuous token streams.
 * It resolves the cyclic loop-carried feedback dependency in floating-point addition
 * (latency of 4-5 cycles on DSP58 primitives) by utilizing a 16-way Round-Robin
 * partial accumulation tree, achieving deterministic Initiation Interval II = 1.
 */

#include "layernorm_kernel.h"
#include <cmath>

extern "C" {
void layernorm_kernel(
    const fp32_t in[MAX_DEPTH],
    const fp32_t gamma[LN_DIM],
    const fp32_t beta[LN_DIM],
    fp32_t out[MAX_DEPTH],
    int total_tokens
) {
    // --------------------------------------------------------------------------
    // AXI Interface Synthesis Pragmas
    // --------------------------------------------------------------------------
    #pragma HLS INTERFACE m_axi port=in    bundle=gmem0 depth=MAX_DEPTH offset=slave max_read_burst_length=64
    #pragma HLS INTERFACE m_axi port=gamma bundle=gmem1 depth=LN_DIM    offset=slave max_read_burst_length=64
    #pragma HLS INTERFACE m_axi port=beta  bundle=gmem2 depth=LN_DIM    offset=slave max_read_burst_length=64
    #pragma HLS INTERFACE m_axi port=out   bundle=gmem3 depth=MAX_DEPTH offset=slave max_write_burst_length=64
    #pragma HLS INTERFACE s_axilite port=total_tokens bundle=control
    #pragma HLS INTERFACE s_axilite port=return       bundle=control

    // --------------------------------------------------------------------------
    // On-Chip Buffers & Hardware Partitioning
    // --------------------------------------------------------------------------
    // 16-way round-robin partial accumulators to decouple loop-carried fadd latency
    fp32_t acc_mu[NACC];
    fp32_t acc_var[NACC];
    #pragma HLS ARRAY_PARTITION variable=acc_mu complete
    #pragma HLS ARRAY_PARTITION variable=acc_var complete

    // On-chip intermediate token buffer with cyclic banking to allow concurrent memory accesses
    fp32_t local_buf[LN_DIM];
    #pragma HLS ARRAY_PARTITION variable=local_buf cyclic factor=16

    // Iterate through all incoming token vectors
    Token_Loop: for (int t = 0; t < total_tokens; t++) {
        int base = t * LN_DIM;

        // Reset partial accumulators to zero
        Init_Acc: for (int a = 0; a < NACC; a++) {
            #pragma HLS UNROLL
            acc_mu[a] = 0.0f;
            acc_var[a] = 0.0f;
        }

        // ======================================================================
        // Stage 1: Mean (mu) Accumulation with II = 1
        // ======================================================================
        Mean_Loop: for (int i = 0; i < LN_DIM; i++) {
            #pragma HLS PIPELINE II=1
            fp32_t val = in[base + i];
            local_buf[i] = val;
            acc_mu[i % NACC] += val; // Interleaved accumulation breaks latency bottleneck
        }

        // Parallel reduction of the 16 partial sums into total mean
        fp32_t sum_mu = 0.0f;
        Reduce_Mu: for (int a = 0; a < NACC; a++) {
            #pragma HLS UNROLL
            sum_mu += acc_mu[a];
        }
        fp32_t mean = sum_mu / (fp32_t)LN_DIM;

        // ======================================================================
        // Stage 2: Variance (sigma^2) Accumulation with II = 1
        // ======================================================================
        Var_Loop: for (int i = 0; i < LN_DIM; i++) {
            #pragma HLS PIPELINE II=1
            fp32_t diff = local_buf[i] - mean;
            acc_var[i % NACC] += diff * diff;
        }

        // Parallel reduction of the 16 partial variance sums
        fp32_t sum_var = 0.0f;
        Reduce_Var: for (int a = 0; a < NACC; a++) {
            #pragma HLS UNROLL
            sum_var += acc_var[a];
        }
        // Compute standard deviation reciprocal with epsilon stability term
        fp32_t inv_std = 1.0f / sqrtf((sum_var / (fp32_t)LN_DIM) + 1e-5f);

        // ======================================================================
        // Stage 3: Normalisation, Affine Scale (gamma) & Shift (beta)
        // ======================================================================
        Norm_Loop: for (int i = 0; i < LN_DIM; i++) {
            #pragma HLS PIPELINE II=1
            fp32_t norm = (local_buf[i] - mean) * inv_std;
            out[base + i] = norm * gamma[i] + beta[i];
        }
    }
}
}
