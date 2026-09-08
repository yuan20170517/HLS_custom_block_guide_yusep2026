/**
 * @file mha_kernel.cpp
 * @brief High-Throughput Quantized Multi-Head Attention Kernel for Vitis HLS.
 *
 * Implements an on-chip, pipelined Multi-Head Attention module for embedded transformer
 * workloads. Incorporates on-chip Q/K/V linear projections, causal triangular masking,
 * LUT-accelerated non-linear Softmax evaluation, and value reduction.
 */

#include "mha_kernel.hpp"

/**
 * @brief Look-up table lookup helper for quantized Softmax non-linearity.
 *
 * Maps a scaled attention score offset (score - max_score) to an 8-bit quantized
 * probability weight via a pre-computed 16-entry LUT.
 *
 * @param[in] x   Relative score offset.
 * @param[in] lut 16-entry exponential Softmax LUT.
 * @return Quantized attention probability weight (0..255).
 */
static uint8_t_hls lut_weight(int16_t_hls x, const uint8_t_hls lut[16]) {
#pragma HLS INLINE
    int idx = (int)(x + 32) >> 2;
    if (idx < 0) idx = 0;
    if (idx > 15) idx = 15;
    return lut[idx];
}

void mha_kernel(
    const int8_t_hls  X[MHA_SEQ][MHA_DIM],
    const int8_t_hls  WQ[MHA_DIM][MHA_DIM],
    const int8_t_hls  WK[MHA_DIM][MHA_DIM],
    const int8_t_hls  WV[MHA_DIM][MHA_DIM],
    const uint8_t_hls softmax_lut[16],
    int16_t_hls       OUT[MHA_SEQ][MHA_DIM]
) {
    // --------------------------------------------------------------------------
    // Intermediate Projection Buffers
    // --------------------------------------------------------------------------
    int16_t_hls Q[MHA_SEQ][MHA_DIM];
    int16_t_hls K[MHA_SEQ][MHA_DIM];
    int16_t_hls V[MHA_SEQ][MHA_DIM];

    // Partition arrays along the feature dimension to enable parallel MAC trees
    #pragma HLS ARRAY_PARTITION variable=X  complete dim=2
    #pragma HLS ARRAY_PARTITION variable=WQ complete dim=1
    #pragma HLS ARRAY_PARTITION variable=WK complete dim=1
    #pragma HLS ARRAY_PARTITION variable=WV complete dim=1
    #pragma HLS ARRAY_PARTITION variable=Q  complete dim=2
    #pragma HLS ARRAY_PARTITION variable=K  complete dim=2
    #pragma HLS ARRAY_PARTITION variable=V  complete dim=2

    // ==========================================================================
    // Stage 1: Linear Query, Key, and Value Projections
    // ==========================================================================
    proj_tokens:
    for (int i = 0; i < MHA_SEQ; ++i) {
        proj_dim:
        for (int o = 0; o < MHA_DIM; ++o) {
            #pragma HLS PIPELINE II=1
            int32_t_hls qa = 0;
            int32_t_hls ka = 0;
            int32_t_hls va = 0;

            dot_reduction:
            for (int d = 0; d < MHA_DIM; ++d) {
                #pragma HLS UNROLL
                qa += X[i][d] * WQ[d][o];
                ka += X[i][d] * WK[d][o];
                va += X[i][d] * WV[d][o];
            }
            Q[i][o] = qa;
            K[i][o] = ka;
            V[i][o] = va;
        }
    }

    // ==========================================================================
    // Stage 2: Multi-Head Scaled Dot-Product Attention & Aggregation
    // ==========================================================================
    head_loop:
    for (int h = 0; h < MHA_HEADS; ++h) {
        seq_query:
        for (int i = 0; i < MHA_SEQ; ++i) {
            uint8_t_hls  weight[MHA_SEQ];
            int16_t_hls  score[MHA_SEQ];
            int16_t_hls  max_score = -32768;
            uint16_t_hls denom = 0;

            #pragma HLS ARRAY_PARTITION variable=weight complete dim=1

            // 1. Calculate Q * K^T dot product with causal masking
            score_loop:
            for (int j = 0; j < MHA_SEQ; ++j) {
                #pragma HLS PIPELINE II=1
                int32_t_hls s = 0;
                head_dim_dot:
                for (int d = 0; d < MHA_HDIM; ++d) {
                    #pragma HLS UNROLL
                    int z = h * MHA_HDIM + d;
                    s += Q[i][z] * K[j][z];
                }
                // Apply causal masking: prevent attention to future tokens (j > i)
                score[j] = (j > i) ? int16_t_hls(-32768) : int16_t_hls(s >> 2);
                if (score[j] > max_score) {
                    max_score = score[j];
                }
            }

            // 2. Compute Softmax probability weights via Look-Up Table
            weight_loop:
            for (int j = 0; j < MHA_SEQ; ++j) {
                #pragma HLS PIPELINE II=1
                weight[j] = (j > i) ? uint8_t_hls(0) : lut_weight(score[j] - max_score, softmax_lut);
                denom += weight[j];
            }

            // 3. Attention weighted sum over Values: OUT = (Weights * V) / Denom
            val_dim:
            for (int d = 0; d < MHA_HDIM; ++d) {
                #pragma HLS PIPELINE II=1
                int z = h * MHA_HDIM + d;
                int32_t_hls a = 0;

                val_accum:
                for (int j = 0; j < MHA_SEQ; ++j) {
                    #pragma HLS UNROLL
                    a += weight[j] * V[j][z];
                }
                OUT[i][z] = (denom != 0) ? int16_t_hls(a / denom) : int16_t_hls(0);
            }
        }
    }
}
