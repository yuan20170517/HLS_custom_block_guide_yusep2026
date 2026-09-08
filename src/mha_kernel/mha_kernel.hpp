#pragma once

/**
 * @file mha_kernel.hpp
 * @brief Quantized Multi-Head Attention (MHA) Hardware Accelerator Header.
 *
 * Implements a complete quantized Multi-Head Attention block:
 *   1. Q, K, V Projections: Linear transform of input tokens using WQ, WK, WV matrices.
 *   2. Scaled Dot-Product Attention: Q * K^T with causal masking.
 *   3. Softmax Activation: Pre-computed 16-entry LUT approximation.
 *   4. Value Aggregation: Attention Weights * V projection.
 */

#include "../common/hls_common.hpp"

// ==============================================================================
// Architectural Sizing Constants
// ==============================================================================
constexpr int MHA_SEQ   = 16;  ///< Maximum supported sequence length (tokens)
constexpr int MHA_DIM   = 16;  ///< Total hidden feature dimension
constexpr int MHA_HEADS = 4;   ///< Number of parallel attention heads
constexpr int MHA_HDIM  = 4;   ///< Dimension per attention head (MHA_DIM / MHA_HEADS)

/**
 * @brief Top-Level Hardware Kernel for Quantized Multi-Head Attention.
 *
 * @param[in]  X           Input token feature matrix [MHA_SEQ][MHA_DIM].
 * @param[in]  WQ          Query projection weight matrix [MHA_DIM][MHA_DIM].
 * @param[in]  WK          Key projection weight matrix [MHA_DIM][MHA_DIM].
 * @param[in]  WV          Value projection weight matrix [MHA_DIM][MHA_DIM].
 * @param[in]  softmax_lut Pre-computed 16-entry Softmax exponential LUT.
 * @param[out] OUT         Output attended token feature matrix [MHA_SEQ][MHA_DIM].
 */
void mha_kernel(
    const int8_t_hls  X[MHA_SEQ][MHA_DIM],
    const int8_t_hls  WQ[MHA_DIM][MHA_DIM],
    const int8_t_hls  WK[MHA_DIM][MHA_DIM],
    const int8_t_hls  WV[MHA_DIM][MHA_DIM],
    const uint8_t_hls softmax_lut[16],
    int16_t_hls       OUT[MHA_SEQ][MHA_DIM]
);
