#pragma once

#include "../common/hls_common.hpp"

// ==============================================================================
// Architectural Sizing Parameters
// ==============================================================================
constexpr int GE_LEN    = 64;  ///< Number of elements processed by GELU kernel per execution
constexpr int EMB_VOCAB = 32;  ///< Embedding vocabulary size (number of token categories)
constexpr int EMB_DIM   = 8;   ///< Embedding feature dimension per token

/**
 * @brief Combined GELU Look-Up Table (LUT) & Token Embedding Hardware Kernel.
 *
 * This hardware kernel implements two parallel non-linear operations commonly
 * required in transformer front-ends and quantized neural network layers:
 *
 * 1. **GELU Non-Linear Activation (LUT-Based):**
 *    Instead of evaluating expensive floating-point polynomial approximations,
 *    an 8-bit quantized activation maps directly into a precomputed 256-entry table.
 *    This eliminates DSP blocks and achieves single-cycle latency (II = 1).
 *
 * 2. **Token Embedding Lookup:**
 *    Retrieves embedding vectors from a weight table indexed by token IDs.
 *
 * @param[in]  X          Input feature vector to be activated by GELU [GE_LEN].
 * @param[in]  token_ids  Input token sequence IDs [EMB_DIM].
 * @param[in]  gelu_lut   Precomputed 256-entry Look-Up Table for GELU non-linearity [256].
 * @param[in]  embed_lut  2D embedding weight matrix [EMB_VOCAB][EMB_DIM].
 * @param[out] gelu_out   Output activated feature vector [GE_LEN].
 * @param[out] embed_out  Output embedding vector [EMB_DIM].
 */
void gelu_embed_kernel(
    const int8_t_hls  X[GE_LEN],
    const uint8_t_hls token_ids[EMB_DIM],
    const int8_t_hls  gelu_lut[256],
    const int8_t_hls  embed_lut[EMB_VOCAB][EMB_DIM],
    int8_t_hls        gelu_out[GE_LEN],
    int8_t_hls        embed_out[EMB_DIM]
);
