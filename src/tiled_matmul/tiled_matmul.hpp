#pragma once

/**
 * @file tiled_matmul.hpp
 * @brief High-Efficiency Tiled Matrix Multiplication (GEMM) Kernel Header.
 *
 * Implements 2D output-stationary tiled matrix multiplication:
 *   C = A * B
 * where A is [TM_ROWS x TM_K], B is [TM_K x TM_COLS], and C is [TM_ROWS x TM_COLS].
 *
 * Architecture Highlights:
 *   - Spatial 2D systolic tile buffer [TM_TILE x TM_TILE] with complete register partitioning.
 *   - Inner accumulation loop pipelined with II = 1.
 *   - Completely unrolled spatial inner dot product evaluating 16 DSP MACs per clock cycle.
 */

#include "../common/hls_common.hpp"

// ==============================================================================
// Matrix Dimensions & Tile Parameters
// ==============================================================================
constexpr int TM_ROWS = 16;  ///< Number of rows in matrix A and C
constexpr int TM_COLS = 16;  ///< Number of columns in matrix B and C
constexpr int TM_K    = 16;  ///< Shared contraction dimension (inner dimension)
constexpr int TM_TILE = 4;   ///< Spatial 2D tile sub-block dimension (4x4 = 16 parallel MACs)

/**
 * @brief Top-Level Hardware Kernel for Tiled Matrix Multiplication.
 *
 * @param[in]  A Input matrix A of dimensions [TM_ROWS][TM_K] (8-bit signed quantized).
 * @param[in]  B Input matrix B of dimensions [TM_K][TM_COLS] (8-bit signed quantized).
 * @param[out] C Output accumulated matrix C of dimensions [TM_ROWS][TM_COLS] (32-bit signed).
 */
void tiled_matmul_kernel(
    const int8_t_hls A[TM_ROWS][TM_K],
    const int8_t_hls B[TM_K][TM_COLS],
    int32_t_hls       C[TM_ROWS][TM_COLS]
);
