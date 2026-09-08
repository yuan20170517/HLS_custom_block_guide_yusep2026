/*
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: X11
*/

/**
 * @file dct.cpp
 * @brief 2-Dimensional 8x8 Discrete Cosine Transform (DCT-II) Kernel for High-Level Synthesis.
 *
 * This module computes an 8x8 2D-DCT by exploiting the mathematical separability
 * of the 2D transform into two successive 1D-DCT passes:
 *   1. Row-wise 1D-DCT transforms each row of the input block.
 *   2. Matrix transposition prepares orthogonal column vectors for row-wise processing.
 *   3. Column-wise 1D-DCT transforms each transposed vector.
 *   4. Final matrix transposition restores natural canonical coordinate ordering.
 *
 * Arithmetic Representation:
 *   - Input / Output / Intermediate Data: 16-bit signed integer (short / dct_data_t).
 *   - Cosine Coefficients: Fixed-point scaled by 2^CONST_BITS (CONST_BITS = 13, scale = 8192).
 *   - Multiplier Accumulator (MAC): 32-bit signed integer (int) with rounding descaling.
 */

#include "dct.h"

/**
 * @brief 1-Dimensional 8-Point Discrete Cosine Transform (DCT-II).
 *
 * Computes:
 *   dst[k] = round( sum_{n=0}^{7} src[n] * C[k][n] / 2^CONST_BITS )
 *
 * @param[in]  src 8-element input vector (16-bit signed).
 * @param[out] dst 8-element transformed output vector (16-bit signed).
 *
 * @note Hardware/HLS Considerations:
 *   - Unrolling 'DCT_Inner_Loop' synthesises an 8-tap parallel multiply-accumulate
 *     tree using dedicated DSP slices.
 *   - Pipelining 'DCT_Outer_Loop' with II=1 achieves an 8-cycle transform per vector.
 *   - Partitioning 'dct_coeff_table' completely (dim=0) maps coefficients directly
 *     to hardwired constants, eliminating ROM read contention.
 */
void dct_1d(dct_data_t src[DCT_SIZE], dct_data_t dst[DCT_SIZE])
{
   unsigned int k, n;
   int tmp;

   // Pre-computed fixed-point DCT coefficient matrix (8x8)
   const dct_data_t dct_coeff_table[DCT_SIZE][DCT_SIZE] = {
#include "dct_coeff_table.txt"
   };

DCT_Outer_Loop:
   for (k = 0; k < DCT_SIZE; k++) {
DCT_Inner_Loop:
      for (n = 0, tmp = 0; n < DCT_SIZE; n++) {
         int coeff = (int)dct_coeff_table[k][n];
         tmp += src[n] * coeff; // 16-bit x 16-bit multiply accumulating into 32-bit
      }
      // Apply symmetrical rounding addition (1 << 12) and shift by CONST_BITS (13)
      dst[k] = DESCALE(tmp, CONST_BITS);
   }
}

/**
 * @brief 2-Dimensional 8x8 DCT Kernel using Separable Row-Column Decomposition.
 *
 * Decomposes 2D matrix transformation into cascaded 1D transforms separated by
 * corner-turn matrix transpositions:
 *   Row_DCT_Loop    -> Compute 1D-DCT across all 8 rows.
 *   Xpose_Row_Loops -> Transpose intermediate matrix (row -> col).
 *   Col_DCT_Loop    -> Compute 1D-DCT across all 8 columns.
 *   Xpose_Col_Loops -> Transpose intermediate matrix back to natural order.
 *
 * @param[in]  in_block  8x8 2D input block.
 * @param[out] out_block 8x8 2D transformed frequency-domain block.
 *
 * @note Memory Architecture:
 *   - Standard dual-port BRAM blocks only allow 2 accesses per cycle.
 *   - Matrix transposition reads across columns while writes occur across rows,
 *     which introduces memory port collisions unless buffers (row_outbuf, col_inbuf)
 *     are partitioned along their respective dimensions (#pragma HLS ARRAY_PARTITION).
 */
void dct_2d(dct_data_t in_block[DCT_SIZE][DCT_SIZE],
      dct_data_t out_block[DCT_SIZE][DCT_SIZE])
{
   // Intermediate staging buffers for ping-pong transform stages
   dct_data_t row_outbuf[DCT_SIZE][DCT_SIZE];
   dct_data_t col_outbuf[DCT_SIZE][DCT_SIZE], col_inbuf[DCT_SIZE][DCT_SIZE];
   unsigned i, j;

   // Stage 1: Compute 1D-DCT across all rows
Row_DCT_Loop:
   for (i = 0; i < DCT_SIZE; i++) {
      dct_1d(in_block[i], row_outbuf[i]);
   }

   // Stage 2: Transpose data matrix (corner-turn) to reuse 1D-DCT row hardware
Xpose_Row_Outer_Loop:
   for (j = 0; j < DCT_SIZE; j++)
Xpose_Row_Inner_Loop:
      for (i = 0; i < DCT_SIZE; i++)
         col_inbuf[j][i] = row_outbuf[i][j];

   // Stage 3: Compute 1D-DCT across all columns (now oriented as rows in col_inbuf)
Col_DCT_Loop:
   for (i = 0; i < DCT_SIZE; i++) {
      dct_1d(col_inbuf[i], col_outbuf[i]);
   }

   // Stage 4: Transpose data back into canonical row-major representation
Xpose_Col_Outer_Loop:
   for (j = 0; j < DCT_SIZE; j++)
Xpose_Col_Inner_Loop:
      for (i = 0; i < DCT_SIZE; i++)
         out_block[j][i] = col_outbuf[i][j];
}

/**
 * @brief Input Staging Utility.
 *
 * Reads 64 continuous samples from a 1D sequential input stream or memory buffer
 * and re-indexes them into an 8x8 2D array structure.
 *
 * @param[in]  input 1D linear array of N=64 elements.
 * @param[out] buf   8x8 2D internal memory buffer.
 */
void read_data(short input[N], short buf[DCT_SIZE][DCT_SIZE])
{
   int r, c;

RD_Loop_Row:
   for (r = 0; r < DCT_SIZE; r++) {
RD_Loop_Col:
      for (c = 0; c < DCT_SIZE; c++) {
         buf[r][c] = input[r * DCT_SIZE + c];
      }
   }
}

/**
 * @brief Output Staging Utility.
 *
 * Flattens the 8x8 2D transformed coefficient matrix into a 1D linear memory
 * buffer or output stream of N=64 elements.
 *
 * @param[in]  buf    8x8 2D transformed result buffer.
 * @param[out] output 1D linear array of N=64 elements.
 */
void write_data(short buf[DCT_SIZE][DCT_SIZE], short output[N])
{
   int r, c;

WR_Loop_Row:
   for (r = 0; r < DCT_SIZE; r++) {
WR_Loop_Col:
      for (c = 0; c < DCT_SIZE; c++) {
         output[r * DCT_SIZE + c] = buf[r][c];
      }
   }
}

/**
 * @brief Top-Level Hardware Kernel Entry Point.
 *
 * Coordinates the execution pipeline:
 *   1. read_data:  Burst read input block into on-chip 2D memory.
 *   2. dct_2d:     Execute 2D separable Discrete Cosine Transform.
 *   3. write_data: Write transformed frequency coefficients to output.
 *
 * @param[in]  input  Input data array (N = 64 samples).
 * @param[out] output Output frequency-domain coefficient array (N = 64 samples).
 *
 * @note Architecture & Throughput Optimisation:
 *   In advanced implementations, applying '#pragma HLS DATAFLOW' allows read_data,
 *   dct_2d, and write_data to execute concurrently in a streaming ping-pong pipeline,
 *   yielding overlapping block processing.
 */
void dct(short input[N], short output[N])
{
   // On-chip intermediate block buffers (inferred as BRAM or registers)
   short buf_2d_in[DCT_SIZE][DCT_SIZE];
   short buf_2d_out[DCT_SIZE][DCT_SIZE];

   // Read input data and populate internal 2D buffer
   read_data(input, buf_2d_in);

   // Perform 2D Discrete Cosine Transform
   dct_2d(buf_2d_in, buf_2d_out);

   // Stream transformed frequency coefficients to output
   write_data(buf_2d_out, output);
}


