/**
 * @file tiled_matmul.cpp
 * @brief Output-Stationary Tiled Matrix Multiplication Hardware Kernel for Vitis HLS.
 *
 * Implements a 2D tiled matrix multiplier optimized for FPGA DSP slices:
 *   - Evaluates a 4x4 spatial sub-block (tile) of matrix C simultaneously.
 *   - Keeps intermediate accumulation values in ultra-fast flip-flop registers (`acc`).
 *   - Streams through contraction dimension k with a single-cycle Initiation Interval (II = 1),
 *     performing 16 parallel 8-bit multiplies and 32-bit accumulations per clock cycle.
 */

#include "tiled_matmul.hpp"

void tiled_matmul_kernel(
    const int8_t_hls A[TM_ROWS][TM_K],
    const int8_t_hls B[TM_K][TM_COLS],
    int32_t_hls       C[TM_ROWS][TM_COLS]
) {
    // --------------------------------------------------------------------------
    // Memory Partitioning for Parallel Memory Access
    // --------------------------------------------------------------------------
    // Partition A columns and B rows to supply 4 parallel elements per cycle
    #pragma HLS ARRAY_PARTITION variable=A complete dim=2
    #pragma HLS ARRAY_PARTITION variable=B complete dim=1
    #pragma HLS ARRAY_PARTITION variable=C complete dim=2

    // Outer tile traversal: step across matrix rows by TM_TILE (4)
    tile_i:
    for (int ii = 0; ii < TM_ROWS; ii += TM_TILE) {
        // Step across matrix columns by TM_TILE (4)
        tile_j:
        for (int jj = 0; jj < TM_COLS; jj += TM_TILE) {
            // Local 2D accumulator registers holding the 4x4 output tile
            int32_t_hls acc[TM_TILE][TM_TILE];
            #pragma HLS ARRAY_PARTITION variable=acc complete dim=0

            // 1. Clear tile accumulator
            init_tile:
            for (int i = 0; i < TM_TILE; ++i) {
                for (int j = 0; j < TM_TILE; ++j) {
                    #pragma HLS UNROLL
                    acc[i][j] = 0;
                }
            }

            // 2. Stream contraction dimension k and accumulate 16 MACs per cycle
            dot_k:
            for (int k = 0; k < TM_K; ++k) {
                #pragma HLS PIPELINE II=1
                for (int i = 0; i < TM_TILE; ++i) {
                    for (int j = 0; j < TM_TILE; ++j) {
                        #pragma HLS UNROLL
                        acc[i][j] += A[ii + i][k] * B[k][jj + j];
                    }
                }
            }

            // 3. Write completed 4x4 tile to output matrix C
            store_tile:
            for (int i = 0; i < TM_TILE; ++i) {
                for (int j = 0; j < TM_TILE; ++j) {
                    #pragma HLS UNROLL
                    C[ii + i][jj + j] = acc[i][j];
                }
            }
        }
    }
}
