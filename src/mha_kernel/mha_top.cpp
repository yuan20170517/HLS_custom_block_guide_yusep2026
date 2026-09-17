/**
 * @file mha_top.cpp
 * @brief Production AXI Top-Level Wrapper for Quantized Multi-Head Attention Kernel.
 *
 * Exposes AXI4-Master and AXI-Lite control interfaces for Vitis Unified IDE
 * and v++ system linking (.xo packaging) on AMD Versal VEK385.
 */

#include "mha_kernel.hpp"

extern "C" {
void mha_top(
    const int8_t_hls* in_tokens,       // [MHA_SEQ * MHA_DIM] in off-chip memory
    const int8_t_hls* weight_q,        // [MHA_DIM * MHA_DIM]
    const int8_t_hls* weight_k,        // [MHA_DIM * MHA_DIM]
    const int8_t_hls* weight_v,        // [MHA_DIM * MHA_DIM]
    int16_t_hls*      out_tokens,      // [MHA_SEQ * MHA_DIM]
    int               num_sequences
) {
    #pragma HLS INTERFACE m_axi port=in_tokens   bundle=gmem0 depth=256 offset=slave max_read_burst_length=64
    #pragma HLS INTERFACE m_axi port=weight_q    bundle=gmem1 depth=256 offset=slave max_read_burst_length=64
    #pragma HLS INTERFACE m_axi port=weight_k    bundle=gmem1 depth=256 offset=slave max_read_burst_length=64
    #pragma HLS INTERFACE m_axi port=weight_v    bundle=gmem1 depth=256 offset=slave max_read_burst_length=64
    #pragma HLS INTERFACE m_axi port=out_tokens  bundle=gmem2 depth=256 offset=slave max_write_burst_length=64
    #pragma HLS INTERFACE s_axilite port=num_sequences bundle=control
    #pragma HLS INTERFACE s_axilite port=return        bundle=control

    // Pre-computed 16-entry Softmax exponential LUT
    const uint8_t_hls softmax_lut[16] = {8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128};

    // On-chip ping-pong / local buffers
    int8_t_hls  local_X[MHA_SEQ][MHA_DIM];
    int8_t_hls  local_WQ[MHA_DIM][MHA_DIM];
    int8_t_hls  local_WK[MHA_DIM][MHA_DIM];
    int8_t_hls  local_WV[MHA_DIM][MHA_DIM];
    int16_t_hls local_OUT[MHA_SEQ][MHA_DIM];

    #pragma HLS ARRAY_PARTITION variable=local_X  complete dim=2
    #pragma HLS ARRAY_PARTITION variable=local_WQ complete dim=1
    #pragma HLS ARRAY_PARTITION variable=local_WK complete dim=1
    #pragma HLS ARRAY_PARTITION variable=local_WV complete dim=1
    #pragma HLS ARRAY_PARTITION variable=local_OUT complete dim=2

    int safe_seqs = (num_sequences <= 0) ? 1 : num_sequences;

    // Load Weight matrices
    Load_W: for (int i = 0; i < MHA_DIM; i++) {
        for (int j = 0; j < MHA_DIM; j++) {
            #pragma HLS PIPELINE II=1
            local_WQ[i][j] = weight_q[i * MHA_DIM + j];
            local_WK[i][j] = weight_k[i * MHA_DIM + j];
            local_WV[i][j] = weight_v[i * MHA_DIM + j];
        }
    }

    Seq_Loop: for (int s = 0; s < safe_seqs; s++) {
        int offset = s * (MHA_SEQ * MHA_DIM);

        // Burst load input tokens
        Load_X: for (int i = 0; i < MHA_SEQ; i++) {
            for (int j = 0; j < MHA_DIM; j++) {
                #pragma HLS PIPELINE II=1
                local_X[i][j] = in_tokens[offset + i * MHA_DIM + j];
            }
        }

        // Call Core MHA Engine
        mha_kernel(local_X, local_WQ, local_WK, local_WV, softmax_lut, local_OUT);

        // Burst writeback attended tokens
        Store_OUT: for (int i = 0; i < MHA_SEQ; i++) {
            for (int j = 0; j < MHA_DIM; j++) {
                #pragma HLS PIPELINE II=1
                out_tokens[offset + i * MHA_DIM + j] = local_OUT[i][j];
            }
        }
    }
}
}
