#include "gelu_embed_kernel.hpp"

/**
 * @brief Implementation of GELU LUT & Embedding hardware kernel.
 */
void gelu_embed_kernel(
    const int8_t_hls  X[GE_LEN],
    const uint8_t_hls token_ids[EMB_DIM],
    const int8_t_hls  gelu_lut[256],
    const int8_t_hls  embed_lut[EMB_VOCAB][EMB_DIM],
    int8_t_hls        gelu_out[GE_LEN],
    int8_t_hls        embed_out[EMB_DIM]
) {
    // --------------------------------------------------------------------------
    // Hardware Memory Partitioning Directives
    // --------------------------------------------------------------------------
    // Completely partition the 256-entry GELU LUT into distributed registers/LUTRAM
    // to provide simultaneous, contention-free read access with 0 BRAM overhead.
    #pragma HLS ARRAY_PARTITION variable=gelu_lut complete dim=1

    // Partition the embedding table along dimension 2 (columns).
    // This provides 8 independent memory banks, allowing parallel reads
    // across all feature dimensions without port contention.
    #pragma HLS ARRAY_PARTITION variable=embed_lut complete dim=2

    // --------------------------------------------------------------------------
    // Stage 1: GELU Non-Linear Activation Loop
    // --------------------------------------------------------------------------
    // Evaluates GELU for 64 elements with a single-cycle Initiation Interval (II=1).
    // Casting signed int8 (-128..127) to unsigned uint8 (0..255) maps the value
    // directly to the corresponding precomputed entry in the 256-entry LUT.
    gelu_loop:
    for (int i = 0; i < GE_LEN; ++i) {
        #pragma HLS PIPELINE II=1
        uint8_t_hls lut_idx = (uint8_t_hls)X[i];
        gelu_out[i] = gelu_lut[lut_idx];
    }

    // --------------------------------------------------------------------------
    // Stage 2: Token Embedding Extraction Loop
    // --------------------------------------------------------------------------
    // Extracts embedding weights for each token ID across the 8 feature dimensions.
    // Pipelined with II=1 to stream one dimension output per clock cycle.
    embed_loop:
    for (int i = 0; i < EMB_DIM; ++i) {
        #pragma HLS PIPELINE II=1
        // Modulo clamping ensures index safety within EMB_VOCAB (32 tokens)
        uint8_t_hls vocab_idx = token_ids[i] % EMB_VOCAB;
        embed_out[i] = embed_lut[vocab_idx][i];
    }
}
