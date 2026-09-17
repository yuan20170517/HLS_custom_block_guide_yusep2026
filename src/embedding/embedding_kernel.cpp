/**
 * @file embedding_kernel.cpp
 * @brief High-Throughput Token + Position Embedding Streamer for AMD Versal.
 * 
 * Fetches 768-dimensional token and position embeddings from LPDDR4/LPDDR5X memory
 * and streams the summed vector directly to the AIE NPU via AXI4-Stream (II = 1).
 */

#include "embedding_kernel.hpp"
#include <cstring>

extern "C" {
void embedding_kernel(
    const fp32_t* wte_table,
    const fp32_t* wpe_table,
    fp32_t* out_embed,
    int token_id,
    int pos_id,
    int embed_dim,
    int write_to_mem,
    hls::stream<axis_float_t>& stream_to_aie
) {
    // --------------------------------------------------------------------------
    // AXI Interface Synthesis Pragmas
    // --------------------------------------------------------------------------
    #pragma HLS INTERFACE m_axi port=wte_table   bundle=gmem0 depth=WTE_TABLE_DEPTH offset=slave max_read_burst_length=64
    #pragma HLS INTERFACE m_axi port=wpe_table   bundle=gmem1 depth=WPE_TABLE_DEPTH offset=slave max_read_burst_length=64
    #pragma HLS INTERFACE m_axi port=out_embed   bundle=gmem2 depth=DEFAULT_EMBED_DIM offset=slave max_write_burst_length=64
    #pragma HLS INTERFACE axis  port=stream_to_aie
    #pragma HLS INTERFACE s_axilite port=token_id     bundle=control
    #pragma HLS INTERFACE s_axilite port=pos_id       bundle=control
    #pragma HLS INTERFACE s_axilite port=embed_dim    bundle=control
    #pragma HLS INTERFACE s_axilite port=write_to_mem bundle=control
    #pragma HLS INTERFACE s_axilite port=return       bundle=control

    // --------------------------------------------------------------------------
    // Operational Bounds Sanitization
    // --------------------------------------------------------------------------
    int safe_token = (token_id < 0) ? 0 : ((token_id >= MAX_VOCAB_SIZE) ? (MAX_VOCAB_SIZE - 1) : token_id);
    int safe_pos   = (pos_id < 0)   ? 0 : ((pos_id >= MAX_POS_LEN)       ? (MAX_POS_LEN - 1)     : pos_id);
    int D          = (embed_dim <= 0 || embed_dim > DEFAULT_EMBED_DIM)   ? DEFAULT_EMBED_DIM    : embed_dim;

    // Base address offsets for row selection
    int token_offset = safe_token * D;
    int pos_offset   = safe_pos   * D;

    // --------------------------------------------------------------------------
    // On-Chip Buffers for Burst Memory Coalescing
    // --------------------------------------------------------------------------
    fp32_t buf_wte[DEFAULT_EMBED_DIM];
    fp32_t buf_wpe[DEFAULT_EMBED_DIM];
    #pragma HLS ARRAY_PARTITION variable=buf_wte cyclic factor=16
    #pragma HLS ARRAY_PARTITION variable=buf_wpe cyclic factor=16

    // Burst Read Row from Token Table (WTE)
    Read_WTE: for (int i = 0; i < D; i++) {
        #pragma HLS PIPELINE II=1
        buf_wte[i] = wte_table[token_offset + i];
    }

    // Burst Read Row from Position Table (WPE)
    Read_WPE: for (int i = 0; i < D; i++) {
        #pragma HLS PIPELINE II=1
        buf_wpe[i] = wpe_table[pos_offset + i];
    }

    // --------------------------------------------------------------------------
    // Vector Addition & Direct AXI4-Stream Push to AIE (II = 1)
    // --------------------------------------------------------------------------
    Stream_Out: for (int i = 0; i < D; i++) {
        #pragma HLS PIPELINE II=1
        fp32_t sum_val = buf_wte[i] + buf_wpe[i];

        // Format 32-bit float into AXI4-Stream packet
        axis_float_t pkt;
        union {
            float f;
            uint32_t u;
        } conv;
        conv.f = sum_val;

        pkt.data = conv.u;
        pkt.keep = -1; // All byte lanes valid
        pkt.last = (i == D - 1) ? 1 : 0; // Assert TLAST on final feature element

        stream_to_aie.write(pkt);

        // Optional writeback to memory for verification / testing
        if (write_to_mem && out_embed != nullptr) {
            out_embed[i] = sum_val;
        }
    }
}
}
