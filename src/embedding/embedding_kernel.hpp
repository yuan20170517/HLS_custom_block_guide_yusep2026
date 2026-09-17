/**
 * @file embedding_kernel.hpp
 * @brief High-Throughput Token & Position Embedding Accelerator Header for AMD Versal.
 *
 * Performs on-the-fly Word Token Embedding (WTE) and Word Position Embedding (WPE)
 * row-vector fetching from off-chip memory (LPDDR4/LPDDR5X) and computes the element-wise sum.
 * Outputs the resultant embedding vector directly to the AIE-ML NPU via AXI4-Stream (PLIO)
 * at deterministic Initiation Interval II = 1, eliminating CPU intervention.
 */

#ifndef EMBEDDING_KERNEL_HPP
#define EMBEDDING_KERNEL_HPP

#include <cstdint>
#include <hls_stream.h>
#include <ap_axi_sdata.h>
#include "../common/types.h"
#include "../common/hls_common.hpp"

// ==============================================================================
// Architectural Sizing Constants (NanoGPT / GPT-2 Baseline)
// ==============================================================================
#define DEFAULT_EMBED_DIM   768    ///< Model hidden dimension (e.g. 768 for NanoGPT)
#define MAX_VOCAB_SIZE      50257  ///< GPT-2 / NanoGPT Vocabulary Table Rows
#define MAX_POS_LEN         1024   ///< Maximum context sequence length / Position Table Rows
#define WTE_TABLE_DEPTH     38597376 ///< 50257 * 768 total float elements
#define WPE_TABLE_DEPTH     786432   ///< 1024 * 768 total float elements

// 32-bit floating point AXI4-Stream packet definition
typedef ap_axis<32, 0, 0, 0> axis_float_t;

extern "C" {
/**
 * @brief Top-Level Hardware Kernel for Token & Position Embedding Streaming.
 *
 * @param[in]  wte_table      Pointer to Word Token Embedding matrix in off-chip memory [50257 x 768].
 * @param[in]  wpe_table      Pointer to Word Position Embedding matrix in off-chip memory [1024 x 768].
 * @param[out] out_embed      Optional memory buffer pointer for storing embedding [embed_dim].
 * @param[in]  token_id       Current vocabulary token ID (0 <= token_id < MAX_VOCAB_SIZE).
 * @param[in]  pos_id         Current sequence position index (0 <= pos_id < MAX_POS_LEN).
 * @param[in]  embed_dim      Active embedding hidden dimension (<= DEFAULT_EMBED_DIM, default 768).
 * @param[in]  write_to_mem   Control flag (1 = write to out_embed buffer, 0 = stream-only).
 * @param[out] stream_to_aie  AXI4-Stream interface directly streaming floats to AIE-ML NPU (PLIO).
 */
void embedding_kernel(
    const fp32_t* wte_table,
    const fp32_t* wpe_table,
    fp32_t* out_embed,
    int token_id,
    int pos_id,
    int embed_dim,
    int write_to_mem,
    hls::stream<axis_float_t>& stream_to_aie
);
}

#endif // EMBEDDING_KERNEL_HPP
