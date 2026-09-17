/**
 * @file sampler_kernel.hpp
 * @brief High-Throughput ArgMax and Top-K Token Sampler Header for NanoGPT / LLM Inference.
 *
 * Implements a single-pass pipelined argmax and top-K candidate extraction engine
 * for Transformer LM Head vocabulary logit distributions (e.g. GPT-2/NanoGPT 50,257 tokens).
 *
 * Architecture Highlights:
 *   - Pipelined AXI4-Master burst reading across 50,257 logits with II = 1.
 *   - On-chip unrolled Top-K shift-register insertion array (K <= 8).
 *   - In-line optional temperature scaling (logits / temperature).
 *   - Direct AXI-Lite register output for zero-overhead host reading of the winning token ID.
 */

#ifndef SAMPLER_KERNEL_HPP
#define SAMPLER_KERNEL_HPP

#include <cstdint>
#include <cmath>
#include "../common/types.h"
#include "../common/hls_common.hpp"

// ==============================================================================
// Vocabulary & Top-K Sizing Parameters
// ==============================================================================
#define DEFAULT_VOCAB_SIZE 50257  ///< GPT-2 / NanoGPT Vocabulary Size
#define TOP_K_MAX          8      ///< Maximum supported hardware top-K candidate list

extern "C" {
/**
 * @brief Top-Level Hardware Kernel for Vocabulary ArgMax and Top-K Candidate Extraction.
 *
 * @param[in]  logits          Input unnormalized logit vector in off-chip memory [DEFAULT_VOCAB_SIZE].
 * @param[out] best_token_id   Winning token ID corresponding to the highest logit (Greedy Search).
 * @param[out] best_logit      Floating-point value of the winning logit.
 * @param[out] top_k_indices   Top-K candidate token indices in descending order [TOP_K_MAX].
 * @param[out] top_k_logits    Top-K candidate logit values in descending order [TOP_K_MAX].
 * @param[in]  vocab_size      Total number of vocabulary tokens to scan (<= DEFAULT_VOCAB_SIZE).
 * @param[in]  k_val           Number of top candidates to return (1 <= k_val <= TOP_K_MAX).
 * @param[in]  temperature     Sampling temperature scaling factor (> 0.0, 1.0 = neutral).
 */
void sampler_kernel(
    const fp32_t logits[DEFAULT_VOCAB_SIZE],
    int* best_token_id,
    fp32_t* best_logit,
    int top_k_indices[TOP_K_MAX],
    fp32_t top_k_logits[TOP_K_MAX],
    int vocab_size,
    int k_val,
    fp32_t temperature
);
}

#endif // SAMPLER_KERNEL_HPP
