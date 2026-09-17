/**
 * @file sampler_kernel.cpp
 * @brief High-Throughput ArgMax and Top-K Token Sampler Kernel for AMD Versal / UltraScale+.
 *
 * Scans vocabulary logit tensors output by Transformer LM Heads (e.g., NanoGPT 50,257 vocab)
 * to perform greedy token prediction (ArgMax) and extract the top-K candidate list
 * in a single pipelined hardware pass with deterministic Initiation Interval II = 1.
 */

#include "sampler_kernel.hpp"
#include <hls_math.h>

extern "C" {
void sampler_kernel(
    const fp32_t logits[DEFAULT_VOCAB_SIZE],
    int* best_token_id,
    fp32_t* best_logit,
    int top_k_indices[TOP_K_MAX],
    fp32_t top_k_logits[TOP_K_MAX],
    int vocab_size,
    int k_val,
    fp32_t temperature
) {
    // --------------------------------------------------------------------------
    // AXI Interface Synthesis Pragmas
    // --------------------------------------------------------------------------
    #pragma HLS INTERFACE m_axi port=logits         bundle=gmem0 depth=DEFAULT_VOCAB_SIZE offset=slave max_read_burst_length=64
    #pragma HLS INTERFACE m_axi port=top_k_indices  bundle=gmem1 depth=TOP_K_MAX          offset=slave max_write_burst_length=16
    #pragma HLS INTERFACE m_axi port=top_k_logits   bundle=gmem1 depth=TOP_K_MAX          offset=slave max_write_burst_length=16
    #pragma HLS INTERFACE s_axilite port=best_token_id bundle=control
    #pragma HLS INTERFACE s_axilite port=best_logit    bundle=control
    #pragma HLS INTERFACE s_axilite port=vocab_size    bundle=control
    #pragma HLS INTERFACE s_axilite port=k_val         bundle=control
    #pragma HLS INTERFACE s_axilite port=temperature   bundle=control
    #pragma HLS INTERFACE s_axilite port=return        bundle=control

    // Bound parameters to safe hardware operational ranges
    int V = (vocab_size <= 0 || vocab_size > DEFAULT_VOCAB_SIZE) ? DEFAULT_VOCAB_SIZE : vocab_size;
    int K = (k_val <= 0 || k_val > TOP_K_MAX) ? TOP_K_MAX : k_val;
    fp32_t inv_temp = (temperature > 1e-4f) ? (1.0f / temperature) : 1.0f;

    // --------------------------------------------------------------------------
    // On-Chip Register Arrays & Initialization
    // --------------------------------------------------------------------------
    fp32_t local_k_logits[TOP_K_MAX];
    int local_k_indices[TOP_K_MAX];
    #pragma HLS ARRAY_PARTITION variable=local_k_logits complete
    #pragma HLS ARRAY_PARTITION variable=local_k_indices complete

    Init_K: for (int k = 0; k < TOP_K_MAX; k++) {
        #pragma HLS UNROLL
        local_k_logits[k] = -1e30f;
        local_k_indices[k] = -1;
    }

    fp32_t current_max = -1e30f;
    int current_best_id = 0;

    // --------------------------------------------------------------------------
    // Single-Pass Pipelined Vocabulary Logit Scan (II = 1)
    // --------------------------------------------------------------------------
    Vocab_Scan_Loop: for (int i = 0; i < V; i++) {
        #pragma HLS PIPELINE II=1

        fp32_t raw_val = logits[i];
        fp32_t val = raw_val * inv_temp;

        // 1. Greedy ArgMax Comparison
        if (val > current_max) {
            current_max = val;
            current_best_id = i;
        }

        // 2. Unrolled Top-K Priority Shift Insertion
        if (val > local_k_logits[TOP_K_MAX - 1]) {
            int insert_idx = -1;
            Find_Pos: for (int p = 0; p < TOP_K_MAX; p++) {
                #pragma HLS UNROLL
                if (val > local_k_logits[p] && insert_idx == -1) {
                    insert_idx = p;
                }
            }

            if (insert_idx != -1) {
                Shift_Down: for (int s = TOP_K_MAX - 1; s > 0; s--) {
                    #pragma HLS UNROLL
                    if (s > insert_idx) {
                        local_k_logits[s] = local_k_logits[s - 1];
                        local_k_indices[s] = local_k_indices[s - 1];
                    }
                }
                local_k_logits[insert_idx] = val;
                local_k_indices[insert_idx] = i;
            }
        }
    }

    // --------------------------------------------------------------------------
    // Write Results Back to Host Control Registers & DDR
    // --------------------------------------------------------------------------
    *best_token_id = current_best_id;
    *best_logit = current_max;

    Write_TopK: for (int k = 0; k < K; k++) {
        #pragma HLS PIPELINE II=1
        top_k_indices[k] = local_k_indices[k];
        top_k_logits[k] = local_k_logits[k];
    }
}
}
