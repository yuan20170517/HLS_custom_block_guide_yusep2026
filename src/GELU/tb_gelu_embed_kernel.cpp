#include "gelu_embed_kernel.hpp"
#include <iostream>
#include <cstdlib>

/**
 * @brief Self-checking C-Simulation Testbench for GELU & Embedding Hardware Kernel.
 */
int main() {
    std::cout << "========================================================\n";
    std::cout << " Starting C-Simulation: GELU & Embedding Hardware Kernel\n";
    std::cout << "========================================================\n";

    // --------------------------------------------------------------------------
    // 1. Allocate Test Stimulus & Reference Buffers
    // --------------------------------------------------------------------------
    int8_t_hls  X[GE_LEN];                         // Input activations
    int8_t_hls  gelu_lut[256];                     // 256-entry GELU LUT
    int8_t_hls  embed_lut[EMB_VOCAB][EMB_DIM];     // 32x8 Embedding table
    uint8_t_hls token_ids[EMB_DIM];                // Token indices for lookup

    int8_t_hls  gelu_out[GE_LEN];                  // Kernel output: GELU activations
    int8_t_hls  embed_out[EMB_DIM];                // Kernel output: Embedding vectors

    // --------------------------------------------------------------------------
    // 2. Initialise Deterministic Stimulus Data
    // --------------------------------------------------------------------------
    // Generate signed input stimulus spanning range [-15, +15]
    for (int i = 0; i < GE_LEN; ++i) {
        X[i] = (int8_t_hls)((i % 31) - 15);
    }

    // Populate synthetic GELU lookup table (threshold test pattern)
    for (int i = 0; i < 256; ++i) {
        gelu_lut[i] = (i > 127) ? (int8_t_hls)63 : (int8_t_hls)-63;
    }

    // Populate 2D embedding matrix with deterministic modular values
    for (int r = 0; r < EMB_VOCAB; ++r) {
        for (int c = 0; c < EMB_DIM; ++c) {
            embed_lut[r][c] = (int8_t_hls)((r + c) % 16);
        }
    }

    // Assign sequential token IDs: {0, 1, 2, ..., EMB_DIM-1}
    for (int i = 0; i < EMB_DIM; ++i) {
        token_ids[i] = (uint8_t_hls)i;
    }

    // --------------------------------------------------------------------------
    // 3. Invoke Hardware Kernel Under Test
    // --------------------------------------------------------------------------
    gelu_embed_kernel(X, token_ids, gelu_lut, embed_lut, gelu_out, embed_out);

    // --------------------------------------------------------------------------
    // 4. Verification & Golden Model Assertions
    // --------------------------------------------------------------------------
    int errors = 0;

    // Verify GELU LUT lookup outputs
    for (int i = 0; i < GE_LEN; ++i) {
        uint8_t_hls expected_idx = (uint8_t_hls)X[i];
        int8_t_hls expected_val = gelu_lut[expected_idx];
        if (gelu_out[i] != expected_val) {
            std::cerr << "[FAIL] GELU mismatch at index " << i
                      << ": Expected " << (int)expected_val
                      << ", Got " << (int)gelu_out[i] << "\n";
            errors++;
        }
    }

    // Verify Embedding table lookup outputs
    for (int i = 0; i < EMB_DIM; ++i) {
        uint8_t_hls expected_vocab_idx = token_ids[i] % EMB_VOCAB;
        int8_t_hls expected_val = embed_lut[expected_vocab_idx][i];
        if (embed_out[i] != expected_val) {
            std::cerr << "[FAIL] Embedding mismatch at dimension " << i
                      << ": Expected " << (int)expected_val
                      << ", Got " << (int)embed_out[i] << "\n";
            errors++;
        }
    }

    // --------------------------------------------------------------------------
    // 5. Test Result Evaluation
    // --------------------------------------------------------------------------
    if (errors == 0) {
        std::cout << "[PASS] All assertions verified: 0 mismatches detected.\n";
        std::cout << ">>> TB_GELU_EMBED PASS <<<\n";
        return 0;
    } else {
        std::cerr << "[FAIL] Total verification errors: " << errors << "\n";
        return 1;
    }
}
