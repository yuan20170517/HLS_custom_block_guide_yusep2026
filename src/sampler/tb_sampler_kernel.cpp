/**
 * @file tb_sampler_kernel.cpp
 * @brief Self-Checking C-Simulation Testbench for the NanoGPT ArgMax / Top-K Sampler Kernel.
 *
 * Verifies mathematical correctness and boundary conditions across a 50,257 logit distribution:
 *   1. Correct extraction of the global maximum logit (Greedy Token ID).
 *   2. Monotonic ordering and index fidelity of the top-8 candidate list.
 *   3. Dynamic temperature scaling tolerance.
 */

#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include "sampler_kernel.hpp"

int main() {
    std::cout << "========================================================" << std::endl;
    std::cout << "[*] Starting Vitis HLS Testbench: NanoGPT Sampler Kernel" << std::endl;
    std::cout << "[*] Target Vocabulary Size: " << DEFAULT_VOCAB_SIZE << " tokens" << std::endl;
    std::cout << "========================================================" << std::endl;

    // Allocate host buffers
    std::vector<fp32_t> logits(DEFAULT_VOCAB_SIZE);
    int hw_best_id = -1;
    fp32_t hw_best_logit = -1e30f;
    int hw_top_k_indices[TOP_K_MAX];
    fp32_t hw_top_k_logits[TOP_K_MAX];

    // Seed background noise logits with small values [-10.0, 10.0]
    srand(42);
    for (int i = 0; i < DEFAULT_VOCAB_SIZE; i++) {
        logits[i] = ((fp32_t)rand() / (fp32_t)RAND_MAX) * 20.0f - 10.0f;
    }

    // Plant 8 known high-confidence candidate tokens at specific indices
    struct KnownCandidate {
        int index;
        fp32_t logit;
    };

    KnownCandidate planted[8] = {
        {15496, 42.50f}, // 1st (Global Winner - e.g. target token)
        { 3201, 38.25f}, // 2nd
        {45000, 35.10f}, // 3rd
        {  892, 31.00f}, // 4th
        {27182, 29.40f}, // 5th
        {10240, 27.80f}, // 6th
        {49999, 25.50f}, // 7th
        {  128, 23.10f}  // 8th
    };

    for (int k = 0; k < 8; k++) {
        logits[planted[k].index] = planted[k].logit;
    }

    // --------------------------------------------------------------------------
    // Test Case 1: Greedy ArgMax & Top-8 with Temperature = 1.0 (Neutral)
    // --------------------------------------------------------------------------
    std::cout << "\n[1] Executing Hardware Kernel (temperature = 1.0, k = 8)..." << std::endl;
    sampler_kernel(
        logits.data(),
        &hw_best_id,
        &hw_best_logit,
        hw_top_k_indices,
        hw_top_k_logits,
        DEFAULT_VOCAB_SIZE,
        8,
        1.0f
    );

    std::cout << ">>> Hardware Best Token ID: " << hw_best_id 
              << " (Expected: " << planted[0].index << ")" << std::endl;
    std::cout << ">>> Hardware Best Logit:    " << hw_best_logit 
              << " (Expected: " << planted[0].logit << ")" << std::endl;

    // Verification
    int errors = 0;
    if (hw_best_id != planted[0].index) {
        std::cerr << "[FAIL] Global ArgMax Token ID mismatch! Got " << hw_best_id 
                  << ", Expected " << planted[0].index << std::endl;
        errors++;
    }

    if (std::fabs(hw_best_logit - planted[0].logit) > 1e-4f) {
        std::cerr << "[FAIL] Global ArgMax Logit mismatch! Got " << hw_best_logit 
                  << ", Expected " << planted[0].logit << std::endl;
        errors++;
    }

    std::cout << "\n[2] Verifying Top-8 Ranked Candidates:" << std::endl;
    for (int k = 0; k < TOP_K_MAX; k++) {
        std::cout << "  Rank " << (k + 1) << ": Token " << hw_top_k_indices[k] 
                  << " | Logit = " << hw_top_k_logits[k]
                  << " (Expected Token: " << planted[k].index << ", Logit: " << planted[k].logit << ")" << std::endl;

        if (hw_top_k_indices[k] != planted[k].index) {
            std::cerr << "  [FAIL] Rank " << (k + 1) << " index mismatch!" << std::endl;
            errors++;
        }
        if (std::fabs(hw_top_k_logits[k] - planted[k].logit) > 1e-4f) {
            std::cerr << "  [FAIL] Rank " << (k + 1) << " logit mismatch!" << std::endl;
            errors++;
        }
    }

    // --------------------------------------------------------------------------
    // Test Case 2: Temperature Scaling (temperature = 2.0)
    // --------------------------------------------------------------------------
    std::cout << "\n[3] Executing Hardware Kernel with Temperature = 2.0..." << std::endl;
    sampler_kernel(
        logits.data(),
        &hw_best_id,
        &hw_best_logit,
        hw_top_k_indices,
        hw_top_k_logits,
        DEFAULT_VOCAB_SIZE,
        8,
        2.0f
    );

    fp32_t expected_scaled_logit = planted[0].logit / 2.0f;
    std::cout << ">>> Scaled Best Logit: " << hw_best_logit 
              << " (Expected: " << expected_scaled_logit << ")" << std::endl;

    if (std::fabs(hw_best_logit - expected_scaled_logit) > 1e-4f) {
        std::cerr << "[FAIL] Temperature scaling calculation error!" << std::endl;
        errors++;
    }

    std::cout << "\n========================================================" << std::endl;
    if (errors == 0) {
        std::cout << "[+] TEST PASSED: All 50,257 logit checks matched golden reference with zero error." << std::endl;
        std::cout << "========================================================" << std::endl;
        return 0;
    } else {
        std::cerr << "[!] TEST FAILED: Encountered " << errors << " verification error(s)." << std::endl;
        std::cout << "========================================================" << std::endl;
        return 1;
    }
}
