/**
 * @file tb_layernorm_kernel.cpp
 * @brief Self-Checking Testbench for Floating-Point Layer Normalization Kernel.
 */

#include "layernorm_kernel.h"
#include <iostream>
#include <vector>
#include <cmath>

int main() {
    std::cout << "========================================================\n";
    std::cout << " Starting C-Simulation: Layer Normalization Kernel      \n";
    std::cout << "========================================================\n";

    const int TOKENS = 2;
    const int TOTAL_ELEMENTS = TOKENS * LN_DIM;

    std::vector<fp32_t> in(MAX_DEPTH, 0.0f);
    std::vector<fp32_t> gamma(LN_DIM, 1.0f);
    std::vector<fp32_t> beta(LN_DIM, 0.0f);
    std::vector<fp32_t> out(MAX_DEPTH, 0.0f);
    std::vector<fp32_t> ref_out(TOTAL_ELEMENTS, 0.0f);

    // Initialise synthetic test data
    for (int t = 0; t < TOKENS; ++t) {
        for (int i = 0; i < LN_DIM; ++i) {
            in[t * LN_DIM + i] = (fp32_t)((i % 17) - 8) * 0.25f;
            gamma[i] = 1.0f + 0.01f * (fp32_t)(i % 5);
            beta[i] = 0.05f * (fp32_t)(i % 3);
        }
    }

    // Compute CPU Golden Reference
    for (int t = 0; t < TOKENS; ++t) {
        int base = t * LN_DIM;
        fp32_t sum = 0.0f;
        for (int i = 0; i < LN_DIM; ++i) {
            sum += in[base + i];
        }
        fp32_t mean = sum / (fp32_t)LN_DIM;

        fp32_t var_sum = 0.0f;
        for (int i = 0; i < LN_DIM; ++i) {
            fp32_t diff = in[base + i] - mean;
            var_sum += diff * diff;
        }
        fp32_t inv_std = 1.0f / std::sqrt((var_sum / (fp32_t)LN_DIM) + 1e-5f);

        for (int i = 0; i < LN_DIM; ++i) {
            ref_out[base + i] = (in[base + i] - mean) * inv_std * gamma[i] + beta[i];
        }
    }

    // Call Top-Level HLS Kernel
    layernorm_kernel(in.data(), gamma.data(), beta.data(), out.data(), TOKENS);

    // Validate assertions
    int errors = 0;
    fp32_t max_diff = 0.0f;
    for (int i = 0; i < TOTAL_ELEMENTS; ++i) {
        fp32_t diff = std::fabs(out[i] - ref_out[i]);
        if (diff > max_diff) max_diff = diff;
        if (diff > 1e-4f) {
            if (errors < 5) {
                std::cout << "Mismatch at [" << i << "]: got " << out[i]
                          << ", expected " << ref_out[i] << " (diff=" << diff << ")\n";
            }
            errors++;
        }
    }

    std::cout << "Max absolute difference: " << max_diff << "\n";
    if (errors == 0) {
        std::cout << ">>> C-SIMULATION PASSED: 0 errors detected. <<<\n";
        return 0;
    } else {
        std::cout << ">>> C-SIMULATION FAILED: " << errors << " errors detected. <<<\n";
        return 1;
    }
}
