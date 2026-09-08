/**
 * @file tb_softmax_kernel.cpp
 * @brief Self-Checking Testbench for Floating-Point Softmax Activation Kernel.
 */

#include "softmax_kernel.h"
#include <iostream>
#include <vector>
#include <cmath>

int main() {
    std::cout << "========================================================\n";
    std::cout << " Starting C-Simulation: Softmax Hardware Kernel         \n";
    std::cout << "========================================================\n";

    const int ROWS = 4;
    const int ROW_LEN = 128;
    const int TOTAL_ELEMENTS = ROWS * ROW_LEN;

    std::vector<fp32_t> in(MAX_DEPTH, 0.0f);
    std::vector<fp32_t> out(MAX_DEPTH, 0.0f);
    std::vector<fp32_t> ref_out(TOTAL_ELEMENTS, 0.0f);

    // Initialise synthetic logit test data (including large values to test stability)
    for (int r = 0; r < ROWS; ++r) {
        for (int c = 0; c < ROW_LEN; ++c) {
            in[r * ROW_LEN + c] = (fp32_t)((c % 13) - 6) * 1.5f + (fp32_t)(r * 10);
        }
    }

    // Compute CPU Golden Reference
    for (int r = 0; r < ROWS; ++r) {
        int base = r * ROW_LEN;
        fp32_t max_v = in[base];
        for (int c = 1; c < ROW_LEN; ++c) {
            if (in[base + c] > max_v) max_v = in[base + c];
        }

        fp32_t sum_exp = 0.0f;
        for (int c = 0; c < ROW_LEN; ++c) {
            ref_out[base + c] = std::exp(in[base + c] - max_v);
            sum_exp += ref_out[base + c];
        }

        fp32_t inv_sum = 1.0f / sum_exp;
        for (int c = 0; c < ROW_LEN; ++c) {
            ref_out[base + c] *= inv_sum;
        }
    }

    // Call Top-Level HLS Kernel
    softmax_kernel(in.data(), out.data(), ROWS, ROW_LEN);

    // Validate assertions
    int errors = 0;
    fp32_t max_diff = 0.0f;
    for (int r = 0; r < ROWS; ++r) {
        fp32_t prob_sum = 0.0f;
        for (int c = 0; c < ROW_LEN; ++c) {
            int idx = r * ROW_LEN + c;
            fp32_t diff = std::fabs(out[idx] - ref_out[idx]);
            if (diff > max_diff) max_diff = diff;
            if (diff > 1e-4f) {
                if (errors < 5) {
                    std::cout << "Mismatch at [" << idx << "]: got " << out[idx]
                              << ", expected " << ref_out[idx] << " (diff=" << diff << ")\n";
                }
                errors++;
            }
            prob_sum += out[idx];
        }
        // Verify probability distribution sums to ~1.0
        if (std::fabs(prob_sum - 1.0f) > 1e-4f) {
            std::cout << "Row " << r << " probability sum violation: " << prob_sum << "\n";
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
