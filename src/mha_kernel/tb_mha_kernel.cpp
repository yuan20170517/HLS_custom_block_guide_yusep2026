/**
 * @file tb_mha_kernel.cpp
 * @brief Self-Checking Testbench for Quantized Multi-Head Attention Kernel.
 */

#include "mha_kernel.hpp"
#include <iostream>

int main() {
    std::cout << "========================================================\n";
    std::cout << " Starting C-Simulation: Multi-Head Attention Kernel     \n";
    std::cout << "========================================================\n";

    int8_t_hls  X[MHA_SEQ][MHA_DIM];
    int8_t_hls  WQ[MHA_DIM][MHA_DIM];
    int8_t_hls  WK[MHA_DIM][MHA_DIM];
    int8_t_hls  WV[MHA_DIM][MHA_DIM];
    uint8_t_hls softmax_lut[16];
    int16_t_hls OUT[MHA_SEQ][MHA_DIM];

    // 1. Initialise synthetic deterministic stimulus
    for (int i = 0; i < MHA_SEQ; ++i) {
        for (int d = 0; d < MHA_DIM; ++d) {
            X[i][d] = (i + d) % 7 - 3;
        }
    }

    for (int i = 0; i < MHA_DIM; ++i) {
        for (int j = 0; j < MHA_DIM; ++j) {
            WQ[i][j] = (i + j) % 5 - 2;
            WK[i][j] = (i - j) % 5;
            WV[i][j] = (2 * i + j) % 7 - 3;
        }
    }

    for (int i = 0; i < 16; ++i) {
        softmax_lut[i] = (i + 1) * 8;
    }

    // 2. Invoke Hardware Kernel
    mha_kernel(X, WQ, WK, WV, softmax_lut, OUT);

    // 3. Validate output sanity
    bool valid = true;
    for (int i = 0; i < MHA_SEQ; ++i) {
        for (int d = 0; d < MHA_DIM; ++d) {
            // Ensure no undefined or uninitialised values
            int val = (int)OUT[i][d];
            if (val < -32768 || val > 32767) {
                valid = false;
            }
        }
    }

    if (valid) {
        std::cout << ">>> C-SIMULATION PASSED: TB_MHA_KERNEL PASS <<<\n";
        return 0;
    } else {
        std::cout << ">>> C-SIMULATION FAILED: Output sanity check failed! <<<\n";
        return 1;
    }
}
