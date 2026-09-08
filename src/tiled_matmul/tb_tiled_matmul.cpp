/**
 * @file tb_tiled_matmul.cpp
 * @brief Self-Checking Testbench for Tiled Matrix Multiplication Hardware Kernel.
 */

#include "tiled_matmul.hpp"
#include <iostream>

int main() {
    std::cout << "========================================================\n";
    std::cout << " Starting C-Simulation: Tiled Matrix Multiplication     \n";
    std::cout << "========================================================\n";

    int8_t_hls  A[TM_ROWS][TM_K];
    int8_t_hls  B[TM_K][TM_COLS];
    int32_t_hls C[TM_ROWS][TM_COLS];
    int errors = 0;

    // 1. Initialise synthetic deterministic matrices
    for (int i = 0; i < TM_ROWS; ++i) {
        for (int k = 0; k < TM_K; ++k) {
            A[i][k] = (i + k) % 8 - 4;
        }
    }

    for (int k = 0; k < TM_K; ++k) {
        for (int j = 0; j < TM_COLS; ++j) {
            B[k][j] = (j - k) % 8;
        }
    }

    // 2. Invoke Hardware Kernel
    tiled_matmul_kernel(A, B, C);

    // 3. Compute Golden Reference & Verify Assertions
    for (int i = 0; i < TM_ROWS; ++i) {
        for (int j = 0; j < TM_COLS; ++j) {
            int ref = 0;
            for (int k = 0; k < TM_K; ++k) {
                ref += (int)A[i][k] * (int)B[k][j];
            }
            if ((int)C[i][j] != ref) {
                if (errors < 5) {
                    std::cout << "Mismatch at [" << i << "][" << j << "]: got "
                              << (int)C[i][j] << ", expected " << ref << "\n";
                }
                errors++;
            }
        }
    }

    if (errors == 0) {
        std::cout << ">>> C-SIMULATION PASSED: All matrix tiles match golden output. <<<\n";
        return 0;
    } else {
        std::cout << ">>> C-SIMULATION FAILED: Total Errors = " << errors << " <<<\n";
        return 1;
    }
}
