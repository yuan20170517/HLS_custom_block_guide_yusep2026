/**
 * @file tb_embedding_kernel.cpp
 * @brief Self-Checking Testbench for Token & Position Embedding Streamer Kernel.
 */

#include "embedding_kernel.hpp"
#include <iostream>
#include <vector>
#include <cmath>

int main() {
    std::cout << "========================================================\n";
    std::cout << " Starting C-Simulation: Token & Position Embedding Kernel \n";
    std::cout << "========================================================\n";

    const int TEST_VOCAB = 64;
    const int TEST_POS   = 16;
    const int TEST_DIM   = DEFAULT_EMBED_DIM; // 768

    // Allocate synthetic memory buffers
    std::vector<fp32_t> wte_table(TEST_VOCAB * TEST_DIM);
    std::vector<fp32_t> wpe_table(TEST_POS * TEST_DIM);
    std::vector<fp32_t> out_mem(TEST_DIM, 0.0f);
    hls::stream<axis_float_t> stream_out;

    // Populate with deterministic test data
    for (int t = 0; t < TEST_VOCAB; t++) {
        for (int d = 0; d < TEST_DIM; d++) {
            wte_table[t * TEST_DIM + d] = std::sin(t * 0.15f) + static_cast<float>(d) * 0.001f;
        }
    }

    for (int p = 0; p < TEST_POS; p++) {
        for (int d = 0; d < TEST_DIM; d++) {
            wpe_table[p * TEST_DIM + d] = std::cos(p * 0.25f) - static_cast<float>(d) * 0.0005f;
        }
    }

    // --------------------------------------------------------------------------
    // Test Case 1: Standard token (Token ID = 7, Position = 3)
    // --------------------------------------------------------------------------
    int test_token = 7;
    int test_pos   = 3;
    std::cout << "[Test 1] Executing embedding for Token=" << test_token 
              << ", Pos=" << test_pos << ", Dim=" << TEST_DIM << "...\n";

    embedding_kernel(
        wte_table.data(),
        wpe_table.data(),
        out_mem.data(),
        test_token,
        test_pos,
        TEST_DIM,
        1, // write_to_mem = true
        stream_out
    );

    // Verify stream packet count and values
    bool pass = true;
    for (int i = 0; i < TEST_DIM; i++) {
        if (stream_out.empty()) {
            std::cerr << "ERROR: Stream underflow at index " << i << "\n";
            return 1;
        }

        axis_float_t pkt = stream_out.read();
        union {
            uint32_t u;
            float f;
        } conv;
        conv.u = pkt.data.to_uint();

        float expected_val = wte_table[test_token * TEST_DIM + i] + wpe_table[test_pos * TEST_DIM + i];
        float actual_val   = conv.f;

        // Check value tolerance
        if (std::abs(actual_val - expected_val) > 1e-5f) {
            std::cerr << "ERROR: Value mismatch at dim " << i 
                      << " Expected=" << expected_val << " Actual=" << actual_val << "\n";
            pass = false;
        }

        // Check memory writeback
        if (std::abs(out_mem[i] - expected_val) > 1e-5f) {
            std::cerr << "ERROR: Memory buffer mismatch at dim " << i << "\n";
            pass = false;
        }

        // Verify TLAST framing: Must be asserted ONLY on the last dimension
        bool expected_last = (i == TEST_DIM - 1);
        if (pkt.last.to_bool() != expected_last) {
            std::cerr << "ERROR: TLAST framing error at dim " << i 
                      << " Expected=" << expected_last << " Actual=" << pkt.last.to_bool() << "\n";
            pass = false;
        }
    }

    if (!stream_out.empty()) {
        std::cerr << "ERROR: Residual packets detected in output stream!\n";
        pass = false;
    }

    // --------------------------------------------------------------------------
    // Test Case 2: Boundary test (Token ID = 0, Position = 0)
    // --------------------------------------------------------------------------
    std::cout << "[Test 2] Executing boundary test for Token=0, Pos=0...\n";
    embedding_kernel(
        wte_table.data(),
        wpe_table.data(),
        out_mem.data(),
        0,
        0,
        TEST_DIM,
        0, // stream only
        stream_out
    );

    for (int i = 0; i < TEST_DIM; i++) {
        axis_float_t pkt = stream_out.read();
        union { uint32_t u; float f; } conv;
        conv.u = pkt.data.to_uint();
        float expected_val = wte_table[0 * TEST_DIM + i] + wpe_table[0 * TEST_DIM + i];
        if (std::abs(conv.f - expected_val) > 1e-5f) {
            pass = false;
        }
    }

    if (pass) {
        std::cout << "\n>>> C-SIMULATION PASSED: TB_EMBEDDING_KERNEL PASS <<<\n";
        std::cout << "All 768-dim embeddings matched with exact TLAST AXI4-Stream packet framing.\n";
        return 0;
    } else {
        std::cerr << "\n>>> C-SIMULATION FAILED: TB_EMBEDDING_KERNEL FAIL <<<\n";
        return 1;
    }
}
