#!/usr/bin/env python3
"""
Unit Test Suite: Softmax C++ Hardware Kernel Verification
Repository: HLS_custom_block_guide_yusep2026

Verifies that the Python testbench properly validates the C++ Softmax hardware
kernel (src/softmax/softmax_kernel.cpp) and its testbench (src/softmax/tb_softmax_kernel.cpp).
"""

import unittest
import math
from pathlib import Path
from src.softmax.tb_softmax import (
    generate_canonical_inputs,
    simulate_cpp_kernel,
    verify_softmax_cpp,
    compute_reference,
    ROWS,
    ROW_LEN
)

class TestSoftmaxCppKernel(unittest.TestCase):
    def setUp(self):
        self.rows = ROWS
        self.row_len = ROW_LEN
        self.inputs = generate_canonical_inputs(self.rows, self.row_len)
        self.outputs = simulate_cpp_kernel(self.inputs)

    def test_canonical_dataset_dimensions(self):
        """Verify that the test dataset matches the C++ testbench dimensions."""
        self.assertEqual(len(self.inputs), 4)
        self.assertEqual(len(self.inputs[0]), 64)
        self.assertEqual(len(self.outputs), 4)
        self.assertEqual(len(self.outputs[0]), 64)

    def test_dataset_formula_consistency(self):
        """Verify the exact mathematical formula matching tb_softmax_kernel.cpp."""
        for r in range(self.rows):
            for c in range(self.row_len):
                expected_in = float(((c % 13) - 6) * 1.5 + (r * 5.0))
                self.assertAlmostEqual(self.inputs[r][c], expected_in, places=5)

    def test_probability_distribution_normalization(self):
        """Verify that C++ simulated output probabilities sum to 1.0 for every row."""
        for r in range(self.rows):
            prob_sum = sum(self.outputs[r])
            self.assertAlmostEqual(prob_sum, 1.0, places=5,
                                   msg=f"Row {r} probabilities do not sum to 1.0")

    def test_numerical_stability_with_large_values(self):
        """Verify stability against overflow when large logits are provided."""
        extreme_logits = [[1000.0, 1005.0, 995.0, 1010.0] + [0.0] * (self.row_len - 4)]
        probs = simulate_cpp_kernel(extreme_logits)[0]
        self.assertFalse(any(math.isnan(p) or math.isinf(p) for p in probs))
        self.assertAlmostEqual(sum(probs), 1.0, places=5)

    def test_full_cpp_kernel_verification(self):
        """Verify the full Softmax C++ kernel testbench passes all assertions."""
        self.assertTrue(verify_softmax_cpp(verbose=False))

    def test_source_files_exist(self):
        """Verify all required C++ kernel, header, C++ testbench, and Python testbench files exist."""
        softmax_dir = Path(__file__).resolve().parent.parent / "src" / "softmax"
        self.assertTrue((softmax_dir / "README.md").exists())
        self.assertTrue((softmax_dir / "softmax_kernel.h").exists())
        self.assertTrue((softmax_dir / "softmax_kernel.cpp").exists())
        self.assertTrue((softmax_dir / "tb_softmax_kernel.cpp").exists())
        self.assertTrue((softmax_dir / "tb_softmax.py").exists())

if __name__ == "__main__":
    unittest.main()
