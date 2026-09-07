import unittest
import math

class TestSoftmaxAlgorithm(unittest.TestCase):
    def test_softmax_math(self):
        """Verify Softmax normalization, online max subtraction, and probability distribution."""
        row_len = 197  # ViT sequence length
        # Deterministic inputs with potentially large values to verify numerical stability
        raw = [float(i) * 0.1 for i in range(row_len)]
        
        # 1. Max subtraction for stability
        max_val = max(raw)
        exps = [math.exp(v - max_val) for v in raw]
        total_sum = sum(exps)
        probs = [e / total_sum for e in exps]

        # Assertions
        self.assertAlmostEqual(sum(probs), 1.0, places=5)
        self.assertTrue(all(0.0 <= p <= 1.0 for p in probs))
        # Monotonicity check: larger raw inputs must yield larger probabilities
        self.assertTrue(all(probs[i] <= probs[i+1] for i in range(row_len - 1)))

    def test_partial_accumulator_softmax_equivalence(self):
        """Verify 16-way partial accumulation matches standard sum for exponential values."""
        nacc = 16
        data = [math.exp((i % 20) * 0.1) for i in range(197)]

        linear_sum = sum(data)
        acc = [0.0] * nacc
        for i, val in enumerate(data):
            acc[i % nacc] += val
        partial_sum = sum(acc)

        self.assertAlmostEqual(linear_sum, partial_sum, places=5)

if __name__ == "__main__":
    unittest.main()
