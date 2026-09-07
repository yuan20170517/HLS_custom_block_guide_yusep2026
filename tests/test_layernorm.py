import unittest
import math

class TestLayerNormAlgorithm(unittest.TestCase):
    def test_layernorm_math(self):
        """Verify LayerNorm reference mathematics (mean, variance, normalisation)."""
        dim = 768
        # Create deterministic synthetic input
        x = [math.sin(i * 0.1) for i in range(dim)]
        gamma = [1.0] * dim
        beta = [0.0] * dim

        mean = sum(x) / dim
        var = sum((xi - mean) ** 2 for xi in x) / dim
        inv_std = 1.0 / math.sqrt(var + 1e-5)

        y = [(xi - mean) * inv_std * g + b for xi, g, b in zip(x, gamma, beta)]

        # Normalized mean should be approx 0, variance approx 1
        norm_mean = sum(y) / dim
        norm_var = sum((yi - norm_mean) ** 2 for yi in y) / dim

        self.assertAlmostEqual(norm_mean, 0.0, places=4)
        self.assertAlmostEqual(norm_var, 1.0, places=4)

    def test_partial_accumulator_equivalence(self):
        """Verify 16-way partial accumulator produces identical sum to linear accumulation."""
        nacc = 16
        data = [float(i * 0.05) for i in range(768)]
        
        # Standard linear sum
        linear_sum = sum(data)

        # 16-way partial accumulator
        acc = [0.0] * nacc
        for i, val in enumerate(data):
            acc[i % nacc] += val
        partial_sum = sum(acc)

        self.assertAlmostEqual(linear_sum, partial_sum, places=5)

if __name__ == "__main__":
    unittest.main()
