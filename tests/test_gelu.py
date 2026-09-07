import unittest
import math

class TestGELUAlgorithm(unittest.TestCase):
    def gelu_ref(self, x: float) -> float:
        """Standard GELU polynomial approximation used in HG-PIPE kernel."""
        sqrt_2_over_pi = 0.7978845608
        coeff = 0.044715
        inner = sqrt_2_over_pi * (x + coeff * (x ** 3))
        tanh_val = math.tanh(inner)
        return 0.5 * x * (1.0 + tanh_val)

    def test_gelu_boundary_conditions(self):
        """Verify GELU asymptotic and zero limits."""
        self.assertAlmostEqual(self.gelu_ref(0.0), 0.0, places=5)
        # For large positive x, GELU(x) ~= x
        self.assertAlmostEqual(self.gelu_ref(10.0), 10.0, places=4)
        # For large negative x, GELU(x) ~= 0
        self.assertAlmostEqual(self.gelu_ref(-10.0), 0.0, places=4)

    def test_gelu_values(self):
        """Verify key known points."""
        # GELU(1.0) ~= 0.8413
        self.assertAlmostEqual(self.gelu_ref(1.0), 0.8413, places=3)
        # GELU(-1.0) ~= -0.1587
        self.assertAlmostEqual(self.gelu_ref(-1.0), -0.1587, places=3)

if __name__ == "__main__":
    unittest.main()
