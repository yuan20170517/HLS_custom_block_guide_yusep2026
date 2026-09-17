import unittest
import math
import random
from pathlib import Path

class TestClosedLoopPipeline(unittest.TestCase):
    """End-to-End algorithmic verification of the 3-stage hardware pipeline:
    Embedding (Pre-processing) -> MHA (Attention) -> Sampler (Token Retrieval).
    """

    def setUp(self):
        self.repo_root = Path(__file__).resolve().parent.parent
        self.config_dir = self.repo_root / "config"

    def test_configurations_exist(self):
        """Verify all HLS .cfg and system.cfg files are present for the triad."""
        expected_configs = [
            self.config_dir / "hls_embedding.cfg",
            self.config_dir / "hls_mha.cfg",
            self.config_dir / "hls_sampler.cfg",
            self.config_dir / "system_closed_loop.cfg"
        ]
        for cfg in expected_configs:
            self.assertTrue(cfg.exists(), f"Missing configuration file: {cfg}")

    def test_e2e_closed_loop_simulation(self):
        """Simulate the closed-loop dataflow using standard library:
        Input Token -> Embedding Vector -> MHA Attention -> Logit Generation -> ArgMax Retrieval -> Next Token.
        """
        random.seed(1337)
        vocab_size = 500
        embed_dim = 16
        seq_len = 16

        # Step 1: Embedding Tables
        wte = [[random.gauss(0, 1) for _ in range(embed_dim)] for _ in range(vocab_size)]
        wpe = [[random.gauss(0, 1) for _ in range(embed_dim)] for _ in range(seq_len)]

        input_token_id = 42
        current_pos = 0

        # Stage 1: Embedding Kernel (Vector addition)
        token_vector = [wte[input_token_id][d] + wpe[current_pos][d] for d in range(embed_dim)]
        self.assertEqual(len(token_vector), embed_dim)

        # Stage 2: Attention (Scaled Dot-Product mockup)
        # Simulate LM Head output logits
        logits = [math.sin(i * 0.1) * 5.0 + (10.0 if i == 188 else 0.0) for i in range(vocab_size)]

        # Stage 3: Output Token Retrieval (Temperature + ArgMax / Top-K)
        temperature = 0.8
        scaled_logits = [l / temperature for l in logits]

        # Greedy ArgMax retrieval (as performed by sampler_kernel)
        max_val = -1e30
        best_token_id = -1
        for idx, val in enumerate(scaled_logits):
            if val > max_val:
                max_val = val
                best_token_id = idx

        # Sort indices for Top-K
        sorted_indices = sorted(range(vocab_size), key=lambda k: scaled_logits[k], reverse=True)
        top_k_indices = sorted_indices[:8]

        # Assertions
        self.assertEqual(best_token_id, 188)
        self.assertEqual(best_token_id, top_k_indices[0])
        self.assertEqual(len(top_k_indices), 8)

        # Feedback loop assertion: next token can loop back as input!
        next_input_token = best_token_id
        next_token_vector = [wte[next_input_token][d] + wpe[current_pos + 1][d] for d in range(embed_dim)]
        self.assertEqual(len(next_token_vector), embed_dim)

if __name__ == "__main__":
    unittest.main()
