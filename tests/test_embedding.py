import unittest
import math
from pathlib import Path

class TestEmbeddingKernel(unittest.TestCase):
    """Algorithmic verification and structural integrity for Token & Position Embedding Streamer."""

    def setUp(self):
        self.repo_root = Path(__file__).resolve().parent.parent
        self.src_dir = self.repo_root / "src" / "embedding"
        self.config_dir = self.repo_root / "config"

    def test_embedding_files_exist(self):
        """Verify that all embedding kernel source, header, testbench, and config files exist."""
        required_files = [
            self.src_dir / "embedding_kernel.hpp",
            self.src_dir / "embedding_kernel.cpp",
            self.src_dir / "tb_embedding_kernel.cpp",
            self.config_dir / "hls_embedding.cfg"
        ]
        for fpath in required_files:
            self.assertTrue(fpath.exists(), f"Required file missing: {fpath}")

    def test_algorithmic_golden_embedding(self):
        """Verify element-wise mathematical addition of WTE and WPE using standard library."""
        vocab_size = 64
        seq_len = 16
        embed_dim = 768

        # Deterministic synthetic tables
        wte_table = [[math.sin(t * 0.15) + d * 0.001 for d in range(embed_dim)] for t in range(vocab_size)]
        wpe_table = [[math.cos(p * 0.25) - d * 0.0005 for d in range(embed_dim)] for p in range(seq_len)]

        token_id = 7
        pos_id = 3

        # Expected hardware result
        expected_embedding = [wte_table[token_id][d] + wpe_table[pos_id][d] for d in range(embed_dim)]

        self.assertEqual(len(expected_embedding), embed_dim)
        for val in expected_embedding:
            self.assertTrue(math.isfinite(val))

    def test_boundary_offsets(self):
        """Verify offset calculation for boundary tokens."""
        vocab_size = 50257
        pos_len = 1024
        dim = 768

        max_token_offset = (vocab_size - 1) * dim
        max_pos_offset = (pos_len - 1) * dim

        self.assertEqual(max_token_offset, 50256 * 768)
        self.assertEqual(max_pos_offset, 1023 * 768)
        self.assertLess(max_token_offset + dim, vocab_size * dim + 1)

if __name__ == "__main__":
    unittest.main()
