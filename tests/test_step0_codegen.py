import unittest
import os
import sys
from pathlib import Path

# Add repo root to import step0
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))
from scripts.step0_case_generation import generate_cases

class TestStep0CaseGeneration(unittest.TestCase):
    def test_vit_base_generation(self):
        config_path = str(repo_root / "statistics" / "vit_base_config.json")
        out_dir = str(repo_root / "src" / "hls" / "instances")
        generate_cases(config_path, out_dir)

        # Check all 3 generated instances exist
        self.assertTrue((Path(out_dir) / "layernorm_kernel_gen.cpp").exists())
        self.assertTrue((Path(out_dir) / "softmax_kernel_gen.cpp").exists())
        self.assertTrue((Path(out_dir) / "gelu_kernel_gen.cpp").exists())

        # Verify parameter substitution in LayerNorm
        ln_content = (Path(out_dir) / "layernorm_kernel_gen.cpp").read_text(encoding="utf-8")
        self.assertIn("Vision_Transformer_Base_16", ln_content)
        self.assertIn("gamma[768]", ln_content)

if __name__ == "__main__":
    unittest.main()
