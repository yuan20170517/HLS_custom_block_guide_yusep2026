import unittest
from pathlib import Path

class TestRepositoryStructure(unittest.TestCase):
    """Sanity checks for HLS Custom Block Guide repository structure."""

    def setUp(self):
        self.repo_root = Path(__file__).resolve().parent.parent
        self.src_dir = self.repo_root / "src"

    def test_src_subdirectories_exist(self):
        """Verify that all core HLS accelerator module directories exist."""
        expected_modules = ["common", "dct", "GELU", "layernorm", "softmax", "mha_kernel", "tiled_matmul"]
        for mod in expected_modules:
            mod_path = self.src_dir / mod
            self.assertTrue(mod_path.exists(), f"Expected module folder missing: {mod_path}")
            self.assertTrue(mod_path.is_dir(), f"Expected module path is not a directory: {mod_path}")

    def test_common_headers_exist(self):
        """Verify that shared HLS headers exist."""
        common_dir = self.src_dir / "common"
        self.assertTrue((common_dir / "hls_common.hpp").exists(), "hls_common.hpp missing")
        self.assertTrue((common_dir / "types.h").exists(), "types.h missing")

    def test_dct_files_exist(self):
        """Verify that 2D DCT kernel, header, coeff table and test files exist."""
        dct_dir = self.src_dir / "dct"
        for fname in ["dct.cpp", "dct.h", "dct_test.cpp", "dct_coeff_table.txt", "in.dat", "out.golden.dat"]:
            self.assertTrue((dct_dir / fname).exists(), f"DCT file missing: {fname}")

if __name__ == "__main__":
    unittest.main()
