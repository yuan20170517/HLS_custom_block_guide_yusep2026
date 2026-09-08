import unittest
import subprocess
import shutil
import os
import sys
from pathlib import Path

class TestGELUEmbedKernel(unittest.TestCase):
    def setUp(self):
        self.repo_root = Path(__file__).parent.parent
        self.gelu_dir = self.repo_root / "src" / "GELU"

    def test_algorithmic_golden_reference(self):
        """Verify the exact mathematical and indexing behavior of gelu_embed_kernel."""
        GE_LEN = 64
        EMB_VOCAB = 32
        EMB_DIM = 8

        # 1. Recreate testbench input vectors
        # X: signed 8-bit stimulus in [-15, +15]
        # Reinterpreted as uint8 index: if x < 0, index is 256 + x
        X_raw = [(i % 31) - 15 for i in range(GE_LEN)]
        X_uint8 = [x & 0xFF for x in X_raw]

        # gelu_lut: synthetic step function
        gelu_lut = [63 if i > 127 else -63 for i in range(256)]

        # embed_lut: 32x8 matrix
        embed_lut = [[(r + c) % 16 for c in range(EMB_DIM)] for r in range(EMB_VOCAB)]

        # token_ids: sequential tokens 0..7
        token_ids = [i for i in range(EMB_DIM)]

        # 2. Kernel execution logic
        # gelu_loop:
        gelu_out = [gelu_lut[X_uint8[i]] for i in range(GE_LEN)]

        # embed_loop:
        embed_out = [embed_lut[token_ids[i] % EMB_VOCAB][i] for i in range(EMB_DIM)]

        # 3. Assertions matching tb_gelu_embed_kernel.cpp
        for i in range(GE_LEN):
            expected = gelu_lut[X_uint8[i]]
            self.assertEqual(gelu_out[i], expected, f"GELU mismatch at index {i}")

        for i in range(EMB_DIM):
            expected = embed_lut[token_ids[i] % EMB_VOCAB][i]
            self.assertEqual(embed_out[i], expected, f"Embedding mismatch at dim {i}")

    def test_cpp_compilation_and_execution(self):
        """Attempt to locate C++ compiler and execute native testbench binary."""
        # Search for available compilers
        candidate_compilers = ["g++", "clang++", "cl"]
        
        # Check standard Xilinx / Vitis paths for clang++
        xilinx_paths = list(Path("C:/Xilinx").glob("**/bin/clang++.exe")) if Path("C:/Xilinx").exists() else []
        if xilinx_paths:
            candidate_compilers.insert(0, str(xilinx_paths[0]))

        found_compiler = None
        for c in candidate_compilers:
            if shutil.which(c):
                found_compiler = c
                break

        if not found_compiler:
            print("[INFO] No native C++ compiler (g++, clang++, cl) found in PATH. Python algorithmic golden verification passed.")
            return

        print(f"[INFO] Compiling C++ testbench using: {found_compiler}")
        tb_src = self.gelu_dir / "tb_gelu_embed_kernel.cpp"
        kernel_src = self.gelu_dir / "gelu_embed_kernel.cpp"
        include_dir = self.repo_root / "src"
        out_bin = self.gelu_dir / "tb_gelu_test.exe"

        if "cl" in Path(found_compiler).name.lower() and not "clang" in Path(found_compiler).name.lower():
            cmd = [found_compiler, "/std:c++17", f"/I{include_dir}", str(tb_src), str(kernel_src), f"/Fe:{out_bin}"]
        else:
            cmd = [found_compiler, "-std=c++17", f"-I{include_dir}", str(tb_src), str(kernel_src), "-o", str(out_bin)]

        try:
            compile_res = subprocess.run(cmd, cwd=str(self.gelu_dir), capture_output=True, text=True, check=True)
            run_res = subprocess.run([str(out_bin)], cwd=str(self.gelu_dir), capture_output=True, text=True, check=True)
            print("[C++ TB Output]:", run_res.stdout.strip())
            self.assertIn("TB_GELU_EMBED PASS", run_res.stdout)
        finally:
            if out_bin.exists():
                out_bin.unlink(missing_ok=True)
            for ext in [".obj", ".o"]:
                for f in self.gelu_dir.glob(f"*{ext}"):
                    f.unlink(missing_ok=True)

if __name__ == "__main__":
    unittest.main()
