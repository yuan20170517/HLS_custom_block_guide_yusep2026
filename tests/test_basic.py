import unittest
import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from main import get_status

class TestBasicSetup(unittest.TestCase):
    def test_repository_status(self):
        info = get_status()
        self.assertEqual(info["status"], "ready")
        self.assertIn("HLS", info["domain"])
        self.assertEqual(info["version"], "0.1.0")

if __name__ == "__main__":
    unittest.main()
