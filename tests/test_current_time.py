import os
import sys
import unittest
from datetime import datetime

# Add src directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from time_utils import get_current_time, print_current_time


class TestCurrentTime(unittest.TestCase):
    def test_print_current_time(self):
        """Test printing current time and verify output format."""
        current_time = print_current_time()
        print(f"[Test Output] Current system time is: {current_time}")
        self.assertIsInstance(current_time, str)
        # Verify that the returned time string matches standard datetime format
        parsed_time = datetime.strptime(current_time, "%Y-%m-%d %H:%M:%S")
        self.assertIsInstance(parsed_time, datetime)

    def test_get_current_time_custom_format(self):
        """Test retrieving current time with custom format."""
        date_str = get_current_time("%Y-%m-%d")
        self.assertEqual(len(date_str), 10)
        parsed_date = datetime.strptime(date_str, "%Y-%m-%d")
        self.assertIsInstance(parsed_date, datetime)


if __name__ == "__main__":
    unittest.main()
