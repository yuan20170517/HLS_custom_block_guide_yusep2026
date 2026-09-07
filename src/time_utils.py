"""
Utility module for time operations.
"""
from datetime import datetime


def get_current_time(fmt: str = "%Y-%m-%d %H:%M:%S") -> str:
    """Return the current local time formatted as a string."""
    return datetime.now().strftime(fmt)


def print_current_time(fmt: str = "%Y-%m-%d %H:%M:%S") -> str:
    """Print and return the current local time."""
    current_time = get_current_time(fmt)
    print(f"Current Time: {current_time}")
    return current_time


if __name__ == "__main__":
    print_current_time()
