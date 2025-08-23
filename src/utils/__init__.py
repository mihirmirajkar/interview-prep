"""Utility functions for interview preparation."""

from typing import Any, list


def print_matrix(matrix: list[list[Any]]) -> None:
    """Pretty print a 2D matrix."""
    for row in matrix:
        print(" ".join(f"{item:>3}" for item in row))


def time_function(func, *args, **kwargs):
    """Time the execution of a function."""
    import time

    start = time.time()
    result = func(*args, **kwargs)
    end = time.time()
    print(f"Function {func.__name__} took {end - start:.6f} seconds")
    return result


def generate_test_array(size: int, min_val: int = 0, max_val: int = 100) -> list[int]:
    """Generate a random test array."""
    import random

    return [random.randint(min_val, max_val) for _ in range(size)]
