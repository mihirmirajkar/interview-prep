"""Test configuration and fixtures."""


import pytest


@pytest.fixture
def sample_array() -> list[int]:
    """Sample array for testing."""
    return [3, 1, 4, 1, 5, 9, 2, 6, 5, 3]


@pytest.fixture
def sorted_array() -> list[int]:
    """Sorted array for testing."""
    return [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]


@pytest.fixture
def empty_array() -> list[int]:
    """Empty array for testing."""
    return []


@pytest.fixture
def single_element_array() -> list[int]:
    """Single element array for testing."""
    return [42]
