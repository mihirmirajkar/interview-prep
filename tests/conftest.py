"""Test configuration and fixtures."""

import pytest
from typing import List, Any


@pytest.fixture
def sample_array() -> List[int]:
    """Sample array for testing."""
    return [3, 1, 4, 1, 5, 9, 2, 6, 5, 3]


@pytest.fixture
def sorted_array() -> List[int]:
    """Sorted array for testing."""
    return [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]


@pytest.fixture
def empty_array() -> List[int]:
    """Empty array for testing."""
    return []


@pytest.fixture
def single_element_array() -> List[int]:
    """Single element array for testing."""
    return [42]
