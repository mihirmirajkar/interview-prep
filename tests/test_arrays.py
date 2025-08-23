"""Tests for array problems."""

from problems.arrays.best_time_to_buy_sell_stock import (
    max_profit,
    max_profit_brute_force,
)
from problems.arrays.two_sum import two_sum, two_sum_brute_force


class TestTwoSum:
    """Test cases for two sum problem."""

    def test_two_sum_basic(self):
        """Test basic two sum cases."""
        assert two_sum([2, 7, 11, 15], 9) == [0, 1]
        assert two_sum([3, 2, 4], 6) == [1, 2]
        assert two_sum([3, 3], 6) == [0, 1]

    def test_two_sum_no_solution(self):
        """Test when no solution exists."""
        assert two_sum([1, 2, 3], 7) is None
        assert two_sum([1], 1) is None

    def test_two_sum_empty_array(self):
        """Test with empty array."""
        assert two_sum([], 0) is None

    def test_two_sum_vs_brute_force(self):
        """Compare optimized vs brute force solutions."""
        test_cases = [
            ([2, 7, 11, 15], 9),
            ([3, 2, 4], 6),
            ([3, 3], 6),
        ]

        for nums, target in test_cases:
            result1 = two_sum(nums, target)
            result2 = two_sum_brute_force(nums, target)
            assert result1 == result2


class TestMaxProfit:
    """Test cases for max profit problem."""

    def test_max_profit_basic(self):
        """Test basic max profit cases."""
        assert max_profit([7, 1, 5, 3, 6, 4]) == 5
        assert max_profit([7, 6, 4, 3, 1]) == 0
        assert max_profit([1, 2, 3, 4, 5]) == 4

    def test_max_profit_edge_cases(self):
        """Test edge cases."""
        assert max_profit([]) == 0
        assert max_profit([5]) == 0
        assert max_profit([1, 2]) == 1

    def test_max_profit_vs_brute_force(self):
        """Compare optimized vs brute force solutions."""
        test_cases = [
            [7, 1, 5, 3, 6, 4],
            [7, 6, 4, 3, 1],
            [1, 2, 3, 4, 5],
        ]

        for prices in test_cases:
            result1 = max_profit(prices)
            result2 = max_profit_brute_force(prices)
            assert result1 == result2
