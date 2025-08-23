"""
Problem: Two Sum
Difficulty: Easy
Category: Arrays
URL: https://leetcode.com/problems/two-sum/

Given an array of integers nums and an integer target, return indices of the two numbers such that they add up to target.
"""



def two_sum(nums: list[int], target: int) -> list[int] | None:
    """
    Find two numbers in the array that add up to the target.

    Args:
        nums: List of integers
        target: Target sum

    Returns:
        Indices of the two numbers that add up to target, or None if not found

    Time Complexity: O(n)
    Space Complexity: O(n)

    Examples:
        >>> two_sum([2, 7, 11, 15], 9)
        [0, 1]
        >>> two_sum([3, 2, 4], 6)
        [1, 2]
        >>> two_sum([3, 3], 6)
        [0, 1]
    """
    seen = {}
    for i, num in enumerate(nums):
        complement = target - num
        if complement in seen:
            return [seen[complement], i]
        seen[num] = i
    return None


def two_sum_brute_force(nums: list[int], target: int) -> list[int] | None:
    """
    Brute force solution for comparison.

    Time Complexity: O(n²)
    Space Complexity: O(1)
    """
    n = len(nums)
    for i in range(n):
        for j in range(i + 1, n):
            if nums[i] + nums[j] == target:
                return [i, j]
    return None


if __name__ == "__main__":
    # Test cases
    test_cases = [
        ([2, 7, 11, 15], 9, [0, 1]),
        ([3, 2, 4], 6, [1, 2]),
        ([3, 3], 6, [0, 1]),
        ([1, 2, 3], 7, None),
    ]

    for nums, target, expected in test_cases:
        result = two_sum_brute_force(nums, target)
        print(f"two_sum({nums}, {target}) = {result} (expected: {expected})")
        assert result == expected, f"Failed for {nums}, {target}"

    print("All tests passed!")
