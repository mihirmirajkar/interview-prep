"""
Problem: Container With Most Water
Category: Array, Two Pointers
LeetCode Link: https://leetcode.com/problems/container-with-most-water/

Given n non-negative integers representing an elevation map where the width of each bar is 1,
find two lines that together with the x-axis form a container, such that the container contains the most water.
"""

def maxArea(height: list[int]) -> int:
    """
    :param height: List of non-negative integers representing bar heights.
    :return: Maximum amount of water that can be contained.
    """
    # TODO: implement this method
    pass


if __name__ == "__main__":
    # Test cases
    tests = [
        ([1, 8, 6, 2, 5, 4, 8, 3, 7], 49),
        ([1, 1], 1),
        ([4, 3, 2, 1, 4], 16),
        ([1, 2, 1], 2),
        ([2, 3, 10, 5, 7, 8, 9], 36),
    ]

    for idx, (height, expected) in enumerate(tests, start=1):
        result = maxArea(height)
        print(f"Test case {idx}: height={height} | result={result} | expected={expected}")
        assert result == expected, f"Test case {idx} failed: expected {expected}, got {result}"

    print("All tests passed!")
