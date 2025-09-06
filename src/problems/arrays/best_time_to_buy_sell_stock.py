"""
Problem: Best Time to Buy and Sell Stock
Difficulty: Easy
Category: Arrays
URL: https://leetcode.com/problems/best-time-to-buy-and-sell-stock/

You are given an array prices where prices[i] is the price of a given stock on the ith day.
You want to maximize your profit by choosing a single day to buy one stock and choosing a different day in the future to sell that stock.
Return the maximum profit you can achieve from this transaction. If you cannot achieve any profit, return 0.
"""



def max_profit(prices: list[int]) -> int:
    """
    Find the maximum profit from buying and selling stock once.

    Args:
        prices: List of stock prices by day

    Returns:
        Maximum profit possible

    Time Complexity: O(n)
    Space Complexity: O(1)

    Examples:
        >>> max_profit([7, 1, 5, 3, 6, 4])
        5
        >>> max_profit([7, 6, 4, 3, 1])
        0
    """
    if len(prices) < 2:
        return 0

    min_price = prices[0]
    max_profit_val = 0

    for price in prices[1:]:
        # Update max profit if we can sell at current price
        max_profit_val = max(max_profit_val, price - min_price)
        # Update minimum price seen so far
        min_price = min(min_price, price)

    return max_profit_val


def max_profit_brute_force(prices: list[int]) -> int:
    """
    Brute force solution for comparison.

    Time Complexity: O(n²)
    Space Complexity: O(1)
    """
    max_profit_val = 0
    n = len(prices)

    for i in range(n):
        for j in range(i + 1, n):
            profit = prices[j] - prices[i]
            max_profit_val = max(max_profit_val, profit)

    return max_profit_val


if __name__ == "__main__":
    # Test cases
    test_cases = [
        ([7, 1, 5, 3, 6, 4], 5),
        ([7, 6, 4, 3, 1], 0),
        ([1, 2, 3, 4, 5], 4),
        ([5], 0),
        ([], 0),
    ]

    for prices, expected in test_cases:
        result = max_profit(prices)
        print(f"max_profit({prices}) = {result} (expected: {expected})")
        assert result == expected, f"Failed for {prices}"

    print("All tests passed!")
