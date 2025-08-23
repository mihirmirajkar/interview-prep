"""
Problem: Valid Anagram
Difficulty: Easy
Category: Strings
URL: https://leetcode.com/problems/valid-anagram/

Given two strings s and t, return true if t is an anagram of s, and false otherwise.
An Anagram is a word or phrase formed by rearranging the letters of a different word or phrase,
typically using all the original letters exactly once.
"""

from collections import Counter


def is_anagram(s: str, t: str) -> bool:
    """
    Check if two strings are anagrams.

    Args:
        s: First string
        t: Second string

    Returns:
        True if strings are anagrams, False otherwise

    Time Complexity: O(n)
    Space Complexity: O(1) - at most 26 characters

    Examples:
        >>> is_anagram("anagram", "nagaram")
        True
        >>> is_anagram("rat", "car")
        False
    """
    if len(s) != len(t):
        return False

    return Counter(s) == Counter(t)


def is_anagram_sorting(s: str, t: str) -> bool:
    """
    Alternative solution using sorting.

    Time Complexity: O(n log n)
    Space Complexity: O(1)
    """
    return sorted(s) == sorted(t)


def is_anagram_manual_count(s: str, t: str) -> bool:
    """
    Manual character counting approach.

    Time Complexity: O(n)
    Space Complexity: O(1)
    """
    if len(s) != len(t):
        return False

    char_count: dict[str, int] = {}

    # Count characters in s
    for char in s:
        char_count[char] = char_count.get(char, 0) + 1

    # Decrement counts for characters in t
    for char in t:
        if char not in char_count:
            return False
        char_count[char] -= 1
        if char_count[char] == 0:
            del char_count[char]

    return len(char_count) == 0


if __name__ == "__main__":
    # Test cases
    test_cases = [
        ("anagram", "nagaram", True),
        ("rat", "car", False),
        ("listen", "silent", True),
        ("evil", "vile", True),
        ("a", "ab", False),
        ("", "", True),
    ]

    for s, t, expected in test_cases:
        result = is_anagram(s, t)
        print(f'is_anagram("{s}", "{t}") = {result} (expected: {expected})')
        assert result == expected, f"Failed for {s}, {t}"

    print("All tests passed!")
