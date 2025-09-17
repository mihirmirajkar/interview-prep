def manacher_longest_palindromic_substring_gpt(s: str) -> str:
    """
    Manacher's algorithm: O(N) time, O(N) space.
    Returns the longest palindromic substring in s.
    Args:
        s (str): The input string.
    Returns:
        str: The longest palindromic substring.
    """
    if not s:
        return ""
    # Transform s to add boundaries to handle even/odd palindromes uniformly
    t = "#" + "#".join(s) + "#"
    n = len(t)
    p = [0] * n  # Array to store palindrome radius at each center
    center = 0
    right = 0
    max_len = 0
    max_center = 0
    for i in range(n):
        mirror = 2 * center - i
        if i < right:
            p[i] = min(right - i, p[mirror])
        # Expand around center i
        a = i + p[i] + 1
        b = i - p[i] - 1
        while a < n and b >= 0 and t[a] == t[b]:
            p[i] += 1
            a += 1
            b -= 1
        # Update center and right boundary
        if i + p[i] > right:
            center = i
            right = i + p[i]
        # Track max palindrome
        if p[i] > max_len:
            max_len = p[i]
            max_center = i
    # Extract the longest palindrome from the original string
    start = (max_center - max_len) // 2
    return s[start:start + max_len]

"""
Problem: Longest Palindromic Substring
Category: Strings
Level: Medium/Hard

Given a string s, return the longest palindromic substring in s.

Example 1:
Input: s = "babad"
Output: "bab"
Explanation: "aba" is also a valid answer.

Example 2:
Input: s = "cbbd"
Output: "bb"

Constraints:
- 1 <= s.length <= 1000
- s consists of only digits and English letters.
"""

from re import L


def longest_palindromic_substring(s: str) -> str:
    """
    Returns the longest palindromic substring in s.
    Args:
        s (str): The input string.
    Returns:
        str: The longest palindromic substring.
    """
    if len(s)<2:
        return s
    
    left = 0
    right = 1
    longest = s[0]
    while left < right and right < len(s):
        if s[right] == s[left]:
            if len(longest) < len(s[left:right+1]):
                longest = s[left:right+1]
            left = max(0, left-1)
            right += 1
        elif right + 1 < len(s) and s[left] == s[right+1]:
            right += 1
            if len(longest) < len(s[left:right+1]):
                longest = s[left:right+1]      
        else:
            left = (left + right)//2 + 1
            right = left + 1
        
    return longest


def main():
    # Test cases
    test_cases = [
        ("babad", ["bab", "aba"]),
        ("cbbd", ["bb"]),
        ("a", ["a"]),
        ("ac", ["a", "c"]),
        ("forgeeksskeegfor", ["geeksskeeg"]),
        ("abccba", ["abccba"]),
        ("", [""]),
        ("abcda", ["a", "b", "c", "d"]),
        ("abcbabcba", ["abcbabcba"]),  # Odd-length palindrome, should pass
        ("abcddcba", ["abcddcba"]),    # Even-length palindrome, should fail with current solution
    ]
    passed = 0
    for i, (s, expected) in enumerate(test_cases):
        result = longest_palindromic_substring(s)
        if result in expected:
            print(f"Test case {i+1} passed.")
            passed += 1
        else:
            print(f"Test case {i+1} failed: input={s}, expected one of {expected}, got {result}")
    print(f"Passed {passed}/{len(test_cases)} test cases.")

if __name__ == "__main__":
    main()
