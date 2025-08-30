"""
Given a string s, find the length of the longest substring without repeating characters.

Example 1:
Input: s = "abcabcbb"
Output: 3
Explanation: The answer is "abc", with the length of 3.

Example 2:
Input: s = "bbbbb"
Output: 1
Explanation: The answer is "b", with the length of 1.

Example 3:
Input: s = "pwwkew"
Output: 3
Explanation: The answer is "wke", with the length of 3.
Notice that the answer must be a substring, "pwke" is a subsequence and not a substring.
"""

import collections
from re import sub

from numpy import long


class Solution:
    def lengthOfLongestSubstringWithoutWindow(self, s: str) -> str:
        # Your solution here
        substring_options = []
        counter = set()
        substring = ""
        i = 0
        while i < len(s):
            c = s[i]
            # print(f'On {i}, {c}')
            if c in counter:
                # print(counter)
                counter = set()
                i = i -len(substring)
                substring_options.append(substring)
                # print(f"found repeated char {c}, substring {substring}, new i = {i}")
                substring = ""
            else:
                substring += c
                counter.add(c)
            i += 1
        substring_options.append(substring)
        # print(substring_options)
        longest_substring = max(substring_options, key=len)
        return longest_substring
    
    def lengthOfLongestSubstring(self, s: str) -> str:
        if not s:
            return ""
        start = 0
        end = 1
        longest_substring = ""
        substring_set = set()
        substring_set.add(s[0]) 
        while end < len(s):
            substring = s[start:end]
            end_char = s[end]
            if end_char in substring_set:
                while start < end and s[start] != end_char:
                    substring_set.remove(s[start])
                    start += 1
                start += 1  # Move past the repeated character

            else:
                substring_set.add(end_char)
                substring += end_char
            if len(substring) > len(longest_substring):
                longest_substring = substring

            
            end += 1
        return longest_substring

def main():
    # Test cases
    solver = Solution()

    # Test Case 1
    s1 = "abcabcbb"
    expected_len1 = 3
    possible_substrings1 = {"abc", "bca", "cab"}
    output_substring1 = solver.lengthOfLongestSubstring(s1)
    test_passed1 = len(output_substring1) == expected_len1 and output_substring1 in possible_substrings1
    print(f"Test Case 1: {'Passed' if test_passed1 else 'Failed'}")
    print(f"Input: '{s1}'")
    print(f"Output Substring: '{output_substring1}' (Length: {len(output_substring1)})")
    print(f"Expected Length: {expected_len1}")
    print(f"Possible Substrings: {possible_substrings1}")
    print("-" * 20)

    # Test Case 2
    s2 = "bbbbb"
    expected_len2 = 1
    possible_substrings2 = {"b"}
    output_substring2 = solver.lengthOfLongestSubstring(s2)
    test_passed2 = len(output_substring2) == expected_len2 and output_substring2 in possible_substrings2
    print(f"Test Case 2: {'Passed' if test_passed2 else 'Failed'}")
    print(f"Input: '{s2}'")
    print(f"Output Substring: '{output_substring2}' (Length: {len(output_substring2)})")
    print(f"Expected Length: {expected_len2}")
    print(f"Possible Substrings: {possible_substrings2}")
    print("-" * 20)

    # Test Case 3
    s3 = "pwwkew"
    expected_len3 = 3
    possible_substrings3 = {"wke", "kew"}
    output_substring3 = solver.lengthOfLongestSubstring(s3)
    test_passed3 = len(output_substring3) == expected_len3 and output_substring3 in possible_substrings3
    print(f"Test Case 3: {'Passed' if test_passed3 else 'Failed'}")
    print(f"Input: '{s3}'")
    print(f"Output Substring: '{output_substring3}' (Length: {len(output_substring3)})")
    print(f"Expected Length: {expected_len3}")
    print(f"Possible Substrings: {possible_substrings3}")
    print("-" * 20)

    # Test Case 4: Empty string
    s4 = ""
    expected_len4 = 0
    possible_substrings4 = {""}
    output_substring4 = solver.lengthOfLongestSubstring(s4)
    test_passed4 = len(output_substring4) == expected_len4 and output_substring4 in possible_substrings4
    print(f"Test Case 4: {'Passed' if test_passed4 else 'Failed'}")
    print(f"Input: '{s4}'")
    print(f"Output Substring: '{output_substring4}' (Length: {len(output_substring4)})")
    print(f"Expected Length: {expected_len4}")
    print(f"Possible Substrings: {possible_substrings4}")
    print("-" * 20)

    # Test Case 5: String with no repeating characters
    s5 = "abcdefg"
    expected_len5 = 7
    possible_substrings5 = {"abcdefg"}
    output_substring5 = solver.lengthOfLongestSubstring(s5)
    test_passed5 = len(output_substring5) == expected_len5 and output_substring5 in possible_substrings5
    print(f"Test Case 5: {'Passed' if test_passed5 else 'Failed'}")
    print(f"Input: '{s5}'")
    print(f"Output Substring: '{output_substring5}' (Length: {len(output_substring5)})")
    print(f"Expected Length: {expected_len5}")
    print(f"Possible Substrings: {possible_substrings5}")
    print("-" * 20)

    # Test Case 6: String with all same characters
    s6 = "aaaaa"
    expected_len6 = 1
    possible_substrings6 = {"a"}
    output_substring6 = solver.lengthOfLongestSubstring(s6)
    test_passed6 = len(output_substring6) == expected_len6 and output_substring6 in possible_substrings6
    print(f"Test Case 6: {'Passed' if test_passed6 else 'Failed'}")
    print(f"Input: '{s6}'")
    print(f"Output Substring: '{output_substring6}' (Length: {len(output_substring6)})")
    print(f"Expected Length: {expected_len6}")
    print(f"Possible Substrings: {possible_substrings6}")
    print("-" * 20)

    # Test Case 7: dvdf
    s7 = "dvdf"
    expected_len7 = 3
    possible_substrings7 = {"vdf"}
    output_substring7 = solver.lengthOfLongestSubstring(s7)
    test_passed7 = len(output_substring7) == expected_len7 and output_substring7 in possible_substrings7
    print(f"Test Case 7: {'Passed' if test_passed7 else 'Failed'}")
    print(f"Input: '{s7}'")
    print(f"Output Substring: '{output_substring7}' (Length: {len(output_substring7)})")
    print(f"Expected Length: {expected_len7}")
    print(f"Possible Substrings: {possible_substrings7}")
    print("-" * 20)

    s = "dvdf"
    print(solver.lengthOfLongestSubstring(s))

if __name__ == '__main__':
    main()
