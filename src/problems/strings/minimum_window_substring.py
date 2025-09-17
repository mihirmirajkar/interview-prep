def min_window_gpt(s: str, t: str) -> str:
    """
    Optimal O(N) solution using sliding window and two hash maps.
    - Count all characters needed from t in a dict (need).
    - Use a window [left, right) and a dict (window_counts) to track chars in the current window.
    - Move right to expand the window until all required chars are included (formed == required).
    - Then move left to shrink the window as much as possible while still containing all required chars.
    - Track the minimum window found.
    """
    if not s or not t:
        return ""
    from collections import Counter, defaultdict
    need = Counter(t)
    window_counts = defaultdict(int)
    required = len(need)
    formed = 0
    l = 0
    ans = float('inf'), None, None  # window length, left, right
    for r, char in enumerate(s):
        window_counts[char] += 1
        if char in need and window_counts[char] == need[char]:
            formed += 1
        while l <= r and formed == required:
            if r - l + 1 < ans[0]:
                ans = (r - l + 1, l, r)
            window_counts[s[l]] -= 1
            if s[l] in need and window_counts[s[l]] < need[s[l]]:
                formed -= 1
            l += 1
    return "" if ans[0] == float('inf') else s[ans[1]:ans[2]+1]

"""
Problem: Minimum Window Substring (Strings / Sliding Window)
Difficulty: Hard

Given two strings s and t, return the minimum window in s which will contain all the characters in t (including duplicates). If there is no such window, return the empty string "".

Note:
- If there is such a window, it is guaranteed that there will always be only one unique minimum window in s.

Example 1:
Input: s = "ADOBECODEBANC", t = "ABC"
Output: "BANC"

Example 2:
Input: s = "a", t = "a"
Output: "a"

Example 3:
Input: s = "a", t = "aa"
Output: ""

Write your solution in the function `min_window(s: str, t: str) -> str` below.
Leave the implementation blank for the candidate to fill in.
Include several test cases in main.
"""
from collections import defaultdict
from typing import List

def contains(s_dict, t_dict):
    for t_key in t_dict:
        if t_key not in s_dict or s_dict[t_key] != t_dict[t_key]:
            return False
    return True

def min_window(s: str, t: str) -> str:
    """Returns the minimum window substring of s containing all characters of t."""
    if not s or not t:
        return ""
    start = 0
    stop = 1
    t_dict = create_dict(t)
    
    shortest = ""
           
    s_dict = create_dict(s[start:stop])
    while stop < len(s) + 1 and start < len(s) and start < stop:
        print(s[start:stop], s_dict)
        while not contains(s_dict, t_dict) and stop < len(s):
            s_dict[s[stop]] += 1
            stop += 1
        if not shortest or len(shortest) > len(s[start:stop]) and contains(s_dict, t_dict):
            shortest = s[start:stop]
        
        s_dict[s[start]]  = max(s_dict[s[start]]-1, 0)
        start += 1
        
        while start < stop and s[start] not in t_dict:
            s_dict[s[start]] = max(s_dict[s[start]]-1, 0)
            start += 1
            
    print(shortest)
    return shortest

def create_dict(t):
    t_dict = defaultdict(int)
    for t_char in t:
        t_dict[t_char] += 1

    return t_dict


def run_tests():
    print("Test 1:", min_window("ADOBECODEBANC", "ABC") == "BANC")
    print("Test 2:", min_window("a", "a") == "a")
    print("Test 3:", min_window("a", "aa") == "")
    print("Test 4:", min_window("ab", "b") == "b")
    print("Test 5:", min_window("bba", "ab") == "ba")
    print("Test 6:", min_window("aa", "aa") == "aa")
    print("Test 7:", min_window("abcdebdde", "bde") == "deb")

if __name__ == "__main__":
    run_tests()
