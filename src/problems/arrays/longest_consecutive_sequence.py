"""
Problem: Longest Consecutive Sequence (Arrays / Hashing)
Difficulty: Medium/Hard

Given an unsorted array of integers nums, return the length of the longest consecutive elements sequence.
You must write an algorithm that runs in O(n) time.

Example 1:
Input: nums = [100,4,200,1,3,2]
Output: 4
Explanation: The longest consecutive sequence is [1, 2, 3, 4]. Therefore its length is 4.

Example 2:
Input: nums = [0,3,7,2,5,8,4,6,0,1]
Output: 9

Write your solution in the function `longest_consecutive(nums: list[int]) -> int` below.
Leave the implementation blank for the candidate to fill in.
Include several test cases in main.
"""
from typing import List

def longest_consecutive(nums: List[int]) -> int:
    """Returns the length of the longest consecutive sequence in nums."""
    tracker = set(nums)
    longest = 0
    for num in nums:
        temp_len = 0
        if num-1 not in tracker:
            temp_len = 1
            next_num = num + 1
            while next_num in tracker:
                
                temp_len += 1
                next_num += 1
        if temp_len > longest:
            longest = temp_len 
    return longest



def run_tests():
    print("Test 1:", longest_consecutive([100,4,200,1,3,2]) == 4)
    print("Test 2:", longest_consecutive([0,3,7,2,5,8,4,6,0,1]) == 9)
    print("Test 3:", longest_consecutive([]) == 0)
    print("Test 4:", longest_consecutive([1,2,0,1]) == 3)
    print("Test 5:", longest_consecutive([10,5,12,3,55,30,4,11,2]) == 4)
    print("Test 6:", longest_consecutive([1]) == 1)
    print("Test 7:", longest_consecutive([1,2,3,4,5,6,7,8,9,10]) == 10)

if __name__ == "__main__":
    run_tests()
