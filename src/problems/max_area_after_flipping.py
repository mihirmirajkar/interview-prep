"""
You are given an n x n binary matrix grid. You are allowed to change at most one 0 to be 1.

Return the size of the largest island in grid after applying this operation.

An island is a 4-directionally connected group of 1s.

 

Example 1:

Input: grid = [[1,0],[0,1]]
Output: 3
Explanation: Change one 0 to 1 and connect two 1s, then we get an island with area = 3.
Example 2:

Input: grid = [[1,1],[1,0]]
Output: 4
Explanation: Change the 0 to 1 and make the island bigger, only one island with area = 4.
Example 3:

Input: grid = [[1,1],[1,1]]
Output: 4
Explanation: Can't change any 0 to 1, only one island with area = 4.
 

Constraints:

n == grid.length
n == grid[i].length
1 <= n <= 500
grid[i][j] is either 0 or 1.
"""

from typing import List


class Solution:
    def largestIsland(self, grid: List[List[int]]) -> int:
        w, h = len(grid[0]), len(grid)
        
        def find_area(i, j, visited):
            if i < 0 or i >= h or j < 0 or j >= w or visited[i][j] or grid[i][j] == 0:
                return 0
            
            visited[i][j] = True
            area = 1
            area += find_area(i+1, j, visited)
            area += find_area(i-1, j, visited)
            area += find_area(i, j+1, visited)
            area += find_area(i, j-1, visited)
            return area

        max_area = 0
        
        # Try flipping each 0
        for i in range(h):
            for j in range(w):
                if grid[i][j] == 0:
                    grid[i][j] = 1
                    visited = [[False for _ in range(w)] for _ in range(h)]
                    temp_area = find_area(i, j, visited)
                    max_area = max(max_area, temp_area)
                    grid[i][j] = 0
        
        # If no 0s found or max_area is still 0, count largest existing island
        if max_area == 0:
            visited = [[False for _ in range(w)] for _ in range(h)]
            for i in range(h):
                for j in range(w):
                    if grid[i][j] == 1 and not visited[i][j]:
                        temp_area = find_area(i, j, visited)
                        max_area = max(max_area, temp_area)

        return max_area

if __name__ == "__main__":
    solution = Solution()
    # Test cases
    tests = [
        ([[1,0],[0,1]], 3),
        ([[1,1],[1,0]], 4),
        ([[1,1],[1,1]], 4),
        ([[0,0],[0,0]], 1),
        ([[1,0,1],[0,1,0],[1,0,1]], 5),
    ]

    for idx, (grid, expected) in enumerate(tests, start=1):
        result = solution.largestIsland(grid)
        print(f"Test case {idx}: grid={grid} | result={result} | expected={expected}")
        assert result == expected, f"Test case {idx} failed: expected {expected}, got {result}"

    print("All tests passed!")
        