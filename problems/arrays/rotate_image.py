"""
You are given an n x n 2D matrix representing an image. Rotate the image by 90 degrees (clockwise).

You have to rotate the image in-place, which means you have to modify the input 2D matrix directly. DO NOT allocate another 2D matrix and do the rotation.

Example 1:
Input: matrix = [[1,2,3],[4,5,6],[7,8,9]]
Output: [[7,4,1],[8,5,2],[9,6,3]]

Example 2:
Input: matrix = [[5,1,9,11],[2,4,8,10],[13,3,6,7],[15,14,12,16]]
Output: [[15,13,2,5],[14,3,4,1],[12,6,8,9],[16,7,10,11]]
"""
from typing import List

class Solution:
    def rotate(self, matrix: List[List[int]]) -> None:
        """
        Do not return anything, modify matrix in-place instead.
        """
        # Your solution here
        # without numpy
        # Transpose matrix
        for i in range(len(matrix)):
            for j in range(i, len(matrix)):
                matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]
   
        for row in matrix:
            row.reverse()
        print(matrix)

if __name__ == '__main__':
    # Test cases
    solver = Solution()

    # Test Case 1
    matrix1 = [[1,2,3],[4,5,6],[7,8,9]]
    solver.rotate(matrix1)
    expected1 = [[7,4,1],[8,5,2],[9,6,3]]
    print(f"Test Case 1: {'Passed' if matrix1 == expected1 else 'Failed'}")
    print(f"Output: {matrix1}")
    print(f"Expected: {expected1}")
    print("-" * 20)

    # Test Case 2
    matrix2 = [[5,1,9,11],[2,4,8,10],[13,3,6,7],[15,14,12,16]]
    solver.rotate(matrix2)
    expected2 = [[15,13,2,5],[14,3,4,1],[12,6,8,9],[16,7,10,11]]
    print(f"Test Case 2: {'Passed' if matrix2 == expected2 else 'Failed'}")
    print(f"Output: {matrix2}")
    print(f"Expected: {expected2}")
    print("-" * 20)

    # Test Case 3: 2x2 matrix
    matrix3 = [[1,2],[3,4]]
    solver.rotate(matrix3)
    expected3 = [[3,1],[4,2]]
    print(f"Test Case 3: {'Passed' if matrix3 == expected3 else 'Failed'}")
    print(f"Output: {matrix3}")
    print(f"Expected: {expected3}")
    print("-" * 20)

    # Test Case 4: 1x1 matrix
    matrix4 = [[1]]
    solver.rotate(matrix4)
    expected4 = [[1]]
    print(f"Test Case 4: {'Passed' if matrix4 == expected4 else 'Failed'}")
    print(f"Output: {matrix4}")
    print(f"Expected: {expected4}")
    print("-" * 20)

    # Test Case 5: 5x5 matrix
    matrix5 = [
        [1, 2, 3, 4, 5],
        [6, 7, 8, 9, 10],
        [11, 12, 13, 14, 15],
        [16, 17, 18, 19, 20],
        [21, 22, 23, 24, 25]
    ]
    solver.rotate(matrix5)
    expected5 = [
        [21, 16, 11, 6, 1],
        [22, 17, 12, 7, 2],
        [23, 18, 13, 8, 3],
        [24, 19, 14, 9, 4],
        [25, 20, 15, 10, 5]
    ]
    print(f"Test Case 5: {'Passed' if matrix5 == expected5 else 'Failed'}")
    print(f"Output: {matrix5}")
    print(f"Expected: {expected5}")
    print("-" * 20)
