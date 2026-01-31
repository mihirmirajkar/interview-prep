# Answer a Query
# Imagine a length-N array of booleans, initially all false. Over time, some values are set to true, and at various points in time you would like to find the location of the nearest true to the right of given indices.
# You will receive Q queries, each of which has a type and a value. SET queries have type = 1 and GET queries have type = 2.
# When you receive a SET query, the value of the query denotes an index in the array that is set to true. Note that these indices start at 1. When you receive a GET query, you must return the smallest index that contains a true value that is greater than or equal to the given index, or -1 if no such index exists.
# Signature
# int[] answerQueries(ArrayList<Query> queries, int N)
# Input
# A list of Q queries, formatted as [type, index] where type is either 1 or 2, and index is <= N
# 1 <= N <= 1,000,000,000
# 1 <= Q <= 500,000
# Output
# Return an array containing the results of all GET queries. The result of queries[i] is the smallest index that contains a true value that is greater than or equal to i, or -1 if no index satisfies those conditions.

"""
Answer a Query - Find nearest true value to the right in a boolean array.

Problem: Handle SET and GET queries on a virtual boolean array.
- SET (type=1): Mark an index as true
- GET (type=2): Find smallest true index >= given index
"""


def binary_search_left(arr: list[int], target: int) -> int:
    """
    Find the leftmost position where target can be inserted to maintain sorted order.

    Args:
        arr: Sorted list of integers
        target: Value to search for

    Returns:
        Index of the leftmost position where target can be inserted

    Time Complexity: O(log n)
    """
    left, right = 0, len(arr)

    while left < right:
        mid = (left + right) // 2
        if arr[mid] < target:
            left = mid + 1
        else:
            right = mid

    return left


def insert_sorted(arr: list[int], value: int, seen: set[int]) -> None:
    """
    Insert value into sorted array if not already present.

    Args:
        arr: Sorted list of integers
        value: Value to insert
        seen: Set tracking already inserted values

    Time Complexity: O(n) due to list insertion
    """
    if value in seen:
        return

    seen.add(value)
    pos = binary_search_left(arr, value)
    arr.insert(pos, value)


def answer_queries(queries: list[list[int]], n: int) -> list[int]:
    """
    Process SET and GET queries on a virtual boolean array.

    Args:
        queries: List of [type, index] where type 1 = SET, type 2 = GET
        n: Size of the virtual boolean array (1-indexed)

    Returns:
        List of results for all GET queries

    Time Complexity: O(Q^2) worst case for insertions, O(log Q) for lookups
    Space Complexity: O(Q) for storing set indices

    Examples:
        >>> answer_queries([[1, 5], [1, 3], [2, 4], [2, 1]], 10)
        [5, 3]
        >>> answer_queries([[2, 1], [1, 5], [2, 5], [2, 6]], 10)
        [-1, 5, -1]
    """
    true_indices: list[int] = []
    seen: set[int] = set()
    results: list[int] = []

    for query_type, index in queries:
        if query_type == 1:  # SET query
            insert_sorted(true_indices, index, seen)
        else:  # GET query (type == 2)
            pos = binary_search_left(true_indices, index)
            if pos < len(true_indices):
                results.append(true_indices[pos])
            else:
                results.append(-1)

    return results


if __name__ == "__main__":
    # Test case 1
    test_queries = [[1, 5], [1, 3], [2, 4], [2, 1]]
    result = answer_queries(test_queries, 10)
    print(f"Test 1: {result}")  # Expected: [5, 3]

    # Test case 2
    test_queries = [[2, 1], [1, 5], [2, 5], [2, 6]]
    result = answer_queries(test_queries, 10)
    print(f"Test 2: {result}")  # Expected: [-1, 5, -1]