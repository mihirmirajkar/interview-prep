"""
Sorting Algorithms Implementation
"""

from typing import List, TypeVar, Callable

T = TypeVar('T')


def bubble_sort(arr: List[T]) -> List[T]:
    """
    Bubble sort implementation.
    
    Time Complexity: O(n²)
    Space Complexity: O(1)
    """
    arr_copy = arr.copy()
    n = len(arr_copy)
    
    for i in range(n):
        swapped = False
        for j in range(0, n - i - 1):
            if arr_copy[j] > arr_copy[j + 1]:
                arr_copy[j], arr_copy[j + 1] = arr_copy[j + 1], arr_copy[j]
                swapped = True
        if not swapped:
            break
    
    return arr_copy


def quick_sort(arr: List[T]) -> List[T]:
    """
    Quick sort implementation.
    
    Time Complexity: O(n log n) average, O(n²) worst
    Space Complexity: O(log n)
    """
    if len(arr) <= 1:
        return arr
    
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    
    return quick_sort(left) + middle + quick_sort(right)


def merge_sort(arr: List[T]) -> List[T]:
    """
    Merge sort implementation.
    
    Time Complexity: O(n log n)
    Space Complexity: O(n)
    """
    if len(arr) <= 1:
        return arr
    
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    
    return merge(left, right)


def merge(left: List[T], right: List[T]) -> List[T]:
    """Helper function for merge sort."""
    result = []
    i = j = 0
    
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    
    result.extend(left[i:])
    result.extend(right[j:])
    return result


def heap_sort(arr: List[T]) -> List[T]:
    """
    Heap sort implementation.
    
    Time Complexity: O(n log n)
    Space Complexity: O(1)
    """
    arr_copy = arr.copy()
    n = len(arr_copy)
    
    # Build max heap
    for i in range(n // 2 - 1, -1, -1):
        heapify(arr_copy, n, i)
    
    # Extract elements from heap one by one
    for i in range(n - 1, 0, -1):
        arr_copy[0], arr_copy[i] = arr_copy[i], arr_copy[0]
        heapify(arr_copy, i, 0)
    
    return arr_copy


def heapify(arr: List[T], n: int, i: int) -> None:
    """Helper function to maintain heap property."""
    largest = i
    left = 2 * i + 1
    right = 2 * i + 2
    
    if left < n and arr[left] > arr[largest]:
        largest = left
    
    if right < n and arr[right] > arr[largest]:
        largest = right
    
    if largest != i:
        arr[i], arr[largest] = arr[largest], arr[i]
        heapify(arr, n, largest)
