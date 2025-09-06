"""
L6/L7 Applied AI/ML Interview Problem: Batch Processing Optimization

Problem Statement:
You're designing a batch processing system for ML inference. Given a stream of data points,
each with a processing cost and memory requirement, you need to group them into batches
that maximize processing efficiency while respecting memory constraints.

Each data point is represented as (processing_cost, memory_requirement).
- processing_cost: positive integer (time units to process)
- memory_requirement: positive integer (memory units required)

Constraints:
- Each batch cannot exceed max_memory units
- You want to minimize total processing time
- Batches are processed sequentially, but items within a batch are processed in parallel
- Batch overhead: each batch has a fixed setup cost of 'batch_overhead' time units

Goal: Return the minimum total time to process all data points.

Example:
data_points = [(3, 2), (5, 3), (2, 1), (4, 2), (1, 1)]
max_memory = 4
batch_overhead = 2

Optimal batching might be:
Batch 1: [(3, 2), (2, 1)] -> memory: 3, time: max(3,2) + 2 = 5
Batch 2: [(5, 3)] -> memory: 3, time: 5 + 2 = 7  
Batch 3: [(4, 2), (1, 1)] -> memory: 3, time: max(4,1) + 2 = 6
Total time: 5 + 7 + 6 = 18

Time Complexity Expected: O(n^2) or better
Space Complexity Expected: O(n) or better
"""

from typing import List, Tuple
import unittest


class BatchProcessor:
    def __init__(self):
        pass
    
    def min_processing_time(self, data_points: List[Tuple[int, int]], 
                          max_memory: int, batch_overhead: int) -> int:
        """
        Find the minimum total processing time for all data points.
        
        Args:
            data_points: List of (processing_cost, memory_requirement) tuples
            max_memory: Maximum memory allowed per batch
            batch_overhead: Fixed time cost added to each batch
            
        Returns:
            Minimum total processing time
            
        TODO: Implement this method
        """
        data_points = sorted(data_points, key=lambda x: x[0], reverse=True)
        processed = [False]*len(data_points)

        processing_time = 0

        for i, (cost, memory) in enumerate(data_points):
            if processed[i]:
                continue
            processed[i] = True
            processing_time += batch_overhead + cost
            current_batch_memory = memory

            for j in range(i, len(data_points)):
                if processed[j]:
                    continue
                _, memory2 = data_points[j]                 
                if current_batch_memory + memory2 > max_memory:
                    break
                processed[j] = True
                current_batch_memory += memory2
        return processing_time
    
    def get_optimal_batches(self, data_points: List[Tuple[int, int]], 
                           max_memory: int, batch_overhead: int) -> List[List[Tuple[int, int]]]:
        """
        Return the actual optimal batching strategy.
        
        Args:
            data_points: List of (processing_cost, memory_requirement) tuples
            max_memory: Maximum memory allowed per batch
            batch_overhead: Fixed time cost added to each batch
            
        Returns:
            List of batches, where each batch is a list of data points
            
        TODO: Implement this method (bonus points)
        """
        # YOUR IMPLEMENTATION HERE  
        data_points = sorted(data_points, key=lambda x: x[0], reverse=True)
        processed = [False]*len(data_points)

        batches = []
        for i, (cost, memory) in enumerate(data_points):
            if processed[i]:
                continue
            processed[i] = True
            # processing_time += batch_overhead + cost
            current_batch = [(cost, memory)]
            current_batch_memory = memory

            for j in range(i, len(data_points)):
                if processed[j]:
                    continue
                cost2, memory2 = data_points[j]                 
                if current_batch_memory + memory2 > max_memory:
                    break
                processed[j] = True
                current_batch.append(data_points[j])
                current_batch_memory += memory2
            batches.append(current_batch)
        return batches



class TestBatchProcessor(unittest.TestCase):
    def setUp(self):
        self.processor = BatchProcessor()
    
    def test_single_item(self):
        """Test with single data point"""
        data_points = [(5, 3)]
        result = self.processor.min_processing_time(data_points, 5, 2)
        expected = 7  # 5 + 2 (overhead)
        self.assertEqual(result, expected)
    
    def test_all_fit_in_one_batch(self):
        """Test when all items fit in one batch"""
        data_points = [(3, 1), (2, 1), (4, 1)]
        result = self.processor.min_processing_time(data_points, 5, 2)
        expected = 6  # max(3,2,4) + 2 = 6
        self.assertEqual(result, expected)
    
    def test_example_case(self):
        """Test the example from problem description"""
        data_points = [(3, 2), (5, 3), (2, 1), (4, 2), (1, 1)]
        result = self.processor.min_processing_time(data_points, 4, 2)
        # One possible optimal solution: [[(3,2), (2,1)], [(5,3)], [(4,2), (1,1)]]
        # Times: [5, 7, 6] = 18 total
        self.assertLessEqual(result, 18)  # Should be optimal or better
    
    def test_memory_constraints(self):
        """Test strict memory constraints forcing single items per batch"""
        data_points = [(1, 3), (2, 3), (3, 3)]
        result = self.processor.min_processing_time(data_points, 3, 1)
        expected = 9  # (1+1) + (2+1) + (3+1) = 9
        self.assertEqual(result, expected)
    
    def test_large_overhead(self):
        """Test case where batch overhead dominates"""
        data_points = [(1, 1), (1, 1), (1, 1), (1, 1)]
        result = self.processor.min_processing_time(data_points, 10, 100)
        expected = 101  # All in one batch: max(1,1,1,1) + 100 = 101
        self.assertEqual(result, expected)
    
    def test_zero_overhead(self):
        """Test with no batch overhead"""
        data_points = [(4, 2), (3, 2), (2, 1)]
        result = self.processor.min_processing_time(data_points, 3, 0)
        # Optimal: [(4,2)], [(3,2), (2,1)] -> times: [4, 3] = 7
        self.assertLessEqual(result, 7)
    
    def test_empty_input(self):
        """Test edge case with empty input"""
        data_points = []
        result = self.processor.min_processing_time(data_points, 5, 2)
        expected = 0
        self.assertEqual(result, expected)


if __name__ == "__main__":
    print("Batch Processing Optimization - L6/L7 Interview Problem")
    print("=" * 60)
    print()
    
    # Interactive test
    processor = BatchProcessor()
    
    # Test case 1
    print("Test Case 1:")
    data_points = [(3, 2), (5, 3), (2, 1), (4, 2), (1, 1)]
    max_memory = 4
    batch_overhead = 2
    print(f"Data points: {data_points}")
    print(f"Max memory: {max_memory}, Batch overhead: {batch_overhead}")
    
    try:
        result = processor.min_processing_time(data_points, max_memory, batch_overhead)
        print(f"Your result: {result}")
        
        # Try to get batches if implemented
        try:
            batches = processor.get_optimal_batches(data_points, max_memory, batch_overhead)
            print(f"Optimal batches: {batches}")
        except:
            print("get_optimal_batches not implemented yet")
            
    except Exception as e:
        print(f"Implementation needed: {e}")
    
    print("\n" + "=" * 60)
    print("Running unit tests...")
    unittest.main(verbosity=2)
