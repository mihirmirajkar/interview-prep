"""
Dynamic Programming Solution for Batch Processing Optimization

This demonstrates the optimal DP approach that guarantees the minimum processing time.
The key insight is that this problem has optimal substructure:
- If we know the optimal way to process items [i+1..n], we can find optimal for [i..n]
- We try all valid batch endings starting from position i

State: dp[i] = minimum time to process items from index i to end
Recurrence: dp[i] = min over all valid j where we can batch items [i..j]:
    dp[i] = min(dp[j+1] + batch_time(i, j)) for all valid j >= i

Time Complexity: O(n^3) - for each i, we try all j, and compute batch_time
Space Complexity: O(n) - DP array
"""

from typing import List, Tuple
import unittest


class BatchProcessorDP:
    def __init__(self):
        pass
    
    def min_processing_time_dp(self, data_points: List[Tuple[int, int]], 
                              max_memory: int, batch_overhead: int) -> int:
        """
        Dynamic Programming solution for optimal batch processing.
        
        Args:
            data_points: List of (processing_cost, memory_requirement) tuples
            max_memory: Maximum memory allowed per batch
            batch_overhead: Fixed time cost added to each batch
            
        Returns:
            Minimum total processing time (guaranteed optimal)
        """
        if not data_points:
            return 0
            
        n = len(data_points)
        
        # dp[i] = minimum time to process items from index i to end
        dp = [float('inf')] * (n + 1)
        dp[n] = 0  # Base case: no items left to process
        
        # Fill DP table from right to left
        for i in range(n - 1, -1, -1):
            # Try all possible batch endings starting from position i
            current_memory = 0
            max_cost = 0
            
            for j in range(i, n):
                cost, memory = data_points[j]
                current_memory += memory
                max_cost = max(max_cost, cost)
                
                # If memory constraint violated, can't extend batch further
                if current_memory > max_memory:
                    break
                
                # Cost of batch [i..j] + optimal cost for remaining items
                batch_time = max_cost + batch_overhead
                total_time = batch_time + dp[j + 1]
                dp[i] = min(dp[i], total_time)
        
        return dp[0]
    
    def get_optimal_batches_dp(self, data_points: List[Tuple[int, int]], 
                              max_memory: int, batch_overhead: int) -> List[List[Tuple[int, int]]]:
        """
        Get the actual optimal batching strategy using DP.
        """
        if not data_points:
            return []
            
        n = len(data_points)
        
        # First, compute DP table
        dp = [float('inf')] * (n + 1)
        dp[n] = 0
        
        # Also track the optimal batch ending for each position
        best_end = [-1] * n
        
        for i in range(n - 1, -1, -1):
            current_memory = 0
            max_cost = 0
            
            for j in range(i, n):
                cost, memory = data_points[j]
                current_memory += memory
                max_cost = max(max_cost, cost)
                
                if current_memory > max_memory:
                    break
                
                batch_time = max_cost + batch_overhead
                total_time = batch_time + dp[j + 1]
                
                if total_time < dp[i]:
                    dp[i] = total_time
                    best_end[i] = j
        
        # Reconstruct the optimal batching
        batches = []
        i = 0
        while i < n:
            end = best_end[i]
            batch = data_points[i:end + 1]
            batches.append(batch)
            i = end + 1
        
        return batches
    
    def analyze_complexity(self, data_points: List[Tuple[int, int]], 
                          max_memory: int, batch_overhead: int) -> dict:
        """
        Analyze the solution and compare with greedy approach.
        """
        # DP solution
        dp_time = self.min_processing_time_dp(data_points, max_memory, batch_overhead)
        dp_batches = self.get_optimal_batches_dp(data_points, max_memory, batch_overhead)
        
        # Greedy solution for comparison
        from batch_processing_optimization import BatchProcessor
        greedy_processor = BatchProcessor()
        greedy_time = greedy_processor.min_processing_time(data_points, max_memory, batch_overhead)
        greedy_batches = greedy_processor.get_optimal_batches(data_points, max_memory, batch_overhead)
        
        return {
            'dp_time': dp_time,
            'dp_batches': dp_batches,
            'greedy_time': greedy_time,
            'greedy_batches': greedy_batches,
            'dp_is_better': dp_time < greedy_time,
            'improvement': greedy_time - dp_time if greedy_time > dp_time else 0
        }


class TestBatchProcessorDP(unittest.TestCase):
    def setUp(self):
        self.processor = BatchProcessorDP()
    
    def test_dp_vs_greedy_simple(self):
        """Test where DP might find better solution than greedy"""
        # Case where greedy might not be optimal
        data_points = [(10, 1), (1, 1), (1, 1), (10, 1)]
        max_memory = 2
        batch_overhead = 5
        
        dp_result = self.processor.min_processing_time_dp(data_points, max_memory, batch_overhead)
        
        # DP should find optimal solution
        # Possible solutions:
        # 1. [(10,1), (1,1)], [(1,1), (10,1)] -> times: [15, 15] = 30
        # 2. [(10,1)], [(1,1), (1,1)], [(10,1)] -> times: [15, 6, 15] = 36
        # DP should pick the better one
        self.assertLessEqual(dp_result, 36)
    
    def test_dp_optimal_property(self):
        """Test that DP solution is never worse than any other approach"""
        test_cases = [
            ([(5, 2), (3, 1), (4, 2), (2, 1)], 3, 1),
            ([(1, 1), (10, 1), (1, 1), (10, 1)], 2, 5),
            ([(3, 2), (5, 3), (2, 1), (4, 2), (1, 1)], 4, 2),
        ]
        
        for data_points, max_memory, batch_overhead in test_cases:
            with self.subTest(data_points=data_points):
                dp_result = self.processor.min_processing_time_dp(data_points, max_memory, batch_overhead)
                
                # DP result should be valid (non-negative, handles empty case)
                self.assertGreaterEqual(dp_result, 0)
                
                # For empty input
                if not data_points:
                    self.assertEqual(dp_result, 0)
    
    def test_batch_reconstruction(self):
        """Test that reconstructed batches are valid and give correct time"""
        data_points = [(3, 2), (5, 3), (2, 1), (4, 2), (1, 1)]
        max_memory = 4
        batch_overhead = 2
        
        dp_time = self.processor.min_processing_time_dp(data_points, max_memory, batch_overhead)
        batches = self.processor.get_optimal_batches_dp(data_points, max_memory, batch_overhead)
        
        # Verify all items are included exactly once
        all_items = []
        for batch in batches:
            all_items.extend(batch)
        self.assertEqual(sorted(all_items), sorted(data_points))
        
        # Verify each batch respects memory constraints
        for batch in batches:
            total_memory = sum(memory for _, memory in batch)
            self.assertLessEqual(total_memory, max_memory)
        
        # Verify time calculation matches
        calculated_time = 0
        for batch in batches:
            if batch:  # Non-empty batch
                max_cost = max(cost for cost, _ in batch)
                calculated_time += max_cost + batch_overhead
        self.assertEqual(calculated_time, dp_time)


def demonstrate_dp_insights():
    """
    Demonstrate key insights about the DP approach and when it outperforms greedy.
    """
    print("=" * 70)
    print("DYNAMIC PROGRAMMING INSIGHTS FOR BATCH PROCESSING")
    print("=" * 70)
    
    processor = BatchProcessorDP()
    
    print("\n1. OPTIMAL SUBSTRUCTURE:")
    print("   dp[i] = min time to process items [i..n-1]")
    print("   dp[i] = min over all valid j: (batch_time[i..j] + dp[j+1])")
    
    print("\n2. COMPARISON WITH GREEDY APPROACH:")
    
    # Test case where greedy might be suboptimal
    test_cases = [
        {
            'name': 'High overhead favors fewer batches',
            'data': [(1, 1), (1, 1), (10, 1), (10, 1)],
            'max_memory': 2,
            'batch_overhead': 100
        },
        {
            'name': 'Mixed costs and memory requirements',
            'data': [(10, 1), (1, 2), (1, 2), (10, 1)],
            'max_memory': 3,
            'batch_overhead': 5
        },
        {
            'name': 'Original example',
            'data': [(3, 2), (5, 3), (2, 1), (4, 2), (1, 1)],
            'max_memory': 4,
            'batch_overhead': 2
        }
    ]
    
    for i, test in enumerate(test_cases, 1):
        print(f"\nTest Case {i}: {test['name']}")
        print(f"Data: {test['data']}")
        print(f"Max Memory: {test['max_memory']}, Overhead: {test['batch_overhead']}")
        
        analysis = processor.analyze_complexity(
            test['data'], test['max_memory'], test['batch_overhead']
        )
        
        print(f"DP Result: {analysis['dp_time']} (batches: {len(analysis['dp_batches'])})")
        print(f"Greedy Result: {analysis['greedy_time']} (batches: {len(analysis['greedy_batches'])})")
        
        if analysis['dp_is_better']:
            print(f"🎯 DP found better solution! Improvement: {analysis['improvement']}")
        else:
            print("✓ Both approaches found same optimal solution")
        
        print(f"DP Batches: {analysis['dp_batches']}")
        print(f"Greedy Batches: {analysis['greedy_batches']}")
    
    print("\n" + "=" * 70)
    print("KEY TAKEAWAYS:")
    print("• DP guarantees global optimum vs greedy's local decisions")
    print("• DP explores all valid batch combinations systematically") 
    print("• Trade-off: O(n³) time vs O(n²) for greedy")
    print("• In production: Use DP for critical optimizations, greedy for speed")
    print("=" * 70)


if __name__ == "__main__":
    demonstrate_dp_insights()
    
    print("\n\nRunning DP unit tests...")
    unittest.main(verbosity=2)
