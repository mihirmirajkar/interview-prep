"""
Optimized Dynamic Programming Solution with Memoization and Pruning

This shows production-ready optimizations for the DP approach.
"""

from typing import List, Tuple, Dict
from functools import lru_cache

class OptimizedBatchProcessorDP:
    def __init__(self):
        self.memo = {}
    
    def min_processing_time_optimized(self, data_points: List[Tuple[int, int]], 
                                    max_memory: int, batch_overhead: int) -> int:
        """
        Optimized DP with several performance improvements:
        1. Memoization to avoid recomputation
        2. Early termination when memory exceeded
        3. Precomputed batch costs
        4. Better loop ordering
        """
        if not data_points:
            return 0
        
        n = len(data_points)
        
        # Precompute all valid batch costs - O(n²) preprocessing
        batch_costs = {}
        for i in range(n):
            current_memory = 0
            max_cost = 0
            for j in range(i, n):
                cost, memory = data_points[j]
                current_memory += memory
                
                # Early termination when memory constraint violated
                if current_memory > max_memory:
                    break
                
                max_cost = max(max_cost, cost)
                batch_costs[(i, j)] = max_cost + batch_overhead
        
        # Standard DP with memoization
        @lru_cache(maxsize=None)
        def dp(start_idx: int) -> int:
            if start_idx >= n:
                return 0
            
            min_time = float('inf')
            
            # Try all valid batch endings from start_idx
            for end_idx in range(start_idx, n):
                if (start_idx, end_idx) not in batch_costs:
                    break  # Memory constraint violated
                
                batch_time = batch_costs[(start_idx, end_idx)]
                remaining_time = dp(end_idx + 1)
                total_time = batch_time + remaining_time
                min_time = min(min_time, total_time)
            
            return min_time
        
        return dp(0)
    
    def min_processing_time_bottom_up_optimized(self, data_points: List[Tuple[int, int]], 
                                              max_memory: int, batch_overhead: int) -> int:
        """
        Bottom-up DP with space and time optimizations.
        """
        if not data_points:
            return 0
        
        n = len(data_points)
        dp = [float('inf')] * (n + 1)
        dp[n] = 0
        
        # Process from right to left
        for i in range(n - 1, -1, -1):
            current_memory = 0
            max_cost = 0
            
            # Try all valid batch endings starting from i
            for j in range(i, n):
                cost, memory = data_points[j]
                current_memory += memory
                
                # Early termination
                if current_memory > max_memory:
                    break
                
                max_cost = max(max_cost, cost)
                batch_time = max_cost + batch_overhead
                
                # Update DP value
                if dp[j + 1] != float('inf'):  # Valid subproblem
                    dp[i] = min(dp[i], batch_time + dp[j + 1])
        
        return dp[0] if dp[0] != float('inf') else 0


def benchmark_approaches():
    """Compare performance of different approaches"""
    import time
    
    # Generate test data
    test_sizes = [10, 20, 50]
    
    for n in test_sizes:
        print(f"\n📊 BENCHMARKING WITH {n} ITEMS")
        print("-" * 40)
        
        # Generate diverse test data
        data_points = [(i % 10 + 1, i % 3 + 1) for i in range(n)]
        max_memory = 5
        batch_overhead = 2
        
        # Test different approaches
        approaches = [
            ("Greedy O(n²)", lambda: greedy_solution(data_points, max_memory, batch_overhead)),
            ("DP Standard O(n³)", lambda: standard_dp(data_points, max_memory, batch_overhead)),
            ("DP Optimized", lambda: optimized_dp(data_points, max_memory, batch_overhead)),
        ]
        
        results = []
        for name, func in approaches:
            start_time = time.time()
            result = func()
            end_time = time.time()
            
            duration = (end_time - start_time) * 1000  # Convert to ms
            results.append((name, result, duration))
            print(f"{name:20}: {result:6} time units ({duration:.2f}ms)")
        
        # Check if all approaches give same result (for correctness)
        if len(set(r[1] for r in results)) > 1:
            print("⚠️  Different results found - need to investigate!")

def greedy_solution(data_points, max_memory, batch_overhead):
    """Greedy baseline for comparison"""
    from batch_processing_optimization import BatchProcessor
    processor = BatchProcessor()
    return processor.min_processing_time(data_points, max_memory, batch_overhead)

def standard_dp(data_points, max_memory, batch_overhead):
    """Standard DP for comparison"""
    from batch_processing_dp_solution import BatchProcessorDP
    processor = BatchProcessorDP()
    return processor.min_processing_time_dp(data_points, max_memory, batch_overhead)

def optimized_dp(data_points, max_memory, batch_overhead):
    """Optimized DP"""
    processor = OptimizedBatchProcessorDP()
    return processor.min_processing_time_bottom_up_optimized(data_points, max_memory, batch_overhead)


if __name__ == "__main__":
    print("🚀 OPTIMIZED DYNAMIC PROGRAMMING APPROACHES")
    print("=" * 60)
    
    print("""
🎯 OPTIMIZATION TECHNIQUES:

1. PRECOMPUTATION:
   - Calculate all valid batch costs upfront O(n²)
   - Avoid recomputing max costs in inner loops

2. MEMOIZATION:
   - Use @lru_cache for top-down DP
   - Avoid redundant subproblem calculations

3. EARLY TERMINATION:
   - Stop when memory constraint violated
   - No need to check larger batches

4. SPACE OPTIMIZATION:
   - Bottom-up DP uses O(n) space
   - Can further optimize to O(1) if only final result needed

5. ALGORITHMIC IMPROVEMENTS:
   - Better loop ordering
   - Efficient data structures
   """)
    
    benchmark_approaches()
    
    print(f"\n💡 PRACTICAL RECOMMENDATIONS:")
    print(f"• For n < 100: Use DP for guaranteed optimality")
    print(f"• For n > 100: Use greedy for speed, DP for critical paths")
    print(f"• In ML pipelines: Cache DP results for repeated patterns")
    print(f"• For real-time: Use greedy with DP validation offline")
