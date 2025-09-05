"""
L6/L7 Applied AI/ML Interview Problem: Distributed Model Inference Scheduler

Problem Statement:
You're building a scheduler for distributed ML inference across multiple GPU workers.
Each worker has different capabilities, and requests have varying requirements and priorities.

GPU Workers are characterized by:
- worker_id: unique identifier
- memory_capacity: total GPU memory (GB)
- compute_power: relative compute strength (1.0 = baseline)
- current_load: current memory usage (GB)

Inference Requests are characterized by:
- request_id: unique identifier  
- memory_required: GPU memory needed (GB)
- compute_time: estimated processing time at baseline compute (seconds)
- priority: higher number = higher priority
- max_wait_time: maximum acceptable wait time (seconds)

Constraints:
- Each worker can handle multiple requests concurrently if memory allows
- Processing time scales inversely with compute_power (stronger GPU = faster)
- Requests must be assigned within their max_wait_time or they fail
- Higher priority requests should generally be processed sooner

Goal: Maximize throughput while respecting priorities and constraints.

Example:
workers = [
    {'id': 'gpu1', 'memory': 16, 'compute': 1.0, 'load': 4},
    {'id': 'gpu2', 'memory': 32, 'compute': 2.0, 'load': 0}
]

requests = [
    {'id': 'req1', 'memory': 8, 'time': 10, 'priority': 3, 'max_wait': 30},
    {'id': 'req2', 'memory': 12, 'time': 5, 'priority': 5, 'max_wait': 20},
    {'id': 'req3', 'memory': 6, 'time': 15, 'priority': 1, 'max_wait': 60}
]

Your scheduler should assign requests to workers optimally.

Time Complexity Expected: O(n log n) or better for n requests
Space Complexity Expected: O(n + m) for n requests, m workers
"""

from typing import List, Dict, Optional, Tuple
import heapq
import unittest
from dataclasses import dataclass
from enum import Enum


class RequestStatus(Enum):
    PENDING = "pending"
    SCHEDULED = "scheduled" 
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class Worker:
    id: str
    memory_capacity: float
    compute_power: float
    current_load: float
    
    def available_memory(self) -> float:
        return self.memory_capacity - self.current_load
    
    def can_handle(self, memory_required: float) -> bool:
        return self.available_memory() >= memory_required
    
    def processing_time(self, base_time: float) -> float:
        return base_time / self.compute_power


@dataclass
class Request:
    id: str
    memory_required: float
    compute_time: float
    priority: int
    max_wait_time: float
    arrival_time: float = 0.0
    status: RequestStatus = RequestStatus.PENDING
    assigned_worker: Optional[str] = None
    start_time: Optional[float] = None
    completion_time: Optional[float] = None


class DistributedInferenceScheduler:
    def __init__(self):
        self.workers: Dict[str, Worker] = {}
        self.requests: Dict[str, Request] = {}
        self.current_time = 0.0
        self.completed_requests = []
        self.failed_requests = []
    
    def add_worker(self, worker_id: str, memory_capacity: float, 
                   compute_power: float, current_load: float = 0.0):
        """Add a GPU worker to the cluster"""
        self.workers[worker_id] = Worker(
            id=worker_id,
            memory_capacity=memory_capacity,
            compute_power=compute_power,
            current_load=current_load
        )
    
    def submit_request(self, request_id: str, memory_required: float,
                      compute_time: float, priority: int, max_wait_time: float):
        """Submit an inference request"""
        request = Request(
            id=request_id,
            memory_required=memory_required,
            compute_time=compute_time,
            priority=priority,
            max_wait_time=max_wait_time,
            arrival_time=self.current_time
        )
        self.requests[request_id] = request
    
    def schedule_requests(self) -> Dict[str, List[str]]:
        """
        Main scheduling algorithm - assign requests to workers optimally.
        
        Returns:
            Dictionary mapping worker_id to list of assigned request_ids
            
        TODO: Implement your scheduling strategy here
        """
        # YOUR IMPLEMENTATION HERE
        pass
    
    def get_scheduling_metrics(self) -> Dict[str, float]:
        """
        Calculate performance metrics for the current schedule.
        
        Returns:
            Dictionary with metrics like throughput, average wait time, etc.
            
        TODO: Implement metric calculation (bonus points)
        """
        # YOUR IMPLEMENTATION HERE
        pass
    
    def simulate_execution(self, schedule: Dict[str, List[str]]) -> Dict[str, any]:
        """
        Simulate the execution of the given schedule.
        
        Args:
            schedule: Dictionary mapping worker_id to list of request_ids
            
        Returns:
            Simulation results including completion times, failures, etc.
            
        TODO: Implement simulation (bonus points)
        """
        # YOUR IMPLEMENTATION HERE
        pass


class TestDistributedScheduler(unittest.TestCase):
    def setUp(self):
        self.scheduler = DistributedInferenceScheduler()
        
        # Add test workers
        self.scheduler.add_worker('gpu1', 16.0, 1.0, 4.0)
        self.scheduler.add_worker('gpu2', 32.0, 2.0, 0.0)
        self.scheduler.add_worker('gpu3', 8.0, 0.5, 2.0)
        
        # Add test requests
        self.scheduler.submit_request('req1', 8.0, 10.0, 3, 30.0)
        self.scheduler.submit_request('req2', 12.0, 5.0, 5, 20.0)
        self.scheduler.submit_request('req3', 6.0, 15.0, 1, 60.0)
        self.scheduler.submit_request('req4', 4.0, 8.0, 4, 25.0)
    
    def test_worker_management(self):
        """Test worker addition and capacity checking"""
        worker = self.scheduler.workers['gpu1']
        self.assertEqual(worker.available_memory(), 12.0)
        self.assertTrue(worker.can_handle(10.0))
        self.assertFalse(worker.can_handle(15.0))
    
    def test_request_submission(self):
        """Test request submission and tracking"""
        self.assertEqual(len(self.scheduler.requests), 4)
        req = self.scheduler.requests['req1']
        self.assertEqual(req.priority, 3)
        self.assertEqual(req.status, RequestStatus.PENDING)
    
    def test_basic_scheduling(self):
        """Test that scheduling produces valid assignments"""
        schedule = self.scheduler.schedule_requests()
        
        if schedule:  # Only test if implementation exists
            # Verify all workers exist
            for worker_id in schedule.keys():
                self.assertIn(worker_id, self.scheduler.workers)
            
            # Verify no request is assigned twice
            all_assigned = []
            for requests in schedule.values():
                all_assigned.extend(requests)
            self.assertEqual(len(all_assigned), len(set(all_assigned)))
    
    def test_capacity_constraints(self):
        """Test that memory constraints are respected"""
        schedule = self.scheduler.schedule_requests()
        
        if schedule:  # Only test if implementation exists
            for worker_id, request_ids in schedule.items():
                worker = self.scheduler.workers[worker_id]
                total_memory = worker.current_load
                
                for req_id in request_ids:
                    if req_id in self.scheduler.requests:
                        req = self.scheduler.requests[req_id]
                        total_memory += req.memory_required
                
                self.assertLessEqual(total_memory, worker.memory_capacity)
    
    def test_priority_ordering(self):
        """Test that higher priority requests are generally favored"""
        schedule = self.scheduler.schedule_requests()
        
        if schedule:
            # This is a heuristic test - higher priority requests should
            # generally be assigned to faster/better workers
            # Implementation depends on your strategy
            pass
    
    def test_edge_cases(self):
        """Test edge cases"""
        # Test with no workers
        empty_scheduler = DistributedInferenceScheduler()
        empty_scheduler.submit_request('req1', 8.0, 10.0, 3, 30.0)
        schedule = empty_scheduler.schedule_requests()
        # Should handle gracefully (return empty or None)
        
        # Test with no requests
        worker_only_scheduler = DistributedInferenceScheduler()
        worker_only_scheduler.add_worker('gpu1', 16.0, 1.0)
        schedule = worker_only_scheduler.schedule_requests()
        # Should handle gracefully


if __name__ == "__main__":
    print("Distributed Model Inference Scheduler - L6/L7 Interview Problem")
    print("=" * 70)
    print()
    
    # Interactive demonstration
    scheduler = DistributedInferenceScheduler()
    
    # Set up example scenario
    print("Setting up example scenario:")
    
    # Add workers
    scheduler.add_worker('gpu1', 16.0, 1.0, 4.0)  # Standard GPU, partially loaded
    scheduler.add_worker('gpu2', 32.0, 2.0, 0.0)  # High-end GPU, available
    scheduler.add_worker('gpu3', 8.0, 0.5, 2.0)   # Budget GPU, partially loaded
    
    print("Workers:")
    for worker_id, worker in scheduler.workers.items():
        print(f"  {worker_id}: {worker.available_memory():.1f}GB available, "
              f"{worker.compute_power}x compute power")
    
    # Add requests
    scheduler.submit_request('req1', 8.0, 10.0, 3, 30.0)   # Medium priority
    scheduler.submit_request('req2', 12.0, 5.0, 5, 20.0)   # High priority, urgent
    scheduler.submit_request('req3', 6.0, 15.0, 1, 60.0)   # Low priority
    scheduler.submit_request('req4', 4.0, 8.0, 4, 25.0)    # High priority
    
    print("\\nRequests:")
    for req_id, request in scheduler.requests.items():
        print(f"  {req_id}: {request.memory_required}GB, {request.compute_time}s, "
              f"priority={request.priority}, max_wait={request.max_wait_time}s")
    
    print("\\n" + "=" * 70)
    print("Your task: Implement the schedule_requests() method")
    print("Consider:")
    print("• Memory constraints per worker")
    print("• Priority-based assignment")
    print("• Compute power differences")
    print("• Wait time constraints")
    print("• Overall throughput optimization")
    
    try:
        schedule = scheduler.schedule_requests()
        if schedule:
            print("\\nYour scheduling result:")
            for worker_id, request_ids in schedule.items():
                print(f"  {worker_id}: {request_ids}")
        else:
            print("\\nImplement the schedule_requests() method to see results!")
            
    except Exception as e:
        print(f"\\nImplementation needed: {e}")
    
    print("\\n" + "=" * 70)
    print("Running unit tests...")
    unittest.main(verbosity=2)
