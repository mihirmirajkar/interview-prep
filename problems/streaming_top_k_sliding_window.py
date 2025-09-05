"""Streaming Top-K in Sliding Time Window

Problem (L6/L7 Difficulty / Applied AI-Systems Flavor)
----------------------------------------------------
You are given an unsorted collection of user interaction events from a logging pipeline.
Each event is a (timestamp, key) pair where:
  - timestamp: int (seconds since epoch)
  - key: str (represents a user, model id, feature name, etc.)

Goal:
  Implement a function `top_k_in_window(events, window_size, k)` that returns the list of the
  top-k most frequent keys within the inclusive time window:
        [T_max - window_size + 1, T_max]
  where T_max is the maximum timestamp present in the provided events.

  If multiple keys share the same frequency, break ties by lexicographical order (ascending).
  If there are fewer than k distinct keys, return all of them ordered by the rules above.

Performance Requirements:
  - Target: O(n log k) time where n = number of events (acceptable: O(n + D log k) where D = distinct keys)
  - Aux space: O(D)

Constraints / Notes:
  - events may be empty -> return []
  - window_size >= 1
  - k >= 1
  - Some events may fall outside the computed window and must be ignored.
  - Timestamps are not guaranteed to be sorted.
  - Large-scale hint: For true streaming you would maintain a deque + counts + min-heap.

Edge Cases to Consider:
  - All events have the same key
  - k larger than number of distinct keys
  - Multiple ties on frequency and lexicographic ordering
  - window_size smaller than overall span (filters most events out)
  - window_size so large it includes all events
  - Single event input

Do NOT implement a fully general streaming class; just implement the pure function operating
on the provided in-memory list for this exercise. Assume inputs fit in memory for now.

Your Task:
  Fill in the body of `top_k_in_window` below. Replace the placeholder implementation.

Suggested Approach (not required):
  1. If events empty -> return []
  2. Find T_max
  3. Define window_start = T_max - window_size + 1
  4. Count keys where window_start <= ts <= T_max
  5. Use a heap of size k (count, key) with inverted sign for max or direct min-heap trimming
  6. Extract, sort by (-count, key) to finalize ordering

Complexity Discussion Points (for interview):
  - Tradeoffs between heap vs. partial sorting vs. using `nlargest`
  - Handling extremely skewed distributions (hot keys)
  - Extending to true sliding windows (queue structure, evictions, watermarking)
  - Probabilistic approximations (Count-Min Sketch, HyperLogLog) for very large cardinality

Do NOT import heavy external libraries.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import List, Sequence, Tuple


@dataclass(frozen=True)
class Event:
    """Represents a single (timestamp, key) event.

    Attributes:
        timestamp: Integer seconds since epoch.
        key: Identifier (e.g., user id, feature name, model id).
    """
    timestamp: int
    key: str


def top_k_in_window(events: Sequence[Event], window_size: int, k: int) -> List[str]:
    """Return the top-k most frequent keys in the sliding window ending at max timestamp.

    Ordering Rules:
        1. Higher frequency first
        2. Tie -> lexicographically smaller key first

    Args:
        events: Sequence of Event objects (may be unsorted).
        window_size: Positive integer window length in seconds.
        k: Positive integer number of keys to return.

    Returns:
        List of up to k keys ordered per rules above.

    Raises:
        ValueError: if window_size < 1 or k < 1

    Implementation Notes:
        Replace the NotImplementedError with an efficient solution.
    """
    if window_size < 1:
        raise ValueError("window_size must be >= 1")
    if k < 1:
        raise ValueError("k must be >= 1")
    print(events)
    max_timestamp = max(*[event.timestamp for event in events] + [-1])
    count_temp = []
    for event in events:
        if max_timestamp-window_size + 1 <= event.timestamp <= max_timestamp:
            count_temp.append(event.key)
    
    count_keys = Counter(count_temp)
    most_common = count_keys.most_common(k)
    print(most_common)
    return [k[0] for k in sorted(most_common, key = lambda k: (-k[1], k[0]))]


# -----------------------------------------------------------------------------
# Test Harness (Runs when executed directly)
# -----------------------------------------------------------------------------

def _run_basic_tests() -> None:
    # Small deterministic cases
    events = [
        Event(100, "a"),
        Event(101, "b"),
        Event(102, "a"),
        Event(103, "c"),
        Event(104, "b"),
        Event(105, "b"),
        Event(106, "d"),
    ]
    # Window includes all (window big)
    # Frequencies: b:3, a:2, c:1, d:1 -> tie between c,d resolved: c < d
    expected_all = ["b", "a", "c"]  # top 3
    try:
        result = top_k_in_window(events, window_size=10_000, k=3)
    except NotImplementedError:
        print("SKIPPED: Implement top_k_in_window to enable tests (_run_basic_tests)")
        return
    assert result == expected_all, f"expected {expected_all}, got {result}"

    # Tight window capturing only last 3 events (timestamps 104-106)
    # Events in window: (104,b),(105,b),(106,d) -> b:2, d:1
    expected_window = ["b", "d"]
    result2 = top_k_in_window(events, window_size=3, k=5)
    assert result2 == expected_window, f"expected {expected_window}, got {result2}"

    # Single event case
    single = [Event(42, "x")]
    result3 = top_k_in_window(single, window_size=1, k=3)
    assert result3 == ["x"], result3

    # Tie on frequency and lexicographic ordering
    tie_events = [Event(10, "z"), Event(11, "y"), Event(12, "z"), Event(13, "y")]  # z:2, y:2
    # Lexicographically smaller is 'y'
    result4 = top_k_in_window(tie_events, window_size=10, k=2)
    assert result4 == ["y", "z"], result4

    # k larger than distinct keys
    result5 = top_k_in_window(tie_events, window_size=100, k=10)
    assert result5 == ["y", "z"], result5

    print("Basic tests passed.")


def _run_randomized_sanity() -> None:
    import random
    random.seed(0)

    # Create clustered frequencies: keys 'k0'..'k9'
    keys = [f"k{i}" for i in range(10)]
    events: List[Event] = []
    base_ts = 1_000_000
    # Assign weights so earlier keys appear more
    weights = [10 - i for i in range(10)]  # k0 highest weight
    population = []
    for key, w in zip(keys, weights):
        population.extend([key] * w)
    for i in range(5_000):
        ts = base_ts - random.randint(0, 400)  # Spread over 400s
        key = random.choice(population)
        events.append(Event(ts, key))

    try:
        top5 = top_k_in_window(events, window_size=500, k=5)
    except NotImplementedError:
        print("SKIPPED: Implement top_k_in_window to enable randomized sanity tests")
        return

    # Expect roughly descending k0..k4 due to weight distribution; allow mild deviations
    assert len(top5) == 5, top5
    assert top5[0] == "k0", f"Expected k0 as most frequent, got {top5}"  # Very likely
    print("Randomized sanity tests passed (non-deterministic validation).")


def _run_edge_cases() -> None:
    # Empty events
    try:
        empty = top_k_in_window([], window_size=10, k=3)
    except NotImplementedError:
        print("SKIPPED: Implement top_k_in_window to enable edge case tests")
        return
    assert empty == [], empty

    # All events outside window
    events = [Event(100, "a"), Event(101, "b"), Event(102, "c")]
    # window_size = 1 -> window = [102,102]
    res = top_k_in_window(events, window_size=1, k=2)
    assert res in (["c"], []), res  # Depending on inclusivity implementation spec expects ['c']

    print("Edge case tests passed.")


if __name__ == "__main__":
    # Execute all test groups
    _run_basic_tests()
    _run_randomized_sanity()
    _run_edge_cases()
    print("ALL TESTS COMPLETED (implement function for full validation).")
