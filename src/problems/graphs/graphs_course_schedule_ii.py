"""Course Schedule II (LeetCode 210 - Medium)

Prompt (Interviewer Version):
You are given the total number of courses (labeled from 0 to n - 1) and a list of prerequisite pairs
where each pair [a, b] indicates you must take course b before course a.
Return one valid ordering of courses you can take to finish all courses. If it's impossible (due to a cycle), return an empty list.

Clarifications to ask (good to voice during interview):
1. Are there duplicate prerequisite pairs? (Assume yes; treat duplicates harmlessly.)
2. Can there be self-dependencies like [1,1]? (Assume yes; that makes it impossible.)
3. If multiple valid orders exist, can I return any? (Yes.)
4. Constraints? 1 <= num_courses <= 10^4, prerequisites length up to 10^5.

Follow-up (possible L6/L7 discussion areas):
- Detecting cycles efficiently (Kahn's BFS vs DFS coloring).
- Memory optimization trade-offs (adjacency list design, using arrays vs dicts).
- Parallelization possibilities if course graph is large and sparse.
- Handling streaming prerequisite input (online topological ordering strategies).

Your Task:
Implement the function `find_course_order(num_courses, prerequisites)` below. Do NOT change the function
signature. Return a list of integers representing a valid topological ordering or [] if impossible.

Complexity Targets:
Time: O(V + E) where V = num_courses, E = len(prerequisites)
Space: O(V + E)

You only need to implement the body; leave the tests as-is.
"""
from __future__ import annotations
from collections import defaultdict
from typing import List, Sequence, Iterable, Tuple


def find_course_order(num_courses: int, prerequisites: Sequence[Sequence[int]]) -> List[int]:
    """Return a valid ordering of courses or [] if impossible.

    Args:
        num_courses: Total number of distinct courses labeled 0..num_courses-1.
        prerequisites: Iterable of pairs [a, b] meaning b must precede a.

    Returns:
        A list of course indices in a valid topological order, or [] if a cycle exists.

    Edge Cases to Consider:
    - No prerequisites -> any permutation (tests expect a specific validation, not exact order).
    - Cycle (including self-loop) -> []
    - Duplicate edges -> should not break logic.
    - Disconnected components -> still produce valid ordering across all components.

    Implementation Guidance (you may remove after implementing):
    - Build adjacency list and indegree array.
    - Use a queue (collections.deque) for nodes with indegree 0.
    - Pop, append to result, decrement neighbors; enqueue when indegree hits 0.
    - If result length != num_courses -> return [] (cycle detected).

    DO NOT implement yet if you are practicing. Fill in during the session.
    """
    #implementing topological sort with adjacency

    adj = defaultdict(list)
    status = [0]*num_courses  # 0 = unvisited, 1 = visiting, 2 = visited
    reverse_order = []
    #building adjacency list
    for pre in prerequisites:
        adj[pre[1]].append(pre[0])
    # topological sort with DFS
    print('adjacency:', adj)

    def dfs(course: int) -> bool:
        print('visiting, course:', course, '    status', status)
        if status[course] == 1:
            return False  # cycle detected
        if status[course] == 2:
            return True  # already processed
        status[course] = 1  # mark as visiting
        for neighbor in adj[course]:
            if not dfs(neighbor):
                return False
        status[course] = 2  # mark as visited
        reverse_order.append(course)
        return True

    for i in range(num_courses):
        if status[i] == 0:
            if not dfs(i):
                return []  # cycle detected

    return reverse_order[::-1]  # Reverse to get correct order

# ---------------------- Test Harness Below ---------------------- #
class _TestCase:
    def __init__(self, name: str, num_courses: int, prerequisites: Sequence[Sequence[int]], expect_empty: bool = False, strict_sequence: List[int] | None = None):
        self.name = name
        self.num_courses = num_courses
        self.prerequisites = prerequisites 
        self.expect_empty = expect_empty
        self.strict_sequence = strict_sequence  # If provided, order must match exactly.


def _is_valid_order(order: List[int], num_courses: int, prerequisites: Sequence[Sequence[int]]) -> bool:
    """Return True if order satisfies all prerequisite constraints.

    Each prerequisite is [a, b] meaning b must appear BEFORE a.
    """
    if len(order) != num_courses:
        return False
    if set(order) != set(range(num_courses)):
        return False
    position = {c: i for i, c in enumerate(order)}
    for pair in prerequisites:
        if len(pair) != 2:
            return False
        a, b = pair
        if a == b:  # self-loop invalidates any ordering
            return False
        # b must come strictly earlier than a
        if position[b] >= position[a]:
            return False
    return True


def run_tests() -> None:
    test_cases = [
        _TestCase(
            name="simple chain",
            num_courses=2,
            prerequisites=[[1, 0]],
        ),
        _TestCase(
            name="cycle simple",
            num_courses=2,
            prerequisites=[[1, 0], [0, 1]],
            expect_empty=True,
        ),
        _TestCase(
            name="diamond / branching",
            num_courses=4,
            prerequisites=[[1, 0], [2, 0], [3, 1], [3, 2]],
        ),
        _TestCase(
            name="no prerequisites",
            num_courses=3,
            prerequisites=[],
        ),
        _TestCase(
            name="linear chain length 5",
            num_courses=5,
            prerequisites=[[1, 0], [2, 1], [3, 2], [4, 3]],
            strict_sequence=[0, 1, 2, 3, 4],
        ),
        _TestCase(
            name="duplicates + cycle",
            num_courses=2,
            prerequisites=[[1, 0], [1, 0], [0, 1]],
            expect_empty=True,
        ),
        _TestCase(
            name="self loop",
            num_courses=3,
            prerequisites=[[0, 0]],
            expect_empty=True,
        ),
    ]

    passed = 0
    failed = 0
    for tc in test_cases:
        try:
            order = find_course_order(tc.num_courses, tc.prerequisites)
        except NotImplementedError:
            print("Implementation missing. Aborting tests early.")
            return
        except Exception as e:  # Unexpected error
            failed += 1
            print(f"[FAIL] {tc.name}: raised unexpected exception: {e}")
            continue

        if tc.expect_empty:
            if order == []:
                passed += 1
                print(f"[PASS] {tc.name} -> correctly detected impossibility")
            else:
                failed += 1
                print(f"[FAIL] {tc.name} -> expected [], got {order}")
            continue

        if tc.strict_sequence is not None:
            if order == tc.strict_sequence:
                passed += 1
                print(f"[PASS] {tc.name} -> strict order matched: {order}")
            else:
                failed += 1
                print(f"[FAIL] {tc.name} -> expected strict {tc.strict_sequence}, got {order}")
            continue

        if _is_valid_order(order, tc.num_courses, tc.prerequisites):
            passed += 1
            print(f"[PASS] {tc.name} -> valid order: {order}")
        else:
            failed += 1
            print(f"[FAIL] {tc.name} -> invalid order returned: {order}")

    total = passed + failed
    if failed == 0:
        print(f"\nAll {total} tests passed ✅")
    else:
        print(f"\nSummary: {passed} passed, {failed} failed (total {total}) ❌")


if __name__ == "__main__":  # pragma: no cover - manual execution only
    run_tests()
