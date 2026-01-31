from reportlab.lib.pagesizes import letter, landscape
from reportlab.platypus import BaseDocTemplate, Frame, PageTemplate, Paragraph, Spacer, PageBreak, Preformatted
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors

out_path = "./Coding_Interview_Cheat_Sheet_Meta_L6_2pages.pdf"

PAGE_SIZE = landscape(letter)
W, H = PAGE_SIZE

margin = 0.32 * inch
gutter = 0.22 * inch
col_count = 2
usable_w = W - 2*margin - gutter
col_w = usable_w / col_count
col_h = H - 2*margin

styles = getSampleStyleSheet()
base = ParagraphStyle(
    "Base",
    parent=styles["Normal"],
    fontName="Helvetica",
    fontSize=6.7,
    leading=7.7,
    textColor=colors.black,
    spaceAfter=1.0,
)
h1 = ParagraphStyle(
    "H1",
    parent=base,
    fontName="Helvetica-Bold",
    fontSize=10.5,
    leading=11.9,
    spaceAfter=3.6,
)
h2 = ParagraphStyle(
    "H2",
    parent=base,
    fontName="Helvetica-Bold",
    fontSize=7.9,
    leading=9.0,
    spaceBefore=2.3,
    spaceAfter=1.0,
    textColor=colors.HexColor("#0B3A66"),
)
h3 = ParagraphStyle(
    "H3",
    parent=base,
    fontName="Helvetica-Bold",
    fontSize=7.1,
    leading=8.1,
    spaceBefore=1.8,
    spaceAfter=0.7,
)
small = ParagraphStyle(
    "Small",
    parent=base,
    fontSize=6.1,
    leading=7.0,
)
mono = ParagraphStyle(
    "Mono",
    parent=small,
    fontName="Courier",
    fontSize=5.7,
    textColor=colors.HexColor("#222222"),
    leading=6.5,
)

def header_footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica-Bold", 10)
    canvas.drawString(margin, H - margin + 4, "Coding Interview Cheat Sheet (Meta L6)")
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#444444"))
    canvas.drawRightString(W - margin, H - margin + 6, "Keep crisp: clarify -> approach -> complexity -> code -> tests -> edge cases")
    canvas.setFillColor(colors.HexColor("#666666"))
    canvas.setFont("Helvetica", 7.5)
    canvas.drawString(margin, margin - 12, "Print: landscape, fit-to-page, margins minimum. Use as a memory jogger (not a script).")
    canvas.drawRightString(W - margin, margin - 12, f"Page {doc.page}")
    canvas.restoreState()

doc = BaseDocTemplate(out_path, pagesize=PAGE_SIZE,
                      leftMargin=margin, rightMargin=margin,
                      topMargin=margin, bottomMargin=margin)

frames = []
for i in range(col_count):
    x = margin + i*(col_w + gutter)
    frames.append(Frame(x, margin, col_w, col_h, showBoundary=0))

doc.addPageTemplates([PageTemplate(id="TwoCol", frames=frames, onPage=header_footer)])

story = []

# PAGE 1
story.append(Paragraph("Coding Interview (Meta L6) — 2-page print cheat sheet", h1))
story.append(Paragraph("<b>Goal:</b> never blank on patterns, templates, edge cases, or communication signals.", base))

story.append(Paragraph("1) Universal problem-solving script (use every time)", h2))
steps = [
    "- Restate problem + constraints; confirm input/output types; clarify duplicates/ordering/ties.",
    "- Ask for bounds: n, value ranges, memory/time limits; streaming? mutability? recursion depth?",
    "- Do 1-2 examples by hand (including tricky/edge).",
    "- Propose solution(s): baseline then optimal; pick one and justify.",
    "- State complexity <b>before</b> coding: time + space; mention dominating term.",
    "- Code in small chunks; narrate invariants; avoid cleverness unless needed.",
    "- Validate with tests: happy path + edges + adversarial; then final complexity recap.",
]
for s in steps:
    story.append(Paragraph(s, base))

story.append(Spacer(1, 4))
story.append(Paragraph("2) Edge-case & bug checklist (quick scan before you hit run)", h2))
bugs = [
    "<b>Empty/single:</b> n=0/1; None; missing keys; empty string.",
    "<b>Bounds:</b> off-by-one, inclusive/exclusive ranges, k=0/k>n, negative numbers.",
    "<b>Overflow:</b> sum of large ints; use Python's bigint or check bounds; in other langs watch int limits.",
    "<b>Duplicates:</b> stable ordering? multiple identical values? set vs multiset behavior.",
    "<b>Cycles:</b> graphs/linked lists; visited handling; parent pointers; self-loops.",
    "<b>Mutability:</b> modifying list while iterating; aliasing; copy vs reference; default mutable args.",
    "<b>Termination:</b> while loops; recursion base cases; pointer movement always progresses.",
    "<b>Floats:</b> precision issues; use abs(a-b) < eps; avoid == on floats.",
    "<b>Complexity:</b> hidden O(n²) (nested loops, list pop(0), string concat, repeated slicing).",
]
for b in bugs:
    story.append(Paragraph("- " + b, small))

story.append(Spacer(1, 4))
story.append(Paragraph("3) Pattern picker (map problem -> tool)", h2))
patterns = [
    "<b>Hash map/set:</b> membership, counts, complements, first occurrence, de-dup.",
    "<b>Two pointers:</b> sorted arrays, remove/partition, palindromes, pair sums.",
    "<b>Sliding window:</b> subarray/substring with constraints (≤k, exactly k, at most k).",
    "<b>Monotonic stack:</b> next greater/smaller, histogram, span, remove k digits.",
    "<b>Heap:</b> top-k, k-way merge, streaming median, schedule/interval greediness.",
    "<b>Binary search:</b> sorted + “first/last true” on monotonic predicate; answer-space search.",
    "<b>BFS/DFS:</b> grid/graph connectivity, shortest path unweighted (BFS), components, topo.",
    "<b>DP:</b> optimal substructure; knapsack; sequences; grid paths; string edit-like.",
    "<b>Union-Find:</b> dynamic connectivity, grouping, cycle detection, Kruskal.",
    "<b>Intervals:</b> merge, sweep line, meeting rooms, overlaps; sort by start/end.",
]
for p in patterns:
    story.append(Paragraph("- " + p, base))

story.append(Spacer(1, 4))
story.append(Paragraph("4) Meta favorites (high-frequency patterns)", h2))
meta = [
    "<b>Trees:</b> LCA, serialize/deserialize, path sums, BST validation, iterators.",
    "<b>Graphs:</b> clone graph, detect cycle, shortest path, number of islands, course schedule.",
    "<b>Arrays:</b> subarray sum, product except self, merge intervals, meeting rooms.",
    "<b>Strings:</b> valid parentheses, decode ways, word break, regex/wildcard.",
    "<b>Design:</b> LRU/LFU cache, iterator patterns, random with weights.",
    "<b>Follow-ups:</b> optimize space, handle streaming, scale to distributed, add concurrency.",
]
for m in meta:
    story.append(Paragraph("- " + m, small))

story.append(Spacer(1, 4))
story.append(Paragraph("5) Core invariants to say out loud (signal seniority)", h2))
invars = [
    "<b>Window invariant:</b> current window always satisfies condition; when violated, shrink left until restored.",
    "<b>Two pointers:</b> pointers move monotonically; each element processed O(1) times => O(n).",
    "<b>Stack:</b> stack maintains monotonic property; each element pushed/popped once => O(n).",
    "<b>BFS:</b> first time you pop a node = shortest distance in unweighted graph.",
    "<b>Dijkstra:</b> min-heap pop finalizes shortest path when edges non-negative.",
    "<b>Binary search:</b> maintain [lo, hi) (or inclusive) + monotonic predicate; prove termination.",
    "<b>DP:</b> define state + transition + base cases + iteration order; confirm no future deps.",
]
for i in invars:
    story.append(Paragraph("- " + i, base))

story.append(Spacer(1, 6))
story.append(Paragraph("6) Complexity cheats (common ops)", h3))
comp = [
    "list append O(1) amort; pop() O(1); pop(0) O(n); sort O(n log n); dict/set avg O(1);",
    "heap push/pop O(log n); deque popleft O(1); bisect O(log n); union-find ~ inverse Ackermann.",
]
story.append(Paragraph(" ".join(comp), small))

story.append(Spacer(1, 6))
story.append(Paragraph("7) L6 signals in coding round (what distinguishes you)", h2))
signals = [
    "<b>Drive:</b> you lead with structure and ask constraints early (no thrash).",
    "<b>Clarity:</b> crisp explanation + invariants; you can prove correctness informally.",
    "<b>Robustness:</b> edge cases, input validation, careful about complexity landmines.",
    "<b>Quality:</b> readable code, good names, small helpers, minimal bugs.",
    "<b>Speed:</b> you converge quickly: identify pattern, state plan, then implement.",
]
for s in signals:
    story.append(Paragraph("- " + s, small))

story.append(Spacer(1, 4))
story.append(Paragraph("8) Handling follow-ups gracefully (L6 must-have)", h2))
followups = [
    "<b>'Can you do better?'</b> → state current complexity, propose next approach (hash/sort/heap), explain tradeoff.",
    "<b>'What if input is huge?'</b> → streaming, external sort, approximate (bloom/HLL), chunking, MapReduce framing.",
    "<b>'What if concurrent?'</b> → locks, CAS, thread-safe structures, immutability; mention race conditions.",
    "<b>'What about scale?'</b> → partitioning/sharding, caching, async, batching; state bottlenecks.",
    "<b>Don't panic:</b> 'Good question—let me think.' Pause, state constraints, propose approach.",
]
for f in followups:
    story.append(Paragraph("- " + f, small))

# COLUMN 2 - Templates
story.append(Paragraph("Python templates (minimal, interview-friendly)", h2))

# Sliding window template
story.append(Paragraph("Sliding window (at most k / generic constraint)", h3))
story.append(Preformatted(
"""l = 0
for r, x in enumerate(arr):
    add(x)
    while not ok():   # window violates constraint
        remove(arr[l]); l += 1
    ans = best(ans, r - l + 1)""", mono))
story.append(Paragraph("Tip: for “exactly k distinct” => f(at_most(k)) - f(at_most(k-1)).", small))

# Binary search template
story.append(Spacer(1, 3))
story.append(Paragraph("Binary search on answer (first True)", h3))
story.append(Preformatted(
"""lo, hi = low_bound, high_bound  # hi exclusive
while lo < hi:
    mid = (lo + hi) // 2
    if feasible(mid):
        hi = mid
    else:
        lo = mid + 1
return lo""", mono))
story.append(Paragraph("Always define feasible(mid) monotonic; state invariants for [lo, hi).", small))

# DFS/BFS
story.append(Spacer(1, 3))
story.append(Paragraph("Graph/Tree traversal", h3))
story.append(Preformatted(
"""from collections import deque

# BFS shortest path (unweighted)
q = deque([src])
dist = {src: 0}
while q:
    u = q.popleft()
    for v in nbrs[u]:
        if v not in dist:
            dist[v] = dist[u] + 1
            q.append(v)""", mono))

story.append(Preformatted(
"""# DFS iterative
stack = [src]; seen = set([src])
while stack:
    u = stack.pop()
    for v in nbrs[u]:
        if v not in seen:
            seen.add(v)
            stack.append(v)""", mono))

# Topological sort
story.append(Spacer(1, 3))
story.append(Paragraph("Topological sort (Kahn)", h3))
story.append(Preformatted(
"""from collections import deque
indeg = {u: 0 for u in nodes}
for u in nodes:
    for v in nbrs[u]: indeg[v] += 1
q = deque([u for u in nodes if indeg[u] == 0])
order = []
while q:
    u = q.popleft(); order.append(u)
    for v in nbrs[u]:
        indeg[v] -= 1
        if indeg[v] == 0: q.append(v)""", mono))
story.append(Paragraph("If len(order) < n => cycle.", small))

# Heap / top-k
story.append(Spacer(1, 3))
story.append(Paragraph("Heap patterns", h3))
story.append(Preformatted(
"""import heapq

# top-k largest via min-heap size k
h = []
for x in nums:
    heapq.heappush(h, x)
    if len(h) > k: heapq.heappop(h)
# h holds k largest""", mono))
story.append(Paragraph("For max-heap: push -x; or store (-key, item).", small))

# Union-Find
story.append(Spacer(1, 3))
story.append(Paragraph("Union-Find (DSU)", h3))
story.append(Preformatted(
"""parent = {x: x for x in items}
rank = {x: 0 for x in items}

def find(x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x

def union(a, b):
    ra, rb = find(a), find(b)
    if ra == rb: return False
    if rank[ra] < rank[rb]: ra, rb = rb, ra
    parent[rb] = ra
    if rank[ra] == rank[rb]: rank[ra] += 1
    return True""", mono))

# DP memo
story.append(Spacer(1, 3))
story.append(Paragraph("DP memoization (avoid recomputation)", h3))
story.append(Preformatted(
"""from functools import lru_cache

@lru_cache(None)
def dp(i, state):
    if i == n: return base
    ans = ...
    return ans""", mono))
story.append(Paragraph("State must be hashable; watch recursion depth; consider iterative DP if deep.", small))

# Backtracking
story.append(Spacer(1, 3))
story.append(Paragraph("Backtracking (permutations/subsets/combinations)", h3))
story.append(Preformatted(
"""def backtrack(path, choices):
    if done(path):
        ans.append(path[:]); return
    for i, c in enumerate(choices):
        if not valid(c): continue
        path.append(c)
        backtrack(path, choices[i+1:])  # or choices[:i]+choices[i+1:] for perms
        path.pop()

ans = []; backtrack([], items)""", mono))
story.append(Paragraph("For permutations: use 'used' set or swap in-place. For pruning: skip invalid early.", small))

# Prefix sum
story.append(Spacer(1, 3))
story.append(Paragraph("Prefix sum (range queries O(1))", h3))
story.append(Preformatted(
"""pre = [0]*(n+1)
for i, x in enumerate(arr):
    pre[i+1] = pre[i] + x
# sum [l..r) = pre[r] - pre[l]

# 2D prefix sum:
pre[i][j] = val + pre[i-1][j] + pre[i][j-1] - pre[i-1][j-1]
# query (r1,c1,r2,c2) = pre[r2][c2] - pre[r1-1][c2] - pre[r2][c1-1] + pre[r1-1][c1-1]""", mono))

# Interval merge
story.append(Spacer(1, 3))
story.append(Paragraph("Interval merge / sweep line", h3))
story.append(Preformatted(
"""intervals.sort(key=lambda x: x[0])
merged = []
for s, e in intervals:
    if merged and s <= merged[-1][1]:
        merged[-1][1] = max(merged[-1][1], e)
    else:
        merged.append([s, e])

# Meeting rooms II (min rooms): sweep line
events = []
for s, e in intervals:
    events.append((s, 1))   # start
    events.append((e, -1))  # end
events.sort()
max_rooms = cur = 0
for _, delta in events:
    cur += delta
    max_rooms = max(max_rooms, cur)""", mono))

# Linked list
story.append(Spacer(1, 3))
story.append(Paragraph("Linked list essentials", h3))
story.append(Preformatted(
"""# Reverse in place
def reverse(head):
    prev = None
    while head:
        nxt = head.next
        head.next = prev
        prev = head
        head = nxt
    return prev

# Detect cycle (Floyd)
def has_cycle(head):
    slow = fast = head
    while fast and fast.next:
        slow, fast = slow.next, fast.next.next
        if slow == fast: return True
    return False

# Find middle (slow at mid when fast at end)""", mono))

# Quick select
story.append(Spacer(1, 3))
story.append(Paragraph("Quick select (k-th smallest, avg O(n))", h3))
story.append(Preformatted(
"""import random
def quickselect(arr, k):  # k is 0-indexed
    if len(arr) == 1: return arr[0]
    pivot = random.choice(arr)
    lo = [x for x in arr if x < pivot]
    eq = [x for x in arr if x == pivot]
    hi = [x for x in arr if x > pivot]
    if k < len(lo): return quickselect(lo, k)
    elif k < len(lo) + len(eq): return pivot
    else: return quickselect(hi, k - len(lo) - len(eq))""", mono))
story.append(Paragraph("Use random pivot to avoid worst case. For top-k, partition around k.", small))

# Quick Python reminders
story.append(Spacer(1, 4))
story.append(Paragraph("Python standard library reminders (fast access)", h2))
py = [
    "<b>collections:</b> deque, Counter, defaultdict",
    "<b>heapq:</b> heappush/heappop/heappushpop/nlargest/nsmallest",
    "<b>bisect:</b> bisect_left/right for insertion points",
    "<b>itertools:</b> accumulate, product, combinations",
    "<b>math:</b> inf, gcd, isqrt",
]
for line in py:
    story.append(Paragraph("- " + line, base))

story.append(Spacer(1, 4))
story.append(Paragraph("Final 60-second close (use if time)", h3))
story.append(Paragraph(
    "Recap approach + invariant. Walk through 1 edge case. State final time/space. Mention any alternative solution and why you didn’t pick it.",
    small
))


# PAGE 2 - Addendum
story.append(Paragraph("Coding Addendum — common algorithms & 'from scratch' snippets", h1))
story.append(Paragraph("<b>Use when the problem screams for it.</b> Keep explanations tight: invariant + why it's optimal + complexity.", small))

# Shortest paths
story.append(Paragraph("1) Shortest paths (graph/grid)", h2))
story.append(Paragraph("Dijkstra (non-negative weights)", h3))
story.append(Preformatted(
"""import heapq
INF = 10**18
dist = {src: 0}
pq = [(0, src)]
while pq:
    d, u = heapq.heappop(pq)
    if d != dist.get(u, INF): 
        continue
    if u == tgt: break
    for v, w in nbrs[u]:
        nd = d + w
        if nd < dist.get(v, INF):
            dist[v] = nd
            heapq.heappush(pq, (nd, v))""", mono))
story.append(Paragraph("Tip: for grid, neighbors are 4/8 dirs; for path reconstruct keep parent[v]=u.", small))

story.append(Spacer(1, 3))
story.append(Paragraph("0-1 BFS (weights are 0/1)", h3))
story.append(Preformatted(
"""from collections import deque
INF = 10**18
dist = {src: 0}
dq = deque([src])
while dq:
    u = dq.popleft()
    for v, w in nbrs[u]:      # w in {0,1}
        nd = dist[u] + w
        if nd < dist.get(v, INF):
            dist[v] = nd
            if w == 0: dq.appendleft(v)
            else: dq.append(v)""", mono))

story.append(Spacer(1, 3))
story.append(Paragraph("A* (grid/pathfinding; admissible heuristic)", h3))
story.append(Preformatted(
"""import heapq, math
def h(u):  # admissible: never overestimates
    return abs(u.x - tgt.x) + abs(u.y - tgt.y)  # manhattan for 4-neigh

INF = 10**18
g = {src: 0}
pq = [(h(src), 0, src)]  # (f=g+h, g, node)
parent = {src: None}
while pq:
    f, gu, u = heapq.heappop(pq)
    if gu != g.get(u, INF): 
        continue
    if u == tgt: break
    for v, w in nbrs[u]:
        nv = gu + w
        if nv < g.get(v, INF):
            g[v] = nv
            parent[v] = u
            heapq.heappush(pq, (nv + h(v), nv, v))""", mono))
story.append(Paragraph("Notes: heuristic must be admissible (and ideally consistent). A* reduces to Dijkstra if h=0.", small))

story.append(Spacer(1, 3))
story.append(Paragraph("Bellman-Ford (handles negative weights; detects neg cycles)", h3))
story.append(Preformatted(
"""INF = 10**18
dist = [INF]*n
dist[src] = 0
for _ in range(n-1):
    changed = False
    for u, v, w in edges:
        if dist[u] != INF and dist[u] + w < dist[v]:
            dist[v] = dist[u] + w
            changed = True
    if not changed: break

# one more pass => negative cycle reachable
neg_cycle = any(dist[u] != INF and dist[u] + w < dist[v] for u, v, w in edges)""", mono))
story.append(Paragraph("Use only when needed; O(VE).", small))

# MST
story.append(Spacer(1, 6))
story.append(Paragraph("2) Minimum spanning tree (MST)", h2))
story.append(Paragraph("Prim (dense-ish graphs)", h3))
story.append(Preformatted(
"""import heapq
seen = set([start])
pq = []
for v, w in nbrs[start]:
    heapq.heappush(pq, (w, start, v))
mst = []
while pq and len(seen) < n:
    w, u, v = heapq.heappop(pq)
    if v in seen: 
        continue
    seen.add(v); mst.append((u, v, w))
    for x, wx in nbrs[v]:
        if x not in seen:
            heapq.heappush(pq, (wx, v, x))""", mono))
story.append(Paragraph("Kruskal + DSU is great for sparse graphs (you already have DSU on page 2).", small))

# Heap from scratch
story.append(Spacer(1, 6))
story.append(Paragraph("3) Heap from scratch (min-heap)", h2))
story.append(Preformatted(
"""# 0-indexed array heap
def heappush(h, x):
    h.append(x)
    i = len(h) - 1
    while i > 0:
        p = (i - 1) // 2
        if h[p] <= h[i]: break
        h[p], h[i] = h[i], h[p]
        i = p

def heappop(h):
    x = h[0]
    last = h.pop()
    if h:
        h[0] = last
        i = 0
        n = len(h)
        while True:
            l = 2*i + 1; r = l + 1
            if l >= n: break
            m = l
            if r < n and h[r] < h[l]: m = r
            if h[i] <= h[m]: break
            h[i], h[m] = h[m], h[i]
            i = m
    return x""", mono))
story.append(Paragraph("Use to show fundamentals if asked; otherwise prefer heapq (less bug risk).", small))

# Trees / range queries
story.append(Spacer(1, 6))
story.append(Paragraph("4) Range queries (BIT / Segment Tree) + Trie", h2))
story.append(Paragraph("Fenwick / BIT (prefix sums)", h3))
story.append(Preformatted(
"""# 1-indexed BIT
bit = [0]*(n+1)
def add(i, delta):
    i += 1
    while i <= n:
        bit[i] += delta
        i += i & -i
def sum_(i):  # sum [0..i)
    s = 0
    while i > 0:
        s += bit[i]
        i -= i & -i
    return s
# range sum [l..r): sum_(r) - sum_(l)""", mono))
story.append(Paragraph("Segment tree is for min/max/gcd or non-invertible ops; BIT is simpler for sums.", small))

story.append(Spacer(1, 3))
story.append(Paragraph("Monotonic stack (next greater/smaller)", h3))
story.append(Preformatted(
"""# Next greater element to right
nge = [-1] * n
stack = []  # indices, decreasing values
for i, x in enumerate(arr):
    while stack and arr[stack[-1]] < x:
        nge[stack.pop()] = x
    stack.append(i)
# For next smaller: change < to >
# For left: iterate backwards or use different approach""", mono))

story.append(Spacer(1, 3))
story.append(Paragraph("Trie (prefix search)", h3))
story.append(Preformatted(
"""class Node:
    __slots__ = ("next", "end")
    def __init__(self):
        self.next = {}
        self.end = False

root = Node()
def insert(s):
    cur = root
    for ch in s:
        cur = cur.next.setdefault(ch, Node())
    cur.end = True""", mono))

# Strings
story.append(Spacer(1, 6))
story.append(Paragraph("5) String algorithms (when brute force times out)", h2))
story.append(Paragraph("KMP prefix function (pattern search)", h3))
story.append(Preformatted(
"""# pi[i] = length of longest proper prefix == suffix for s[:i+1]
pi = [0]*m
j = 0
for i in range(1, m):
    while j > 0 and p[i] != p[j]:
        j = pi[j-1]
    if p[i] == p[j]: j += 1
    pi[i] = j

# scan text
j = 0
for i, ch in enumerate(t):
    while j > 0 and ch != p[j]:
        j = pi[j-1]
    if ch == p[j]: j += 1
    if j == m: 
        return i - m + 1""", mono))
story.append(Paragraph("Alternative: Rabin-Karp for average-case hashing; KMP is deterministic O(n+m).", small))

# LRU cache + OrderedDict
story.append(Spacer(1, 6))
story.append(Paragraph("6) Systems-y coding: LRU cache (common ask)", h2))
story.append(Preformatted(
"""from collections import OrderedDict

class LRUCache:
    def __init__(self, cap):
        self.cap = cap
        self.od = OrderedDict()

    def get(self, k):
        if k not in self.od: return -1
        self.od.move_to_end(k)
        return self.od[k]

    def put(self, k, v):
        if k in self.od: self.od.move_to_end(k)
        self.od[k] = v
        if len(self.od) > self.cap:
            self.od.popitem(last=False)""", mono))
story.append(Paragraph("If they demand “from scratch”, implement doubly-linked list + hashmap.", small))

story.append(Spacer(1, 6))
story.append(Paragraph("Last-moment reminders", h2))
rem = [
    "- If a graph has <b>negative weights</b>: Dijkstra is wrong (use Bellman-Ford / SPFA-ish; or reframe).",
    "- If asking for <b>shortest path on grid</b>: BFS (unweighted), 0-1 BFS (0/1), Dijkstra (weighted), A* (with heuristic).",
    "- For <b>state-space search</b>: define state, transitions, and visited key; beware exponential blowups.",
    "- When implementing from scratch: keep invariants, write tiny helper functions, test small cases.",
]
for r in rem:
    story.append(Paragraph(r, base))


doc.build(story)
out_path

