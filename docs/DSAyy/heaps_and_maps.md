
## Problem 1: Ways to Form a Max Heap

### Problem Statement

Given `A` distinct integers, count the number of distinct **Max Heaps** that can be formed using all of them. The heap must satisfy:

1. It is a complete binary tree.
    
2. Each parent node has a value strictly greater than those of its children.
    

Return the result modulo $10^9 + 7$.

---

### Combinatorial Insight

Let the distinct integers be labeled as ${1, 2, \dots, A}$.  
In any max heap, the largest element $A$ must be at the root.

Let:

- $n = A$
    
- $h = \lfloor \log_2 n \rfloor$
    
- $\text{nodesAbove} = 2^h - 1$
    
- $\text{lastLevel} = n - \text{nodesAbove}$
    
- $\text{leftLast} = \min(\text{lastLevel}, 2^{h-1})$
    

Then the number of nodes in the **left subtree** is:

L=(2h−1−1)+leftLastL = (2^{h-1} - 1) + \text{leftLast}

---

### Recurrence Relation

Let $D[n]$ denote the number of valid Max Heaps that can be built with $n$ distinct values.

D[0]=D[1]=1D[0] = D[1] = 1 D[n]=(n−1L)⋅D[L]⋅D[n−1−L]D[n] = \binom{n - 1}{L} \cdot D[L] \cdot D[n - 1 - L]

where:

- $\binom{n - 1}{L}$ is the number of ways to choose elements for the left subtree,
    
- $L$ is the number of nodes in the left subtree.
    

---

### Clean Python Implementation

```python
import math

MOD = 10**9 + 7
MAXN = 105

fact = [1] * MAXN
invfact = [1] * MAXN
dp = [0] * MAXN

def modexp(base, exp=MOD - 2):
    result = 1
    while exp:
        if exp % 2:
            result = result * base % MOD
        base = base * base % MOD
        exp //= 2
    return result

def precompute(n):
    for i in range(1, n + 1):
        fact[i] = fact[i - 1] * i % MOD
    invfact[n] = modexp(fact[n])
    for i in range(n, 0, -1):
        invfact[i - 1] = invfact[i] * i % MOD

def choose(n, k):
    if k < 0 or k > n:
        return 0
    return fact[n] * invfact[k] % MOD * invfact[n - k] % MOD

def count_max_heaps(A):
    precompute(A)
    dp[0] = dp[1] = 1

    for n in range(2, A + 1):
        h = n.bit_length() - 1
        nodes_above = (1 << h) - 1
        last_level = n - nodes_above
        left_last = min(last_level, 1 << (h - 1))
        L = (1 << (h - 1)) - 1 + left_last

        ways = choose(n - 1, L)
        dp[n] = ways * dp[L] % MOD * dp[n - 1 - L] % MOD

    return dp[A]
```

---

### Time and Space Complexity

- **Precomputation** of factorials and inverse factorials: $O(A)$
    
- **DP computation** for all $n \leq A$: $O(A)$
    
- **Total Time Complexity**: $O(A)$
    
- **Space Complexity**: $O(A)$
    
---
## Problem 2: N Maximum Pair Sum Combinations

### Problem Statement

Given two integer arrays $A$ and $B$, each of size $N$, compute the $N$ **largest sums** of the form $A_i + B_j$ and return them in **descending order**.

Formally, compute:

Output: {s1,s2,…,sN}where each sk=Aik+Bjk\text{Output: } \{s_1, s_2, \dots, s_N\} \quad \text{where each } s_k = A_{i_k} + B_{j_k}

such that $s_1 \geq s_2 \geq \dots \geq s_N$ and each $(i_k, j_k)$ is a valid index pair.

---

### Key Insight

- Sort both arrays $A$ and $B$ in descending order.
    
- The largest sum is $A_0 + B_0$.
    
- From position $(i, j)$ in the implicit $N \times N$ sum matrix, you can move:
    
    - Right to $(i, j+1)$
        
    - Down to $(i+1, j)$
        

Use a **max-heap** to always choose the next largest unvisited sum from this matrix.

Maintain a `seen` set to avoid pushing duplicate index pairs.

---

### Clean Python Implementation

```python
import heapq

def max_pair_sums(A, B):
    N = len(A)
    A.sort(reverse=True)
    B.sort(reverse=True)

    max_heap = []
    seen = set()
    result = []

    heapq.heappush(max_heap, (-(A[0] + B[0]), 0, 0))
    seen.add((0, 0))

    while len(result) < N:
        total, i, j = heapq.heappop(max_heap)
        result.append(-total)

        if i + 1 < N and (i + 1, j) not in seen:
            heapq.heappush(max_heap, (-(A[i + 1] + B[j]), i + 1, j))
            seen.add((i + 1, j))

        if j + 1 < N and (i, j + 1) not in seen:
            heapq.heappush(max_heap, (-(A[i] + B[j + 1]), i, j + 1))
            seen.add((i, j + 1))

    return result
```

---

### Time and Space Complexity

- **Sorting** both arrays: $O(N \log N)$
    
- **Heap operations**: Up to $2N$ pushes, $N$ pops → $O(N \log N)$
    
- **Space**: $O(N)$ for heap and visited set
    
---
## Problem 3: K Largest Elements

### Problem Statement

Given an integer array $A$ of size $N$ and an integer $B$ ($1 \leq B \leq N$), return any $B$ **largest elements** from the array. The output can be in **any order**.

---

### Key Insight

There are two efficient approaches:

#### 1. Partition-Based (Quickselect/Nth Element)

- Use a partitioning algorithm to position the $(N-B)$-th smallest element at index $N-B$.
    
- All elements from index $N-B$ to $N-1$ are the $B$ largest elements (in any order).
    
- This approach uses in-place partitioning and avoids full sorting.
    

#### 2. Min-Heap of Size $B$

- Build a min-heap with the first $B$ elements.
    
- For every new element $x$ in the rest of the array:
    
    - If $x$ is greater than the root, replace the root with $x$.
        
- The heap contains the $B$ largest elements.
    

---

### Clean Python Implementation (Min-Heap Approach)

```python
import heapq

def k_largest_elements(A, B):
    if B == 0:
        return []

    min_heap = A[:B]
    heapq.heapify(min_heap)

    for x in A[B:]:
        if x > min_heap[0]:
            heapq.heappushpop(min_heap, x)

    return min_heap
```

If you'd prefer the partition-based approach (like `nth_element`), here's an alternative using `heapq.nlargest`:

```python
import heapq

def k_largest_elements_quick(A, B):
    return heapq.nlargest(B, A)
```

---

### Time and Space Complexity

#### Min-Heap Approach:

- Build heap: $O(B)$
    
- Remaining $N - B$ insertions: $O((N - B) \log B)$
    
- Total time: $O(N \log B)$
    
- Space: $O(B)$
    

#### Quickselect/Nth Element (via `heapq.nlargest`):

- Time: $O(N + B \log B)$ average
    
- Space: $O(B)$
    

---

## Problem 4: Profit Maximisation

### Problem Statement

You are given an array $A$ of size $N$, where $A_i$ is the number of **vacant seats** in row $i$ of a stadium. You need to sell tickets to $B$ people, **one ticket at a time**.

- Each ticket sold from row $i$ earns profit equal to the **current** number of vacant seats in that row.
    
- After selling one ticket in row $i$, the vacancy count in that row decreases by 1.
    

Compute the **maximum total profit** from selling all $B$ tickets.

---

### Key Insight

To maximize profit:

- Always sell the next ticket from the row with the **most vacant seats**.
    
- Use a **max-heap** to always access the row with the highest seat count.
    

Steps:

1. Add all seat counts to a max-heap.
    
2. For $B$ iterations:
    
    - Pop the largest seat count.
        
    - Add it to the total profit.
        
    - Decrease the seat count by 1 and push it back if it’s still positive.
        

This greedy strategy ensures the highest marginal gain at every step.

---

### Clean Python Implementation

```python
import heapq

def max_profit(A, B):
    MOD = 10**9 + 7
    max_heap = [-x for x in A]  # simulate max-heap using min-heap
    heapq.heapify(max_heap)

    profit = 0
    for _ in range(B):
        if not max_heap:
            break
        top = -heapq.heappop(max_heap)
        profit = (profit + top) % MOD
        if top > 1:
            heapq.heappush(max_heap, -(top - 1))

    return profit
```

---

### Time and Space Complexity

- Heap initialization: $O(N)$
    
- Selling $B$ tickets:
    
    - Each ticket → 1 pop and possibly 1 push → $O(\log N)$
        
- Total Time: $O((N + B) \log N)$
    
- Space: $O(N)$ for the heap
    


---

## Problem 5: Merge K Sorted Arrays

### Problem Statement

You are given $K$ sorted integer arrays, each of length $N$, forming a $K \times N$ matrix $A$. Merge them into one single **sorted array** of length $K \cdot N$.

**Input**:  
A list of $K$ sorted lists:  
A=[A0,A1,…,AK−1]A = [A_0, A_1, \dots, A_{K-1}]  
Each $A_i$ is sorted in non-decreasing order.

**Output**:  
A single list with all elements from all $K$ arrays in sorted order.

---

### Key Insight

Use a **min-heap** to merge $K$ sorted arrays efficiently:

1. Initialize the heap with the **first element** of each array:  
    Each heap entry is a tuple:  
    (value,row index,element index)(\text{value}, \text{row index}, \text{element index})
    
2. Repeatedly extract the minimum element from the heap and push the **next element** from the same row (if any).
    
3. Append the extracted element to the result list.
    

This way, you always pull the smallest available value across all $K$ arrays.

---

### Clean Python Implementation

```python
import heapq

def merge_k_sorted_arrays(arrays):
    min_heap = []
    result = []

    # Initialize heap with first element of each array
    for i, row in enumerate(arrays):
        if row:
            heapq.heappush(min_heap, (row[0], i, 0))

    while min_heap:
        val, row, idx = heapq.heappop(min_heap)
        result.append(val)

        if idx + 1 < len(arrays[row]):
            next_val = arrays[row][idx + 1]
            heapq.heappush(min_heap, (next_val, row, idx + 1))

    return result
```

---

### Time and Space Complexity

- Total elements to process: $K \cdot N$
    
- Each heap operation: $O(\log K)$
    
- **Time complexity**:  
    O(KNlog⁡K)O(KN \log K)
    
- **Space complexity**:
    
    - Heap: $O(K)$
        
    - Output: $O(KN)$
        

---

## Problem 6: Connect Ropes

### Problem Statement

You are given an array $A$ of rope lengths. Your task is to connect all ropes into one single rope.  
Each time you connect two ropes of lengths $x$ and $y$, the cost is $x + y$.  
Return the **minimum total cost** to connect all ropes into one.

**Input**:  
An array $A = [a_1, a_2, \dots, a_N]$, where $a_i$ is the length of the $i$-th rope.

**Output**:  
Minimum total cost to connect all ropes.

---

### Key Insight

This is a classic **greedy problem** similar to building a **Huffman tree**.

To minimize total cost:

1. Always **connect the two shortest ropes** first.
    
2. Repeat until all ropes are merged.
    

Use a **min-heap** to efficiently extract the two smallest ropes at each step.

---

### Clean Python Implementation

```python
import heapq

def connect_ropes(A):
    heapq.heapify(A)
    total_cost = 0

    while len(A) > 1:
        x = heapq.heappop(A)
        y = heapq.heappop(A)
        cost = x + y
        total_cost += cost
        heapq.heappush(A, cost)

    return total_cost
```

---

### Example

```python
connect_ropes([1, 2, 3, 4, 5])  # Output: 33
```

Steps:

- Connect 1 + 2 = 3 → heap: [3, 3, 4, 5]
    
- Connect 3 + 3 = 6 → heap: [4, 5, 6]
    
- Connect 4 + 5 = 9 → heap: [6, 9]
    
- Connect 6 + 9 = 15 → heap: [15]
    
- Total = 3 + 6 + 9 + 15 = 33
    

---

### Time and Space Complexity

- Heap initialization: $O(N)$
    
- Merging steps: $(N - 1)$ pops + pushes, each $O(\log N)$
    
- **Time**: $O(N \log N)$
    
- **Space**: $O(N)$ for the heap
    

---


## Problem 7: Magician and Chocolates

### Problem Statement

You are given an integer $A$ (number of time units) and an array $B$ of size $N$, where $B_i$ denotes the initial number of **chocolates** in bag $i$.

Every second (for $A$ seconds), a kid performs the following:

1. Chooses the bag with the **maximum number of chocolates**.
    
2. Eats all chocolates in that bag.
    
3. The magician refills the bag with $\left\lfloor \frac{x}{2} \right\rfloor$ chocolates.
    

Return the **maximum number of chocolates** the kid can eat over $A$ seconds, modulo $10^9 + 7$.

---

### Key Insight

At every step, to maximize chocolates eaten:

- Always choose the **bag with the most chocolates**.
    
- Use a **max-heap** to track the current state of all bags.
    
- For each second:
    
    - Pop the max,
        
    - Add it to total,
        
    - Push back $\left\lfloor \frac{x}{2} \right\rfloor$.
        

---

### Clean Python Implementation

```python
import heapq

def max_chocolates(A, B):
    MOD = 10**9 + 7
    max_heap = [-x for x in B]
    heapq.heapify(max_heap)

    total = 0
    for _ in range(A):
        if not max_heap:
            break
        x = -heapq.heappop(max_heap)
        total = (total + x) % MOD
        heapq.heappush(max_heap, -(x // 2))

    return total
```

---

### Example

```python
max_chocolates(3, [6, 5])  # Output: 14
```

Steps:

- Eat 6 → refill with 3
    
- Eat 5 → refill with 2
    
- Eat 3 → refill with 1
    
- Total eaten: 6 + 5 + 3 = 14
    

---

### Time and Space Complexity

- Heap initialization: $O(N)$
    
- Each second:
    
    - 1 pop and 1 push → $O(\log N)$
        
- **Total time**: $O((A + N) \log N)$
    
- **Space**: $O(N)$ for the heap
    

---

## Problem 8: Maximum Sum Combinations

### Problem Statement

Given two integer arrays $A$ and $B$ of length $N$ and an integer $C$, return the **top $C$ maximum sums** of the form $A_i + B_j$, in **non-increasing order**.

**Input**:

- Two arrays: $A, B \in \mathbb{Z}^N$
    
- An integer $C$ such that $1 \leq C \leq N$
    

**Output**:

- An array of length $C$ containing the largest $C$ sums formed by adding one element from $A$ and one from $B$
    

---

### Key Insight

- Sort both $A$ and $B$ in **descending** order.
    
- The largest sum is always $A[0] + B[0]$.
    
- Model the problem as exploring a **virtual matrix** $M[i][j] = A[i] + B[j]$:
    
    - Each row and column is non-increasing.
        
- Start at $(0, 0)$ and always pick the next largest sum from the **max-heap**.
    
- Use a **visited set** to avoid revisiting index pairs.
    

Each time you pop $(i, j)$:

- Push $(i+1, j)$ and $(i, j+1)$ if not already visited.
    

---

### Clean Python Implementation

```python
import heapq

def max_sum_combinations(A, B, C):
    N = len(A)
    A.sort(reverse=True)
    B.sort(reverse=True)

    max_heap = [(-(A[0] + B[0]), 0, 0)]
    visited = set((0, 0))
    result = []

    while len(result) < C:
        total, i, j = heapq.heappop(max_heap)
        result.append(-total)

        if i + 1 < N and (i + 1, j) not in visited:
            heapq.heappush(max_heap, (-(A[i + 1] + B[j]), i + 1, j))
            visited.add((i + 1, j))

        if j + 1 < N and (i, j + 1) not in visited:
            heapq.heappush(max_heap, (-(A[i] + B[j + 1]), i, j + 1))
            visited.add((i, j + 1))

    return result
```

---

### Example

```python
A = [1, 4, 2, 3]
B = [2, 5, 1, 6]
C = 4

max_sum_combinations(A, B, C)  # Output: [10, 9, 9, 8]
```

---

### Time and Space Complexity

- Sorting: $O(N \log N)$
    
- Heap operations: Up to $2C$ pushes, $C$ pops → $O(C \log C)$
    
- **Total time**: $O(N \log N + C \log C)$
    
- **Space**: $O(C)$
    

---

## Problem 9: Merge K Sorted Lists

### Problem Statement

Given a list of $K$ sorted singly linked lists, merge them into a single sorted linked list and return its head.

**Input**:

- A list of $K$ pointers to sorted linked lists.
    

**Output**:

- A single sorted linked list containing all the nodes from the $K$ input lists.
    

---

### Key Insight

This is a classical **heap-based k-way merge** problem.

Use a **min-heap** to repeatedly select the node with the smallest value:

1. Initialize the heap with the **head node** of each list.
    
2. At each step:
    
    - Pop the node with the smallest value.
        
    - Append it to the merged list.
        
    - If the node has a `.next`, push that into the heap.
        

This guarantees the merged list remains sorted.

---

### Clean Python Implementation

```python
import heapq

class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
    def __lt__(self, other):  # necessary for heapq
        return self.val < other.val

def merge_k_lists(lists):
    heap = []
    for node in lists:
        if node:
            heapq.heappush(heap, node)

    dummy = ListNode(0)
    tail = dummy

    while heap:
        smallest = heapq.heappop(heap)
        tail.next = smallest
        tail = tail.next
        if smallest.next:
            heapq.heappush(heap, smallest.next)

    return dummy.next
```

---

### Example

```python
# Helper to build a list from Python list
def build_linked_list(values):
    dummy = ListNode(0)
    cur = dummy
    for v in values:
        cur.next = ListNode(v)
        cur = cur.next
    return dummy.next

# Usage
l1 = build_linked_list([1, 4, 5])
l2 = build_linked_list([1, 3, 4])
l3 = build_linked_list([2, 6])
merged = merge_k_lists([l1, l2, l3])
```

---

### Time and Space Complexity

- Total number of nodes: $N$
    
- Each push/pop in heap: $O(\log K)$
    
- **Total time**: $O(N \log K)$
    
- **Space**: $O(K)$ for the heap
    

---

## Problem 10: Distinct Numbers in a Window

### Problem Statement

Given an integer array $A$ of size $N$ and an integer $B$, return a list of length $N - B + 1$ where each element represents the **number of distinct integers** in the subarray (window) of size $B$ starting at that index.

If $B > N$, return an empty list.

---

### Key Insight

Use a **sliding window** with a **frequency map** (dictionary):

- Initialize the first window and count distinct elements.
    
- As the window slides:
    
    - **Remove** the outgoing element from the frequency map.
        
        - If its count drops to zero, it's no longer in the window.
            
    - **Add** the incoming element.
        
        - If it's a new element, increment the distinct count.
            

This gives an $O(N)$ solution using a hash map to track counts.

---

### Clean Python Implementation

```python
from collections import defaultdict

def distinct_in_window(A, B):
    N = len(A)
    if B > N:
        return []

    freq = defaultdict(int)
    result = []
    distinct = 0

    # First window
    for i in range(B):
        if freq[A[i]] == 0:
            distinct += 1
        freq[A[i]] += 1
    result.append(distinct)

    # Slide the window
    for i in range(B, N):
        out_elem = A[i - B]
        freq[out_elem] -= 1
        if freq[out_elem] == 0:
            distinct -= 1

        in_elem = A[i]
        if freq[in_elem] == 0:
            distinct += 1
        freq[in_elem] += 1

        result.append(distinct)

    return result
```

---

### Example

```python
distinct_in_window([1, 2, 1, 3, 4, 3], 3)
# Output: [2, 3, 3, 2]
```

Explanation:

- Window [1, 2, 1] → 2 distinct
    
- Window [2, 1, 3] → 3 distinct
    
- Window [1, 3, 4] → 3 distinct
    
- Window [3, 4, 3] → 2 distinct
    

---

### Time and Space Complexity

- Each element enters and exits the window exactly once.
    
- **Time**: $O(N)$
    
- **Space**: $O(B)$ for the frequency map
    

---

## Problem 11: LRU Cache

### Problem Statement

Design a **Least Recently Used (LRU) Cache** with the following operations:

- `get(key)`: Return the value if the key exists in the cache, otherwise return `-1`. Accessing a key marks it as most recently used.
    
- `set(key, value)`: Insert or update the `(key, value)`. If the cache exceeds capacity, evict the least recently used item.
    

### Constraints

- All operations should run in **O(1)** average time.
    
- Capacity $C$ is fixed at initialization.
    

---

### Key Insight

Use a combination of two data structures:

1. **Doubly Linked List** (DLL): to store `(key, value)` pairs ordered from **most-recently** to **least-recently** used.
    
2. **Hash Map**: maps each `key` to its corresponding node in the DLL.
    

On every access or insertion:

- Move the node to the **front** of the DLL (most recently used).
    
- If inserting causes overflow, remove the node from the **back** of the DLL (least recently used).
    

---

### Clean Python Implementation (Using `OrderedDict`)

```python
from collections import OrderedDict

class LRUCache:
    def __init__(self, capacity):
        self.cache = OrderedDict()
        self.cap = capacity

    def get(self, key):
        if key not in self.cache:
            return -1
        self.cache.move_to_end(key, last=False)
        return self.cache[key]

    def set(self, key, value):
        if key in self.cache:
            self.cache.move_to_end(key, last=False)
            self.cache[key] = value
        else:
            if len(self.cache) == self.cap:
                self.cache.popitem(last=True)
            self.cache[key] = value
            self.cache.move_to_end(key, last=False)
```

---

### Example Usage

```python
lru = LRUCache(2)
lru.set(1, 10)
lru.set(2, 20)
lru.get(1)      # Returns 10
lru.set(3, 30)  # Evicts key 2
lru.get(2)      # Returns -1 (not found)
```

---

### Time and Space Complexity

- `get`, `set`: $O(1)$ average time (due to `OrderedDict`)
    
- Space: $O(C)$ where $C$ is the cache capacity
    

---

