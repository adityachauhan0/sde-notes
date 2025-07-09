
Given a sorted array, that has been rotated at some unknown pivot, find the minimum element in A.

Ex: 7 2 4 5, output = 2

```python
def findMin(A):
	n = len(A)
	l,r = 0, n - 1
	while l < r:
		m = l + (r-l)//2
		if A[m] > A[r]: #minimum is on the right.
			l = m + 1
		else:
			r = m
	return A[l]
```

## Search in Bitonic Array
Given a bitonic sequence (first increasing, then decreasing) of N distinct elements. Find the index of element B in A in O(logN) time.

If it does not exist, return -1.

Ex: A = 3 9 10 20 17 5 1, B = 20. Output: 3
### How

First find the peak, then do two binary searches each side.

```python
def searchBito(A):
	N = len(A)
	lo = 0; hi = N-1
	while lo < hi:
		mid = (lo + hi) // 2
		if A[mid] > A[mid+1]:
			hi = mid
		else:
			lo = mid + 1
	peak = lo
	#bs in increasing part
	lo , hi = 0, peak
	while lo <= hi:
		mid = (lo + hi)//2
		if A[mid] == B:
			return mid
		if A[mid] < B:
			lo = mid + 1
		else:
			hi = mid - 1
	#bs in decreasing part
	lo,hi = peak + 1, N-1
	while lo <= hi:
		mid = (lo + hi)//2
		if A[mid] == B:
			return mid
		if A[mid] < B:
			hi = mid - 1
		else:
			lo = mid + 1
	return -1
```

## Smaller or equal elements.

Given a sorted array A, and int B, count the number of elements in A that are $\leq$ B.

```python
def leq(A,B):
	N = len(A)
	lo,hi = 0, N-1
	ans = N
	while lo <= hi:
		mid = (lo + hi)//2
		if A[mid] > B:
			ans = mid
			hi = mid - 1
		else:
			lo = mid + 1
	return ans
```

## Wood Cutting Made easy

Given an array of tree heights, and the amount of wood B to be collected. Find the maximum height H such that if you cut all trees taller than H down to height H, you collect at least B metres of wood.

```python
def wood(A,B):
	lo,hi = 0, max(A)
	best = 0
	while lo <= hi:
		mid = (lo + hi)//2
		wood = sum(max(0, ai - mid) for ai in A)
		if wood >= B:
			best = mid
			low = mid + 1 #try higher cut
		else:
			high = mid - 1 #try lower cut
	return best
```

## Matrix Search

Given a $N \times M$ matrix where each row is sorted and first element of each row is $\geq$ last element of previous row.

Given an integer B, determine if B exists in the matrix.

Ex:

| 1   | 3   | 5   | 7   |
| --- | --- | --- | --- |
| 10  | 11  | 16  | 20  |
| 23  | 30  | 34  | 50  |

Flattening it, its just a sorted list.

an index k would correspond to $A[row][col]$ where row = $\lfloor \frac{k}{M} \rfloor$ and col = $k\space mod\space M$

```python
def searchMatrix(A,B):
	n,m = len(A), len(A[0])
	low, high = 0, n*m - 1
	while low <= high:
		mid = (low + high)//2
		row = mid//m
		col = mid % m
		val = A[row][col]
		if val == B:
			return 1
		elif val < B:
			low = mid + 1
		else:
			high = mid - 1
	return 0
```

## Search for a Range.
Given a sorted array, and int B. Find the starting point and ending point of B in A.

Bhai bas upperbound - lowerbound hai.

```python
def searchRange(A,B):
	n = len(A)
	res = [-1,-1]
	lo,hi = 0, n-1
	while lo <= hi: #search for first occurence
		mid = (lo + hi) //2
		if A[mid] < B:
			lo = mid + 1
		elif A[mid] > B:
			hi = mid - 1
		else:
			res[0] = mid
			hi = mid - 1 #keep search better on left
	if res[0] == -1: return res
	#find last occ
	lo, hi= 0, n-1
	while lo <= hi:
		mid = (lo + hi)//2
		if A[mid] < B:
			lo = mid + 1
		elif A[mid] > B:
			hi = mid - 1
		else:
			res[1] = mid
			lo= mid + 1 #keep searching right
	return res
```

## Sorted Insertion Position
Given a sorted array, tell the index where B is found. If not found, where should it be inserted.

```python
def searchInsert(A,B):
	n = len(A)
	lo,hi = 0, n
	while lo < hi:
		mid = (lo + hi) // 2
		if A[mid] < B:
			lo = mid + 1
		else:
			hi = mid
	return lo
```

## Capacity to Ship packages within B days.
Conveyor belt has packages that must be shipped within another B days.

The $i^{th}$ package on the conveyor belt has weight $A[i]$. Each day, we load the packages (not more than the maximum weight capacity).

Return the least weight capacity of the belt such that all the packages can be shipped within B days.

$A = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], B = 5$
Output = 15

```python
def maxCap(A,B):
	lo,hi = max(A), sum(A)
	answer = hi
	def canBeShipped(C):
		days = 1
		currentLoad = 0
		for weight in A:
			if currentLoad + weight <= C:
				currentLoad += weight
			else:
				days += 1
				currentLoad = weight
		return days <= B
	#bs on the capacity
	while lo <= hi:
		cap = (lo + hi)//2
		if canBeShipped(cap):
			answer = mid
			hi = mid - 1
		else:
			lo = mid + 1
	return answer
```

## Matrix Median
Given a matrix $N \times M$, where each row is sorted in non-decreasing order. Find and return the overall median of the matrix.

Use bs on value range, not indices.

Median is just the kth smallest element where $k = \frac{N\times M-1}{2}$

For a guessed value of mid, count how many elements $\leq$ mid using upper_bound in each row, since each row is sorted.

```python
import bisect
def findMedian(matrix):
	n = len(matrix)
	m = len(matrix[0])
	lo = min(row[0] for row in matrix)
	hi = max(row[-1] for row in matrix)
	desired = (n*m+1) // 2
	def notDesiredCount(x):
		count = 0
		for row in matrix:
			#count elements <= mid using upperbound
			count += bisect.bisect_right(row,mid)
		return count < desired
	while lo < hi:
		mid = (lo + hi)//2
		if notDesiredCount(mid):
			lo = mid + 1
		else:
			hi = mid
	return lo
```

## Square root of Integer
Given a non-neg A, compute and return $\sqrt A$ .
```python
def sqrt(A):
	if A < 2:
		return A
	lo,hi = 1, A//2
	ans = 1
	while lo <= hi:
		mid = (lo + hi)//2
		if mid*mid == A:
			return mid
		if mid*mid < A:
			ans = mid
			lo = mid + 1
		else:
			hi = mid - 1
	return ans
```

## Allocate Books
Given N books with pages count, and B students. Allocate books to students such that, books for a student are consecutive. 

Minimize the maximum pages assigned to a student.

A = 12 34 67 90, B = 2. Ans = 113 limit per student.

```python
def allocateBooks(A,B):
	def studentsFit(pageLimit):
		students = 1
		pages = 0
		for p in A:
			if pages + p > pageLimit:
				students += 1
				pages = p
			else:
				pages += p
		return students <= B
	if B > len(A): return -1
	lo = max(A)
	hi = sum(A)
	ans = hi
	while lo <= hi:
		limit = (lo + hi)//2
		if studentsFit(limit):
			ans = limit
			hi = limit - 1
		else:
			lo = limit + 1
	return ans
```

## Painter's Partition 
Given A painters, each taking B units of time per length, and and array of board lengths.

Paint all boards while minimizing the maximum time any painter spends painting.

A = 2, B = 5, C = 1,10 : Answer = 50. (each painter gets one board)

```python
def minTime(A,B,C):
	MOD = 100000003
	if A >= len(C): #each painter paints one board
		return (max(c)*B) % MOD #max length a painter paints.
	def paintersDontExceed(maxLen):
		#if every painter paints at max this length, all boards are painted with the given painters.
		painters = 1
		currSum = 0
		for boardLength in C:
			if currSum + boardLength > maxLen:
				painters += 1
				currSum = boardLength
			else:
				currSum += boardLength
		return painters <= A
	lo,hi = max(C), sum(C)
	best = hi
	while lo <= hi:
		length = (lo + hi) // 2
		if paintersDontExceed(length):
			best = length
			hi = length-1
		else:
			lo = length + 1
	return (best*B) % MOD
```

# Red Zone Detection Problem

## Problem Statement

You're given **N orange zones** on an infinite 2D plane. Each zone is located at coordinate `A[i] = (x_i, y_i)`.  
Every day, each orange zone spreads influence in a circular region with increasing radius $d$ (Euclidean distance).  

A location becomes a **red zone** if it lies within distance $d$ of **at least B** orange zones.

**Objective:**  
Find the **minimum integer value of $d$** such that there exists **at least one** point covered by **at least $B$ orange zones**.

### Constraints:
- $2 \leq B \leq N \leq 100$
- $0 \leq x_i, y_i \leq 10^9$

---

## Mathematical Explanation

Given a fixed distance $r$:

1. For each pair of centers $(P, Q)$, if distance $d \leq 2r$, their circles intersect.
2. Calculate the **intersection points** (at most two) between the two circles.
   - Let $d = |P - Q|$
   - Midpoint: $$ M = \frac{P + Q}{2} $$
   - Height to intersection point:  
     $$ h = \sqrt{r^2 - \left(\frac{d}{2}\right)^2} $$
   - Direction vector from $P$ to $Q$:  
     $$ \vec{u} = \frac{Q - P}{|Q - P|} $$
   - Perpendicular unit vector $\vec{u}^\perp$ is perpendicular to $\vec{u}$
   - Intersection points:  
     $$ M \pm h \cdot \vec{u}^\perp $$

3. Also check each circle center, as a red zone might occur at the center.
4. Count how many circles each candidate point lies within.

---

## Python Code

```python
import math
from typing import List, Tuple
from itertools import combinations

def distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

def circle_intersections(p1, p2, r) -> List[Tuple[float, float]]:
    d = distance(p1, p2)
    if d > 2 * r or d == 0:
        return []
    
    mx = (p1[0] + p2[0]) / 2
    my = (p1[1] + p2[1]) / 2
    dx = (p2[0] - p1[0]) / d
    dy = (p2[1] - p1[1]) / d
    h = math.sqrt(r * r - (d / 2) ** 2)

    px = mx + h * dy
    py = my - h * dx
    qx = mx - h * dy
    qy = my + h * dx

    return [(px, py), (qx, qy)]

def count_covering(point, centers, r) -> int:
    return sum(distance(point, c) <= r + 1e-7 for c in centers)

def check(centers: List[Tuple[float, float]], B: int, r: float) -> bool:
    candidates = list(centers)
    for (p1, p2) in combinations(centers, 2):
        candidates.extend(circle_intersections(p1, p2, r))

    return any(count_covering(pt, centers, r) >= B for pt in candidates)

def min_radius(A: List[List[int]], B: int) -> int:
    centers = [(float(x), float(y)) for x, y in A]
    lo, hi = 0, 2_000_000_000
    answer = hi
    while lo <= hi:
        mid = (lo + hi) // 2
        if check(centers, B, mid):
            answer = mid
            hi = mid - 1
        else:
            lo = mid + 1
    return answer
```

# Modular Power Function: Compute $(x^n) \mod d$

## Problem Statement

Given integers $x$, $n$, and $d$, compute:

$$
(x^n) \mod d
$$

### Constraints:
- $-10^9 \leq x \leq 10^9$
- $0 \leq n \leq 10^9$
- $1 \leq d \leq 10^9$

If the result is negative, convert it to a non-negative number in the range $[0, d-1]$.

---

## Mathematical Insight: Binary Exponentiation

Naive computation of $x^n$ is inefficient for large $n$.  
Instead, use **Exponentiation by Squaring** to reduce time complexity from $O(n)$ to $O(\log n)$.

### Recurrence:
$$
x^n =
\begin{cases}
(x^{n/2})^2 & \text{if } n \text{ is even} \\
x \cdot (x^{(n-1)/2})^2 & \text{if } n \text{ is odd}
\end{cases}
$$

### Key Points:
- Always take modulo $d$ at each step to prevent overflow.
- Convert $x$ to a non-negative base in $[0, d-1]$ before starting.
- Result is always normalized to $[0, d-1]$.

---

## Python Code

```python
def power_mod(x: int, n: int, d: int) -> int:
    base = x % d
    if base < 0:
        base += d

    result = 1
    while n > 0:
        if n & 1:
            result = (result * base) % d
        base = (base * base) % d
        n >>= 1

    return result % d
```


# Simple Queries

## Problem Statement

Given an array $A$ of $N$ integers, perform the following operations:

1. Generate all subarrays of $A$.
2. For each subarray, take its **maximum element** and collect them into array $G$.
3. Replace each element $x$ in $G$ with $f(x)$, the **product of all divisors** of $x$, modulo $10^9 + 7$.
4. Sort $G$ in descending order.

For $Q$ queries, each giving a value $k$, return the $k^{\text{th}}$ element of $G$.

### Constraints:
- $1 \leq N, Q \leq 10^5$
- $1 \leq A[i], k \leq 10^5$

---

## Step 1: Count Occurrences of Each Maximum

For each index $i$ in array $A$:
- Let $L$ = number of consecutive elements to the **left** that are **strictly less than** $A[i]$ (including $i$).
- Let $R$ = number of elements to the **right** (including $i$) that are **less than or equal to** $A[i]$.
- Then the number of subarrays where $A[i]$ is the maximum = $L \times R$

Efficiently compute:
- `prev[i]`: index of previous greater element to the left
- `next[i]`: index of next greater or equal element to the right  
Using **monotonic stacks**

---

## Step 2: Product of Divisors $f(x)$

If the prime factorization of $x$ is:

$$
x = \prod p_i^{a_i}
$$

Then:
- Number of divisors $D = \prod (a_i + 1)$
- Product of divisors:  
  $$ f(x) = \prod p_i^{(D \cdot a_i)/2} $$

To compute $f(x)$:
- Use **smallest prime factor (SPF)** for fast factorization
- Use **modular exponentiation** for large powers

---

## Step 3: Build List $G$ and Answer Queries

1. For each unique $x$ in $A$, calculate how many subarrays it is max in → `count[x]`
2. Compute $f(x)$ for each $x$
3. Each $f(x)$ appears `count[x]` times in $G$
4. Sort $(f(x), count[x])$ by descending $f(x)$
5. Build prefix sums of counts
6. For each query $k$, use binary search over the prefix sums to get answer

---

## Python Code

```python
from typing import List
from collections import defaultdict
import bisect

MOD = 10**9 + 7
MAX_A = 100005

spf = list(range(MAX_A))

def build_spf():
    for i in range(2, MAX_A):
        if spf[i] == i:
            for j in range(i*i, MAX_A, i):
                if spf[j] == j:
                    spf[j] = i

def factorize(x):
    factors = defaultdict(int)
    while x > 1:
        p = spf[x]
        factors[p] += 1
        x //= p
    return factors

def mod_pow(a, b):
    result = 1
    while b > 0:
        if b % 2:
            result = (result * a) % MOD
        a = (a * a) % MOD
        b //= 2
    return result

def product_of_divisors(x):
    factors = factorize(x)
    D = 1
    for a in factors.values():
        D *= (a + 1)
    result = 1
    for p, a in factors.items():
        exp = (D * a) // 2
        result = (result * mod_pow(p, exp)) % MOD
    return result

def solve(A: List[int], queries: List[int]) -> List[int]:
    n = len(A)
    build_spf()
    
    prev = [-1] * n
    next_ = [n] * n

    stack = []
    for i in range(n):
        while stack and A[stack[-1]] < A[i]:
            stack.pop()
        if stack:
            prev[i] = stack[-1]
        stack.append(i)

    stack = []
    for i in range(n-1, -1, -1):
        while stack and A[stack[-1]] <= A[i]:
            stack.pop()
        if stack:
            next_[i] = stack[-1]
        stack.append(i)

    count = defaultdict(int)
    for i in range(n):
        left = i - prev[i]
        right = next_[i] - i
        count[A[i]] += left * right

    freq = []
    for x in count:
        fx = product_of_divisors(x)
        freq.append((fx, count[x]))

    freq.sort(reverse=True)

    prefix = []
    total = 0
    for val, cnt in freq:
        total += cnt
        prefix.append((total, val))

    result = []
    for k in queries:
        idx = bisect.bisect_left(prefix, (k, -1))
        result.append(prefix[idx][1])

    return result
```

# Median of Two Sorted Arrays

## Problem Statement

Given two sorted arrays $A$ and $B$ of lengths $m$ and $n$, compute the **median** of the merged array in $O(\log(\min(m, n)))$ time.

If the total number of elements is even, return the average of the two middle elements.

### Constraints:
- $0 \leq |A|, |B| \leq 10^6$
- $1 \leq |A| + |B| \leq 2 \cdot 10^6$

---

## Approach

Naively merging and sorting the arrays takes $O(m+n)$ time, which is too slow.  
Instead, use **binary search on the smaller array** to find the correct partition:

Let $i$ be the partition index in $A$, and $j$ in $B$, such that:
$$
i + j = \left\lfloor \frac{m+n+1}{2} \right\rfloor
$$

Let:
- $a_{\text{left}} = A[i-1]$ or $-\infty$ if $i=0$
- $a_{\text{right}} = A[i]$ or $+\infty$ if $i=m$
- $b_{\text{left}} = B[j-1]$ or $-\infty$ if $j=0$
- $b_{\text{right}} = B[j]$ or $+\infty$ if $j=n$

The correct partition satisfies:
$$
a_{\text{left}} \leq b_{\text{right}} \quad \text{and} \quad b_{\text{left}} \leq a_{\text{right}}
$$

- If true, we can compute the median:
  - If total is odd:  
    $$ \max(a_{\text{left}}, b_{\text{left}}) $$
  - If total is even:  
    $$ \frac{\max(a_{\text{left}}, b_{\text{left}}) + \min(a_{\text{right}}, b_{\text{right}})}{2} $$

- If $a_{\text{left}} > b_{\text{right}}$, move $i$ left.
- If $b_{\text{left}} > a_{\text{right}}$, move $i$ right.

---

## Python Code

```python
def findMedianSortedArrays(A, B):
    if len(A) > len(B):
        A, B = B, A

    m, n = len(A), len(B)
    total = m + n
    half = (total + 1) // 2

    lo, hi = 0, m
    while lo <= hi:
        i = (lo + hi) // 2
        j = half - i

        a_left = float('-inf') if i == 0 else A[i - 1]
        a_right = float('inf') if i == m else A[i]
        b_left = float('-inf') if j == 0 else B[j - 1]
        b_right = float('inf') if j == n else B[j]

        if a_left <= b_right and b_left <= a_right:
            if total % 2 == 1:
                return max(a_left, b_left)
            else:
                return (max(a_left, b_left) + min(a_right, b_right)) / 2
        elif a_left > b_right:
            hi = i - 1
        else:
            lo = i + 1

    return 0.0
```

# Rotated Sorted Array Search

## Problem Statement

Given a rotated sorted array $A$ (with distinct integers) and an integer $B$,  
find the index of $B$ using $O(\log N)$ time, or return $-1$ if $B$ is not present.

---

## Key Insight

Although the array is rotated, **at least one half** of any subarray $[lo, hi]$ is **sorted**.  
This allows us to perform binary search.

---

## Binary Search Strategy

1. Compute $mid = \left\lfloor \frac{lo + hi}{2} \right\rfloor$
2. If $A[mid] == B$, return $mid$
3. Otherwise, check which half is sorted:
   - **If** $A[lo] \leq A[mid]$ (left half sorted):
     - If $A[lo] \leq B < A[mid]$: search **left**
     - Else: search **right**
   - **Else** (right half sorted):
     - If $A[mid] < B \leq A[hi]$: search **right**
     - Else: search **left**

---

## Python Code

```python
def search_rotated_array(A, B):
    lo, hi = 0, len(A) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if A[mid] == B:
            return mid
        if A[lo] <= A[mid]:  # Left half is sorted
            if A[lo] <= B < A[mid]:
                hi = mid - 1
            else:
                lo = mid + 1
        else:  # Right half is sorted
            if A[mid] < B <= A[hi]:
                lo = mid + 1
            else:
                hi = mid - 1
    return -1
```
