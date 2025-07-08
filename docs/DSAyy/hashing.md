## Colorful Numbers
Given an int A, determine whether its colorful or not.

Number if colorful if product of digits of every contiguous subsequence is unique.

### How??
Slide a window of size k over the digits and multiply as you go. If two windows ever yield the same product, the number is not colorful.

```python
def is_colorful(A):
	digits = [int(d) for d in str(A)]
	seen = set()
	for i in range(len(digits)):
		prod = 1
		for j in range(i,len(digits)):
			prod *= digits[j]
			if prod in seen:
				return 0
			seen.add(prod)
	return 1
```

## Largest Continuous Sequence Zero Sum
Given an $int$ array, find the longest contiguous subarray whose elements sum to zero. If multiple exist, return the one that appears first.

### How
Compute the prefix sum, and store the first index where each sum occurs in a hash map. Whenever the same sum reappears at index $j$, the subarr $(firstIdx  + 1) .. j$ sums to zero.

Track the maximum length.

```python
def longest_zero_sum_subarray(A):
	first_occ = {0: -1}
	total, best_len, best_start = 0,0,0
	for i in range(len(A)):
		total += A[i]
		if total in first_occ:
			length = i - first_occ[total]
			if length > best_len:
				best_len = length
				best_start = first_occ[total] + 1
		else:
			first_occ[total] = i
	return A[best_start:best_start + best_len]
```

## Longest Subarray Length
Given a binary array, find the length of the longest contiguous subarray in which the number of 1's is exactly one more than number of 0's.

### How
Treat each 1 as +1, and each 0 as -1. Compute the prefix sum.

Subarray $[j+1...i]$ has $cnt(1) - cnt(0) = 1$ when $S[i] - S[j] = 1$, or $S[j] = S[i] - 1$

Store in a hash_map the earliest index where each prefix sum occurs. At every index, look for $S[i] - 1$ to extend the longest valid segment ending at i.

```python
def longest_subarray_len(A):
	first_occ = {0: -1}
	total = 0
	max_len = 0
	for i in range(len(A)):
		total += 1 if A[i] == 1 else -1
		if (total - 1) in first_occ:
			max_len = max(max_len, i - first_occ[total - 1])
		if total not in first_occ:
			first_occ[total] = i
	return max_len
```

## First Repeating Element.
Find the first repeating element in an array.

```python
def first_repeating(A):
	seen = set()
	candidate = None #will hold the left-most repeating value
	#scan right to left
	for x in reversed(A):
		if x in seen:
			candidate =x
		else:
			seen.add(x)
	return candidate if candidate is not None else -1
```

## Two Sum
Given an int array, and a target B, find two distinct elements whose sum is $B$. Return their 1-based indexes, $i < j$.  If multiple exists, choose the pair with the smallest $j$: if tied, smallest i. 
Nahi mila toh return empty list.

```python
def two_sum(A,B):
	first_pos = {}
	for j in range(len(A)):
		want = B - A[j]
		if want in first_pos:
			return [first_pos[want] + 1, j+1]
		if A[j] not in first_pos:
			first_pos[A[j]] = j
	return []
```

## 4 Sum
Given an int array, and a target B. Find all unique quadruplets $(a,b,c,d)$ in A such that $a + b + c + d = B$, where $a \leq b \leq c \leq d$.

Return the list in lex order without duplicates.

Sort arr, two nested loops, remaining use two pointer.

```python
def fourSum(A,B):
	A.sort()
	A = len(A)
	res = []
	for i in range(n-3):
		if i > 0 and A[i] == A[i-1]:
			continue
		for j in range(i+1,n-2):
			if j > i+1 and A[j] == A[j-1]:
				continue
			left,right = j+1,n-1
			while left < right:
				total = A[i] + A[j] + A[left] + A[right]
				if total == target:
					res.append(A[i],A[j],A[left], A[right])
					while left < right and A[left] == A[left + 1]:
						left += 1
					while left < right and A[right] == A[right-1]:
						right -= 1
					left += 1
					right -=1
				elif total < target:
					left += 1
				else: right -=1
	return res
```

## Valid Sudoku

Given a 9 \times 9 sudoku board filled with digits from `1` to `9`, and empty cells as `.`. Determine if the filled cells form a valid sudoku.

Just no row and col duplicates, and each $3 \times 3$ sub box has no duplicate digits.

### How
$3 \times 3$ boxes can be tracked using bitmasks. Same for rows and cols.

Boxes can be indexed as $b = 3 \times (i/3) + (j/3)$.

```python
def is_valid_sudoku(A):
	rows,cols,boxes = [0]*9,[0]*9,[0]*9
	for i in range(9):
		for j in range(9):
			c = A[i][j]
			if c == '.':
				continue
			d = int(c) -1 #convert 1-9 to 0-8
			mask = 1 << d
			b = 3*(i//3) + (j//3)
			if rows[i] & mask or cols[j] & mask or boxes[i] & mask:
				return 0
			rows[i] |= mask
			cols[j] |= mask
			boxes[b] |= mask
	return mask
```

## Diffk II
Given an int arr, and a int B, find if there are two distinct indices i and j, such that $A[i] - A[j] = B$.

Return 1 if such a pair exists, otherwise 0.

### How
For every x in A:
 Just check in the hashset if you have seen $x+B$ or $x-B$

```python
def diffk(A,B):
	seen = set()
	for x in A:
		if (x+B) in seen or (x-B) in seen:
			return 1
		seen.insert(x)
	return 0
```

## Pairs with given XOR.
Given an int arr, and int B, count the unique pairs $(x,y)$ in A such that $x \oplus y = B$ 

```python
def count_pairs_xor(A,B):
	s = set(A)
	count = 0
	for x in A:
		y = x^B
		if y in S and x < y:
			count += 1
	return count
```

## Anagrams
Given an array of lowercase strings, return all groups of anagrams. Represent the group by a list of 1-based indices in A. Within each group, maintain the original relative order of strings. Group themselves should be ordered by the first occurence of any member.

### How
Two strings are anagrams iff they ahve identical character counts.

For each string, compute a signature encoding of its letter frequencies. Use hashmap from signature to list of idx. 

Track the order in which distinct signatures first appear to output groups in correct order.

```python
def group_anagrams(A):
	from collections import defaultdict
	groups = defaultdict(list)
	keys = []
	for i,word in enumerate(A):
		count = [0]*26
		for c in word:
			count[ord(c) - ord('a')] += 1
		sig = tuple(count) #use tuple of count as signature
		if sig not in groups:
			keys.append(sig)
		groups[sig].append(i+1) #store 1-based
	result = []
	for sig in keys:
		result.append(groups[sig])
	return result
```

## Equal
Given an int array, find indices a,b,c,d such that: 
$$
	A[a] + A[b] = A[c] + A[d]
$$
and $a < b\space c < d\space a < c \space b \neq c,\space b \neq d$
Return the lexicographically smallest one.

```python
def find_equal_quadruple(A):
	first_pair = {}
	best = []
	n = len(A)
	for i in range(n-1):
		for j in range(i+1,n):
			s = A[i] + A[j]
			if s not in first_pair:
				first_pair[s] = (i,j)
			else:
				p,q = first_pair[s]
				if p < i and q != i and q != j:
					cand = [p,q,i,j]
					if not best or cand < best:
						best = cand
	return best
```

## Copy List with Random Pointer
A singly LL with an extra random pointer (may be null or any node). Return a deep copy of the list. 

```python
class Node:
	def __init__(self,label,next = None, random = None):
		self.label = label
		self.next = next
		self.random = random
def copy_random_list(head):
	if head is None:
		return None
	#interleave copied nodes with og nodes
	cur = head
	while cur:
		copy = Node(cur.label)
		copy.next = cur.next
		cur.next = copy
		cur = copy.next
	#assign random pointers
	cur = head
	while cur:
		if cur.random:
			cur.next.random = cur.random.next
		cur = cur.next.next
	#separate og and copied nodes
	pseudo_head = Node(0)
	copy_iter = pseudo_head
	cur = head

	while cur:
		copy = cur.next
		cur.next = copy.next
		copy_iter.next = copy
		copy_iter = copy
		cur = cur.next
	return pseudo_head.next
```

## Check palindrome possible
Given a lowercase string, check if it can be rearranged to form a palindrome.

```python
def can_form_palindrome(A):
	freq = [0]*26
	for c in A:
		freq[ord(c) - ord('a')] += 1
	odd_count = 0
	for count in freq:
		if count % 2 == 1:
			odd_count += 1
			if odd_count > 1:
				return 0
	return 1
```

## Fraction to Recurring Decimal
Given two integers A and B (numerator and denominator), return their fraction in string form. If the decimal part is repeating, enclose the repeating sequence in parenthesis.

```python
def fraction_to_decimal(A,B):
	if A == 0:
		return "0"
	res = []
	sign = "-" if (A < 0) ^ (B < 0) else ""
	num,den = abs(A),abs(B)
	q = num // den
	res.append(str(q))
	rem = num % den
	if rem == 0:
		return sign + "".join(res)
	res.append(".")
	seen = {}
	while rem != 0:
		 if rem in seen:
			 pos = seen[rem]
			 rem.insert(pos, "(")
			 res.append(")")
			 break
		seen[rem] = len(res)
		rem *= 10
		d = rem // den
		res.append(str(d))
		rem %= den
	return sign + "".join(res)
```

## Points on the Straight Line
Given N points $(A[i], B[i])$ on the 2D plane, find maximum points that lie on the same straight line.

### How
The trick is to generate a unique slope key.

For two points ($x_i,y_i$) and ($x_j,y_j$), let $dx = x_j - x_i$ and $dy = y_j - y_i$.

If both dx and dy are zero, they coincide, otherwise compute $g = gcd(dy,dx)$, and $dy` = \frac{dy}{y}$ and $dx` = \frac{dx}{y}$ .
Normalize the sign so that $dx`$ is $\geq 0$ . Represent vertical lines as (1,0) and horizontal lines as (0,1). 

Now we use $dy`, dx`$ pair as unique slope key.

```python
from math import gcd
from collections import defaultdict
def max_points_on_line(A,B):
	n = len(A)
	if n < 2: return n
	result = 1
	for i in range(n):
		slope_count = defaultdict(int)
		duplicates = 1
		local_max = 0
		for j in range(i+1,n):
			dx = A[j] - A[i]
			dy = B[j] - B[i]
			if dx == 0 and dy == 0:
				duplicates += 1
			else:
				g = gcd(dx,dy)
				dy //= g
				dx //= g
				if dx < 0:
					dx = -dx
					dy = -dy
				if dx == 0:
					dy = 1
				if dy == 0:
					dx = 1
				slope_count[(dy,dx)] += 1
				local_max(local_max, slope_count[(dy,dx)])
		result = max(result, local_max + duplicates)
	return result
```

## An Increment Problem

Given a stream of integers, for each incoming element $x$, if it has appeared before, increment the first occ by $1$, then append the new $x$ to the end. Return the final stream after processing all arrivals.

Maintain :

- a dynamic arr for stream.

- For each val $v$, a min-heap of indices in stream where $v$ currently resides.
When x arrives:

- If heap for $x$ is nonempty, pop the smallest index $j$, increment the stream there, and push $j$ into the heap for $x+1$

- Append $x$ to the end at index $k$, and push $k$ into heap for $x$.

```python
from collections import defaultdict
import heapq
def process_increment_stream(A):
	stream = []
	idx_heap = default_dict(list)
	for x in A:
		if idx_heap[x]:
			j = heapq.heappop(idx_heap[x])
			stream[j] += 1
			heapq.heappush(idx_heap[x+1],j)
		k = len(stream)
		stream.append(x)
		heapq.heappush(idx_heap[x],k)
	return stream
```

## Subarray with Given XOR
Given int array $A$ and an int $B$, count the subarrays whose bitwise xor of elements equals $B$.

### How
Bhai prefix sum toh xor ka bhi hota.
$$
xor(A[L..R]) = prefix[R] \oplus prefix[L]
$$

```python
from collections import defaultdict

def count_subarrays_with_xor(A,B):
	freq = defaultdict(int)
	freq[0] = 1
	prefix = 0
	ans = 0
	for x in A:
		prefix ^= x
		need = prefix ^ B
		ans += freq[need]
		freq[prefix] += 1
	return ans
```

## Two out of Three
Given three int arrays, A B and C, return a sorted list of all numbers that appear in atleast two of the arrays. Arrays may contain duplicates internally, but each value is counted at mmost once per array.

### How
Use a small integer mask per value to record which of the tree arrays it appears in. bit 0 for A, 1 for B and samajh le. Scan every array once, OR-ing appropriate bit into $mask[x]$. Finally any value whose mask has atleast two bits set belongs to the answer.

```python
def two_out_of_three(A,B,C):
	max_val = max(max(A), max(B), max(C), 0)
	mask = [0]*(max_val + 1)
	for x in set(A):
		mask[x] |= 1
	for x in set(B):
		mask[x] |= 2
	for x in set(C):
		mask[x] |= 4
	result = []
	for x in range(1,max_val + 1):
		if bin(mask[x]).count('1') >= 2:
			result.append(x)
	return result
```

## Substring Concatenation
Given a string S of length n, and list L of m words, each of length k. Find all starting indices in S where a substring is formed by adding each word in L exactly once without any gaps. Return the list of indices in any order.

- This algorithm finds all starting indices in `S` where a concatenation of all words in `L` occurs **exactly once** and **without any intervening characters**.
    
- Uses a **sliding window** technique with an offset loop to handle all alignment positions modulo `k`.
    
- `target` counts how many times each word is expected.
    
- `window` tracks how many times each word appears in the current sliding window.
### How
Simple sliding window.

```python
from collections import defaultdict, Counter
def find_substring_concatenation(S,L):
	if not S or not L or not L[0]:
		return []
	n,m,k = len(S), len(L), len(L[0])
	target = counter(L)
	result = []
	for r in range(k):
		window = defaultdict(int)
		count = 0
		left = r
		for right in range(r,n-k+1,k):
			w = S[right:right+k]
			if w not in target:
				window.clear()
				count = 0
				left = right + k
			else:
				window[w] += 1
				if window[w] <= target[w]:
					count += 1
				else:
					while window[w] > target[w]:
						w_left = S[left:left + k]
						window[w_left] -= 1
						if window[w_left] < target[w_left]:
							count -= 1
						left += k
				if count == m:
					result.append(left)
					w_left = S[left:left + k]
					window[w_left] -= 1
					count -= 1
					left += k
	return result
```


## Subarray with exactly B odd numbers

Given an int array, and an int B, count how many subarrays contain exactly B odd numbers.

```python
def count_subarrays_with_b_odds(A,B):
	n = len(A)
	odd_pos = [i for i,x in enumerate(A) if x % 2 == 1]
	#special case: B == 0, count all even only subarrays
	if B == 0:
		ans = 0
		length = 0
		for x in A:
			if x % 2 == 0:
				length += 1
			else:
				ans += length*(length+1)//2
				length = 0
		ans += length*(length + 1) // 2
		return ans
	m = len(odd_pos)
	if m < B:
		return 0
	ans = 0
	for i in range(1,m-B+2):
		L = odd_pos[i] - odd_pos[i-1]
		R = odd_pos[i+B] - odd_pos[i+B-1]
		ans += L*R
	return ans
```

## Minimum Window Substring
Given a string S of length n and a pattern T of length m. Find the smallest substring (window) of S that contains all characters of T.
If no such window, return empty string. If multiple exists, return the one with smallest starting index.

### How
Sliding Sliding maa chod dunga google ki.
```python
from collections import Counter, defaultdict
def min_window_substring(S,T):
	n,m = len(S), len(T)
	if n < m:
		return ""
	need = Counter(T)
	have = defaultdict(int)
	required = len(need)
	formed = 0
	l = r =0
	min_len = float('inf')
	min_l = 0
	while r < n:
		c = S[r]
		have[c] += 1
		if c in need and have[c] == need[c]:
			formed += 1
		while l <= r and formed == required:
			window = r - l + 1
			if window_len < min_len:
				min_len = window_len
				min_l = l
			d = S[l]
			have[d] -= 1
			if d in need and have[d] < need[d]:
				formed -= 1
			l += 1
		r += 1
	return "" if min_len == float('inf') else S[min_l:min_l + min_len]
```

## Longest Substring without repeating characters.

```python
def longest_unique_substring(S):
	last = [-1]*256 #assuming ASCII
	start = best = 0
	for i,c in enumerate(S):
		ascii_c = ord(c)
		start = max(start, last[ascii_c] + 1)
		best = max(best, i - start + 1)
		last[ascii_c] = i
	return best
```
