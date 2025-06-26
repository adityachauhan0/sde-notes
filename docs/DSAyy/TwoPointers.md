## Pair with Given Difference
Given 1D arr `A` of `N` integers, and an int `B`, determine if there is a pair of elements whose difference in B. 

A = 5 10 3 2 50 80, B = 78. Ans? Yes

Keep a hash set, just $O(n)$ time complexity.

```python
def pairSum(A,B):
	seen = set()
	for x in A:
		if (x-B) in seen or (x+B) in seen:
			return 1
		seen.insert(x)
	return 0
```


## 3 Sum closest.

Given an arr `A`, find 3 int in A such that their sum is the closest to a given num `B`.
Return sum of those 3 int.

A = -1 2 1 -4, B = 1. Output = 2

### How
Sort the arr and use 2 pointers method.
Iterate one pointer, `l` and `r` the other 2 to moderate.

```python
def threeSumClosest(A,B):
	sort(A)
	N = len(A)
	best = A[0] + A[1] + A[2]
	for i in range(0,N-2):
		l = i+1, r = N-1
		while l < r:
			sm = A[i] + A[l] + A[r]
			if sm == B: return B
			if abs(sm - B) < abs(best - B): best = sm
			if sm < B: l += 1
			if sm > B: r -= 1
	return best
```

## Counting Triangles
Given arr `A` of non-neg numbers. Each $A_i$ represents length of a line segment. Count num of distinct triangles that can be formed from these edge lengths.

A = 1 1 1 2 2
Output: 4

### How
$$
a + b > c
$$
We just have to find all values that satisfy this.
- Sort the arr
- For each pos largest side $A_k$ , use two pointers
	- i starts from 0, j from k-1
	- for each k, count pairs ($A_i,A_j$) such that $A_i + A_j  > A_k$ 
	- if thats true, all ind from i to j-1 also satisfy the condition.
```python
def nTriang(A):
	sort(A)
	count = 0
	for k in range(N-1,1,-1): #n-1 down to 2
		i,j = 0, k-1
		while i < j:
			if A[i] + A[j] > A[k]:
				count += (j-i)
				j -= 1
			else:
				i += 1
	return count % (10**9 + 7)
```


## Diffk (Find pair with given diff in a sorted arr)
Given a **sorted** arr A of int, and non neg `B`. Find whether there exists two indices sucht that their el diff is B
$$
A[j] - A[i] = B,\space i \neq j
$$
Return 1 if exists warna 0

```python
def diffPossible(A,B):
	i, j = 0 , 1
	N = len(A)
	while i < N and j < N:
		if i != j:
			diff = A[j] - A[i]
			if diff == B: return 1
			if diff < B: j += 1
			else : i += 1
		else:
			j += 1
	return 0
```


## Max 1's after modification. (sliding window)
Given bin arr `A`, and an int `B`. Find length of longest subseg of consec 1's you can obtain, by changing at most B zeroes in A to ones.

```python
def maxOnes(A,B):
	N = len(A)
	l = 0
	zeroes = 0
	best = 0
	for right in range(N):
		if A[right] == 0:
			zeroes += 1
		while zeroes > B:
			if A[left] == 0:
				zeroes -= 1
			left += 1
		best = max(best, right - left + 1)
	return best
```

## Counting Subarrays
Given array of non-neg arr, and a non-neg int `B`, find num of subarrs whose sum $<$ B

A = 2 5 6, B = 10
 Ans = 4 (2 5 6 (2 5) )

### How
Sliding window baby
If the window is valid, count += (right - left + 1)

```python
def countSubarrs(A,B):
	left,sm,count = 0
	for right in range(N):
		sm += A[right]
		while left <= right and sm >= B:
			sm -= B
			left += 1
		count += (right - left + 1)
	return count
```


## Subarrs with Distinct Int
Given an arr with pos int, count subarrays (continuous) which are good. Good meaning num of distinct int is exactly `B`. Return the number of good subarrays.

```python
from collections import defaultdict
def subarrays_with_k_distinct(A,K):
	return countAtMostK(A,K) - countAtMostK(A,K-1)
def countAtMostK(A,K):
	freq = defaultdict(int)
	left,count,distinct = 0
	for right in range(len(A)):
		if freq[A[right]] == 0:
			distinct += 1
		freq[A[right]] += 1
		while distinct > K:
			freq[A[left]] -= 1
			if freq[A[left]] == 0:
				distinct -= 1
			left += 1
		count += right - left + 1
	return count
```

## Max Cont Series of 1s
Given binary array, find **max sequence of continuous 1s**, that can be formed by replacing atmost `B` zeroes with ones.

Return the indices of max cont series of 1s in order. If multiple soln exists, return the sequence with min starting index.

A = 1 1 0 1 1 0 0 1 1 1, B = 1, Output = (0,1,2,3,4)

### How
Sliding window. Window is valid is zero_count > B
```python
def maxone(A,B):
	left = zeroes = 0
	bestLen = bestLeft = 0
	for right in range(len(A)):
		if A[right] == 0:
			zeros += 1
		while zeros > B:
			if A[left] == 0:
				zeros -= 1
			left += 1
		if right - left + 1 > bestLen:
			bestLen = right - left + 1
			bestLeft = left
	ans = [bestLeft + i for i in range(bestLen)]
	return ans
```

## Array 3 Pointers
Given 3 sorted arrays, A, B, C. Find indices i,j,k such that
$$
max(|A[i] - B[j]|, |B[j] - C[k]|, |C[k] - A[i]|)
$$
is minimized.

Return the min value.

### How?
take i,j,k = 0, then calc the curMax - curMin, of all elements.

Advance the ptr to the min element, since we need to decrease the gap.

```python
def minimize(A,B,C):
	i = j = k = 0
	ans = 10**18
	while i < len(A) and j < len(B) and k < len(C):
		x,y,z = A[i], B[j], C[k]
		curMax = max(x,y,z)
		curMin = min(x,y,z)
		ans = min(ans, curMax - curMin)
		if curMin == x: i += 1
		else if curMin == y: j +=1 
		else: k += 1
	return ans
```


## Container With Most Water (Trapping rain water)
Given arr of non-neg int `A`, where  $A_i$ is a wall's height. Find the area of most water you can contain in this.

A = 1 5 4 3, Output = 6 (trap between 5 and 3)

### How
1. left = 0, right = n-1
2. Area between left and right is: height =  $min(a_{left},a_{right})$, width = right - left.
3. Move pointer of the shorter wall inwards, since it is holding us back.
4. Repeat until left $\leq$ right.

```python
def maxArea(A):
	left,right = 0, len(A) - 1
	maxAr = 0
	while left < right:
		height = min(A[left],A[right])
		width = right - left
		maxAr = max(maxAr, height * width)
		if A[left] < A[right]: left += 1
		else: right -= 1
	return maxAr
```

## Merge Two Sorted Lists II
Given two sorted arrs, modify first one inplace to contain the merged sorted arr of both.

### How
1. Expand A to size $m + n$ 
2. Merge from the end, to avoid overwriting in A that hasn't been moved.
3. Use 3 pointers:
	1. i = m-1 (end of old A)
	2. j = n-1 (end of B)
	3. k = m+n -1 (end of expanded A)
4. Compare `A[i]` and `B[j]`, put the larger at `A[k]`.
5. If B has more elements left, just copy them over.
```python
def merge(A,B):
	m,n = len(A), len(B)
	i,j,k = m-1,n-1,m+n - 1
	while i>= 0 and j >= 0:
		if A[i] > B[j]:
			A[k] = A[i]
			i -= 1
		else:
			A[k] = B[j]:
			j -= 1
		k -= 1
	while j >= 0:
		A[k] = B[j]
		k -= 1
		j -= 1
```

## Intersection of Sorted Arrays
Given two sorted arrays A and B, Find their intersection and preserve element's frequencies.

### How
1. i = 0, j = 0
2. if `A[i]` < `B[j]`: incr i
3. `A[i]` > `B[j]`, incr j
4. equal? add to the res
```python
def intersect(A,B):
	i=j=0
	res = []
	while i < len(A) and j < len(B):
		if A[i] < B[j]:
			i += 1
		elif A[i] > B[j]:
			j += 1
		else: 
			res.append(A[i])
			i += 1
			j += 1
	return res
```

## Remove Duplicated From Sorted Array
Remove all duplicates inplace from the sorted array, and return the length of sorted array with only distinct elements. Also update A inplace.

### How
Use 2 pointers, read and write. First read then write.

```python
def removeDuplicates(A):
	if not A: return 0
	write = 1 #where we write next unique
	for read in range(1,n):
		if A[read] != A[read -1]: #whenever we see new value
			A[write] = A[read]
			write += 1
	return write
```

## Remove Duplicates From Sorted Array II
Remove duplicates from sorted array in-place, so that each element appears atmost **twice**. Return the new length. Update A in-place and return the new length.

A = 1 1 1 2, it becomes A = 1 1 2, and output is 3

### How
1. Duplicates are adjacent.
2. Use a write ptr
3. For every $A[read] \neq A[write - 2]$ (current el is not equal to two elements before it), copy it to $A[write]$ 
```python
def removeDuplicatesII(A):
	n = len(A)
	if n <= 2: return n
	write = 2
	for read in range(2,n):
		if A[read] != A[write -2]:
			A[write] = A[read]
			write += 1
	return write
```

## Remove Element From Array
Remove all instances of `B` from `A`. Update array in place, and return the num of elements left after operation.

if $A[read] \neq B$ , then $A[write] = A[read]$ 
```python
def removeElement(A,B):
	write = 0
	for read in range(len(A)):
		if A[read] != B:
			A[write] = A[read]
			write += 1
	return write
```

## Sort by Color (dutch national flag problem)

Sort the array, consisting only of 0 1 2. 

Let `low` be boundary between 0 and 1, let mid be cur el, right be boundary between 1 and 2

```python
def sortColors(A):
	low = mid = 0
	high = len(A) - 1
	while mid <= high:
		if A[mid] == 0:
			A[low],A[mid] = A[mid],A[low]
			low += 1
			mid += 1
		elif A[mid] == 1:
			mid += 1
		else:
			A[mid],A[high] = A[high],A[mid]
			high -= 1
```
