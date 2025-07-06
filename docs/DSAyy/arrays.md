# Array Simulation
## Spiral Order Matrix
### Question kya hai

Matrix A: Size M x N, return all elements in the spiral order. (clockwise starting from top-left)

Matrix A:

| 1 | 2 | 3 |
|:-:|:-:|:-:|
| 4 | 5 | 6 |
| 7 | 8 | 9 |

the output would be $1,2,3,6,9,8,7,4,5$

### How to do this

Take 4 pointers and continuously run for loops on that bitch. 
Bas run top first, then right, then down, then left

```cpp
vector<int> spiralOrder(vector<vector<int>> &A){
	int M = A.size(), N = A[0].size();
	int u = 0, d = M-1, l = 0, r = N-1;
	vector<int> spiral;
	while (l <= r && u <= d){
		for (int i = l; i <= r; ++i)
			spiral.push_back(A[u][i]);
		++u;
		for (int i = u; i <= d; ++i)
			spiral.push_back(A[i][r]);
		--r;
		if (u <= d){
			for (int i = r; i >= l; --i)
				spiral.push_back(A[d][i]);
			--d;
		}
		if (l <= r){
			for (int i = d; i >= u; --i)
				spiral.push_back(A[i][l]);
			++l;
		}
	}
	return spiral;
}
```

Iski time complexity is $O(n \times m)$
Space complexity bhi same

## Large Factorial
### Question
Given integer A,
compute A ! as a string, coz kuch zyaada hi bada number hai.

### Kaise karna hai ye

Dekh bro as a string return karna hai answer toh legit make a multiply function for strings and karle solve. Kya hi dumb shit hai ye.
Just know ki digits would be reversed for the convenience of the carry shit. 
Toh reverse pointer se string meh add kariyo.

```cpp
string factorial(int A){
	vector<int> digits {1}; // har factorial meh 1 toh hota hi hai
	
	auto multiply = [&](int i) {
		int carry = 0;
		for (int &d : digits){
			long long prod = (long long)d * i + carry;
			d = prod % 10; // same time digit update kar diya
			carry = prod / 10;
		}
		while (carry){
			digits.push_back(carry % 10);
			carry /= 10;
		}
	};
	
	for (int i = 2; i <= A; ++i) // multiply sabkuch from 2 to A
	{
		multiply(i); // multiple every number into 2
	}
	string s;
	// put all the digits into a string
	for (auto it = digits.rbegin(); it != digits.rend(); ++it){
		s.push_back('0' + *it); 
	}
	return s;
}
```

## Max Non-Negative Subarray
### Question kya hai
Array A of N integers, find the subarray with max sum.
agar tied, choose the longer one.
still tied? smallest starting index

Sunn BEHENCHOD, Subarray means continuous, sab kuch subsequence nahi hota
### Karna kaise hai

kadane kadane khelenge
agar negative number mila, that is where we stop and process the answer.
By process i mean, bas compare karke check karlenge if its max

End meh bhi ek baar check kar lena coz when the loop ends, ek baar remaining computation bhi toh update karni hai.

```cpp
vector<int> maxSet(vector<int> &A){
	int n = A.size();
	long long bestSum = -1, curSum = 0;
	int bestStart = 0, bestEnd = -1, bestLen = 0;
	int curStart = 0;
	for (int i = 0; i < n; ++i){
		if (A[i] >= 0)
			curSum += A[i];
		else {
			int curLen = i - curStart;
			if (curSum > bestSum || (curSum == bestSum && curLen > bestLen)){
				bestSum = curSum;
				bestStart = curStart;
				bestEnd = i - 1;
				bestLen = curLen;
			}
			curSum = 0;
			curStart = i+1;
		}
	}
	if (curStart < n){
		int curLen = n - curStart;
		if (curSum > bestSum || (curSum == bestSum && curLen > bestLen)){
			bestSum = curSum;
			bestStart = curStart;
			bestEnd = n - 1;
			bestLen = curLen;
		}
	}
	if (bestEnd < bestStart) return {};
	return vector<int>(A.begin()+bestStart, A.begin()+bestEnd + 1);
}
```

Time complexity is O(n), space complexity is O(1)


## Pick from Both Sides
Array A of N elements. Pick exactly B elements from either left ya right end, and just get the max sum.

### Karna kaise hai
Imagine kar ek sexy sa sliding window, but instead on inside the array, ye saala bahar se aa raha hai.
like the right pointer is left meh and left wala is right meh.
ye leke bas max sum with B elements karle.
Start the right pointer at B - i, and keep the left wala at n - i, and baaju baaju shift and update karte ja.
Keep a sum of first B elements, and fir middle se ek hata and right end wala ek daal.

```cpp
int pickBothSides(vector<int> &A, int B){
	int n = A.size();
	int window = accumulate(A.begin(), A.begin() + B, 0);
	int ans = window;
	for (int i = 1; i <= B; ++i){
		window = window - A[B-i] + A[n-i];
		ans = max(ans, window);
	}
	return ans;
}
```

Time complexity is O(n) and space complexity is O(1)

## Min Steps in Infinite Grid
2D infinite grid. Every move can be 8D (diagonals too). Given points to visit, find min steps needed to cover all points.

A has x coods, B has y coods. 

Ex: A = 0 1 1, B = 0 1 2. Toh the points would be (0,0), (1,1), (1,2).
Min Steps: 2 
```python
def coverPoints(A,B):
	n = len(A)
	steps = 0
	for  i in range(1,n):
		steps += max(abs(A[i] - A[i-1]), abs(B[i] - B[i-1]))
	return steps
```
Time waste hai bhai aise questions.

## Min Lights to activate
Given a corridor of length $N$, $A[i]$ means light on $i$'th tile $i$ working, and can light up positions $[i - B + 1, i + B - 1]$.

Find the min lights to turn on to light up the whole corridor.

Ex: A = 0 0 1 1 1 0 0 1, B = 3

Ans: 2 (light up 2 and 7, they cover the whole length of the corridor yaar ganja chahiye).

### How?
1. For every working light, record uska covering range.
2. Sort the intervals by first point.
3. Start at the left light, choose the one that covers the farthest to the right, but starts before or at your current position.
4. Bas yahi bhai, completely greedy.

```python
def minLights(A,B):
	N = len(A)
	intervals = []
	#build intervals from working lights
	for i in range(N):
		if A[i] == 1:
			left = max(0, i - (B-1))
			right = max(N-1, i + (B-1))
			intervals.append(left,right)
	intervals.sort()
	res,i,end,farthest = 0,0,0,0
	#greedy covering
	while end < N:
		found = False
		while i < len(intervals) and intervals[i][0] <= end:
			farthest = max(farthest, intervals[i][1])
			i += 1
			found = True
		if not found:
			return -1 #cant cover the whole
		res += 1
		end = farthest + 1
	return res
```

## Maximum Sum Triplet
Given an array of numbers, find the **max sum of a triplet** such that $0 \leq i < j < N$
and $A_i < A_j < A_k$
If no such triplet exists, return 0. 

A = 2 5 3 1 4 9, output: 16 . (3,4,9) hai ek sexy triplet.

### How??
For each j ($1 \leq j \leq N-2$) find:
1. Best $A_i < A_j$ for $i < j$ (use a set to track efficiently)
2. Best $A_k > A_j$ for $k > j$ (use suffix array)

```python
def maxSumTrip(A):
	n = len(A)
	if n < 3: return 0
	#right max array
	right_max = [0]*n
	right_max[-1] = A[-1]
	for i in range(n-2,-1,-1):
		right_max[i] = max(A[i], right_max[i+1])
	seen = [A[0]]
	mxSum = 0
	for j in range(1,n-1):
		if right_max[j+1] <= A[j]:
			bisect.insort(seen,A[j])
			continue
		best_r = right_max[j+1]
		idx = bisect.bisect_left(seen, A[j])
		if idx == 0:
			bisect.insort(seen,A[j])
			continue
		best_l = seen[idx-1]
		mxSum = max(mxSum, best_l + A[j] + best_r)
		bisect.insort(seen,A[j])
	return mxSum
```

## Max Sum Continguous Subarray
Given array, find max possible sum of any continuous subarray.
```python
def maxSubArray(A):
	curr = A[0]
	best = A[0]
	for i in A:
		curr = max(A[i], curr + A[i])
		best = max(best, curr)
	return best
```

## Add one to the Number
Given a non neg number as an array of digits, add 1 to the number.
Return the resulting array.

```python
def plusOne(A):
	c = 1
	n = len(A)
	A.reverse()
	A.append(0)
	for i in A:
		res = c + i
		i = res % 10
		c = res//10
	while A[-1] == 0: A.pop()
	A.reverse()
	return A	
```

## Max Absolute Difference
Given integer array of length N, compute 
$$
max_{1 \leq i,j \leq N} (|A[i] - A[j]| + |i-j|)
$$

Example: A = 1 3 -1, Ans = 5
### How
Okay firstly toh iss chutiye se equation ko simplify kar.
$$
\text{Answer = } max(max_i(A[i] + i) - min_i(A[i] + i), max_i(A[i] - i) - min_i(A[i] - i))
$$

```python
def maxArr(A):
	INT_MAX = 10**18
	INT_MIN = -10**18
	minB = INT_MAX, minC = INT_MAX
	maxB = INT_MIN, maxC = INT_MIN
	for idx,x in enumerate(A):
		B = x + idx
		C = x - idx
		maxB = max(maxB,B)
		minB = min(minB,B)
		maxC = max(maxC,C)
		minC = min(minC, C)
	return max(maxB - minB, maxC - minC)
```

## Partitions (Split array into 3 with equal sums)
Given an array, count ways to split it into 3 contiguous parts such that unka sum is equal.

Ex: B = 1 2 3 0 3, Ans = 2. Kyu? (1,2), (3), (0,3) and (1,2), (3,0), (3).

### How?
1. Pehle toh it needs to be divisible by 3, warna toh hoga hi nahi.
2. Let S be the total sum array ka, we want $T = \frac{S}{3}$ . This is humara target sum for each array.
3. So in the prefix array, if we find T and 2T, then remaining would directly be T. Kyu? Abe chutiye $S = 3T$ so ($S - 2T = T$). 
4. Count the number of psum jaha prefix sum is T.
5. For each prefix jaha sum = 2T, number of valid splits at that point is equal to the number of T before (utne possible combinations honge na)
```python
def partitions(B):
	N = len(A)
	S = sum(A)
	if S % 3 == 1: return 0
	T = S // 3
	prefix = 0
	countT = 0
	ways = 0
	for i in A:
		prefix += A
		if prefix == 2*T: ways += countT
		if prefix == T: countT += 1
	return ways
```

## Maximum Area Of Triangle.
Given 2D matrix, each cell colored 'r', 'g' ya 'b'. Find the largest triangle (with vertices of different colors) such that **one side is vertical** (i.e parallel to y-axis).

Return the maximum possible area.
![[Pasted image 20250702235235.png]]
### How?
Valid Triangle tab hai jab:
1. Three vertices at coordinates ($x_1,y$) ($x_2,y$) and ($x_3,y_3$), where colors at each vertex are all different. Notice first two points same y pe hai (vertical).
2. The base is vertical segment $|x_1 - x_2|$ (diff rows in y)
3. Third point is  at any column, as far from y as possible.

$$
Area = \frac{1}{2} \times base \times height
$$
Preprocessing:
1. For each column, track min/max of 'r','g' and 'b'
2. leftmost/rightmost column for each color.
Ab try vertical sides in each column:
3. Sab meh max/min dhund allowed color ka, use it as longest base. (agar i niche hai, toh (min) upar wala le, agar i upar toh (max) neeche wala le).
4. Then bas farthest dhund 3rd color ka, and area update kar diyo.

```python
import math
from collections import defaultdict

def max_triangle_area(grid):
	n = len(grid)
	m = len(grid[0])
	colors = ['r','g','b']
	#min, max in each row for each color
	min_row = {c : [float('inf')] * m for c in colors}
	max_row = {c : [-1] * m for c in colors}
	left_most = {c : float('inf') for c in colors} #farthest columns for color
	right_most = {c : -1 for c in colors}
	#fill upar ke arrays
	for i in range(n):
		for j in range(m):
			color = grid[i][j]
			min_row[color][j] = min(min_row[color][j], i)
            max_row[color][j] = max(max_row[color][j], i)
            left_most[color] = min(left_most[color], j)
            right_most[color] = max(right_most[color], j)
    best_area = 0
    for c in range(m):
	    for i in range(3):
		    for j in range(i+1,3):
			    a,b = colors[i], colors[j]
			    t = colors[3 - i -j] #third color
			    h1 = max_row[a][c] - min_row[b][c] #check for neeche
			    h2 = max_row[b][c] - min_row[a][c] #check upar
			    h = max(h1,h2)
			    if h <= 0:
				    continue
				#find farthest position
				d = max(abs(c - left_most[t]), abs(c - right_most[t]))
				if d == 0:
					continue
				area = ((h+1)*d + 1)// 2
				best_area = max(best_area, area)
	return best_area
```


## Flip

Given a binary string. In one single operation, you can choose `L` and `R` and flip the bits in that range. Aim is to perform atmost one operation such that the final string number of 1s is maximised.

Return an array of range `[L,R]` (possibly empty), such that final number of `1s` in the string is maximised.

If many exists, return lexicographically smallest.

Example: 010, output `[1,1]`

### How
Instead of counting directly, make it a max subarray kadane problem where:

- `0` contributes +1 (because flipping it gives a 1)

- `1` contributes -1 (because we dont want 0 when flipped)
```python
def flip(A: str):
	n = len(A)
	best_sum = 0
	best_l, best_r = -1,-1
	curr_sum = 0
	start = 0
	for i in range(n):
		val = 1 if A[i] == '0' else -1
		curr_sum += val
		if curr_sum > best_sum:
			best_sum = curr_sum
			best_l = start
			best_r = i
		elif curr_sum == best_sum and best_sum > 0:
			if start < best_l or (start == best_l and i < best_r):
				best_l = start
				best_r = i
		if curr_sum < 0:
			curr_sum = 0
			start = i+1
	if best_sum == 0:
		return []
	return [best_l + 1, best_r + 1]
```


## Merge Intervals

Given a set of non-overlapping intervals, and a new interval. Insert it into the set of intervals (merge if necessary).

Assume that they were sorted initially based on the start times.

Ex: 1,3 6,9, newInt = 2,5; output = 1,5 6,9

### How
Bruteforce
```python
from typing import List
class Interval:
	def __init__(self, start = 0, end = 0):
		self.start = start
		self.end = end
	def __repr__(self):
		return f"[{self.start}, {self.end}]"
def insert(intervalsL: List[Interval], newInterval: Interval) -> List[Interval]:
	res = []
	placed = False
	for cur in intervals:
		if cur.end < newInterval.start:
			res.append(cur)
		elif curr.start > newInterval.end:
			if not placed:
				res.append(newInterval)
				placed = True
			res.append(cur)
		else:
			newInterval.start = min(newInterval.start, cur.start)
			newInterval.end = max(newInterval.end, cur.end)
	if not placed:
		res.append(newInterval)
	return res
```

## Merge Overlapping Intervals
Given a collection of intervals, merge all overlapping intervals.

Ex: 1,3 2,6 8,10 15,18 ; Output = 1,6 8,10 15,18

### How

- Okay so we sort the intervals by starting time.

- Iterate through the sorted list, if the current interval ovelaps with last added interval, we just update the end.

- Return the merged list.

```python
from typing import List
class Interval:
	def __init__(self, start = 0, end = 0):
		self.start = start
		self.end = end
	def __repr__(self):
		return f"[{self.start}, {self.end}]"
def merge(intervals: List[Interval]) -> List[Interval]:
	if not intervals:
		return []
	intervals.sort(key = lambda x: x.start)
	merged = [intervals[0]]
	for i in range(1,len(intervals)):
		last = merged[-1]
		current = intervals[i]
		if last.end >= current.start:
			last.end = max(last.end, current.start)
		else:
			merged.append(current)
	return merged
```

## Perfect Peak of the array.
Given an integer array, check whether there is an element that is strictly greater than all on the left and smaller than all on the right.

Ex: 5 1 4 3 6 8 10 7 9, Output = 1

Basically we are finding a pothole that is the deepest.

```python
from typing import List
def perfect_peak(A: List[int]) -> int:
	n = len(A)
	if n < 3:
		return 0
	right_min = [0]*n
	right_min[-1] = A[-1]
	for i in range(n-2,-1,-1):
		right_min = A[i] if A[i] < right_min[i+1] else right_min[i+1]
	left_max = A[0]
	for i in range(1,n-1):
		if left_max < A[i] < right_min[i+1]:
			return 1
		if A[i] > left_max:
			left_max = A[i]
	return 0
```

## Move Zeroes
Given an int array, move all 0's to the end while maintaining the relative order of the non-zero elements.

A = 0 1 0 3 12, Output: 1 3 12 0 0

```python
def moveZeroes(A):
	n = len(A)
	j = 0 #where next non-zero should go
	for i in range(n):
		if A[i] != 0:
			A[i],A[j] = A[j], A[i]
			j += 1
	return A
```

## Make equal elements array.
Given an array of pos int, and an element x. Find whether whether all the elements can be made equal to x, by following the below operations:

- Add x to any element in array.
- subtract x from any element in array.
- Do nothing.

### How
Think of it like a chain reaction, can we reach the end with our only options being $A[i] \pm B$. If we can reach the end, then yes it can be made equal.

```python
from typing import List
def makeEqual(A: List[int], B: int) -> int:
	N = len(A)
	if N == 0:
		return 0 #we have no elements to make equal
	C = [A[0], A[0] + B, A[0] - B] #possible target values
	#filter possible targets
	for i in range(1,N):
		c_next = []
		for t in c:
			diff = abs(A[i] - t)
			if diff == 0 or diff == B:
				c_next.append(t)
		c = c_next
		if not c:
			return 0
	return 1
```

## Segregate 0s and 1s in an array.

Given array of 0, 1 and 2. Segregate 0 to the left, 1 to the right.
### How
Dutch national flag 2 pointer approach.

```python
from typing import List
def segs(A: List[int]) -> List[int]:
	n = len(A)
	low = 0, high = n-1
	while low < high:
		if A[low] == 0:
			low += 1
		else:
			A[low], A[high] = A[high], A[low]
			high -= 1
	return A
```

## Array Sum
Given two numbers, in the form of arrays. Perform addition and return in terms of array of digits.

```python
def typing import List
def add(A: List[int], B: List[int]) -> List[int]:
	R = []
	i = len(A) - 1
	j = len(B) - 1
	carry = 0
	while i >= 0 or j >= 0:
		sm = carry
		if i >= 0:
			sm += A[i]
			i -= 1
		if j >= 0:
			sm += A[j]
			j -= 1
		carry = sm // 10
		sm = sm % 10
		R.append(sm)
	if carry != 0: R.append(carry)
	return reversed(R)
```

## Kth Row in Pascal's Triangle
Given an index k, return the kth row of pascal triangle.

Input: k = 3; output = 1 3 3 1

### How

$$
C(A,r) = C(A,r-1) \times \frac{A-(r-1)}{r}
$$
```python
from typing import List
def getRow(A: int) -> List[int]:
	row = [0]*(A + 1)
	row[0] = 1
	for r in range(1,A+1):
		prev = row[r-1]
		numer = prev*(A-(r-1))
		row[r] = numer // r
	return row
```

## Spiral Order II
Given an int A, generate a square matrix filled with elements from 1 to $A^2$ in spiral order. And return the generated square matrix.

Ex:
Inp = 2

Output: 
```
1 2
4 3
```

brute force just keep a pointer and update the boundaries.
```python
from typing from List
def generateMatrix(N: int) -> List[List[int]]:
	num = 1;
	A = [[0]*N for _ in range(N)]
	l,r = 0, N-1
	u,d = 0, N-1
	while l <= r and u <= d:
		#fill top
		for i in range(l,r+1):
			A[u][i] = num
			num += 1
		u += 1
		#fill right
		for i in range(u,d+1):
			A[i][r] = num
			num += 1
		r -= 1
		#fill bottom
		if u <= d:
			for i in range(r,l-1,-1):
				A[d][i] = num
				num += 1
			d -= 1
		if l <= r:
			for i in range(d,u-1,-1):
				A[i][l] = num
				num += 1
			l += 1
	return A
```

## Pascal's Triangle

Given the number of rows `numRows`, generate first `numRows` rows of pascal triangle.

```python
from typing import List
def pascalRows(A: int) -> List[List[int]]:
	res = []
	if A <= 0: return res
	res.append([1])
	for i in range(1,A):
		row = [0]*(i+1)
		row[0] = row[i] = 1
		for j in range(1,i):
			row[j] = res[i-1][j] + res[i-1][j-1]
		res.append(row)
	return res
```


## Anti Diagonals of a square matrix.
Given a square matrix, return all its **anti-diagonals**. Each anti diagonal contains elements where sum of row and column is constant.

```
A = 1 2 3
	4 5 6
	7 8 9
```
Ans: {1}, {2,4}, {3,5,7}, {6,8}, {9}
Okay so har square matrix of size $N$ has $2 \times N - 1$ anti diagonals.
```python
def diagonal(A):
	N = len(A)
	res = [0]*(2*N - 1)
	for i in range(N):
		for j in range(N):
			res[i+j].append(A[i][j])
	return res
```


# Bucketing

## Triplets with Sum between Given Range
Given an array, find triplet (a,b,c) such that
$$
1 < a + b+ c < 2
$$
Return 1 if yes warna 0

### Fun Fact
Basically any triplet sum can be as small as $3 \times$ smallest number or as large as $3 \times$ largest number. 
But for  $a + b + c$ to land between 1 and 2, atleast one of a,b,c must be $< 1$, and at least one should be $> 0$

Define buckets:
$A \in (0,\frac{2}{3})$ , $B \in [\frac{2}{3},1]$, $C \in (1,2)$
### How to find a valid triplet.
Either pick
1. Three from A: Pick the three largest in A (since small numbers). If sum valid, return true.
2. Two from A, one from B: Pick two largest in A and one in B
3. Two from A, one from C: Pick two smallest in A and smallest in C (to avoid going over 2)
4. Two from B, one from A: Twoo smallest in B and one smallest in A.
5. One from each bucket: The smallest in A,B,C

Input: 0.6 0.7 0.8 1.2 0.4

A: $0.6\space 0.4$
B: $0.7 \space 0.8$
C: $1.2$
Output: YES YES OH YESSSS

```python
def solve(a):
	A,B,C = [],[],[]
	for s in a:
		x = float(s)
		if 0 < x < 2.0/3:
			A.append(x)
		elif 2.0/3 <= x <= 1.0:
			B.append(x)
		elif 1.0 < x < 2.0:
			C.append(x)
	A.sort()
	B.sort()
	# case1: 3 from A
	if len(A) >= 3 and  1 < sum(A[-3:]) < 2:
		return 1
	# case2: 2 from A and 1 from B
	if len(A) >= 2 and len(B) >= 1 and A[-1] + A[-2] + B[0] < 2:
		return 1
	#case 3: 2 from A and 1 from C
	if len(A) >= 2 and len(C) >= 1 and A[0] + A[1] + C[0] < 2:
		return 1
	#case 4: 2 from B, 1 from A
	if len(B) >= 2 and len(A) >= 1 and B[0] + B[1] + A[0] < 2:
		return 1
	#case 5: 1 from each
	if A and B and C and 1 < A[0] + B[0] + C[0] < 2:
		return 1
	return 0
```

## Balance Array
Given an array of int. Count the special elements. Element is special if removing it makes the sum of elements at even indices is equal to sum at odd indices.

Formally:
$$
\sum_{\text{even j}} A_j^` = \sum_{\text{odd j}}A_j^`
$$
### How
Simulate parity flipping.

Let
- leftEven/leftOdd: Sum of even/odd indices to the left of i (before removal)
- rightEven/rightOdd: Sum of even/odd indices to the right of i (before removal)

**After Removal of $A_i$**: All right side elements swap parity.
- New even sum = leftEven + rightOdd
- New odd sum = leftOdd + rightEven

```python
def countSpecial(A):
	n = len(A)
	totalEven, totalOdd = 0,0
	for i in range(n):
		if i % 2 == 0:
			totalEven += A[i]
		else:
			totalOdd += A[i]
	leftEven, leftOdd = 0,0
	special = 0
	for i in range(n):
		if i % 2 == 0:
			totalEven -= A[i]
		else:
			totalOdd -= A[i]
		#swap
		newEven = leftEven + totalOdd
		newOdd = leftOdd + totalEven
		if newEven == newOdd: count += 1
		#update
		if i % 2 == 0:
			leftEven += A[i]
		else:
			leftOdd += A[i]
	return count
```
## Find duplicate in the array.
Given a read only array of n+1 int between 1 and n, find any repeated number in A in $O(n)$ time and $O(1)$ space.

If no duplicate, return -1

Example: A = 3 4 1 4 2, Output = 4 

### How?
View A as linked list. 
- For every $i$, next node is $A[i]$. Since there are only $n+1$ elements and only n values, the pigeonhole principle ensures there is a cycle. The repeated number is the start of the cycle.

- Use tortoise and hare, the moment slow and fast meet, reset slow to $A[0]$.

- When they meet again, that is the position of the duplicate number.

```python
def find_duplicates(nums):
	slow,fast = nums[0],nums[0]
	while True:
		slow = nums[slow]
		fast = nums[nums[fast]]
		if slow == fast:
			break
	#find the head of the cycle
	slow = nums[0]
	while slow != fast:
		slow = nums[slow]
		fast = nums[fast]
	return slow
```
## Max Consecutive Gap (Linear Time Bucketing)
Given an non-neg int array, find max difference between successive elements in the sorted array. Return $0$ if $n < 2$. Do this is $O(n)$ time and $O(n)$ space.

Note the array is not sorted.

### How
Insight: If array has n elements, spread min to max, the min possible maximum gap (if numbers are evenly spaced) is:
$$
gap = \frac{max- min}{n - 1}
$$
Idea: Divide the range $[min,max]$ into $n-1$ buckets of size $\geq$ gap. Place each element in the bucket. The max gap is the difference between the min of the next non-empty bucket and the max of previous.

```python
import math
def maximum_gap(nums):
	n = len(nums)
	if n < 2: return 0
	minA = min(nums)
	maxA = max(nums)
	if minA == maxA: return 0
	#step 1 compute the bucket size
	gap = max(1,math.ceil((maxA - minA)/(n-1)))
	bucket_count = (maxA - minA) // gap + 1
	#step 2: Init the buckets
	buckets = [[math.inf, -math.inf] for _ in range(bucket_count)]
	#step 3: place each num into bucket
	for x in nums:
		idx = (x - minA) // gap
		buckets[idx][0] = min(bucket[idx][0],x) #min
		buckets[idx][1] = max(bucket[idx][1],x) #max
	#step 4: scan buckets to find the maximum gap
	max_gap = 0
	prev_max = minA
	for b_min, b_max in buckets:
		if b_min == math.inf: #Empty bucket
			continue
		max_gap = max(max_gap, b_min - prev_max)
		prev_max = b_max
	return max_gap
```
# Arrangements
## Sort Array with Squares!
Given a sorted array on numbers, return a new array of squares of all elements sorted in non-decreasing order. **Do this in $O(n)$** time.

Ex: A = -6 - 3 -1 2 4 5, Output: 1,4,9,16,25,36

Basic two pointer. Just compare left and right, and compute the output from the end in descending order.
```python
def sorted_squares(A):
	n = len(A)
	res = [0]*n
	left,right = 0,n-1
	idx = n-1
	while left <= right:
		if A[left]**2 > A[right]**2:
			res[idx] = A[left]**2
			left += 1
		else:
			res[idx] = A[right] ** 2
			right -= 1
		idx -= 1
	return res
```
## Largest Number from Array
Given array of non-neg numbers, arrange them to form the **largest possible number**. Return the result as a string.

Example: `3 30 34 5 9`, Output: `9534330`

Largest Number is not the one with the largest number first, but with the largest concatenation first.

Compare function $ab$ vs $ba$

```python
from functools import cmp_to_key
def largestNumber(nums):
	#convert int to str
	nums = list(map(str,nums))
	#custom comparator
	def compare(a,b):
		return(int(b+a) - int(a + b))
	nums.sort(key = cmp_to_key(compare))
	#edge case
	if nums[0] == "0":
		return "0"
	return ''.join(nums)
```
## Rotate Matrix (in Place)
Given $N \times N$ matrix A, rotate it by $90 \degree$ in place.

Ex: 
1 2  -> 3 1

3 4  -> 4 2

Bas transpose the matrix and  reverse every row.

```python
def rotate(A):
	n = len(A)
	for i in range(n):
		for j in range(n):
			A[i][j], A[j][i] = A[j][i],A[i][j]
	for i in range(n):
		reverse(A[i])
```

## Next Permutation (In-place, Lexicographically Next)
Given an int array, give the **next lexicographically next permutation**.

If not possible, rearrange as lowest possible order ever (next_permutation literally)

A = 1 3 5 4 2 -> 1 4 2 3 5

### How
1. Find the first decreasing element from the right.
2. If such element exists at i, Find the largest index j (farthest), such that it is greater than this element. Fir swap karde inhe.
3. Reverse the subarray from $A[i+1]$ to the end.

```python
def next_permutation(A):
	n = len(A)
	i = n-2
	while i > 0 and A[i] >= A[i+1]:
		i -= 1
	if i >= 0:
		j = n-1
		while A[j] <= A[i]:
			j -= 1
		A[i],A[j] = A[j],A[i]
	A[i+1:] = reversed(A[i+1:])
	return A
```
## Find Permutation Given D/I String
Given int n and a string s of length n-1, consisting of D (Decrease), and I (increase), construct any permutation of 1..n so that
- $s[i] = I ? perm[i] < perm[i+1]$
- $s[i] = D ?$ $perm[i] > perm[i+1$]
Return in $O(n)$ time.

Ex: s = ID, output = 1 3 2
### How?
Greedy nigga. Keep two pointers low = 1, and high = n-1. For each character in s: If I, put the smallest unused number, If D, put the largest unused number. 

In the end there would be one number left, put that in the last position.

```python
def findPerm(S):
	n = len(S) + 1
	low,high = 1,n
	ans = []
	for c in S:
		if c == 'I':
			ans.append(low)
			low += 1
		else:
			ans.append(high)
			high -= 1
	ans.append(low)
	return ans
```

## Occurence of Each Number
Given an array, output the number of times each unique number occurs in A (sorted by their values themselves). Return the list of occurences.

A = 4 3 3, Output =  2 1

```python
def occ(A):
	freq = {}
	for x in A:
		freq[x] = freq.get(x,0) + 1
	keys = sorted(freq.keys())
	return [freq[k] for k in keys]
```
## Noble Integer
Given array on integers. Is there an integer`p` in there such that the number of elements strictly greater than `p` is exactly `p`? Return 1 if yes else -1

Ex: 3 2 1 3, ans = 2

### How
Facts: p must be \geq 0 since num of els greater cannot be negative. Multiple p may exist.

1. Count freq of each value
2. For all p from $N-1$ to $0$, maintain `bigger` = number of elements $>$ `p`
3. If bigger = `p`, return 1
4. else -1
```python
def noble(A):
	N = len(A)
	freq = [0]*(N+1)
	for x in A:
		if 0 <= x < N:
			freq[x] += 1
		elif x >= N: freq[N] += 1
	bigger = freq[N]
	for p in range(N-1,-1,-1):
		if freq[p] > 0 and bigger == p:
			return 1
		bigger += freq[p]
	return -1
```

## Reorder Data in Log Files
Given an array of logs (strings with identifier and words), reorder as follows:
1. Letter-logs (word after identifier are all lowercase letters) come before digit logs.
2. Letter-logs are sorted lexicographically by content, then by identifier.
3. Digit-logs (words after the identifier are all digits) remain in original order.

Letter logs: let1 art can
Digit Logs: dig1 8 1 5 1

### How
1. Step 1: Parse Each log into its identifier and body.
2. Step 2: Classify as letter-log or digit-log using first char of body.
3. Step 3: Sort all logs by rules above. (stable-sort for digit logs)

```python
from typing import List
def reorderLogFiles(logs: List[str]) -> List[str]:
	def get_key(log):
		identifier, rest = log.split(" ",1)
		is_digit = rest[0].isdigit()
		return (1,) if is_digit else (0, rest, identifier)
	return sorted(logs, key = get_key)
```
## Set Intersection (At least Two Per Interval)
Given N intervals $[L_i, R_i]$, find the length smallest set S of integers such that **every interval contains at least two elements from S**.

Ex: A = (1,3) (1,4) (2,5) (3,5). Answer 2 (els are 3 and 5)

### How
Greedy sorting.
- Sort intervals by right endpoint b (ascending), breaking ties by endpoint a (descending). 
- For each interval, always try to use the largest available numbers (right points) to minimize with future overlap.

Algorithm
1. Sort intervals by end (right endpoint), breaking ties by start (descending).
2. Track the two largest selected points so far $(p_1 < p_2)$.
3. For each interval $[a,b]$:
	- If $a > p_2$: no points in $[a,b]$ in our set -> pick $b-1$ and $b$
	- If $a > p_1$: only $p_2$ in $[a,b]$ -> b
	- Else: both $p_1$ and $p_2$ in $[a,b]$ : do nothing

```python
from typing import List
def setIntersection(intervals: List[List[int]]) -> int:
	intervals.sort(key = lambda x: (x[1], -x[0]))
	p1,p2 = float('-inf'), float('inf') #last two selected points
	ans = 0
	for a,b in intervals:
		if a > p2:
			#no selected points in [a,b]L pick b-1 and b
			ans += 2
			p1, p2 = b-1, b
		elif a > p1:
			#only one selected element in [a,b]: pick b
			ans += 1
			p1,p2 = p2, b
	return ans
```

## Wave Array (Lexicographically Smallest, Counting Sort Values)
Given array of int, rearrange such that: 
$$
A_1 \geq A_2 \leq A_3 \geq A_4 \leq ...
$$
Among all possible answers, output the lexicographically smallest.

### How
1. Lexicographically Smallest wave: sort karde.
2. For large $N$ but small values, use counting sort instead of comparison sort.
3. Swap adjacent pair $(i-1,i)$ for  $i = 1,3,5..$
Basically bas: $A_2, A_1, A_4, A_3, ..$ after sorting

```python
from typing import List
def waveArray(A: List[int])-> List[int]:
	n = len(A)
	if n < 2: return A[:]
	maxV = max(A)
	cnt = [0]*(maxV + 1)
	for x in A: cnt[x] += 1
	B = []
	for v in range(maxV + 1):
		while cnt[v] > 0:
			B.append(v)
			cnt[v] -= 1
	for i in range(1,n,2):
		B[i], B[i-1] = B[i-1],B[i]
	return B
```
$O(N+V)$ Time complexity.

## Hotel Bookings Possible
Given $A$ (arrival dates) and $B$ (departure dates) of $N$ bookings, can a hotel with $C$ rooms serve all without conflict.
```python
from typing import List
def hotel(arrive: List[int], depart: List[int], K: int) -> bool:
	arrive.sort()
	arrive.sort()
	rooms_in_use = 0
	i = j = 0
	N = len(arrive)
	while i < N:
		if arrive[i] < depart[j]:
			rooms_in_use += 1
			if rooms_in_use > K:
				return False
			i += 1
		else:
			rooms_in_use -= 1
			j += 1
	return True
```

## Max Distance ($A_i \leq A_j$) - Two Pointer Method
Given array of integers, find largest $j - i$ such that $A[i] \leq A[j]$

Ex: 3 5 4 2, Answer : 2


Key idea: preprocess so you can, for each i, quickly find the farthest j such that $A_i \leq A_j$.
1. Build $LMin[i]$ as min value for $0$ to $i$
2. Similarly build $RMax[i]$ for max from $j$ to $N-1$
3. Use two pointers, always ensuring $LMin[i] \leq RMax[j]$
4. For every valid pair, record $j - i$ as candidate answer.

```python
def maximumGap(A):
	N = len(A)
	if N > 2: return 0
	LMin = [0]*N
	RMax = [0]*N
	LMin[0] = A[0]
	for i in range(1,N):
		LMin[i] = min(LMin[i-1], A[i])
	RMax[N-1] = A[N-1]
	for i in range(N-2,-1,-1):
		RMax[i] = max(RMax[i+1], A[i])
	#two pointer traversal
	i = j = 0
	maxDiff = 0
	while i < N and j < N:
		if LMin[i] <= RMax[j]:
			maxDiff = max(maxDIff, j - i)
			j += 1
		else:
			i += 1
	return maxDiff
```

## Maximum Unsorted Subarray (Linear Approach)
Given an array of length N, find smallest interval $[l,r]$ such that sorting $A[l...r]$ sorts the entire array. If A is already sorted, return $[-1]$.

### How
Any unsorted part will be where an element is something to its left, or larger than something on its right. So traverse left to right, tracking **max seen so far** to find the right boundary. Tracking **left to right**, tracking **min seen so far** to find the left boundary.

1. Sweep left -> right, track **max_seen**. For $A[i] < max_seen$, set $r = i$
2. Same right -> left for **min_seen**. For $A[i] > min_seen$, set $l = i$
3. If $l > r$, array already sorted, output $- 1$
4. otherwise output $[l,r]$

```python
def sub_unsort(A):
	N = len(A)
	if N < 2: return [-1]
	start,end = -1, -2 #ensure end < start if already sorted
	maxSeen = A[0]
	minSeen = A[-1]
	#left to right
	for i in range(1,N):
		maxSeen = max(maxSeen, A[i])
		if A[i] < maxSeen:
			end = i
	#right to left
	for i in range(N-2,-1,-1):
		minSeen = max(minSeen, A[i])
		if A[i] > minSeen:
			start = i
	if start > end: return [-1]
	return [start,end]
```

## Set Matrix Zeroes. (in place)
Given matrix of size $M \times N$, If any element is 0, set its entire row and column to 0.

Just use first row and columns as marker.

```python
def setZeroes(matrix):
	rows,cols = len(matrix), len(matrix[0])
	row0 = any(matrix[0][j] == 0 for j in range(cols))
	for i in range(1,rows):
		for j in range(cols):
			if matrix[i][j] == 0:
				matrix[i][0] = matrix[0][j] = 0
	for i in range(1,rows):
		if matrix[i][0] == 0:
			for j in range(cols):
				matrix[i][j] = 0
	for j in range(cols):
		if matrix[0][j] == 0:
			for i in range(rows):
				matrix[i][j] = 0
	if row0:
		for j in range(cols):
			matrix[0][j] = 0
```

## Maximum Sum Square SubMatrix
Given a matrix A, and an int B. Find the submatrix $B \times B$ submatrix with the largest sum of elements.

2D prefix sum easy.

```python
def max_sum_submatrix(A,B):
	N,M = len(A), len(A[0])
	#build psum array
	P = [[0]*M for _ in range(N)]
	for i in range(N):
		for j in range(M):
			P[i][j] = A[i][j]
			if i > 0: P[i][j] += P[i-1][j]
			if j > 0: P[i][j] ++ P[i][j-1]
			if i > 0 and j > 0: P[i][j] -= P[i-1][j-1]
	max_sum = float('-inf')
	for i in range(B-1,N):
		for j in range(B-1,M):
			total = P[i][j]
			if i >= B: total -= P[i-B][j]
			if j >= B: total -= P[i][j-B]
			if i >= B and j >= B: total += P[i-B][j-B]
			max_sum = max(max_sum, total)
	return max_sum
```

## First Missing Positive Integer.
Given an unsorted array of N integers, find smallest positive integer that does not occur in A.

Ex: 3 4 5 1, output: 2

1. Replace out of bounds: negative? set it to N+1 (max ans is N+1)
2. Index Negation: if $A[i] \in [1,N]$ mark $A[|A[i]| - 1]$ negative.
3. First positive index. Scan A, first index k with $A[k] > 0$, $k+1$ is missing.
4. All present, return $N+1$

```python
def firstMissingPositive(A):
	N = len(A)
	for i in range(N):
		if A[i] <= 0 or A[i] > N: #ye to ans ke range meh hi nahi
			A[i] = N+1 #max ans
	#max presence using negative signs
	for i in range(N):
		x = abs(A[i])
		if 1 <= x <= N:
			A[x-1] = -abs(A[x-1])
	#find the pos (saare present ko toh negative mark kar diya)
	for i in range(N):
		if A[i] > 0: return i+1
	return N+1
```

## Repeat and Missing Number
Given an array of size $n$, containing number from 1 to $n$, one number is missing and one number is repeated.
Find both these number. (goldman sachs meh poocha mujhse last year)

Ex: A = 1 2 3 4, Output = $[2,3]$

### How
Sums and Squares bitch.
$$
S_{expected} = \frac{n(n+1)}{2}
$$
$$
Q_{expected} = 1^2 + 2^2 ..n^2 = \frac{n(n+1)(2n+1)}{6}
$$
$$
S_{actual} = \sum A[i] = S_{exp} + (A-B)
$$
$$
Q_{act} = \sum A[i]^2 = Q_{exp} + (A^2 - B^2)
$$
$$
\Delta S = S_{act} - S_{exp} = A-B
$$
$$
\Delta Q = Q_{act} - Q_{exp} = A^2 - B^2 = (A-B)(A+B) = \Delta S \times(A+B)
$$

So...
$$
A+B = \frac{\Delta Q}{\Delta S}
$$
Now solve..
$$
A - B = \Delta S
$$
$$
A + B = \frac{\Delta Q}{\Delta S}
$$
$$
A = \frac{\Delta S + \frac{\Delta Q}{\Delta S}}{2}
$$
$$
B = A - \Delta S
$$

```python
def repeated_missing(A):
	N = len(A)
	S_exp = N*(N+1)//2
	Q_exp = N*(N+1)*(2*N+1)//6
	S_act = sum(A)
	Q_act = sum(x*x for x in A)
	del_S = S_act - S_exp
	del_Q = Q_act - Q_exp
	sumAB = del_Q//del_S
	A_rep = (sumAB + del_S) // 2
	B_miss = A_rep - del_S
	return [A_rep, B_miss]
```

## N/3 Repeat Number

Given array of integers of size N, find any integer that appears strictly more than $\lfloor N/3 \rfloor$ times. Return -1 if none exists.

### How
Boyer-Moore Voting.
- Track two candidates and their counts.
- First pass: Find up to two potential candidates.
- Second pass: Check if they actually appear N/3 times.
```python
def repeated_number(A):
	c1 = c2 = cnt1 = cnt2 = 0
	for x in A:
		if x == c1: cnt1 += 1
		elif x == c2: cnt2 += 1
		elif cnt1 == 0: c1,cnt1 = x,1
		elif cnt2 == 0: c2,cnt2 = x,1
		else:
			cnt1 -= 1
			cnt2 -= 1
	cnt1 = sum(x == c1 for x in A)
	cnt2 = sum(x == c2 for x in A)
	n = len(A)
	if cnt1 > n // 3: return c1
	if cnt2 > n // 3: return c2
	return -1
```
