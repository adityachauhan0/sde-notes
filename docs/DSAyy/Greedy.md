## Highest Product of Three Numbers
Given an int array, find the highest possible sum from any $3$ elements in the array.

A = 1 2 3 4, ans: 24

```python
def maxp3(A):
	max1 = max2 = max3 = float('-inf') #max3
	min1 = min2 = float('inf') #min2
	for x in A:
		if x > max1:
			max3,max2,max1 = max2,max1,x
		elif x > max2:
			max3,max2 = max2,x
		elif x > max3:
			max3 = x
		if x < min1:
			min2, min1 = min1, x
		elif x < min2:
			min2 = x
	prod1 = max1*max2*max3
	prod2 = max1*min1*min2
	return max(prod1, prod2)
```

## Bulbs
Given $array$ of bulbs. Each bulb has a switch. However due to faulty wiring, pressing a switch toggles all the bulbs to the right.

Find min number of switches to turn all on.

$A = 0 1 0 1$

Output: $4$

### How

Just simulate the effect of toggles.

Maintain a $flips$ counter

If a bulb appears $off$, it must be toggled $on$.

Go left to right.

```python
def bulbs(A):
	flips,ans = 0,0
	for idx,switch in enumerate(A):
		#determine state
		curr = switch if flips % 2 == 0 else 1 - switch	
		if curr == 0:
			ans += 1
			flips += 1
	return ans
```

## Disjoint Intervals

Given $N$ intervals, find max number of mutually disjoint intervals that can be selected. Disjoint meaning they dont share any point.

Ex: $1,4$ $2,3$ $4,6$ $8,9$ : Output = $3$

### How
Sort all in ascending order of ending times.

Select an interval if its starting time is greater than end of the last one.

```python
def disj(A):
	if not A: return 0
	
	A.sort(key = lambda x: x[1])
	count = 0
	last_end = -1
	for start,end in A:
		if start > last_end:
			count += 1
			last_end = end
	return count
```

## Largest Permutation
Given a permutation of 1 to N, and int B, representing max swaps allowed, return the lexicographically largest permutation by performing atmost B swaps.

Ex: A =  1 2 3 4, B = 1: Output: 4 2 3 1

### How
To achieve the largest:

- For every element $i$ in $A$, if $i$ is not the maximum possible $(N-i)$, we find the position of the desired value and swap it with $A[i]$.

- Maintain a map of positions to values for $O(1)$ lookup.

- After each swap, decrement $B$.

```python
def solve(A,B):
	N = len(A)
	pos = {val: i for i,val in enumerate(A)}
	i = 0
	while i < N and B > 0:
		desired = N-i
		if A[i] != desired:
			idx = pos[desired]
			pos[A[i]] = idx
			pos[desired] = i
			A[i],A[idx] = A[idx], A[i]
			B-= 1
		i += 1
	return A
```

## Meeting Rooms
Given time intervals representing meetings, find minimum rooms required to accommodate all.

If a meeting ends at $t$, and another meeting starts at $t$, they can use the same room.

Ex: $0,30$ $5,10$ $15,20$ Output: $2$

```python
def meetingRoomsII(A):
	N = len(A)
	if N == 0: return 0
	starts = sorted([interval[0] for interval in A])
	ends = sorted([interval[1] for interval in A])
	used = 0
	maxRooms = 0
	i = j = 0
	while i < N:
		if starts[i] < ends[j]:
			used += 1
			maxRooms = max(maxRooms, used)
			i += 1
		else:
			used -= 1
			j += 1
	return maxRooms
```

## Distribute Candy
$N$ children in a line, Each has a $rating$. Distribute candies such that

- Each child gets atleast one candy.

- Child with a higher rating than the immediate neighbor must get more candies than that neighbor.

Return the min candies required.

Ex: $1,2$ Output $3$

### How

Perform two passes, 

1. Left to Right: If current rating is higher than the left neigbor, give one more candy than the left neighbor.

2. Right to left: If current rating is higher than the right neighbor, ensure current has more candies.

Sum all these values.

```python
def candy(A):
	n = len(A)
	if n == 0: return 0
	candies = [1]*n
	for i in range(1,n):
		if A[i] > A[i-1]:
			candies[i] = candies[i-1] + 1
	for i in range(n-2, -1 -1):
		if A[i] > A[i+1]:
			candies[i] = max(candies[i], candies[i+1] + 1)
	return sum(candies)
``` 

## Seats
Given a string representing row of seats.

- Each seat is either empty `.` or occupied `x`

- A group of people is randomly scattered the row(at `x`).

- Move the people such that they sit together in adjacent seats, minimizing the number of total jumps.

- One jump allows a person to move to an adjacent seat.

Return the minimum number of jumps.

Ex: $....x..xx...x..$ Output: 5 ($6-4 + 12 - 9$)

Make them sit together from 6 to 9.

### How
Store indices of occupied seats in a list $p$.

Define transformed list $r[i] = p[i] - i$, which aligns people to a contiguous block.

The optimal center for the group is the median $r$.

Number of jumps is $sum(r[i] - median)$

```python
def seats(A):
	MOD = 10000003
	positions = [i for i,ch in enumerate(A) if ch == 'x']
	k = len(positions)
	if k <= 1:
		return 0 #no jumps
	#adjust to normalised relative positions
	r = [positions[i]-i for i in range(k)]
	m = r[k//2] #median minimizes the difference
	ans = sum(abs(val - m) for val in r) % MOD
	return ans
```

## Assign Mices to Holes
There are N mice and N holes, placed along a straight line. Each mouse and hole is positioned at a coordinate on this line:

- A mouse can move one step left or right in one minute.
- Each hole can hold only one mouse.

The goal is to assign each mouse to a hole such that the time taken for the last mouse to enter a hole is minimized.

Ex: Mouse = -4 2 3, Hole: 0 -2 4 Output: 2

### How
Sort both arrays.

Pair the smallest unassigned mouse with the smallest unassigned hole, second smallest with second smallest etc.

For each pair, compute absolute difference and return the max.

```python
def mice(A,B):
	N = len(A)
	A.sort()
	B.sort()
	return max(abs(A[i] - B[i]) for i in range(n))
```

## Majority Element
Given an int array, find the majority element. The majority element is the element that appears more than $\lfloor N/2 \rfloor$ times.

Majority element always exists.

Ex: 2 1 2, Output = 2

```python
def majorityElement(A):
	count,candidate = 0,0
	for x in A:
		if count == 0:
			candidate = x
			count = 1
		elif x == candidate:
			count += 1
		else:
			count -= 1
	return candidate
```

## Gas Station
Given two int arrays A and B of size N. There are N gas stations arranged in a circle:

- $A[i]$ amount of gas at station $i$

- $B[i]$ cost of gas to travel from station $i$ to station $(i+1)$ mod N

Return the minimum index of the gas station from which you can start and complete full circle exactly once, or -1 if no such start exists.

A = 1 2

B = 2 1

Output = 1 (zero based)

```python
def canCompleteCircuit(A,B):
	total_tank, curr_tank = 0,0
	start = 0
	n = len(A)
	for i in range(n):
		diff = A[i] - B[i]
		total_tank += diff
		curr_tank += diff
		if curr_tank < 0:
			start = i+1
			curr_tank = 0
	return start if total_tank >= 0 else -1
```
