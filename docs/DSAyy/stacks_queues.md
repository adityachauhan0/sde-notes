## Valid Parenthesis
Given a string containing bracket sequences, determine if the string is valid.
brackets can be `(` ,`{` `[`

```python
def validParent(S):
	stack = []
	bracket_map = {')' : '(', '}' : '{', ']' : '['}
	for char in S:
		if char in bracket_map.values(): #opening bracket
			stack.append(char)
		elif char in bracket_map: #closing bracket
			if not stack or stack[-1] != bracket_map[char]:
				return False
			stack.pop()
	return not stack
```

## Reverse String Using Stack
Given a string S, reverse the string using a stack.
```python
def reverse(s):
	stack  = []
	for char in S:
		stack.append(char)
	for i in range(len(S)):
		S[i] = stack.pop()
	return S
```

## Balanced Parentheses
Given a string, consisting only of `(` and `)`. Determine if this sequence is balanced.
```python
def balanced(S):
	balance = 0
	for char in S:
		balance += 1 if char == '(' else -1
		if balance < 0: return 0
	return 1
```

## Simplify Directory Path
Given a string representing absolute path for a file, return the simplified absolute path.
Ex:
1. "/home/" -> "/home"
2. "/a/./b/../../c/" -> "/c"

### How
1. Split the input path on '/' to get tokens.
2. For .., pop the last directory fron the stack.
3. Otherwise push directory name into the stack.
```python
def simplify_path(path):
	stack = []
	tokens = path.split('/')
	for token in tokens:
		if token == '' or token == '.':
			continue
		elif token == '..':
			if stack:
				stack.pop()
		else:
			stack.append(token)
	return '/' + '/'.join(stack)
```

## Redundant Braces in Expressions
Given a string denoting arithmetic expression (+ - * /), check whether it has redundant braces.

Ex:
1. ((a+b)) : yes
2. (a + (a+b)) : no

### How
1. Push everything onto the stack except when you see a closing parenthesis.
2. On seeing ), pop untill you find a (.
3. If no operator is found inside, they are redundant.

```python
def has_redundant_braces(expression):
	stack = []
	for char in expression:
		if char == ')':
			has_operator = False
			while stack and stack[-1] != '(':
				top = stack.pop()
				if top in ['+','-','*','/']:
					has_operator = True
			if stack:
				stack.pop() #pop the (
			if not has_operator:
				return True
		else:
			stack.append(char)
	return False
```

## Min Stack: Stack with constant time minimum
Design a stack with push pop top, and getMin(): retrieves the min element.

getMin and top should return -1 on an empty stack.
pop should do nothing on an empty stack.

### How
Use two stacks, one for elements, other for min.
Also push to minSt if its a new minimum, when you pop, also pop from minSt if it was the current minimum.

```python
class MinStack:
	def __init__(self):
		self.st = []
		self.minSt = []
	def push(self,x):
		self.st.append(x)
		if not self.minSt or x <= self.minSt[-1]:
			self.minSt.append(x)
	def pop(self):
		if not self.st:
			return
		val = self.st.pop()
		if self.minSt and val == self.minSt[-1]:
			self.minSt.pop()
	def top(self):
		if not self.st:
			return -1
		return self.st[-1]
	def getMin(self):
		if not self.minSt:
			return -1
		return self.minSt[-1]
```


## MAXPPROD: Maximum Special Product
Given an array of integers, define for each index i:
1. LeftSpecialValue(LSV): the max index `j` such that $A[j] > A[i]$ . If none, LSV = 0
2. RightSpecialValue(RSV): the min index `i` such that $A[j] > A[i]$ if none, RSV = 0
3. Special Product for i is $LSV \times RSV$ 

### How
basically nearest greatest element to the left, and nearest greatest element to the right.

```python
def max_special_product(A):
	n = len(A)
	left = [0]*n
	right = [0]*n
	stack = []
	#compute left special
	for i in range(n):
		while stack and A[stack[-1]] <= A[i]:
			stack.pop()
		left[i] = stack[-1] if stack else 0
		stack.append(i)
	stack.clear()
	#compute right special
	for i in range(n-1,-1,-1):
		while stack and A[stack[-1]] <= A[i]:
			stack.pop()
		right[i] = stack[-1] if stack else 0
		stack.append(i)
	#compute maxProd
	maxProd = 0
	for i in range(n):
		prod = left[i]*right[i]
		maxProd = max(maxProd, prod)
	return maxProd % (10 **9 + 7)
```


## Nearest Smaller Element.
Monotonic Stack basic. Find nearest smaller element to the left. Return $G[i]$ aka all the values of nearest smaller. If no element, $G[i] = -1$ .

```python
def prevSmaller(A):
	n = len(A)
	G = [0]*(n)
	stack = []
	for i in range(n):
		while stack and A[stack[-1]] >= A[i]:
			stack.pop()
		G[i] = stack[-1] if stack else -1
		stack.append(A[i])
	return G
```
## Largest Rectange in Histogram.
Given an array containing height of histogram bars (Each width 1). Find the area of largest histogram.

Ex: A = 2 1 5 6 2 3, Output = 10

### How
1. Use a monotonic increasing stack to keep the value of left greatest.
2. For each bar, while stack top is greater than current bar, pop and commute height times width. width is determined by the index.
```python
def largestRectArea(A):
	n = len(A)
	stack = []
	maxArea = 0
	for i in range(n+1):
		h = 0 if i == n else A[i]
		while stack and h < A[stack[-1]]:
			height = A[stack.pop()]
			j = stack[-1] if stack else -1
			width = i - j - 1
			area = height * width
			maxArea = max(maxArea, area)
		stack.append(i)
	return maxArea
```

## Hotel Service (Nearest Hotel in a grid)
Given a matrix of $N \times M$ of 0s and 1s. $1 == hotel$ and `Q` queries `B` (coordinates). Find the shortest distance from each query cell to nearest hotel. (measured in 4D).

A = 0 0       B = 1 1
    1 0           2 1
                1 2
Output: 1 0 2

### How
Use multisource BFS
1. Stack BFS from every hotel cell ($A[i][j] = 1$) simultaneously.
2. Fill out a dist grid so that minimal steps to each coordinate is recorded.
3. After BFS, answer each query in $O(1)$

```python
from collections import deque
def nearest_hotel_bfs(A,queries):
	N = len(A)
	M = len(A[0]) if N > 0 else 0
	dist = [[-1]*M for _ in range(N)]
	Q = deque()
	#enqueue all hotels
	for i in range(N):
		for j in range(M):
			if A[i][j] == 1:
				dist[i][j] = 0
				Q.append((i,j))
	directions = [(-1,0), (1,0), (0,-1), (0,1)]
	while Q:
		x,y = Q.popleft()
		for dx, dy in directions:
			nx, ny = x + dx, y + dy
			if 0 <= nx < N and 0 <= ny < M and dist[nx][ny] == -1:
				dist[nx][ny] = dist[nx][ny] + 1
				Q.append((nx,ny))
	results = []
	for X,Y in queries:
		results.append(dist[X-1][Y-1])
	return results
```

## First Non-repeating character in stream

Given string representing a stream of lowercase letters, construct B such that $B[i]$ is the first non repeating chracter in the prefix $A[0...i]$ of the stream.
If none, append #.

A = abadbc, output = aabbdd

### How
for each char
1. increase its frequency
2. push c into a queue if its a candidate for non repetition.
3. while the char at front of queue is a repeater($\text{frequency > 1}$), pop it.
4. If the queue is empty, append `#` 

```python
from collections import deque
def first_non_repeating_char_stream(A):
	freq = [0]*26 
	Q = deque()
	B = []
	for c in A:
		idx = ord(c) - ord('a')
		freq[idx] +=  1
		Q.append(c)
		while Q and freq[ord(Q[0]) - ord('a')] > 1:
			Q.popleft()
		if not Q:
			B.append('#')
		else:
			B.append(Q[0])
	return ''.join(B)
```

## Sliding Window Maximum
Given an array and a window size, for each window of that size moving from left to right, find the maximum in that window.

A: 1 3 -1 -3 5 3 6 7

B: 3

Output: 3 3 5 5 6 7

### How
Use a double-ended queue, to maintain a list of candidates for the maximum (monotonically decreasing)
1. When the window moves, remove the indices that are out of range from the front.
2. Remove indices from the back that are less than the current element (not maximum).
3. The front of the deque always gives maximum for the current window.

```python
from collections import deque
def sliding_window_maximum(A,B):
	n = len(A)
	if B > n: return [max(A)]
	dq = deque()
	C = []
	for i in range(n):
		if dq and dq[0] == i - B: #out of window index
			dq.popleft()
		#remove indices whose val < A[i]
		while dq and A[dq[-1]] < A[i]:
			dq.pop()
		dq.append(i)
		#record max for current window
		if i >= B-1:
			C.append(A[dq[0]])
	return C
```

## Evaluate Expression (Reverse Polish Notation)
Given a string `A` representing arithmetic expression in **Reverse Polish Notation**, evaluate and return its value.

A = 2 1 + 3 * , Output = 9

### How
traverse the tokens
1. If a number, push to stack.
2. If an operator, pop two numbers, compute the result, and push it back.

```python
def RPN(tokens):
	stack = []
	for token in tokens:
		if token in ["+", "-", "*", "/"]:
			b = stack.pop()
			a = stack.pop()
			if token == "+":
				res = a+b
			elif token == "-":
				res = a-b
			elif token  == "*":
				res = a*b
			else:
				res = int(a/b)
			stack.append(res)
		else:
			stack.append(int(token))
	return stack[-1]
```


## Trapping Rain Water (total this time)
Given `A[i]` representing the height of the wall. width of each wall is `1`.
Compute total units of water that can be trapped after it rains.

A = 0 1 0 2 1 0 1 3 2 1 2 1, Output = 6

### How
Water is trapped if there are taller bars on both sides.
$$
\text{water at i = } min(maxLeft_i, maxRight_i) - A[i]
$$
Instead of precomputing maxLeft and maxRight, we can traverse both ends and track the highest bar from left to right on the go.

1. Two pointers (left and right) at both ends, and two variables (left_max and right_max).
2. Move the smaller inwards (if left is smaller):
	1. If $A[left] \geq \text{left\_max}$, update left_max.
	2. Else, add left_max - A[left] to water.
	Increment left.
3. Else do the same for right.
4. Continue until pointers meet.
```python
def trap_rain_water(A):
	n = len(A)
	left, right = 0, n-1
	left_max, right_max = 0, 0
	water = 0
	while left <= right:
		if A[left] < A[right]:
			if A[left] >= left_max:
				left_max = A[left]
			else:
				water += left_max - A[left]
			left += 1
		else:  
			if A[right] >= right_max:
				right_max = A[right]
			else:
				water += right_max - A[right]
			right -= 1
	return water
```
