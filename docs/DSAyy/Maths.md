## Verify Prime
Given a number, find whether it is prime.
```python
def is_prime(N):
	if n < 2:
		return False
	if n in (2,3):
		return True
	if n % 2 == 0:
		return False
	for i in range(3, int(n**0.5)+ 1, 2):
		if n % i == 0:
			return False
	return True
```

## Prime Numbers (Linear Sieve)

Given an int A, return all prime numbers $\leq$ A. Sorted in increasing order.

```python
def primes_upto(A):
	if A< 2:
		return []
	sieve = [True]*(A+1)
	sieve[0], sieve[1] = False, False
	for i in range(2,int(A**0.5)+1):
		if sieve[i]:
			for j in range(i*i, A+1, i):
				sieve[j] = False
	return [i for i, is_prime in enumerate(sieve) if is_prime]
```

## Prime Representation
Given a non-neg int N, find its binary representation as a string.
```python
def find_digits_in_binary(N):
	if N == 0:
		return "0"
	bin_str = ""
	while N > 0:
		bin_str += str(N & 1)
		N >>= 1
	return bin_str[::-1]
```

## All Factors
Given an int A, find and return all its positive factors (divisors).
### How
Every factor $d$ has to be $\leq$ $\sqrt{A}$ pairs with another factor $\frac{A}{d}$ 
```python
def all_factors(A):
	small = []
	large = []
	i = 1
	for i*i <= A:
		if A % i == 0:
			small.append(i)
			if i != A//i:
				large.append(A//i)
	return small + large[::-1]
```

## Adhoc
Given the pos (A,B) of a bishop on a standard $8 \times 8$ chess board, count the total number of squares the bishop can move to in one move.

Ex: A = 4, B = 4, Total = 13

$$
Total = NE + NW + SE + SW
$$
```python
def bishopMoves(A,B):
	n = 8
	nw = min(A-1,B-1)
	ne = min(A-1, n-B)
	sw = min(n-A, B-1)
	se = min(n-A,n-B)
	return nw + ne + sw + se
```

## Distribute in Circle
Given $A$ items to be delivered in circle of size $B$, starting at position $C$. Find the position where the $A$th item will be delivered. Items are delivered to adjacent positions.

$$
\text{Position = } (C+A - 1) mod \space B
$$
```python
def distToCircle(A,B,C):
	pos = (C + A - 1) % B
	return B if pos == 0 else pos
```

## Prime Sum
Given an even number $A > 2$, return two prime numbers whose sum equals $A$. Return the lexicographically smallest pair $[a,b]$

### How
This is based on **Goldbach's Conjecture** which states that every even integer $> 2$ can be expressed as sum of two prime numbers.

Approach:

- Generate all primes up to A using linear sieve.

- For each $a$ from $2$ to $A/2$:
	
	- If both $a$ and $A-a$ are primes, return $[a,A-a]$

```python
def primeSum(A):
	is_prime = [True]*(A+1)
	is_prime[0] = is_prime[1] = False
	for i in range(2,int(A**0.5) + 1):
		if is_prime[i]:
			for j in range(i*i, A+1, i):
				is_prime[j] = False
	for a in range(2, A//2):
		if is_prime[a] and is_prime[A-a]:
			return [a,A-a]
	return []
```

## Sum of Pairwise Hamming Distance
Given an array of int, compute the sum of hamming distance of all ordered pairs in A.

**Hamming Distance**: The hamming distance between two integers is the number of positions at which their binary representations differ.
(basically popcount of xor)

### How
For each position b (from 0 to 30), count how many number have 1 at that position. $zeroes_b = N - ones_b$ , then we just have to find how many 1's at that bit, we can pair with 0's since that creates a hamming distance of 1.

Since (i,j) and (j,i) both count, we have to count duplicates.

So 
$$
\text{Contribution} = 2 \times ones_b \times zeroes_b
$$
Sum this over all bits, to get the answer.

```python
def hamming_distance(A):
	MOD = 10**9 + 7
	N = len(A)
	cnt = [0]*31 #counts of 1s in each bit position
	for x in A:
		for b in range(31):
			if x & (1 << b):
				cnt[b] += 1
	ans = 0
	for b in range(31):
		ones = cnt[b]
		zeroes = N - ones
		contrib = (ones*zeroes) % MOD
		contrib = (contrib*2) %MOD
		ans = (ans + contrib) % MOD
	return ans
```

## Step by Step

Given a target position $A$ on a infinite number line, starting at position $0$, you can move in the $i$-th move by exactly $i$ steps (either forward or backward). What is the minimum number of moves required to reach the target?

### How
**Crazy baat:** after k moves, sum of steps = $\frac{k(k+1)}{2}$ . So if sum = A, we can reach just by going forward. 

If $S > A$, then difference $S- A$ can be compensated by reversing the directions of some steps. But you can only make $S-A$ even since flipping the direction of a move changes the total by $2 \times$ (that move's length).

So we need the smallest $k$ such that:
$$
S = \frac{k(k+1)}{2} \geq |A| \text{ and } S - |A| \text{ is even}
$$

So lets solve for k:
$$
\frac{k(k+1)}{2} \geq \text{target} \equiv k^2 + k - 2\times target \geq 0
$$

Using the quadratic formula:
$$
k = \lceil{ \frac{-1 + \sqrt{1 + 8 \times target}}{2} }\rceil
$$


```python
import math
def solve(A):
	target = abs(A)
	#minimal k such that k*(k+1)//2 >= target
	k = math.ceil((-1.0 + math.sqrt(1.0 + 8.0 * target)) / 2.0)
	S = k*(k+1)//2
	if (S - target) % 2 == 0:
		return k
	elif ((S+k+ 1 - target) % 2 == 0):
		return k+1
	else:
		return k+2
```

## Is Power?
Given an int A $>$ 0, can $A$ be written as $a^p$ for some integers $a \geq 2$ , $p \geq 2$ ?

We just have to check if its a perfect power.

```python
import math
def is_power(A):
	if A == 1:
		return 1
	for p in range(2, int(math.log2(A)) + 1):
		a = round(A ** (1/p))
		if a >= 2 and a**p == A:
			return 1
	return 0
```

## Excel Column Number
Given a string A representing a column title. Return its corresponding column number.

Ex: `AB` output: 28

For a string $A = a_1a_2...a_k$ :
$$
\text{Value} = \sum_{i = 1}^k (\text{value of } a_i)\times 26^{k-i}
$$
Where value of $a_i$ $= ascii[a_i] - ascii['A'] + 1$

```python
def titleToNumber(A):
	result = 0
	for char in A:
		digit = ord(char) - ord('A') + 1
		result = result*26 + digit
	return result
```

## Excel Column Title
Given a positive int $A$, return its corresponding Excel Column Title.

$$
\text{current letter } = chr('A' + (A-1) \space \% \space 26)
$$
$$
A = \lfloor \frac{A-1}{26} \rfloor
$$

```python
def convToTitle(A):
	s = ""
	while A > 0:
		A -= 1 #zero based
		r = A % 26
		s += chr(ord('A') + r)
		A //= 26
	return s[::-1]
```

## Digit Ops
Determine whether an int A is a palindrome.

```python
def is_palindrome_num(A):
	if A < 0 or (A%10 == 0 and A != 0):
		return False
	rev = 0
	n = A
	while n > rev:
		rev = rev*10 + n%10
		n //= 10
	return n == rev or n == rev//10
```

## Next Smallest Palindrome
Given a numeric string $A$ (no leading zeroes). Find the smallest palindrome $> A$.

**Crazy Shit**: If $A$ only has $9$s, the answer is always $100..001$. For $n$ digits: $1$ followed by $n-1$ zeroes then $1$. 

For other cases, we can try to mirror the left half into right half to form a palindrome.

If that is not big enough, increment the middle, handle the carries, then mirror again.

```python
def next_palindrome(A):
	A = str(A)
	n = len(A)
	if A == '9'*n:
		return '1' + '0'*(n-1) + '1'
	#mirror left to right
	def mirror(s):
		half = s[:(n//2)]
		if n%2 == 0:
			return half + half[::-1]
		else:
			return half + s[n//2] + half[::-1]
	mirrored = mirror(A)
	if mirrored > A:
		return mirrored
	#incr middle and mirror again
	t = list(A)
	mid = n//2
	carry = 1
	if n % 2 == 1:
		i = mid
		t[i] = str((int(t[i]) + carry) % 10)
		carry = (int(A[i]) + 1) // 10
		left = i-1
		right = i+1
	else:
		left = mid - 1
		right = mid
	while left >= 0 and carry:
		num = int(t[left]) + carry
		t[left] = str(num % 10)
		carry = num // 10
		t -= 1
	#mirror again
	for i in range(n//2):
		t[-(i+1)] = t[i]
	return ''.join(t)
```
