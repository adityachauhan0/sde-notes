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
\text{current letter } = chr('A' + (A-1) \space mod \space 26)
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


## Number Theory

### Greatest Common Divisor (GCD)
Given two non-neg int $A$ and $B$, find their $GCD$. Don't use library functions.

Simple recurrence for this.
$$
gcd(A,B) = gcd(B, A \space mod \space B)
$$
```python
def __gcd(a,b):
	if b == 0:
		return a
	else:
		return gcd(b, a % b)
```

## Find N'th Fibonacci Number (mod)
Given int A, find the Ath fibonacci number mod $10^9 + 7$

### How
We can be a noob and use $F_n = F_{n-1} + F_{n-2}$, but we can use something way more big dick energy.

**Mathematical Trick** : Fast Doubling/ Matrix Exponentiation.
We can compute this shit in $log(n)$ using:
$$
F_{2k} = F_k \times [2F_{k+1} - F_k]
$$
$$
F_{2k + 1} = F_k^2 + F_{k+1}^2
$$
```python
def fib_pair(n):
	if n == 0:
		return (0,1)
	(a,b) = fib_pair(n//2)
	c = a*(2*b - a) % MOD
	d = (a*a + b*b) % MOD
	if n % 2 == 0:
		return (c,d)
	else:
		return (d, (c+d)% MOD)
def nthFib(A):
	return fib_pair(A)[0]
```

## Divisible by 60
Given a large number represented in array of digits, determine if we can rearrange it to form a number divisible by 60.

### How
To be divisible by 60, it needs to be divisible by 5, 4 and 3.

Divisible by 5: Last digit must be 5 or 0

Divisible by 4: Last two digits form a number divisible by 4.

Divisible by 3: Sum of digits must be divisible by 3.

How to check for an arrangement?

- There must be atleast one 0 digit (to end with 0)

- There must be atleast one more even digit, so that the last two digits can be even digit 0.

- Sum of digits must be divisible by 3.

```python
def divisibleBy60(A):
	if len(A) == 1:
		return 1 if A[0] == 0 else 0
	sum_digits = 0
	zero_count = 0
	even_count = 0
	for d in A:
		sum_digits += d
		if d == 0: zero_count += 1
		if d % 2 == 0: even_count += 1
	if sum_digits % 3 == 0 and zero_count >= 1 and even_count >= 2:
		return 1
	else:
		return 0
```

## Powerful Divisors
Given an array of integers, for each X in A, find the count of Y such that $1 \leq Y \leq X$ and number of divisors of Y is a power of 2.

Return an array with answer for every X in A.

### How

- Let $d(Y)$ be the num of positive divisors of Y.

- A number is a power of 2 if it can be written as $2^k$ for some k $\geq$ 0

For each X in A, count the Y's in $[1,X]$ such that $d(Y)$ is a power of 2.

Now to compute d(Y), we can use a modified sieve to compute $d(y)$ for all $y$ $\leq A_{max}$ 

Step 2: Precompute prefix counts of **good** numbers.

Let $P(x)$ be the count of numbers in $[1,x]$ with $d(y)$ a power of 2.

So for each $X$ in $A$, our answer is $P(X)$.

To check if a number is a power of $2$, $n > 0$ and $n \& (n-1) == 0$.

```python
from math import isqrt
def powerfulDivisors(A):
	N = len(A)
	if N == 0: return []
	M = max(A) #precompute till this
	divs = [0]*(M+1) #div count sieve
	for i in range(1,M+1):
		for j in range(i, M+1, i):
			divs[j] += 1
	def is_power_of_two(x):
		return x > 0 and (x &(x-1)) == 0
	pref = [0]*(M+1) #pref sum of good numbers
	for i in range(1,M+1):
		pref[i] = pref[i-1]+ (1 if is_power_of_two(divs[i]) else 0)
	return [pref[x] for x in A]
```

## Trailing Zeroes in Factorial

Given an int, return the number of trailing zeroes in A!.

### How

We need to count the 5 $\times$ 2 s. The count of 2 is always gonna be more than the count of 5s. So just check:

Just check how many times 5 divides A factorial. 
$$
\lfloor \frac{A}{5} \rfloor + \lfloor \frac{A}{25} \rfloor ...
$$

So the Trailing Zeroes in A! $= \sum_{k = 1}^{\infty}\lfloor \frac{A}{5^k} \rfloor$ 

```python
def trailingZeroes(int A):
	count = 0
	p = 5
	while p <= A:
		count += A//p
		p *= 5
	return count
```

## Sorted Permutation Rank
Given a string $A$ with no repeated chars, return the rank of $A$ among all its lexicographically sorted permutations. Return the ans modulo $1000003$.

### How
For each char in string from left to right, cound how many unused chars are smaller than the current one.

For each such smaller chars, all permutations that can start with that char and  use the rest of the letters come before A.

$n = |A|$ and $A_i$ be the $i$-th character. For each $i$ $0 \leq i \leq n$:

smaller = number of unused chars $< A_i$

Permutations with $A_0...A_{i-1}$ then a smaller char : $smaller \times (n-1-i)!$

So the total rank: $1 + \sum_{i = 0}^{n-1}(\text{number of unused chars < } A_i \times (n-1-i)!)$

```python
from collections import Counter
MOD = 10**9 + 7
def findRank(A):
	n = len(A)
	rank = 1
	fact = [1]*(n+1)
	for i in range(1,n+1):
		fact[i] = (fact[i-1]*i) % MOD
	freq = Counter(A) #freq count of all chars
	for i,c in enumerate(A):
		freq[c] -= 1
		smaller = sum(freq[ch] for ch in freq if ch < c)
		rank = (rank + smaller*fact[n-1-i]) % MOD
	return rank
```


## Largest Coprime Divisor
Given two positive int $A$ and $B$. find largest int $X$ such that $X$ divides $A$ and $gcd(X,B) = 1$. (that is, $X$ and $B$ are coprime).

Ex: A = 30, B = 12, Answer = 5

$$
\text{while gcd(A,B) > 1}: A \equiv \frac{A}{gcd(A,B)} 
$$
Then A would be the answer.

```python
def largCD(A,B):
	while gcd(A,B) > 1:
		A = A // gcd(A,B)
	return A
```

## Sorted Permutation Rank with Repeats

Given a string A with repeated chars, find the Rank of A among all its permutations.

Return the rank modulo $10^6 + 3$

Ex: A = "aba": Rank of aba is 2.
### How

Suppose A has n chars, possibly repeated. The number of unique permutations of A is:
$$
Total = \frac{n!}{\prod_c(freq[c])!}
$$
The rank of A is found by counting how many unique strings are lexicographically smaller than A.

At each index i ($0$ to $n-1$):

1. For each char $c$ strictly less than $A[i]$ that is still available,

2. Place c at position i, then count how many unique permutations can be made with the remaining multiset.

$$
count = \frac{(n-i-1)!}{\prod_x(freq^`[x])!}
$$
```python
from math import factorial
from collections import Counter
MOD = 1000003
def modinv(a,mod):
	return pow(a,mod - 2, mod)
def findRank(A):
	n = len(A)
	fact = [1]*(n+1)
	inv_fact = [1]*(n+1)
	for i in range(n):
		fact[i] = (fact[i-1]*i) % MOD
	inv_fact[n] = modinv(fact[n], MOD)
	for i in range(n-1,-1,-1):
		inv_fact[i] = (inv_fact[i+1]*(i+1)) % MOD
	freq = Counter(A)
	chars = sorted(freq.keys())
	ans = 0
	for i in range(n):
		cur = A[i]
		rem = n - 1 - i
		for c in filter(lamda x: x < cur, chars):
			if freq[c] == 0:
				continue
			freq[c] -= 1
			#compute permutations of remaining chars
			perms = fact[rem]
			for val in freq.values():
				perms = (perms*inv_fact[val])% MOD
			ans = (ans + perms) % MOD
			freq[c] += 1
		freq[cur] -= 1
		if freq[cur] == 0:
			chars.remove(cur)
	return (ans + 1) % MOD
```

## K-th Permutation
Given an integer A (length of the permutation) amd an int B, return the Bth permutation of A.

### How

There are A! permutations in total.

Fix the first element: for $[1,2....A]$, the first element can be any of the A numbers, and for each such choice, there are (A-1)! ways to permute the rest.

To find the Bth permutation, we can decide the first number, by figuring out how many blocks of size (A-1)! fit into B-1 (0-based index).

Ex: 

A = 3; All perms = $[1, 2, 3], [1, 3, 2], [2, 1, 3], [2, 3, 1], [3, 1, 2], [3, 2, 1]$ 

Block size for first element: 2! = 2

So permutations are grouped as:
$$Start with 1: [1, 2, 3], [1, 3, 2] \space
Start with 2: [2, 1, 3], [2, 3, 1] \space
Start with 3: [3, 1, 2], [3, 2, 1]$$
If B = 3, zero-based: B-1 = 2, 2/2 = 1. So second group ($[2,1,3]$). Within group, 2 mod 2 = 0, so first element in group: $[2,1,3]$

Let S be the set of available numbers, At position i ($i \leq i \leq A$)
$$
index = \lfloor \frac{B'}{(A-1-i)!} \rfloor
$$
Pick the $index$-th unused element, Update $B' = B' \space mod \space (A-1-i)!$

### Mathematical Diss

Partitioning List into equal sized blocks.

Fix the first element, 

- If the first element is 1, the remaining A-1 numbers can appear in $(A-1)!$ ways.

- IF the first element is 2, same $(A-1)!$ ways.

Hence the full list can be viewed as $(A-1)!$ sized blocks.

Now turning the rank b into factorial based digits.

r = B - 1 (zero-based)

Divide R successively by factorials.

| step (position) | divisor  | quotient = digit                          | remainder               | interpretation                      |
| --------------- | -------- | ----------------------------------------- | ----------------------- | ----------------------------------- |
| 1st element     | (A-1)!   | $d_{A-1}=\big\lfloor r/(A-1)!\big\rfloor$ | $r\gets r \bmod (A-1)!$ | pick the $d_{A-1}-th$ unused number |
| 2nd element     | (A-2)!   | $d_{A-2}=\big\lfloor r/(A-2)!\big\rfloor$ | $r\gets r \bmod (A-2)!$ | pick the $d_{A-2}-th$ unused number |
| …               | …        | …                                         | …                       | …                                   |
| last element    | 0!=10!=1 | $d_{0}=r$                                 | 0                       | only one number left                |

```python
from math import factorial

def kth_permutation(A,B):
	B -= 1 #zero based
	symbols = list(range(1,A+1))
	result = []
	#repeatedly pick the correct element by factorial base digits.
	for remaining in range(A,0,-1):
		block = factorial(remaining - 1)
		idx, B = divmod(B, block)
		result.append(symbols.pop(idx))
	return result
```

## City Tour

Given $A$ cities from $1$ to $A$. $M$ of them are already visited, given by an array $B$ (of length $M$). From any visited city, you can visit adjacent unvisited cities $(i-1)$ or $(i+1)$. At each step, you can choose any available city that is adjacent to a visited city. Find the number of ways to visit all cities, modulo $10^9 + 7$.

For A = 5, B = $[2,5]$, All poss ways are:

1. 1→3→4

2. 1→4→3

3. 3→1→4

4. 3→4→1

5. 4→1→3

6. 4→3→1

### How

Partition the cities into segments.

$L_0$ : before the first visited city.

$L_1..L_{M-1}$: Between the visited cities.

$L_M$ after the visited city.


So the edge segments have only one way to travel, but..

MIddle segments have $2^{k-1}$ ways to travel (since we can choose left or right)

So the ways:
$$
ways = \frac{N!}{L_0!...L_M!} \prod_{i=1}^{M-1} 2^{L_i - 1}
$$

Given;
$$A = 5, B = [2,5] \space
L_0 = 2 - 1 = 1 \space L_1 = 5 − 2 − 1 = 2 \space L_2 = 5-5 = 0 \space N = 5-2 = 3
$$
$$
ways = \frac{3!}{1!2!0!} \times 2^{2-1} = 6
$$

```python
MOD = 1_000_000_007
def city_tour(A,B):
	B = sorted(B)
	gaps = [B[0] - 1] + [B[i] - B[i-1] - 1 for i in range(1,len(B))] + [A-B[-1]]
	unvisited = A - len(B)
	fact = [1]*(A+1)
	for i in range(1,A+1):
		fact[i] = fact[i-1]*i % MOD
	inv_fact = [1]*(A+1)
	inv_fact[A] = pow(fact[A], MOD-2, MOD)
	for i in range(A,0,-1):
		inv_fact[i-1] = inv_fact[i]*i % MOD
	ans = fact[unvisited]
	for g in gaps:
		ans = ans * inv_fact[g] % MOD
	#internal gaps add more
	for g in gaps[1:-1]:
		if g >= 2:
			ans = ans * pow(2,g-1,MOD) % MOD
	return ans
```

## Grid Unique Paths
You are on the topleft of $A \times B$ grid. Robot can move only down and right.
How many unique paths are there from top-left to bottom-right.

### How
Each unique path is a sequence of A-1 down moves and B-1 right moves.

In total we make: $t = (A-1) + (B-1) = A + B - 2$ moves. Out of which $A-1$ are down.

So total unique paths would be:
$$
\text{Unique Paths } = \binom{A+B-2}{A-1} = \frac{(A+B-2)!}{(A-1)! \cdot (B-1)!}
$$
we can also write
$$
\binom{t}{k} = \prod_{i = 1}^{k} \frac{t - k + i}{i}
$$
So that means no factorial needed.

```python
def uniquePaths(A,B):
	n,m = A-1, B-1
	k = min(m,n)
	t = m+n
	res = 1
	for i in range(1,k+1):
		res *= (t-k+1)//i
	return res
```

## Highest Score
You are given a N \times 2 string array A. Where each row consists of students name and their marks.
Find maximum average marks for any student (rounding down).

```python
def highestScore(records):
	from collections import defaultdict
	scores = defaultdict(lambda: [0,0])
	for name, mark in records:
		mark = int(mark)
		scores[name][0] += mark #total score
		scores[names][1] += 1 #count
	best = 0
	for total,count in score.values():
		avg = total // count
		best = max(best,avg)
	return best
```

## Addition without summation.
Add two numbers without summation.

So basically sumBits = $A \oplus B$ and carryBits = $(A \& B) << 1$ 

```python
def add(A,B):
	while B != 0:
		sumBits = A^B
		carryBits = (A&B) << 1
		A = sumBits
		B = carryBits
	return A
```

## Next Similar Number

Given a numeric string A, find the next_permutation that is greater than A.

Find the pivot (first index i such that $A[i] < A[i+1]$). If there is none, we have no next number. 

Find the successor (smallest digit from the right of $i$, that is $> i$)

Swap $A[i]$ and $A[j]$.

Reverse from $A[i+1]$ to the end.

```python
def next_permutation(A):
	A = list(A)
	n = len(A)
	i = n-2

	while i >= 0 and A[i] >= A[i+1]:
		i -= 1
	if i < 0:
		return "-1"
	j = n-1
	while A[j] <= A[i]:
		j -= 1
	A[i],A[j] = A[j], A[i]
	A[i+1:] = reversed(A[i+1:])
	return ''.join(A)
```

## Rearrange Array 
Given an array, rearrange such that $A[i]$ becomes $A[A[i]]$.

All elements are in range $[0,N-1]$, and distinct.

### How
Let $A[i] = \text{ old value } + N \times \text{ new value}$

then decode by $A[i] \gets \lfloor \frac{A[i]}{N} \rfloor$

```python
def arrange(A):
	N = len(A)
	for i in range(N):
		old = A[i]
		newv = A[old] % N
		A[i] = old + newv*N
	A = [i//N for i in A]
```

## Number of length B and value less than C
Given a set of digits $A$ (sorted, may contain 0), and int $B$ and $C$.
Count how many $B$-digit numbers are strictly less than $C$.

A = 0 1 5, B = 1, C = 2. Output = 2 (0,1)

### How

Let C have digits: $C_1C_2..C_B$ We iterate over each digit position $i$ from 0 to $B-1$

For each position, count how many digits in $A$ are $< C_i$ (for $i = 0$, skip $0$ if $B > 1$)

For every such choice, the rest of $B-1- i$ digits can be anything in A.

If $C_i$ itself not in $A$, stop.

```python
from bisect import bisect_left

def countNumbers(digits,length,upper):
	digits = sorted(set(digits))
	d = len(digits)
	if d == 0 or length == 0:
		return 0
	upper_str = str(upper)
	u_len = len(upper_str)
	if u_len < length:
		return 0
	# precomp powers: p[k] = d**k k <= length
	p = [1]
	for _ in range(length):
		p.append(p[-1]*d)
	# ── 1.  C has more digits than `length` → only “length-digit universe” counts
    #     First digit cannot be 0 if length > 1
    if u_len > length:
	    if length == 0:
		    return d
		non_zero_first = d - (1 if 0 in digits else 0)
		return non_zero_first * p[length -1]
	# ── 2.  u_len == length:  digit-by-digit scan
    total = 0
    for i in range(length):
	    current_digit = int(upper_str[i])
	    #how many avail are smaller than cur digit
	    smaller = bisect_left(digits,current_digit)
	    if i == 0 and length > 1 and 0 in digits:
		    smaller -= 1 #cant start from 0
		# add combinations for selecting a “smaller” digit here
		total += smaller*p[length - 1 -i]
		# if the current digit of C is not present in our digit set,
        # we cannot continue matching the prefix ⇒ stop the scan
		if current_digit not in digits:
			break
	return total
```
