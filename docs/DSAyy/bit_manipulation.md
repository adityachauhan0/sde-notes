## Number of 1 bits (Hamming / Popcount)
Given a non-neg int A, return the number 1 bits in it.

```python
def countSetBits(A):
	count = 0
	while n:
		n &= (n-1) #clears the least significant set bit
		count += 1
	return count
```

## Trailing Zeroes in Binary Representation
Given an int, count the trailing zeroes in binary representation.

```python
def trailBinZer(A):
	count = 0
	while A&1 == 0:
		count += 1
		A >>= 1
	return count
```

## Reverse Bits
Given a 32 bit unsigned int, return the value with all bits reversed.

A = 3 (00000000000000000000000000000011), Output: 11000000000000000000000000000000 = 3221225472

```python
def reverse(A):
	rev = 0
	for i in range(32):
		rev = (rev << 1) | (A & 1)
		A >>= 1
	return rev
```

## Divide Integers (without multiplication, div or mod)
Given two numbers, divide them without multiplication, division or modulus operations.

If overflow, return INT_MAX ($2^{31} - 1$) 
### How
For each bit from 31 to 0, check if (B << shift) fits in the dividend.

If yes, set the corresponding bit in the quotient, and subtract (B << shift) from the dividend.

Finally assign the corresponding sign.

```python
def divide(A,B):
	INT_MAX = 2**31 - 1
	INT_MIN = -2**31
	if A == INT_MIN and  B == -1:
		return INT_MAX
	negative = (A < 0) != (B < 0)
	dividend = abs(A) #converting to 64 bit unsigned
	divisor = abs(B)
	result = 0
	for shift in range(31,-1,-1):
		if (dividend >> shift) >= divisor:
			result |= (1 << shift)
			dividend -= (divisor << shift)
	if negative:
		result = -result
	return result
```

## Different Bits Sum Pairwise
Given N positive int $A_1...A_N$, compute the sum:
$$
\sum_{i=1}^{N}\sum_{j=1}^{N}f(A_i,A_j)
$$
where f(X,Y) is the number of different bits between x and y. Return the result modulo $10^9 + 7$

Ex: A = 1,3,5 Output = 8
### How
Instead of $N^2$ , use the **linearity of counting**- For each bit position (0 to 31),
count how many numbers have bit set and unset (ones and zeroes). Each differing pair adds 1 to $f(A_i,A_j)$ for both $(i,j)$ and $(j,i)$.

$$
cntPairs = 2 \times ones \times zeros
$$
because both $(i,j)$ and $(j,i)$ count.

```python
def cntBits(A):
	MOD = 10**9 + 7
	n = len(A)
	bitCount = [0]*32
	for x in A:
		for b in range(32):
			if x & (1 << b):
				bitcount[b] += 1
	ans = 0
	for b in range(32):
		ones = bitcount[b]
		zeros = n - ones
		ans = (ans + (2*ones*zeros) % MOD) % MOD
	return ans
```

## Count Total Set Bits from 1 to A
Given a positive int $A$, count the total set bits in the binary representation of all numbers from $1$ to $A$. Return ans modulo $10^9 + 7$

### How
Cant be naive, I am standing on business, isn't that clocking to you.

For each bit position, observe nigga, Bit $i$ in all numbers cycles between blocks of 0s and 1s of length $2^i$.

Over a range 0 to A, for each full block length of $2^{i+1}$, bit i is 1 for exactly $2^i$ times.

How many full cycles in A? $fullCycles = \frac{A+1}{2^{i+1}}$ 

Each full cycle contributes $2^i$ set bits.

Now for the leftovers, if $rem = (A+1)\%2^{i+1}$ the bit is 1 for atmost $max(0,rem - 2^i)$ numbers.

Sum for all i.

```python
def countTotalSetBits(A):
	MOD = 10**9 + 7
	n = A+1
	ans = 0
	for i in range(31):
		bitMask = 1 << i
		cycleLen = 1 << (i+1)
		fullCycles = n // cycleLen
		ans += fullCycles*bitMask
		rem = n % cycleLen
		extra = max(0, rem - bitMask)
		ans += extra
	return ans % MOD
```

## Palindromic Binary Representation

Given an int $A$, find the $A$th positive integer, whose **binary representation is a palindrome**.

Note: first number is 1.

### How
We can generate Ath instead of checking each one.

For every bit length L, we can count how many palindromic exists. Left half of the bit determines the palindrome, right is just a mirror.

Highest bit is always $1$.

Steps:

1. Find the bit-length L such that Ath palindrome lies within.

2. Within palindromes of length L, determine the index of our target.

3. Construct the left half. (with top bit 1 and the rest from binary counting)

4. Mirror this to complete the palindrome.

Ex: Find the 9th one

- L = 5 (1 + 2 + 2 + 4 = 9 so it falls in this length)

- Half is $\lceil 5/2 \rceil = 3$ bits. The prefix ranges from $100_2$ to $111_2$ (4 possibilites)

- $9$th means prefix is $100_2$ since $9$ is the first among $L = 5$ palindromes.

- $11011_2 = 27_{10}$

```python
import math
def palindromicBinary(A):
	remaining = A
	for L in range(1,64): #find the length of palindromic binary
		half = (L+1)//2 
		count = 1 << (half - 1)
		if remaining > count:
			remaining -= count
		else:
			break
	idx = remaining - 1
	prefix = (1 << (half - 1)) | idx #add leading 1 to maintain length
	result = prefix
	toMirror = prefix >> (L%2) #skip middle if L is odd

	for _ in range(half):
		result = (result << 1)|(toMirror & 2)
		toMirror >>= 1
	return result	
```

## XOR-ing the subarrays

Given an array A of N integers, perform the following:

1. For every contiguous subarray, XOR its elements.

2. XOR all these results, return the final value.

### How
An element affects the XOR only if it appears odd number of times in all subarrays.

Number of subarrays containing $A[i] = (i+1) \times (N-i)$

**Fact**: When N is even, every $(i+1) \times (N-i)$ is even for all i, so ans is $0$.

When N is odd, only the elements at even indices (0-based), have $(i+1) \times (N-i)$ odd.

When N is odd, just XOR the even index elements.

```python
def XORSub(A):
	N = len(A)
	if (N&1)==0: return 0
	ans = 0
	for i in range(0,N,2):
		ans ^= A[i]
	return ans
```

## Single Number (Every Number Appears Twice Except One)
Int array A, every element appears twice except one, find.

Just XOR the whole thing.

```python
def singleNumber(A):
	ans = 0
	for x in A:
		ans ^= x
	return ans
```

## Single Number II (Every Element Appears Thrice Except 1)
Given an array A, every element appears thrice except one, find it.

### How
Hashmap nononono

Sigma: use variables $ones$ and $twos$, to record the bitwise modulo $3$ sum for all numbers seen so far. Each bit in $ones$ and twos$ $forms a $base-3$ counter for that bit position.

- ones records the bits seen once modulo 3
- twos records bits seen twice modulo 3
- for new x:
	
	- Add $x$'s bits to $ones$ where not already in $twos$
	
	- Add $x$'s bits to $A$ where not already in updates $ones$.

After all elements, $ones$ contains the answer.

```python
def singleNumber(A):
	ones = 0 #bits seen exactly once (modulo 3)
	twos = 0 #bits seen exactly twice (modulo 3)
	for x in A:
		ones = (ones ^ x) & ~twos
		twos = (twos ^ x) & ~ones
	return ones
```


### 🥊 **"The One Among Fakes"**

In a world full of noise, where fakes come in threes,  
I stand like a king, with code that don’t freeze.  
Most of 'em copy, repeat, disappear —  
But one’s got that spark, that truth crystal clear.

---

**"I track with two masks,"** I boldly declare,  
Ones and twos — yeah, I split it with flair.  
First time you show up? You land in my grip.  
Second time? I move you — tighten the zip.

But come at me thrice? You're gone, erased.  
This empire’s mine — I don't leave a trace.  
Only the rare stays in my throne,  
While impostors fall like kings de-throned.

---

Bit by bit, I scan with might,  
With logic sharper than a knight.  
When the dust settles and lies are gone,  
The truth remains — the chosen one.

---

**So when they ask, “Who’s left?” — I say with pride,**  
_"The man who showed once, and never had to hide."_  
He walks through zeros, untouchable flame —  
And `ones` holds his code, forever the name.
