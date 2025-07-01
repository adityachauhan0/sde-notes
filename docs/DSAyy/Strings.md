## Palindrome String
Given a string, determine if its palindrome.

```python
def isPalindrome(A):
	i = 0
	j = len(A) - 1
	while i < j:
		while i < j and not isalnum(A[i]):
			i += 1
		while i < j and not isalnum(A[j]):
			j -= 1
		if tolower(A[i]) != tolower(A[j]):
			return 0
		i += 1
		j -= 1
	return 1
```
## Vowel and Consonant Substrings
Given string A of lowercase characters. Find the number of substrings in A which start with a vowel and end with a consonant, or vice-versa.

```python
def substrings(A):
	MOD = 10**9 + 7
	V = 0 #vowels
	for char in A:
		if char in ['a', 'e', 'i', 'o', 'u']:
			V += 1
	C = len(A) - V
	return (V * C) % MOD
```

## Remove Consecutive Characters
Given a string, and an int `B`, remove all consecutive identical characters that have length exactly `B`.

Ex: `aabbccd` Output = `d`

```python
def remCon(A,B):
	runs = []
	for c in A:
		if not runs or runs[-1][0] != c: #if its not the same as last seen char
			runs.append([c,1])
		else:
			runs[-1][1] += 1
	ans = ""
	for char, count in runs:
		if count == B:
			continue
		ans += char * count
	return ans
```

## Serialize
Given an array of strings. **Serialize** the array and return the resulting substring.

To serialize, append:
1. og string s
2. its length
3. a delimiter(~).

Ex: "interviewbit" = "interviewbit12"

[`scaler`,`academy`] -> scaler6~academy7~

```python
def serialize(A):
	result = ""
	for s in A:
		result += s
		result += to_string(len(s))
		result += '~'
	return result
```

## Deserialize
Given a serialized string, deserialize it.
```python
def deserialize(A):
	res = []
	i = 0
	while i < len(A):
		#get word
		j = i
		while j < len(A) and A[j].isalpha():
			j += 1
		#get digit
		k = j
		while k < len(A) and A[k].isdigit():
			k += 1
		#A[k] should be delimiter
		res.append(A[i:j])
		i = k + 1
	return res
```


## String and Its Frequency
Given a string of lowercase chars, return a string in which for each character, its total freq is written immediately after its appearance.

Ex: `abbhuabcfghh` Output: `a2b3h3u1c1f1g1`

```python
def strFreq(A):
	freq = [0]*26
	seen = [False]*26
	order = []
	for c in A:
		idx = ord(c) - ord('a')
		freq[idx] += 1
		if not seen[idx]:
			seen[idx] = True
			order.append(c)
	ans = ""
	for c in order:
		idx = ord(c) - ord('a')
		ans += c + str(freq[idx])
	ans
```

## Bulls and Cows

Write down a secret number, your friend makes a guess (of equal length, all digits).
For each guess, provide a hint.
- Bulls: Digits that match both value and position.
- Cows: Digits in the guess that are in the secret but in the wrong position.
- Each digit can be counted only once, as a bull or as a cow.
- Output is formatted as "xAyB" where $x$ is bulls, and $y$ is cows.
- Both strings contain digits only.

Input:
- secret: string of digits
- guess: string of digits
Output:
- A string of format "xAyB"

Ex: secret: "1807", guess = "7810". Output = "1A3B"

```python
def bullsCows(secret, guess):
	bulls = 0
	cows = 0
	cnt = [0]*10
	for i in range(len(secret)):
		s = int(secret[i])
		g = int(secret[i])
		if s == g:
			bulls += 1
		else:
			if cnt[g] > 0:
				cows += 1
			if cnt[s] < 0:
				cows += 1
			cnt[s] += 1
			cnt[g] -= 1
	return f"{bulls}A{cows}B"
```

## Self Permutation
Given 2 strings A and B, determine if there exists a permutation where bot A and B are equal.

Basically check if they are anagrams.

```python
def permuteStrings(A,B):
	if len(A) != len(B):
		return 0
	freq = [0]*26
	for c in A:
		freq[ord(c) - ord('a')] += 1
	for c in B:
		freq[ord(c) - ord('a')] += 1
		if freq[ord(c) - ord('a')] < 0:
			return 0
	return 1
```


## Longest Common Prefix
A = abcde, aefg, asdasd, Output = `a`

### How
1. Find length of the shortest string.
2. Stop and return the prefix so far on the first mismatch.

```python
def longestCommonPrefix(A):
	if not A:
		return ""
	minLen = min(len(s) for s in A)
	for i in range(minLen):
		c = A[0][i]
		for s in A:
			if s[i] != c:
				return A[0][:i]
	return A[0][:minLen]
```

## Count and Say
1. Start with 1
2. Then say the count and the key of the count.
- 1211 -> 111221
- 21 -> 1211
Given int `n`, generate the nth term, in the count and say sequence.
So
1. n = 2, output = 11
2. n = 5, 111221
	- 1; 11; 21; 1211; 111221
	- Each term describes the previous term.
```python
def countAndSay(n):
	res = "1"
	for seq in range(2,n+1):
		next = ""
		i = 0
		while i < len(res):
			count = 1 #count the freq of cur el
			while i + 1 < len(res) and res[i] = res[i+1]:
				i += 1
				count += 1
			next += str(count) + res[i] #add it to the res
			i += 1
		res = next
	return res
```

## Amazing Subarrays
Given a string, find the num of amazing substrings.

Subtring is amazing if it starts with a vowel.
Substrings are continuous.

### How
For each index i, if $A[i]$ is a vowel, add (n-i) to the answer. Since there are (n-i) substrings starting fron i.

```python
def amazingSubs(A):
	MOD = 10003
	ans = 0
	for i,c in enumerate(A):
		if c.lower() in ['a','e','i','o','u']:
			ans += len(A) - i
			if ans >= MOD: ans %= MOD
	return ans % MOD
```

## Implement StrStr
Given a string (haystack) and another string (needle), return the index of the first occurence of `B` in `A`, or `-1` if `B` does not occur in `A`.

### How
Knuth Morris Pratt algorithm.
Precompute an `LPS` (longest prefix suffix) of the needle in the haystack.
Slide this pattern over the haystack, when a mismatch occurs, jump using the LPS array to avoid unnecesarry comparisons.

```python
def strStr(A,B):
	if len(B) == 0:
		return -1
	if len(A) == 0 or len(B) > len(A):
		return -1
	lps = buildLPS(B)
	i = j = 0
	while i < len(A):
		if A[i] == B[j]:
			i += 1
			j += 1
			if j == len(B):
				return i - j
		else:
			if j != 0:
				j = lps[j-1]
			else:
				i += 1
	return -1
def buildLPS(pat):
	lps = [0]*len(pat)
	length = 0 #len of the longest prefix suffix
	i = 1
	while i < len(pat):
		if pat[i] == pat[length]:
			length += 1
			lps[i] = length
			i+= 1
		else :
			if length != 0:
				length = lps[length - 1]
			else:
				lps[i] = 0
				i += 1
	return lps
```

## Stringoholics
Given arr of strings, each made only from `a` and `b`. At each time i, every string is circularly rotated to the left by (`i` % `length`) letters.

After some time, a string returns to its original form.

**Goal**: Find the min time `t` where all strings in A are at their original state simultaneously.
Output: t mod $10^9 + 7$ 

Example:
1. A = a, ababa, aba; Output = 4
2. A = a, aa; Output = 1

### How
1. For each string , find the first time `t` such that repeated by that time it returns to its starting form.
2. For each string, its **reset period** p is found via KMP (smallest period of repetition in the string)
3. For each p, find minimal t such that $\frac{t(t+1)}{2}$ mod p = 0 (the net number of rotations modulo p is 0)
4. The answer is lcm(all t) (smallest time that aligns)

```python
from math import gcd
from functools import reduce
MOD = 10**9 + 7
def lcm(a,b):
	return a* b//gcd(a,b)
def lcm_list(lst):
	return reduce(lcm,lst)
def buildLPS(s):
	n = len(s)
	lps = [0]*n
	length = 0
	i = 1
	while i < n:
		if s[i] == s[length]:
			length += 1
			lps[i] = length
			i += 1
		else:
			if length != 0:
				length = lps[length - 1]
			else:
				lps[i] = 0
				i += 1
	return lps
def get_period(s):
	lps = buildLPS(s)
	n = len(s)
	length = lps[-1]
	if length > 0 and n % (n - length) == 0:
		return n - length
	else:
		return n
def minimal_t(p):
	t = 1
	while (t * (t+1))//2 % p != 0:
		t += 1
	return t
def stringoholics(A):
	t_list = []
	for s in A:
		p = get_period(s)
		t = minimal_t(p)
		t_list.append(t)
	return lcm_list(t_list) % MOD
```

## Min Characters to Make a string palindrome.
Given a string, the only allowed operation is to insert chars at the **beginning** of the string. Find the min chars that must be inserted to make A a palindrome.

A = `ABC`, Output = `2`

### How
Intuition
- To minimize the insertions, find the **longest palindromic prefix** of A.
- All the characters after this prefix must be mirrored at the front.
KMP
- Let B be reverse of A
- Build T = A + `#` + reverse(A)
- Compute the LPS(T)
- L = lps(end) gives length of the longest palindromic prefix in A
- answer = n - L, where $n = |A|$ .

```python
def buildLPS(s):
	n = len(s)
	lps = [0]*n
	length = 0
	i = 1
	while i < n:
		if s[i] == s[length]:
			length += 1
			lps[i] = length
			i += 1
		else:
			if length != 0:
				length = lps[length - 1]
			else:
				lps[i] = 0
				i += 1
	return lps
def minCharsToMakePalindrome(A):
	revA = A[::-1]
	T = A + '#' + revA
	lps = buildLPS(T)
	LPS_len = lps[-1]
	return len(A) - LPS_len
```


## Convert to Palindrome
Given a string of lowercase chars, determine if it is possible to make it a palindrome by removing exactly one character.

A = `abcba`, Output = 1

A = `abecbea`, Output = 0

```python
def isConvertable(A):
	l,r = 0, len(A) - 1
	while l < r:
		if A[l] == A[r]:
			l += 1
			r -= 1
		else:
			return isPalindrome(A, l+1, r) or isPalindrome(A,l, r-1)
	return 1
def isPalindrome(A,i,j):
	while i < j:
		if A[i] != A[j]: return False
		i += 1
		j -= 1
	return True
```
