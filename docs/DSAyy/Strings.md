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
