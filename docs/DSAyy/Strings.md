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
		freq[ord(c) - ord('a')] -= 1
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

## Min Appends for Palindrome
Given a string, find min appends at the end of the string to make it a palindrome.

### How
We want to append as little as possible. So find the longest common palindromic suffix of A.

Part before this suffix must be mirrored and appended.

How do we do this.
1. Let R = reverse(A)
2. Build string T = R + `#` + A
3. Compute LPS (longest prefix suffix array) for T.
4. L = lps(T.length - 1) is the length of the longest palindromic suffix in A.
5. min appends is n - L

```python 
def computeLPS(s):
	lps = [0]*len(s)
	length = 0 #len of prev longest prefix suffix
	for i in range(1,len(s)):
		while length > 0 and s[i] != s[length]:
			length = lps[length - 1]
		if s[i] == s[length]:
			length += 1
			lps[i] = length
	return lps
def minAppendsForPalindrome(A):
	R = A[::-1]
	T = R + '#' + A
	lps = computeLPS(T)
	L = lps[-1]
	return len(A) - L
```

## Min Parentheses
Given a string containing bracket sequence or `(` and `)`. Find the min number of parenthesis you must add at any position to make A a valid sequence.

```python
def minParents(A):
	open = 0
	inserts = 0
	for c in A:
		if c == '(':
			open += 1
		else:
			if open > 0:
				open -=1
			else:
				inserts += 1
	return inserts + open
```


## Longest Palindromic Substring (LPS)
Given a string, find and return the longest palindromic substring in A.
If multiple answers exist, return the one with least starting index.

### How
- For every center, expand outwards while left/right match.
- Track the longest palindrome seen.
- Return the first in case of tie.

```python
def longestPalindrome(A):
	n = len(A)
	if n == 0: return ""
	bestLen = 1
	bestStart = 0
	def expand(left, right):
		nonlocal bestLen, bestStart
		while left >= 0 and right < n and A[left] == A[right]:
			curLen = right - left + 1
			if curLem > bestLen or (curLen == bestLen and left < bestStart):
				bestLen = curLen
				bestStart = left
			left -=1
			right += 1
	for i in range(n):
		expand(i,i) #odd len palindrome
		expand(i,i+1)
	return A[bestStart: bestStart + bestLen]
```

## Salutes
Given a string, representing soldiers walking in a hallway:
- `¿` : soldier walking from left to right.
- `¡` : soldier walking right to left.
Whenever two soldiers cross, they both salute. Return the total number of salute.
```python
def countSalutes(A):
	rightCount = 0
	salutes = 0
	for c in A:
		if c == '>': rightCount += 1
		elif c == '<': salutes += rightCount
	return salutes
```

## Integer to Roman
Given a number, convert it to roman numeral.

```python
def intToRoman(A):
	table = [
        (1000, "M"),
        (900, "CM"),
        (500, "D"),
        (400, "CD"),
        (100, "C"),
        (90, "XC"),
        (50, "L"),
        (40, "XL"),
        (10, "X"),
        (9, "IX"),
        (5, "V"),
        (4, "IV"),
        (1, "I")
    ]
	res = ""
	for value, symbol in table:
		while A >= value:
			res += symbol
			A -= value
	return res
```

## Roman to Integer
```python
def romanToInt(A):
	value = {
        'I': 1,
        'V': 5,
        'X': 10,
        'L': 50,
        'C': 100,
        'D': 500,
        'M': 1000
    }
    res = 0
    n = len(A)
    for i in range(n):
	    v = value[A[i]]
	    if i + 1 < n and v < value[A[i+1]]:
		    res -= v
		else:
			res += v
	return res
```

## Add Binary Strings
Given two binary strings, add them and return the result binary string.
```python
def addBinary(A,B):
	i,j = len(A) - 1, len(B) - 1
	carry = 0
	res = []
	while i >= 0 or j >= 0 or carry:
		total = carry
		if i >= 0:
			total += ord(A[i]) - ord('0')
			i -= 1
		if j >= 0:
			total += ord(B[j]) - ord('0')
			j -= 1
		res.append(str(total % 2))
		carry = total // 2
	result.reverse()
	return ' '.join(result)
```

## Power of 2
Given a string representing non-negative int (extremely large). Determine if it can be written as a power of 2 : $2^k$ for some $k \geq 1$
How:
1. Simulate division by 2, use a big integer algorithm.
	- Remove leading zeroes.
	- If string is `1`, okk, else `0`? $2^0$ is not allowed.
	- Convert string to vector of base $10^9$ chunks
	- While number is not 1, check for oddness.
	- If divisible by 2 down to 1, it is a power of 2.
```python
def isPowerOfTwoString(A):
	A = A.lstrip('0')
	if A == "1": return False
	#convert A into base-1e9 chunks (little endian)
	def parse_chunks(A):
		chunks = []
		while s:
			s, chunk = s[:-9], s[-9:]
			chunks.append(int(chunk))
		return chunks
	def remove_leading_zeroes(limbs):
		while len(limbs) > 1 and limbs[-1] == 0:
			limbs.pop()
	def divide_by_two(limbs):
		carry = 0
		for i in reversed(range(len(limbs))):
			current = limbs[i] + carry*10**9
			limbs[i] = current // 2
			carry = current % 2
		remove_leading_zeroes(limbs)
		return carry #return remainder
	limbs = parse_chunks(A)
	while limbs != 0:
		if limbs[0] % 2 == 1:
			return False
		divide_by_two(limbs)
	return True
```

## Multiply Strings
Given two numbers as strings, return their product as a string with no leading zeroes.

### How
Simulating elementary school multiplication.
```python
def multiply(A,B):
	if A == "0" or B == "0":
		return "0"
	m,n = len(A), len(B)
	res = [0]*(m+n)
	for i in range(m-1,-1,-1):
		for j in range(n-1,-1,-1):
			res[i+j+1] += int(A[i])*int(B[j])
	#handle carry
	for k in range(m+n - 1, 0, -1):
		carry = res[k] // 10
		res[k] %= 10
		res[k-1] += carry
	#no leading zeroes
	i = 0
	while i < len(res) and res[i] == 0:
		i += 1
	return ''.join(map(str, res[i:]))
```

## Convert the Amount in Number to Words
Given an amount, write its number name. Check if it matches with B.
```python
def solve(A,B):
	A = lstrip('0') or '0'
	if A == '0':
		return 1 if B == 'zero' else 0
	num = int(A)
	crores = num // 10**7
	lakhs = (num // 10**5)% 100
	thousands = (num // 10**3) % 100
	rem = num % 1000
	tok = []
	def num_to_words(n):
		units = ["", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
                 "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
                 "sixteen", "seventeen", "eighteen", "nineteen"]
        tens = ["", "", "twenty", "thirty", "forty", "fifty",
                "sixty", "seventy", "eighty", "ninety"]
        if n == 0:
	        return ""
	    elif n < 20:
		    return units[n]
		else:
			return tens[n//10] + ("-" + units[n%10] if n % 10 != 0 else "")
	def process_group(value, label):
		if value:
			tok.append(num_to_words(value) + '-' + label)
	process_group(crores,"crore")
	process_group(lakhs,"lakh")
	process_group(thousands,"thousand")
	if rem:
		hundreds = rem // 100
		tens_ones = rem % 100
		part = ""
		if hundreds:
			part += num_to_words(hundreds) + "-hundred"
			if tens_ones:
				part += num_to_words(tens_ones)
		if part:
			tok.append(part)
	formed = '-'.join(tok)
	return 1 if formed == B else 0
```

## Compare Version Numbes
Given two versions of number `A` and `B`, Compare them if:
- Return $1$ if $A > B$
- Return $-1$ if $A < B$
- Return $0$ if they are equal

1. Version contains digits and `.`
2. Can have leading zeroes
3. Compare them left to right
4. MIssing are treated as 0

Example: `"0.1" < "1.1" < "1.2" < "1.13" < "1.13.4"`
### How
- Parse each segment between the dots
- Remove leading zeroes
- Compare int values of each segment.
- If one string has more segments, the missing one is treated as `0`.

```python
def compareVersion(A,B):
	i,j = 0,0
	n,m = len(A), len(B)
	while i < n and j < m:
		#parse next numeric segment from A
		segA = ""
		while i < n and A[i] != '.':
			segA += A[i]
			i += 1
		i+= 1 #skip the .

		segB = ""
		while j < m and B[j] != '.':
			segB += B[j]
			j += 1
		j += 1 #skip .

		segA = segA.lstrip('0') or '0'
		segB = segB.lstrip('0') or '0'
		if len(segA) < len(segB):
			return -1
		elif len(segA) > len(segB):
			return 1
		else:
			#lexographical comparison
			if segA < segB:
				return -1
			elif segA > segB:
				return 1
	return 0
```

## Atoi (string to int conversion)
Return INT_MAX or INT_MIN if overflow.
```python
def atoi(A):
	INT_MAX = 2**31 - 1
	INT_MIN = -2**31
	i = 0
	n = len(A)
	res = 0
	sign = 1
	#slip leading whitespace
	while i < n and A[i].isspace():
		i += 1
	#parse sign if there is
	if i < n and A[i] in ['+', '-']:
		sign = -1 if A[i] == 'i' else 1
		i += 1
	digit_seen = False
	while i < n and A[i].isdigit():
		digit = int(A[i])
		digit_seen = True
		#check overflow
		if result > (INT_MAX - digit) // 10:
			return INT_MAX if sign == 1 else INT_MIN
		result = result*10 + digit
		i += 1
	if not digit_seen:
		return 0
	return sign* result
```

## Valid IP Addresses
Given a string only containing digits, restore it by returning all possible valid IP address combinations, in sorted order.

Valid IP is in the form of A.B.C.D, where each segment is an int from 0 to 255, with:
- No segment has leading zeroes, except itself is `0`
- 1 $\leq |A| \leq 12$

### How
DFS
- At each step, choose next segment of length $\leq 3$ digits
- Check if the segment is valid (value, and no leading zeroes)
- Recur to form the next segment
- WHen 4 segments formed, all digs used, add to the result.

Prune early if it is impossible to form 4 segments.

```python
def restoreIPAddresses(A):
	result = []
	def dfs(pos,segs):
		if len(segs) == 4:
			if pos == len(A):
				result.append('.'.join(segs))
			return
		for length in [1,2,3]:
			if pos + length > len(A):
				break
			segment = A[pos:pos + length]
			#no leading zeroes
			if segment[0] == '0' and length > 1:
				break
			if int(segment) > 255:
				continue
			dfs(pos + length, segs + [segment])
	dfs(0,[])
	result.sort()
	return result
```

## Length of Last Word
Given a string of alphabets and space characters, return the length of the last word in the string.

```python
def lenOfLastWord(A):
	i = len(A) - 1
	#skip trailing spaces
	while i >= 0 and A[i] == ' ':
		i -= 1
	length = 0
	#count chars
	while i >= 0 and A[i] != ' ':
		length += 1
		i -=1
	return length
```

## Reverse The String (Word by Word)
Given string, return the string reversed word by word.
String will be a sentence.

A = teri maa ki chut, Output: chut ki maa teri

Output must have no leading or trailing spaces.
Multiple space becomes single space.

```python
def reversedWords(A):
	i = len(A) - 1
	result = ""
	while i >= 0:
		#skip trail zeroes
		while i >= 0 and A[i] == ' ':
			i -= 1
		if i < 0:
			break
		end= i
		#move to the start of the word
		while i >= 0 and A[i] != ' ':
			i -= 1
		start = i + 1
		if result:  result += ' '
		result += A[start:end + 1]
	return result	
```

## Zigzag String Conversion
Given a string and an int, write A in zigzag order on B rows and then read the pattern row by row.

A = "PAYPALISHIRING", B = 3, Output = "PAHNAPLSIIGYIR"
How?
Row 0: P   A   H   N
Row 1: A P L S I I G
Row 2: Y   I   R

### How
Simulate writing in zigzag
move down row by row, if bottom, change the direction

```python
def convert(A,B):
	if B <= 1 or B >= len(A):
		return A
	rows = [""]*B
	cur = 0
	direction = 1
	for c in A:
		rows[cur] += c
		if cur == 0:
			direction = 1
		elif curr == B-1:
			direction = -1
		cur += direction
	return ''.join(rows)
```

## Justified Text
Justifiy the string (indentation).
Format such that:
- Each line has exactly B characters.
- Each line is left and right justified.
- Distribute spaces as evenly as possible between words
	- if uneven, left gets more spaces
- Last line is left justified. 
- No word is ever split, must fit in B.

Input: A: list of strings, B = line width.

Output: List of justified lines, each of len B.

### How
- Greedily fit as many words as possible per line.
- For each line
	- IF it is the last line or has only one word: left-justify, pad right.
	- Else: distribute space evenly, starting from left to right.

```python
def fullJustify(words, L):
	result = []
	idx = 0
	while idx < len(words):
		start = idx
		lineLen = len(words[idx])
		idx += 1
		while idx < len(words) and lineLen + 1 + len(words[idx]) <= L:
			lineLen += 1 + len(words[idx])
			idx += 1
		cnt = idx - start
		isLast = idx == len(words)
		lineWords = words[start:idx]
		if cnt == 1 or isLast:
			line = ' '.join(lineWords).ljust(L)
		else:
			totalLen = sum(len(w) for w in lineWords)
			totalSpaces = L - totalLen
			gaps = cnt - 1
			spaceWidth, extra = divmod(totalSpaces, gaps)
			line = ''
			for i in range(gaps):
				line += linewords[i]+' '*(spacewidth + (1 if i < extra else 0))
			line += lineWords[-1]
		result.append(line)
	return result
```

## Pretty JSON Formatter
Given a string representing a JSON object, return an array with proper indentation:
- Every inner brace increases the indentation
- Every closing brace decreases the indentation
- Braces: only `{}` and `[]` are valid.
- Spaces may be ignored.
```python
def prettyJSON(A):
    result = []     # Final list of formatted lines
    indent = 0      # Current indentation level
    cur = ""        # Current line being built
    i = 0
    n = len(A)

    while i < n:
        c = A[i]

        # If opening bracket/brace
        if c in ['{', '[']:
            if cur.strip():                # Flush any existing line
                result.append(cur)
            cur = '\t' * indent + c        # Add opening with current indent
            result.append(cur)
            indent += 1                    # Increase indentation for next level
            cur = ""

        # If closing bracket/brace
        elif c in ['}', ']']:
            if cur.strip():                # Flush current buffer if not empty
                result.append(cur)
            indent -= 1                    # Decrease indent before printing
            line = '\t' * indent + c       # Start line with correct indent
            if i + 1 < n and A[i + 1] == ',':
                line += ','                # Append comma if it follows immediately
                i += 1                     # Skip the comma
            result.append(line)
            cur = ""

        # If comma separating elements
        elif c == ',':
            cur += ','                     # Add comma to current line
            result.append(cur)             # Flush the line
            cur = ""

        # Any other character (part of key, value, etc.)
        else:
            if not cur:
                cur = '\t' * indent        # Begin new line with indent
            cur += c                       # Add character to current line

        i += 1

    # Append any remaining content
    if cur.strip():
        result.append(cur)

    return result
```

