## Longest Common Subsequence

## Question Statement
Given 2 strings, find the length of uska longest common subsequence.
Note subsequence does not have to be continuous.

Example: A = $abbcdgf$ and B = $bbadcgf$  toh the output would be 5 (bbcgf is the lcs)

### Kaise karna hai
Thode subproblems meh divide karte hai isse.
Let $\text{LCS[i][j] be LCS of substrings A[0...i] and B[0....j]}$ .
Toh the obvious relation we can find is
if $\text{A[i-1] == B[j-1] fir LCS[i][j] is just LCS[i-1][j-1] + 1}$ 
which means ki humme ek element same mil gaya, toh length would be 1 + the substrings removing those used indexes dono strings se.
else $\text{take max of LCS[i-1][j] and LCS[i][j-1]}$ coz wahi dono possibilities are left. 
which means dono string se ek element skip karke check karlo
Humhe continuity bhi maintain karni hai.
Thats it literally.

```cpp
int LCS(string A, string B){
	int n = A.size(), m = B.size();
	vector<vector<int>> lcs(n+1, vector<int>(m+1,0));
	for (int i = 1; i <= n; ++i)
		for (int j = 1; j <= m; ++j){
			if (A[i-1] == B[j-1]) lcs[i][j] = 1 + lcs[i-1][j-1];
			else lcs[i][j] = max(lcs[i-1][j], lcs[i][j-1]); 
		}
	return lcs[n][m];
}
```
Time complexity $O(n \times m)$  and and same with space
Now the sexy part is hum isse aur optimize kar sakte hai

Abhi let $\text{prev}$ be the results of the i-1 row, and $curr$ be the result of the current row i


$\text{LCS[i-1][j-1] = previous row ka j-1}$
$\text{LCS[i-1][j] = previous row ka j}$
$\text{LCS[i][j-1] = cur row ka j-1}$

```cpp
int LCS(string A, string B){
	int n = A.size(), m = B.size();
	if (m > n){
		// keeping the rows as the bigger one, since lcs ka upper limit toh chotta wala hi hoga
		swap(A,B);
		swap(m,n);
	}
	vector<int> prev(m+1, 0), cur(m+1,0);
	for (int row = 1; row <= n; ++row){
		for (int j = 1; j <= m; ++j){
			if (A[i-1] == B[j-1])
				cur[j] = prev[j-1] + 1; 
				// we found a similar el, toh prev row ke result se ek zyada hoga
			else 
				cur[j] = max(prev[j], cur[j-1]);
		}
		swap(prev,cur)
	}
	return prev[M]
}
```

Isme the space complexity changed from $O(n\times m)$ se $O(min(n,m))$ 

## Longest Palindromic Subsequence

### Question Statement
Given a string A, find length of the longest palindromic subsequence.
Example: $A = beebeeed$  the output would be 4 coz LPS is $eeee$

### Karna kaise hai
Abe chutiye, Longest palindromic substring is just the LCS of A and reversed(A)

I am not even gonna waste my time fuck off.
$$
LPS(A) = LCS(A, reverse(A))
$$

## Edit Distance

### Problem Statement
Given 2 strings, find min steps required to convert A to B given in one step we can
- Insert a char
- Delete a char
- Replace a char
Example:
- A = abad, B = abac. The output is 1 coz sirf c ko d se replace karna

### Kaise karna
Let $edit[i][j]$ be the minimum dist to convert $A[0...i-1] \space and \space B[0...j-1]$ 
Toh iska rishta is 
$$
edit[i][j] = edit[i-1][j-1] \space \text{if A[i-1] and B[j-1] same hai}
$$
$$
\text{else if its different, edit[i][j] is gonna be 1 + min(edit[i-1][j], edit[i-1][j-1], edit[i-1][j-1])}
$$
$edit[i-1][j]$ means hum $A[i-1]$ delete kar rahe hai
same with $edit[i][j-1]$ meaning $B[j-1]$ delete
$edit[i-1][j-1]$ matlab bro humne dono ko replace kar diya ek dusre se and both are updated

### Base Cases
Kuch baate sacch hoti hai
Jaise, $edit[0][j] = j$ -> means $B[0..j]$ ko empty karne ke liye j operations are needed.
Similarly. $edit[i][0] = i$

Bas bhai ab chalao for loop, sabke level nikalenge.

```cpp
int editDistance(string A, string B){
	int n = A.size(), B = B.size();
	vector<vector<int>> edit(n+1, vector<int> (m+1, 0));
	// sach baate
	for (int i = 1; i <= n; ++i) edit[i][0] = i;
	for (int j = 1; j <= m; ++j) edit[0][j] = j;

	for (int i = 1; i <= n; ++i){
		for (int j = 1; j <= m; ++j){
			if (A[i-1] == B[j-1]) 
				edit[i][j] = edit[i-1][j-1];
				// no extra edits needed
			else {
				edit[i][j] = 1 + min({
					edit[i-1][j], edit[i][j-1], edit[i-1][j-1]
				});
			}
		}
	}
	return dp[n][m];
}
```

Time and Space dono in this are $O(n\times m)$


## Repeating Subsequence
### Problem kya yap kar raha
String A, check kar if its longest repeating subsequence is $\geq$ 2. 
Repeating subsequence is basically repeating subsequence.

Example: A = $abab$, Output is 1 (subseq $ab$ repeats)

### How to solve?

$$
\text{Repeating subsequence (A) is LCS(A,A) but i } \neq
 j $$
  Aur kuch bhi nahi.

bas 
$$
\text{return dp[n][n] } \geq 2
$$


## Distinct Subsequences

### Problem Statement
Given 2 sequences A and B, count ways to form B as a subseq of A.

Example:
- A = rabbbit, B = rabbit, Output 3
	- Basically all 3 `b` can be removed.

### Karu kaise?
Let $ways[j]$ be ways to form $B[0...j]$  as a subseq of first `i` elements of A. 
Isse 1D meh fit kar sakte if we update j from right to left, toh $ways[j-1]$ hoga from previous `i` 

Toh the rishta would be 
$$
ways[j] = ways[j] + ways[j-1] \text{      if A[i-1] == B[j-1]}
$$
warna $ways[j]$ would remain 0, coz agar element hi same nai toh string kaise banega with those indexes.
Note ki $ways[j-1]$ humare previous i ke iteration se hoga, so we have already computed pehle ka.

```cpp
int distSubseq(string A, string B){
	int n = A.size(), m = B.size();
	vector<long long> ways(m+1,0);
	ways[0] = 1; //empty banane ke liye kya hi chahiye
	for (int i = 1; i <= n; ++i){
		for (int j = m; j >= 1; --j){
			if (A[i-1] == B[j-1]){
				ways[j] += ways[j-1];
			}
		}
	}
	return (int)ways[m];
}
```


## Scramble String

### What does the dog say?
Given 2 strings, bata if the other string can be made by scrambling A.
Now what the fuck is scrambling?
B is scramble(A) if -> A can be represented as a binary tree by recursively partitioninng into two non-substrings, and by swapping left and right children any number of times, we can get B.
What the helly?

Input: A= `we` and B = `we`  Ans = 1.

### Bhai mar jau meh?

Iske liye we use 3D dp. Whether  `A[i...i+len - 1]` can be scrambled into `B[j....j+len - 1]`
Let `scramble[i][j][len]` be true if `A.substr(i,len)` can be scrambled into `B.substr(j,len`

### Hard Truth
Bhai pata nahi yaar ye scrambling shit kya hai
**Base Case**: For length 1, `scramble[i][j][1] = (A[i] == B[j])` 
	Like bhai ek hi toh length hai, same string hi hogaya ye toh
For each length $l$ from 2 to n, for all `A.substr(i,l)` and `B.substr(j,l)` har ek split `k` try kar le
- No swap: `scramble[i][j][l]` is true if $scramble[i][j][k] \space \& \space scramble[i][j][l-k]$ meaning dono continuous partitions valid hai.
- Swap: `scramble[i][j][l]` is true if `scramble[i][j+l-k][k]` and `scramble[i+k][j][l-k]` dono true hai
Also quick check: If dono sorted are different, tab toh they cant scramble.

```cpp
bool isScramble(string A, string B){
	int n = A.size();
	if (n != B.length()) return 0;
	//quick sorted check
	{
		array<int,256> freq = {0};
		for (char c : A) freq[(unsigned char)c]++;
		for (char c : B) freq[(unsigned char)c]--;
		for (int x : freq) if (x != 0) return 0;
	}
	static bool scramble[51][51][51];
	memset(dp,false, sizeof(scramble));
	// base case
	for (int i = 0; i < n; ++i)
		for (int j = 0; j < n; ++j)
			scramlbe[i][j][1] = (A[i] == B[i]);
	for (int len = 2; len <= n; ++len){
		for (int i = 0; i + len <= n; ++i){
			for (int j = 0; j + len <= n; ++j){
				for (int k = 1; k < len; k++){
					if ((scramble[i][j][k] && scramble[i+k][j+k][len-k]) || 
						(scramble[i][j+len - k][k] && scramble[i+k][j][len - k]){
						scramble[i][j][len] = 1;
						break;
						}
					)
				}
			}
		}
	}
	return scramble[0][0][n] ? 1 : 0;
}
```
Time Complexity: $O(n^4)$

## WildCard Pattern Matching

### Problem kya hai
Given 2 strings, find the wildcard pattern between them
- `?` matches any single character
- `*` matches any sequence of character
This match must cover the entire string

Return if A can be formed with the B pattern
- A = `aa` and B = `a*` , then ans is 1
- A = `aab` and B = `c*a*b`, then output 0 hai
### Karna kaise hai
Iske liye 2 pointer greedy karna padega with some backtracking for the `*` wala part.
- If the character at B is `*`, remember uska position and current index in A, and try to match it with 0 chars first.
- If there is a mismatch, then look for the previous `*`, since that can save us, and then advance the A pointer and try to match with more.
- If the current pointer in B is `?` ya fir it matches A, then advance both the pointers
- If A khatam hogaya, then skip the trailing `*` in B
- If dono khatam, then ans is 1

```cpp
bool isMatch(string A, string B){
	int n = A.size(), m = B.size();
	int i = 0, j = 0;
	int last_star = -1, match_i = 0;
	while (i < n){
		if (j < m && (B[j] == A[i] || B[i] == '?')) 
			i++,j++;
		else if (j < m && B[j] == '*'){
			last_star = j;
			match_i = i;
			j++;
		}
		else if (last_star != -1){
			j = last_star + 1;
			match_i++;
			i = match_i;
		}
		else 
			return 0;
	}
	while (j < m && B[j] == '*') j++;
	return (j == m)? 1: 0;
}
```

##  Pattern Matching . and *
### Problem kya hai
Again pattern matching but,
- `.` means atleast ek element hai here
- `*` ek element nahi bhi hoga toh chalega
Example: match(`aa`, `.*`) is 1, but match(`aa`,`.`) is 0.
### Kaise karna hai
Yaha we finally use dynamic programming.
`match[i][j]` means `a[0...i-1]` matches `b[0..j-1]`
But isse bhi optimize karenge for only two rows.

### Rishte
- If `B[j-1]` is `.` or `a[i-1]` matches one character, tab `match[i][j] = match[i-1][j-1]`
- If `B[j-1]` is `*`, tab it can match zero or more previous element:
	-  Zero occurence: $match[i][j] \space | \space  match[i][j-2]$
	- One or more: If `A[i-1]` matched `B[j-2]` or `B[j-2]` is `.`, then `match[i][j]` |= `match[i-1][j]`
- else `match[i][j] = 0`

### Truth
- `match[0][0]` = 1 (empty toh match hoga hi)
- `match[0][j]` = 1 if `B[0...j-1]` can represent empty 
- `match[i][0] = 0` for i > 0 coz non-empty match nahi kar sakta empty se

The less optimized one but easier to understand:
```cpp
/*
   Regex-like pattern match for
     .  = exactly one arbitrary character
     *  = zero or more copies of the PREVIOUS pattern symbol
   dp[i][j]  == true  ⇔   A[0 .. i-1] matches  B[0 .. j-1]
   (so the table has   (n+1) × (m+1)   entries)
*/
bool isMatch2D(const string& A, const string& B)
{
    int n = A.size(), m = B.size();
    vector<vector<bool>> dp(n + 1, vector<bool>(m + 1, false));
    // ➊ empty pattern vs. empty text
    dp[0][0] = true;
    // ➋ first row: empty text vs. longer & longer pattern
    //    Only a chain like  x* y* z*  can match emptiness
    for (int j = 2; j <= m; ++j)
        if (B[j - 1] == '*')
            dp[0][j] = dp[0][j - 2];
    // ➌ fill the whole grid
    for (int i = 1; i <= n; ++i)
    {
        for (int j = 1; j <= m; ++j)
        {
            char pc = B[j - 1];          // current pattern symbol
            if (pc != '*')               // case 1: normal char or '.'
            {
                bool same = (pc == '.' || pc == A[i - 1]);
                dp[i][j] = same && dp[i - 1][j - 1];
            }
            else                         // case 2: we’re at a '*'
            {
                //   pc == '*'  ← it always modifies B[j-2]
                //   let prev = B[j - 2]
                // ─────────────────────────────────────────────
                // zero copies of prev*
                bool zero      = dp[i][j - 2];
                // one-or-more copies  ⇒  prev must match A[i-1]
                bool oneOrMore = false;
                char prevPat   = B[j - 2];
                if (prevPat == '.' || prevPat == A[i - 1])
                    oneOrMore = dp[i - 1][j];
                dp[i][j] = zero || oneOrMore;
            }
        }
    }
    return dp[n][m];
}
```
The optimized one:
```cpp
bool isMatch(string A, string B){
	int n = A.size(), m = B.size();
	vector<bool> prev(m+1, false), cur(m+1, false);
	prev[0] = true;
	for (int j = 1; j <= m; ++j){
		if (B[j-1] == '*' && j >= 2)
			prev[j] = prev[j-2];
		else 
			prev[j] = false;
	}
	for (int i = 1; i <= n; ++i){
		cur[0] = false;
		for (int j = 1; j <= m; ++j){
			if (B[j-1] == '.' || B[j-1] == A[i-1])
				cur[j] = prev[j-1];
			else if (B[j-1] == '*'){
				bool matchZero = (j >= 2) ? cur[j-2] : false;
				bool matchOneOrMore = (j >= 2 && 
					(B[j-2] == '.' || B[j-2] == A[i-1])
					)? 
					prev[j] : false;
				cur[j] = matchZero || matchOneOrMore;
			}
			else cur[j] = false;
		}
		swap(cur,prev);
	}
	return prev[m] ? 1 : 0;
}
```

## Length of Longest Subsequence

### Problem yap
Given array A, find length of longest sequence whihc is strictly increasing  then strictly decreasing. (mountain peak type shit)

Example:
- A = `[1,11,2,10,4,5,2,1]`, output is 6 (`1,2,10,4,2,1`)

### How????
So this mountain thing is called bitonic subsequence.
- For each i, compute `lis[i]` which is the length of longest increasing subsequence ending at i.
- then also compute `lds[i]` which is length of longest decreasing subsequence starting at i
- Ab bas har ek index pe compute dono ka sum  $lis[i] + lds[i] - 1$

```cpp
int longSubseq(vector<int> &A){
	int n = A.size();
	if (n == 0) return 0;
	vector<int> lis(n,1);
	for (int i = 0; i < n; ++i)
		for (int j = 0; j < i; ++j)
			if (A[j] < A[j])
				lis[i] = max(lis[i], lis[j] + 1);
	vector<int> lds(n,1);
	for (int i = n-1; i >= 0; --i)
		for (int j = i+1; j < n; ++j)
			if (A[j] < A[i])
				lds[i] = max(lds[i], lds[j] + 1);
	int ans = 0;
	for (int i = 0; i < n; ++i) 
		ans = max(ans, lis[i] + lds[i] - 1);
	return ans;
}
```


## Smallest subsequence given the primes.
### Sexy ass statement
Given 3 prime numbers, and an integer D, find the first smallest D positive numbers which only have A,B,C, or a combination of them as their prime factors.

Input: A,B,C,D, Output: Array of size D

- A= 2, B = 3, C = 5, D = 5
	- the output is `2,3,4,5,6`

### How to do this shit.

What they ask is `Ugly Numbers` (numbers with only prime as their factors)
We use 3 pointers, with a min-merge approach
- Maintain an array `res` initialized with the sequence {1}
- For each index, multiple previous seq elements by A,B,C and pick the smallest new candidate.
- Increment pointers which produced the candidate to avoid duplicates.
$$
res[next] = min(res[i_1]\times A,res[i_2]\times B, res[i_3]\times C)
$$

```cpp
vector<int> subseq(int A, int B, int C, int D){
	vector<long long> primes = {(long long)A,(long long)B, (long long)C};
	sort(primes.begin(),primes.end());
	long long p1 = primes[0], p2 = primes[1], p3 = primes[2];
	vector<unsigned long long> res(D+1);
	res[0] = 1ull;
	int i1 = 0, i2 =0, i3 = 0;
	for (int idx = 1; idx <= D; idx++){
		unsigned long long 
			nextA = res[i1] * p1,
			nextB = res[i2] * p2,
			nextC = res[i3] * p3;
		unsigned long long val = min({nextA, nextB, nextC});
		res[idx] = val;
		if (val == nextA) i1++;
		if (val == nextB) i2++;
		if (val == nextC) i3++;
	}
	vector<int> ans;
	for (int k = 1; k <= D; ++k){
		ans.push_back((int)res[k]);
	}
	return ans;
}
```

## Largest Area of Rectangle with Permutations
### What does this bitch yap
Given a binary grid A of $N \times M$, find the area of the largest rectangle containing only 1s. 
We are allowed to permute the columns of the matrix in any order.

Example
$$
A = [[1,0,1],[0,1,0],[1,0,0]]
$$
The output is 2
### How to do this?
- For each cell (i,j), compute `H[i][j]` = number of consecutive 1's ending at (i,j) upto row i.
- For each row, treat `H[i]` as histogram of column heights, and since `H[i]` can be permuted, sort `H[i]` in descending order.
- For each width k (the k largest column heights), the maximal rectangle is $height \times width = H[i][k-1] \times k$ 
- Take the best over all i and k.
Example dry run:
```
A
row0  1 0 1 1 0 1
row1  1 1 1 1 1 1
row2  1 1 1 0 1 1
row3  0 1 1 1 1 0
row4  1 1 1 1 1 1
```

|row i|formula per column j|**H[i]**|
|---|---|---|
|0|first row → copy `A`|**1 0 1 1 0 1**|
|1|if `A[1][j]==1` then `1+H[0][j]` else 0|**2 1 2 2 1 2**|
|2|same rule|**3 2 3 0 2 3**|
|3|same rule|**0 3 4 1 3 0**|
|4|same rule|**1 4 5 2 4 1**|

| row | original H[i] | after sort ↓    | `k`                                 | height · width |
| --- | ------------- | --------------- | ----------------------------------- | -------------- |
| 0   | 1 0 1 1 0 1   | **1 1 1 0 0 0** | 1 → 1 2 → 2 3 → 3                   | **3**          |
| 1   | 2 1 2 2 1 2   | **2 2 2 2 1 1** | 1 → 2 2 → 4 3 → 6 4 → 8 5 → 5 6 → 6 | **8**          |
| 2   | 3 2 3 0 2 3   | **3 3 3 2 2 0** | 1 → 3 2 → 6 3 → 9 4 → 8 5 → 10      | **10**         |
| 3   | 0 3 4 1 3 0   | **4 3 3 1 0 0** | 1 → 4 2 → 6 3 → 9 4 → 4             | **9**          |
| 4   | 1 4 5 2 4 1   | **5 4 4 2 1 1** | 1 → 5 2 → 8 3 → **12** 4 → 8        | **12**         |

--- 

Hence the largest we find is $4 \times 3 = 12$
```cpp
int maximalRectangle(vector<vector<int>> & A){
	int N = A.size();
	if (N == 0) return 0;
	int M = A[0].size();
	// building H matrix
	vector<vector<int>> H(N, vector<int> (M,0));
	for (int j = 0; j < M; ++j){
		H[0][j] = A[0][j];
		// starting meh consecutive ones is the 1 if the value at that ind is 1
	}
	for (int i = 1; i < N; ++i)
		for (int j = 0; j < M; ++j){
			if (A[i][j] == 1)
				H[i][j] = H[i-1][j] + 1;
			else H[i][j] = 0;
		}
	int ans = 0;
	for (int i = 0; i < N; ++i){
		vector<int> row = H[i];
		sort(row.rbegin(),row.rend());
		for (int k = 1; k <= M; ++k){
			int height = row[k-1];
			int area = height * k;
			ans = max(ans, area);
		}
	}
	return ans;
}
```

## Tiling with Dominoes

### Problem Statement
Given integer A, find total ways to tile a $3 \times A$ board with $2 \times 1$ dominoes.
Return answer % $10^9 + 7$.

Example: 
- A = 2, Output = 3
- A = 1, Output = 0
### Maths behind this
Let `f(n)` be the number of ways to tile a $3 \times n$ board.
- `f[0]` = 1. (empty board)
- `f[1]` = 0 (cannot be completely tiled)
- `f[2]` = 3
- `f[3]` = 0. (odd ke liye toh impossible hai completely tile karna, always some remainder)
- For n $\geq$ 4, even n, $f[n] = 4 \times f[n-2] - f[n-4]$
- For odd n, `f[n]` = 0

```cpp
int domino(int A){
	const int MOD = 1e9 + 7;
	vector<int> f(A+1,0);
	if (A >= 0) f[0] = 1;
	if (A >= 1) f[1] = 0;
	if (A >= 2) f[2] = 1;
	if (A >= 3) f[3] = 0;
	for (int n = 4; n <= A; ++n){
		if (n & 1) f[n] = 0;
		else {
			long long x = (4LL * f[n-2]) % MOD;
			x = (x - f[n-4] + MOD) % MOD;
			f[n] = (int)x;
		}
	}
	return f[A];
}
```


## Paint House

N houses in a row, each can be painted with RGB.  Painting each house with a certain color has a given cost, represented by $n \times 3$ matrix A, where `A[i][j]` is the cost to paint the house `i` with cost `j`  (0 -> red, 1-> blue, 2-> green).
Paint such that
- No two adjacent houses have the same color.
- Minimize the total painting cost.
Input $N\times 3$ matrix, output -> min cost to paint all.
Example:
$$
	A = \begin{matrix} 1 && 2  && 3\\ 10 && 11 && 12 \end{matrix}
$$
Output: 12
Paint 0 with R, 1 with G: 1 + 11 = 12

### How to do this painting

Let `cost[i][c]` be the min cost to paint houses 0 to i with house i painted color c. But since each row only depends on the previous,, we can just use 2 arrays.

$$
cost[i][0] = A[i][0] + min(cost[i-1][1], cost[i-1][2])
$$
$$
cost[i][1] = A[i][1] + min(cost[i-1][0], cost[i-1][2])
$$
$$
cost[i][2] = A[i][2] + min(cost[i-1][0], cost[i-1][1])
$$
Iss se simple dp ho nahi sakta
Thoda simple karne ke liye
let `prev_cost[c]` be cost of painting the previous house with color c.
`prev_cost[c] = cost[i-1][c]`

```cpp
int minCost(vector<vector<int>> &A){
	int N = A.size();
	if (N == 0) return 0;
	long long prev_cost[3];
	for (int c = 0; c < 3; ++c) 
		prev_cost[c] = A[0][c];
	for (int i = 1; i < N; ++i){
		long long cost[3];
		cost[0] = A[i][0] + min(prev_cost[1], prev_cost[2]);
		cost[1] = A[i][1] + min(prev_cost[0], prev_cost[2]);
		cost[2] = A[i][2] + min(prev_cost[0], prev_cost[1]);
		for (int c = 0; c < 3; ++c) prev_cost[c] = cost[c];
	}
	long long ans = min({prev_cost[0], prev_cost[1], prev_cost[2]});
	return (int)ans;
}
```


## Ways to decode
Given an encoded string `A` consisting of digits.
- 'A' = 1, 'B' = 2 ... 'Z' = 26
find the total number of ways of decoding A modulo $10^9 + 7$.

Input: String A, Output: total number of decoding ways.

Example:
- A = '8', Output = 1 ("H")
- B = '12', Output = 2 ("AB", "L")

### Karna kaise hai
Let `ways[i]` be number of ways to decode `A[0...i]` (first i characters)

Toh iss ke rishte kuch aise honge:
- If $A[0] = '0'$, single digit is valid, add `ways[i-1]`
- If `A[i-2,i-1]` form a valid 2 digit number between 10 and 26, add `ways[i-2]`

Base cases:
- $ways[0] = 1$ (empty string banane ka there is only one way)
- $ways[1] = 1 \text{ if A[0] } \neq '0', \text{ else 0}$

```cpp
int numDecodings(string A){
	int n = A.length();
	const int MOD = 1e9 + 7;
	if (n == 0) return 0;
	vector<int> ways(n+1,0);
	ways[0] = 1;
	ways[1] = (A[0] != '0')? 1 : 0;
	for (int i = 2; i <= n; ++i){
		char c1 = A[i-1], c0 = A[i-2];
		if (c1 != '0')
			ways[i] = (ways[i] + ways[i-1]) % MOD;
		if (c0 == '1' || (c0 == '2' && c1 <= '6'))
			ways[i] = (ways[i] + ways[i-2]) % MOD;
	}
	return ways[n];
}
```


## Stairs
### Legendary beginner problem
You are climbing a staircase with A steps. You can climb either 1 or 2 steps;
How many distinct ways can you reach the top?

Input: A = 2, Output = 2 (`[1,1],[2]`)

### Karna kaise hai
Let $waysToStep[n]$ be the number of ways to reach step n

$$
	waysToStep[i] = waysToStep[i-1] + waysToStep[i-2]
$$
Some facts:
$waysToStep[0] = 1$ (1 way to stay at the bottom)
$waysToStep[1] = 1$ (climb one step)

```cpp
int climbStairs(int A){
	if (A <= 1) return 1;;
	int prev = 1, curr = 2;
	for (int i = 3; i <= A; ++i) {
		int next = prev + curr;
		prev = curr;
		curr = next;
	}
	return curr;
}
```


## Longest Increasing Subsequence

### Problem kya hai
Given array of integers A, find the length of Longest Increasing Subsequence.

Example:
- `A = [1,2,1,5]` Output: 3 (LIS = `[1,2,5]` )

### Karna kaise hai
BINARY SEARCH BITCH
- Maintain a list `tail` where `tail[i]` is the smallest possible tail value of an increasing subsequence of length `i+1`
- `tail[i]` is the smallest possible value that can end a increasing subsequence of length `i+1` 
- For each x in A:
	- Use lower_bound to find the first element in tail $\geq$ x.
	- If none, append x (increase the LIS length)
	- Otherwise, replace it. (keep tail as small as possible for future extensions)
		- the older value was bigger than x, and it wouldn't be a valid increasing subsequence ending at `i` so we replace the bigger value withh this new pookie.
Length of the tail at the end is the ans

```cpp
int lis(vector<int> &A){
	vector<int> tail;
	for (int x : A){
		auto it = lower_bound(tail.begin(),tail.end(), x);
		// find first el >= x
		if (it == tail.end())
			tail.push_back(x);
		else 
			*it = x;
	}
	return tails.size();
}
```

