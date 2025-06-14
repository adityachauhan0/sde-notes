# Array Simulation
## Spiral Order Matrix
### Question kya hai

Matrix A: Size M x N, return all elements in the spiral order. (clockwise starting from top-left)

$$ 
\matrix{
{1,2,3,} \\ {4,5,6} \\ {7,8,9}
}
$$
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

