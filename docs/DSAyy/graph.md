
## Path in a directed graph
Directed graph A nodes, node labelled 1 to A. Directed edges B.

Determine if a path exists from node 1 to node A.

### How
BFS. duh

```python
from collections import deque, defaultdict
def reachable(A,B) 
	adj = defaultdict(list)
	for u,v in B:
		adj[u].append(v)
	visited = [False]*(A+1)
	q = deque()
	visited[0] = True
	q.append(1)
	while q:
		q.popleft()
		if u == A: return 1
		for v in adj[u]:
			if not visited[v]:
				visited[v] =True
				q.append(v)
	return 0	
```

## Water Flow
Given a N $\times$ M matrix A. It represents height of land cells. Determine how many cells can allow water to flow to **both** the:

Blue Lake: which touches top and left border

Red Lake: which touches bottom and right border

Water can flow from a cell to its neighbors if the neighbors height $\leq$ cur height

### How?
Do two BFS traversal

one from blue lake connected cells

one from red lake connected cells

Only flow if height of neighbor $\geq$ our height

```python
from collections import deque
def waterFlow(A):
	n,m = len(A), len(A[0])
	blue = [[False]*(m+1) for _ in range(n)]
	red = [[False]*(m+1) for _ in range(n)]
	qb,qr = deque(), deque()
	for i in range(n):
		blue[i][0], red[i][m-1] = True, True
		qb.append([i,0])
		qr.append([i,m-1])
	for j in range(m):
		blue[0][j], red[n-1][j] = True, True
		qb.append([0,j])
		qr.append([n-1,j])
	dirs = [[1,0],[-1,0], [0,1], [0,-1]]
	def bfs(q,vis):
		while q:
			i,j = q.popleft()
			for dx,dy in dirs:
				ni,nj = i + dx, j + dy
				if 0 <= ni < n and 0 <= nj < m and not vis[ni][nj]:
					if A[ni][nj] >= A[i][j]:
						vis[ni][nj] = True
						q.append([ni,nj])
	bfs(qb, blue)
	bfs(qr,red)
	ans = 0
	for i in range(n):
		for j in range(m):
			if blue[i][j] and red[i][j]: ans += 1
	return ans
```

## Stepping number
step num: if abs dif between every pair of adjacent digits = 1

find all step num in range A to B

123 and 321 are stepping nums, but 358 is not

### How
Perform bfs for each starting from 1 to 9 (and include 0 is A = 0)

For each num extract the last digit

Append digit that differ by 1 to the current num
	
$12 \times 10 + (2-1) = 121$

$12 \times 10 + (2+1) = 123$

bas yahi karte reh until you reach B
```cpp
vector<int> stepNum(int A, int B){
	vector<int> result;
	if (A == 0) result.push_back(0);
	for (int start - 1; start <= 9; ++start){
		queue<int> q; q.push(start);
		while (!q.empty()){
			int num = q.front(); q.pop();
			if (num > B) continue;
			if (num >= A) result.push_back(num);
			int last = num % 10;
			if (last > 0) q.push(num * 10 + (last - 1));
			if (last < 9) q.push(num * 10 + (last + 1))	;
		}
	}
	sort(results.begin(),results.end());
	return result;
}
```

## Capture regions on board.
Given a 2d character matrix A size $N \times M$ , each cell contains either `X` or `0`.
Capture all regions surrounded by `X`

A region is captured by flipping 0s into Xs in that surrounded region.

Dont output anything, just change inplace.

### How
Find all boundary connected 0s,

Mark them as # using BFS.

Then traverse the entire board and convert unmarked Os into Xs

Then convert marked # into 0s

```python
from collections import deque
def captureRegions(A):
	n,m = len(A), len(A[0])
	q = deque()
	for i in range(n):
		for j in range(m):
			if A[i][j] == 'O':
				A[i][j] = '#'
				q.append((i,j))
	dirs = [[1,0],[-1,0], [0,1], [0,-1]]
	while q:
		i,j = q.popleft()
		for dx,dy in dirs:
			x,y = i+dx,j + dy
			if 0 <= x < n and 0 <= y < m and A[x][y] == 'O':
				A[x][y] = '#'
				q.append((x,y))
	for i in range(n):
		for j in range(m):
			if A[i][j] == 'O': A[i][j] = 'X'
			if A[i][j] == '#': A[i][j] = '0'	
```

## Word Search on a Board
2D char board and a word, determine if word exists in the grid.
Word must be constructed through sequentially adjacent cells.
Same letter can be used more than once on a cell

Board = 

ABCE

SFCS

ADEE
### How

Perform DFS from every cell in the board, check if the word can be constructed starting at that cell. Dont mark visited cells, since it can be resused.

```python
def exist(board,word):
	n,m = len(board), len(board[0])
	dirs = [[1,0],[-1,0], [0,1], [0,-1]]
	vis = [[False]*(m) for _ in range(n)]
	def dfs(i, j ,pos):
		if board[i][j] != word[pos]: return False
		if pos == len(word) - 1: return True
		vis[i][j] = 1
		for dx,dy in dirs:
			x,y = i + dx, j + dy
			if 0 <= x < n and 0 <= y < m:
				if dfs(i,j,pos+1):
					return True
		vis[i][j] = 0
		return False
	
	for i in range(n):
		for j in range(m):
			if dfs(i,j,0): return True
	return False
```

## Path with Good Nodes
Given a tree with N nodes. Each node is marked good/bad
`A[i]` = 1 means node i+1 is good, warna its bad


A 2D array B of (N-1) $\times$ 2 representing undirected edges of the tree.

int C representing max good nodes on a path

Compute num of root-to-leaf paths in the tree with atmost C good nodes.

### How
Perform a dfs from node 1, keeping track of good nodes on the path
If good nodes exceed C, prune the path.

```python
from collections import defaultdict
def goodPath(A,B,C):
	adj = defaultdict(list)
	for u,v in B:
		adj[u].append(v)
		adj[v].append(u)
	ans = 0
	def dfs(node, parent, good_count):
		if u != 1 and len(adj[u]) == 1:
			if goodCnt <= C: ans += 1
			return
		for v in adj[u]:
			if v == parent: continue
			nxtGood = goodCnt + A[v-1]
			if nxtGood <= C:
				dfs(v,u,nxtGood)
	dfs(1,0,A[0])
	return ans
```

## Largest Dist between nodes of a tree
Given an unweighted tree with N nodes numbered from 0 to N-1

Find largest path in the tree.

Given the parent array.

if $parent[i]$ = -1? i is the root, warna $parent[i]$ has an edge with i

### How
Find farthest node from root, then find farthest node from that node.
The dist is the ans

```python
from collections import defaultdict, deque

def largestDist(A):
	n = len(A)
	if n <= 1: return 0
	adj = defaultdict(list)
	root = 0
	for i in range(n):
		if A[i] == -1:
			root = i
		else:
			p = A[i]
			adj[i].append(p)
			adj[p].append(i)
	def bfs(start):
		dist = [-1]*(N)
		q = deque()
		dist[start] = 0
		q.append(start)
		farNode = start
		while q:
			u = q.popleft()
			for v in adj[u]:
				if dist[v] != -1: continue
				dist[v] = dist[u] + 1
				q.push(v)
				farNode = v if dist[v] > dist[farNode] else farNode
		return (farNode, dist[farNode])
	u,du = bfs(root)
	v,dv = bfs(u)
	return dv
```

## Good Graph
Directed Graph, N nodes. Each node points to exactly one node. 

We are given the parent array.

Node is good if

it is node 1

it is connected to node 1

Find min number of edge changes such that all nodes become good.

### How
So the input is a functional graph where all nodes have out degree 1

This may contain many disjoint cycles.

We aim to detect all cycles.

If a cycle contains node 1, it is already good. 

If it does notm we must change atleast one point in the cycle to point towards node 1 or another good node.

```python
def goodGraph(A):
	N = len(A)
	next = [0]*(N)
	for i in range(N): # convert to 0 based indexing
		next[i] = A[i] - 1
	vis = [0]*(N)
	ans = 0
	for i in range(N):
		if vis[i]: continue
		path = []
		curr = i
		while True:
			vis[curr] = 1
			path.append(curr)
			nxt = next[curr]
			if vis[nxt] == 0:
				curr = nxt
			elif vis[nxt] == 1: # this is a cycle
				containsOne = False
				n = len(path)
				for k in range(n-1,-1,-1): # look if 1 is in the cycle
					if path[k] == nxt:
						for t in range(k,n):
							if path[t] == 0:
								containsOne = True
								break
						break
				if not containOne: 
					ans += 1
				break # break once done doin the cycle shit
			else : break # break if we come across an already processed node
		for v in path:
			vis[v] = 2 # mark processed
	return ans	
```

## Cycle in directed graph

A nodes and M edges Directed, check if it has a cycle

B is a list of edges

### How?
If the graph can be topo sorted, it has no cycles.

Topo sort is just bfs but

compute indegree of all nodes

nodes with indegree 0 go to queue

while queue is not empty,

increment processed for front node

decrease indegree of all neighbors, and if 0, add to queue

if processed count is not the same as A, cycle exists

```python
from collections import defaultdict, deque
def hasCycle(A,B):
	adj = defaultdict(list)
	indegree = [0]*(A+1)
	for u,v in B:
		adj[u].append(v)
		indegree[v] += 1
	q = deque()
	processed = 0
	for i in range(1,A+1):
		if indegree[i] == 0:
			q.append(i)
	while q:
		u = q.popleft()
		processed += 1
		for v in adj[u]:
			indegree[v] -= 1
			if indegree[v] == 0:
				q.append(v)
	return processed != A
```

## Delete Edge!
Given rooted undirected tree with N nodes and arr A having weight of each node.
Delete one node such that the product of the sum of weights of the two resulting subtrees is maximised.

Example:

A = 10 5 12 6
	
B = (1,2) (1,4) (4,3)

Output = 270
	
Removing (1,4): output = 15 $\times$ 18 = 270
### How
Post order dfs to compute subtree sum at each node
for every edge, compute:
$$
\text{Product = Sum of Subtree} \times (\text{Total Sum - Sum of Subtree})
$$
```python
from collections import defaultdict
MOD = 10**9 + 7

def mxDelEdge(A,B):
	adj = defaultdict(list)
	for u,v in B:
		adj[u].append(v)
		adj[v].append(u)
	n = len(A)
	weights = [0]+A
	totalSum = sum(weights)
	maxProd = 0
	subtreeSum = [0]*(n+1)
	def dfs(u,parent):
		nonlocal maxProd
		subtreeSum[u] = weights[u]
		for v in adj[u]:
			if v != parent:
				dfs(v,u)
				subtreeSum[u] += subtreeSum[v]
		prod = subtreeSum[u]*(totalSum - subtreeSum[u])
		maxProd = max(maxProd,prod)
	dfs(1,-1)
	return maxProd % MOD
```
