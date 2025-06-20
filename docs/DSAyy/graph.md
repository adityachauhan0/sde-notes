
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

## Two Teams? (if graph is bipartite)
A people, 1 to A. Divide them into two teams. Given list B of $M\times 2$ having edges. Person u and v are enemies and cant be on the same team.
Determine if its possible to divide them into two teams.

A = 5

B = (1,2) (2,3) (1,5) (2,4)

Output: 1 (Yes)

### How
Bipartite Coloring kar, if not possible then ret 0

BFS to traverse path

Assign opp colors to the neighbors

If adj nodes have same color, ret 0

check all connected components

```python
from collections import deque, defaultdict
def twoTeamsPossible(A,B):
	adj = defaultdict(list)
	for u,v in B:
		adj[u].append(v)
		adj[v].append(u)
	color = [0]*(A+1)
	q = deque()
	for i in range(1,A+1):
		if color[i]: continue # Already processed
		color[i] = 1
		q.append(i)
		while q:
			u = q.popleft()
			for v in adj[u]:
				if color[v] == 0:
					color[v] = -color[u]
					q.append(v)
				elif color[v] == color[u]:
					return 0
	return 1
```

## Valid Path

Grid of size $x \times y$ and N circular obstacles of radius R centered inside the grid. Determine if its to go from (0,0) to (x,y) without touching any circular region.

Movement:

8D including the diagonals

Cannot go outside grid boundaries (chutiya hai kya)

Cannot step into a cell that intersects any circle

Note: a circle blocks a cell (i,j) if the dist from the centre is less than or equal to R

Example:

x = 2, y= 3, N = 1, R = 1, A = (2), B = (3)
Output? No.
Circle at (2,3) blocks the path to dest
### How
Preprocess Blocked Cells

$$
(x-a)^2 + (y-b)^2 \leq R^2
$$
Check this for every cell

Perform BFS from 0,0 and check if any one reaches x,y


```python
from collections import deque
def validPath(x,y,N,R,A,B):
	blocked = [[False]*(y) for _ in range(x)]
	center= []
	for i in range(N):
		center.append((A[i],B[i]))
	for i in range(x):
		for j in range(y):
			for a,b in center:
				if (i-a)**2 + (j-b)**2 <= R**2:
					blocked[i][j] = True
	if blocked[0][0] or blocked[x][y]:
		return 0
	q = deque()
	q.append((0,0))
	visited = [[False]*(y) for _ in range(x)]
	dir = [(-1, -1), (-1, 0), (-1, 1),
                  (0, -1),         (0, 1),
                  (1, -1),  (1, 0),  (1, 1)]
	while q:
		i,j = q.popleft()
		if (i,j) == (x,y): return 1
		for dx,dy in dir:
			nx = i+dx, ny= j+dy
			if 0 <= nx <= x and 0 <= ny <= y:
				if not blocked[nx][ny] and not visited[nx][ny]:
					visited[nx][ny] = 1
					q.append((nx,ny))
	return 0
```


## Region in binary matrix
Given binary matrix $N \times M$ . Each cell is either 1 (filled) or 0 (not).

Region is formed by one or more connected components in 8D.

Return the size of the largest connected component.

Example

A = 0 0 1 1 0
	1 0 1 1 0
	0 1 0 0 0
	0 0 0 0 1
Ans : 6

### How
Simple bfs from every unvisited cell and counting all reachable cells and marking them visited.

```python
from collections import deque
def largestRegion(A):
	n,m = len(A), len(A[0])
	maxRegion = 0
	seen = [[False]*(m) for _ in range(n)]
	dir = [(-1, -1), (-1, 0), (-1, 1),
                  (0, -1),         (0, 1),
                  (1, -1),  (1, 0),  (1, 1)]

	for i in range(n):
		for j in range(m):
			if not seen[i][j] and A[i][j]:
				regionSize = 0
				q = deque()
				q.append((i,j))
				seen[i][j] = 1
				while q:
					x,y = q.popleft()
					regionSize += 1
					for dx,dy in dir:
						nx,ny = x + dx , y + dy
						if 0 <= nx < n and 0 <= ny < m:
							if not seen[nx][ny] and A[nx][ny]:
								seen[nx][ny] = 1
								q.append((nx,ny))
				maxRegion = max(maxRegion,regionSize)
	return maxRegion						
```

## Path in Matrix
$N \times M$ matrix where 0 -> wall (cant move), 1 -> source, 2 -> dest, 3 -> blank (can move through)

4D movement. Check if path exists from source to destination

### How
BFS and explore the matrix and find the path.

```python
from collections import deque
def checkPath(A):
	n,m = len(A), len(A[0])
	source,dest = (-1,-1),(-1,-1)
	for i in range(n):
		for j in range(m):
			if A[i][j] == 1:
				source = (i,j)
			elif A[i][j] == 2:
				dest = (i,j)
	if source == (-1,-1) or dest == (-1,-1):
		return 0
	q = deque()
	visited = [[False]*(m) for _ in range(n)]
	q.append(source)
	visited[source[0]][source[1]] = 1
	dirs = [[1,0],[-1,0], [0,1], [0,-1]]
	while q:
		cur = q.popleft()
		if cur == dest: return 1
		i,j = cur
		for dx,dy in dir:
			x,y = i+dx, j+dy
			if 0<= x < n and 0 <= y < m:
				if not visited[x][y] or A[x][y] != 0:
					visited[x][y] = 1
					q.append((x,y))
	return 0
```

## Level Order Traversal of Binary Tree
Given binary tree, return its level order traversal as 2D array.
Level order is visiting nodes level by level starting from root

### How
BFS.
```python
from collections import deque
def levelOrder(root):
	if not root: return []
	q = deque()
	q.append(root)
	result = []
	while q:
		levelSize = len(q)
		level = []
		for i in range(levelSize):
			node = q.popleft()
			level.append(node)
			if node.left: q.append(node.left)
			if node.right: q.append(node.right)
		result.append(level)
	return result
```

## Smallest multiple with 0 and 1
Given a pos int A, find smallest mult of A such that it only consists of 0 or 1.
Output the num as string.

Example, A = 55, ans = 110. 

### How
Shortest path search, each node is a remainder mod A.

Edge are like adding 0 or 1 to current number

start from `1`

for $d \in {0,1}$ : $newRem = (r\times 10 + d) mod \space A$
Track parent and digit to reconstruct once we reach remainder 0.

```python
from collections import deque

def multiple(A):
    if A == 0:
        return "0"
    
    q = deque()
    parent = [-1] * A
    digit = [''] * A
    visited = [False] * A

    rem = 1 % A
    visited[rem] = True
    digit[rem] = '1'
    q.append(rem)

    while q:
        r = q.popleft()
        if r == 0:
            break

        for d in ['0', '1']:
            newRem = (r * 10 + int(d)) % A
            if not visited[newRem]:
                visited[newRem] = True
                parent[newRem] = r
                digit[newRem] = d
                q.append(newRem)

    # Reconstruct the answer from remainder 0
    rem = 0
    result = []
    while rem != -1:
        result.append(digit[rem])
        rem = parent[rem]
    
    return ''.join(reversed(result))
```

## Snake Ladder Problem

Given a $10 \times 10$ snake ladder board, numbered from 1  to 100 with list of `N` ladders, each ladder is a pair `(u,v)` indicating a move from `u` to `v` (u < v)

A list of M snakes `(u,v)` where it moves from `v` to `u`

Find the min dice rolls, to reach from square `1` to square `100`

Rule: if you land on a ladder or a snake, you have to take it. (note no overlaps in ladder and snakes)

### How? 
BFS, node (1,100) where

each node has edge to atmost 6 next nodes (dice roll)

each edge leads to a destination after taking ladder or snake

BFS, track min dice rolls to reach each square. Try all values of dice roll at each edge.

```python
def snakeLadder(A,B):
	jumps = [i for i in range(1,101)]
	for u,v in A:
		jump[u] = v
	for u,v in B:
		jump[v] = u
	visited = [False]*(101)
	q = deque()
	visited[1] = 1
	q.push((1,0)) # square , roll count
	while q:
		sq,rolls = q.popleft()
		if sq == 100:
			return rolls
		for d in range(1,7):
			nxt = sq + d
			if nxt > 100: break
			nxt = jump[nxt]
			if not visited[nxt]:
				visited[nxt] = 1
				q.append((nxt,rolls + 1))
	return -1
```

## Min Cost Path
Given matrix `C` of $A \times B$ where each cell has `U`, `D`, `L`, `R`
Start from `0,0` and reach `A-1,B-1`

Rules:

Following the dir on the current cell, move costs `0`

if you move in diff direction, it costs `1`

return min total cost to reach the destination

Example:

C = RRR
	DDD
	UUU
Output = 1

### How?
New Topic: (0-1) BFS

If you move in the given dir, it costs `0`. (enqueue it at front)

if you move in other dir, it costs `1` (enqueue it at the back)

Simple deque bfs that always prefers lower cost

```python
def minCost(A,B,C):
	if A == 1 and B == 1: return 0
	INF = 10**9
	dist = [[INF]*(B) for _ in range(A)]
	dist[0][0] = 0
	q = deque()
	q.append((0,0))
	dir = {'U': (-1,0),'D': (1,0), 'L': (0,-1),'R': (0,1)}
	while q:
		i,j = q.popleft()
		curCost = dist[i][j]
		for d in ['U','D','L','R']:
			dx,dy = dir[d]
			x,y = i + dx, j + dy
			if not (0 <= x < A and 0 <= y < B): continue
			cost = 0 if C[x][y] == d else 1
			if curCost + cost < dist[x][y]:
				dist[x][y] = curCost + cost
				if cost == 0: q.appendleft((x,y))
				else: q.append((x,y))
	return dist[A-1][B-1]
```

## Permutation Swaps
Given two perm A and B of int from 1 to N, and list of M good pairs C. good pair $(i,j)$ meaning you can swap those $(A[i],A[j])$ as many times.

Find if its possible to transform A into B.

Example:

A = 1 3 2 4
B = 1 4 2 3
C = (2,4)

Output = Yes

### How 
DSU babyyy

Build DSU from good pairs.

In each component (group of indices), collect A and B at those indices

Sort both subarrs. If the sorted val match, they can be rearranged.

If any component mismatch, return 0

```python
from collections import defaultdict
class DSU:
	def __init__(self, n):
		self.parent = list(range(n))
		self.rank = [0]*n
	def find(self, x):
		if self.parent[x] != x:
			self.parent[x] = self.find(self.parent[x])
		return self.parent[x]
	def unite(self,x,y):
		x_root = self.find(x)
		y_root = self.find(y)
		if x_root == y_root:
			return
		if self.rank[x_root] < self.rank[y_root]:
			self.parent[x_root] = y_root
		elif self.rank[y_root] < self.rank[x_root]:
			self.parent[y_root] = x_root
		else:
			self.parent[y_root] = x_root
			self.rank[x_root] += 1
def can_transform(A,B,C):
	if A== B:
		return true
	N = len(A)
	dsu = DSU(N)
	for u,v in C:
		dsu.unite(u-1,v-1)
	groups = defaultdict(list)
	for i in range(N):
		groups[dsu.find(i)].append(i)
	for indices in groups.values():
		a_vals = sorted([A[i] for i in indices])
		b_vals = sorted([B[i] for i in indices])
		if a_vals != b_vals:
			return False
	return True
```

## Commutable Islands (MST)
Given A islands and M bidirectional bridges. 
Each bridge has a cost. Your goal is to find the **min total cost** to connect all islands such that all of them are in one component.

A= num of islands, B = $M \times 3$ arr with weighted undirected edges

Output: Minimum total cost

A = 4
B = 1 2 1
	2 3 4
	1 4 3
	4 3 2
	1 3 10
Output = 6

### How

Kruskals Algorithm for MST with DSU

Sort all bridges by ascending cost.

Use disjoint set union to maintain connected components

Iteratively add A-1 lowest ccost bridges which connects two different components.

```python
class DSU:
	def __init__(self,n):
		self.parent = list(range(n))
		self.rank = [0]*n
	def find(self, x):
		if self.parent[x] != x:
			self.parent[x] = self.find(self.parent[x])
		return self.parent[x]
	def unite(self,x,y):
		xr, yr = self.find(x), self.find(y)
		if xr == yr:
			return False
		if self.rank[xr] < self.rank[yr]:
			self.parent[xr] = yr
		if self.rank[yr] < self.rank[xr]:
			self.parent[yr] = xr
		else:
			self.parent[yr] = xr
			self.rank[xr] += 1
		return True
def minCost(A,B):
	B.sort(key = lambda x: x[2])
	dsu = DSU(A)
	total_cost = 0
	edges_used = 0
	for u,v in B:
		if  dsu.unite(u-1,v-1):
			total_cost += cost
			edges_used += 1
			if edges_used == A-1:
				break
	return total_cost
```


## Possibility of Finishing all courses (Cycles in directed Graph, TopoSort)
Given A courses, 1 to A. 
pair $B[i],C[i]$ means to take course $C[i]$ you have to take $B[i]$ first
Determine if its possible to finish all courses.

Example:
A = 3, B = 1 2 
		C = 2 3
	Output = 1

Take 1 then 2 then 3

### How
Topological Sort using Kahn's Algorithm

Let courses be nodes and $prereq(u,v)$ be directed node from $(u,v)$

To check that the courses can be completed, we need to make sure it has no cycles.

Construct adj, and indegrees

Toposort using BFS

If all nodes are visited at the end(no cycle), return 1, else 0


```python
from collections import deque, defaultdict
def topoSort(A,B,C):
	adj = defaultdict(list)
	indeg = [0]*(A+1)
	M = len(B)
	for i in range(M):
		adj[B[i]].append(C[i])
		indegree[C[i]]+= 1
	q = deque()
	for i in range(1,A+1):
		if indegree[i] == 0:
			q.append(i)
	processed = 0
	while q:
		u = q.popleft()
		processed += 1
		for v in adj[u]:
			indegree[v]-= 1
			if indegree[v] == 0:
				q.append(v)
	return processed == A
```

## Cycles in undirected graph

A nodes from 1 to A. M edges represented in $M \times 2$ arr B, check whether graph has a cycle

Return if a cycle exists

### How?
DSU

for each node (u,v), if they are already connected, adding this edge would create a cycle.

```python

class DSU:
	def __init__(self,n):
		self.parent = list(range(n))
		self.rank = [0]*n
	def find(self, x):
		if self.parent[x] != x:
			self.parent[x] = self.find(self.parent[x])
		return self.parent[x]
	def unite(self,x,y):
		xr, yr = self.find(x), self.find(y)
		if xr == yr:
			return False
		if self.rank[xr] < self.rank[yr]:
			self.parent[xr] = yr
		elif self.rank[yr] < self.rank[xr]:
			self.parent[yr] = xr
		else:
			self.parent[yr] = xr
			self.rank[xr] += 1
		return True
def cycleExists(A,B):
	dsu = DSU(A)
	for u,v in B:
		if not unite(u,v):
			return 1
	return 0
```


## Mother Vertex
A vertices 1 to A, M directed edges given in a $M \times 2$ matrix B. Determine whether there is a mother. 
Mother is a vertex from which all the vertices are reachable.

Output boolean

### How

Candidate find: perform dfs, keep track of vertex with max finishing time. Then waha se verify by running a dfs and checking is all are visited.

```python
from collections import defaultdict

def motherVertex(A,B):
	adj = defaultdict(list)
	for u,v in B:
		adj[u].append(v)
	visited = [False]*(A+1)
	candidate = 1
	def dfs(u):
		visited[u] = True
		for v in adj[u]:
			if not visited[v]:
				dfs(v)
	for i in range(1,n+1):
		if not visited[i]:
			dfs(i)
			candidate = i
	visited = [False]*(A+1)
	dfs(candidate)
	for i in range(1,A+1):
		if not visited[i]:
			return 0
	return 1
```

## File Search
Organise records into sets. Each record is a pair (`ID`, `parentID`). If a record `(X,Y)` exists, then both `X` and `Y` belong to the same set. 
Find max number of sets into which the records can be partitioned, such that the condition holds.

### How
Just DSU and return the number of disjoint sets.

```python
class DSU:
	def __init__(self,n):
		self.parent = list(range(n))
		self.rank = [0]*(n)
	def find(self,x):
		if self.parent[x] != x:
			self.parent[x] = self.find(self.parent[x])
		return self.parent[x]
	def unite(self,x,y):
		rx,ry = self.find(x), self.find(y)
		if rx == ry:
			return False
		if self.rank[rx] < self.rank[ry]:
			self.parent[rx] = ry
		elif self.rank[ry] < self.rank[rx]:
			self.parent[ry] = rx
		else:
			self.parent[ry] = rx
			self.rank[rx] += 1
def fileSearch(A,B):
	dsu = DSU(A)
	for u,v in B:
		dsu.unite(u,v)
	groups = set()
	for i in range(A+1):
		groups.insert(dsu.find(i))
	return len(groups)
```

## Black Shapes
Given a grid A, having char `O` (white cells) and `X` black cells. Count the number of black shapes in the grid. 
Black shape is just the connected component of black cells.

### How
Just count components using BFS

```python
from collections import deque
def blackShapes(A):
	if not A: return 0
	n, m = len(A), len(A[0])
	dirs = [[1,0],[-1,0], [0,1], [0,-1]]
	count = 0
	for i in range(n):
		for j in range(m):
			if A[i][j] == 'X':
				count += 1
				q = deque()
				q.append((i,j))
				A[i][j] = 'O'
				while q:
					x,y = q.popleft()
					for dx,dy in dirs:
						a,b = x + dx, y + dy
						if 0 <= a < n and 0 <= b < m:
							if A[a][b] == 'X':
								A[a][b] = 'O'
								q.append((a,b))
	return count
```

## Convert Sorted List to Binary Tree
Given a linked list, where elements are in ascending order, convert it into a height balanced BST.

Height balanced maane depth of two subtrees of a node does not differ by more than 1

### How
Simulate $in-order$ traversal over the BST and advance the LL pointer in parallel

Count the length of linked list: n

Recursively build the BST using a helper function with start and end indices.

The function constructs left subtree first, then creates root using the current list node, then constructs the right subtree.


```python
class ListNode:
	def __init__(self, val = 0, next = None):
		self.val = val
		self.next = next
class TreeNode:
	def __init__(self,val = 0, left = None, right = None):
		self.val = val
		self.left = left
		self.right = right
def sortedListToBST(head: ListNode) -> TreeNode:
	# get length of LL
	def getLength(node):
		count = 0
		while node:
			count += 1
			node =node.next
		return count
	n = get_length(head)
	current = head
	# Recursively build BST
	def build_BST(start,end):
		nonlocal current
		if start > end: return None
		mid = (start + end) // 2
		left= build_BST(start,mid - 1)
		root = TreeNode(current.val)
		root.left = left
		current = current.next
		root.right = build_BST(mid + 1,end)
		return root
	return build_BST(0,n-1)
```

## Sum of Fibonacci Numbers

Given a pos int `A`, return min num of fibonacci numbers such that their sum is `A`. Repetition of numbers is allowed.

Example

A = 4, Output = 2 (2 + 2)
A = 7, Output = 2 (5 + 2)

### How
Generate all fibonacci numbers $\leq$ A.

Repeatedly subtract largest fibonacci number $\leq$ remaining value.

Count steps until `A` = 0

This works due to Zeckendorf's Theorem that every number can be represented as the sum of non consecutive fibonacci numbers. 

```python
import bisect
def fibSum(A):
	fib = [1,1]
	while True:
		nxt = fib[:-1] + fib[:-2]
		if nxt > A: break
		fib.append(nxt)
	# greedy subtraction
	count = 0
	while A > 0:
		idx = bisect.bisect_right(fib,A) # bisect right = upper bound
		A -= fib[idx]
		count += 1
	return count
```

## Knight on chess board

Given a chessboard of size $A \times B$, a knight starts at pos $(C,D)$ and wants to reach pos $(E,F)$. Find min num of moves to reach the destination.
If not pos, return -1

### How
Its simple bfs idk bro

```python
from collections import deque
def knight(A,B,C,D,E,F):
	if C == E and D == F: return 0
	dirs = [[2,1],[2,-1],[-2,1], [-2,-1], [1,2],[1,-2],[-1,2], [-1,-2]]
	visited = [[False]*(B+1) for _ in range(A+1)]
	q = deque((C,D,0))
	visited[C][D] = 1
	while q:
		x,y,dist = q.popleft()
		for dx,dy in dirs:
			i,j = x + dx, y+dy
			if 1 <= i <= A and 1 <= j  <= B and not visited[i][j]:
				if (i,j) == (E,F):
					return dist + 1
				visited[i][j] = 1
				q.append((i,j,dist+1))
	return -1
```

## Useful Extra Edges
A Nodes, undirected weighted edges $B[i] = [u,v,w]$ .

Given source C and dest D. List of extra edges $E[j] = [u,v,w]$ 
You are allowed to add atmost one extra edge from E.

Return the shortest path length from C to D. If no such path, return -1.

### How
Run dijkstra from source, and compute `distFromC[x]` 

Run dijkstra from destination, and compute distFromD[x]

let $best = distFromC[D]$ 
For each extra edge (u,v,w)
$$
minpath = min(distFromC[u] + w + distFromD[v], distFromC[v] + w + distFromD[u])
$$
Update the best dist if a shorter path is found.

```python
import heapq
from collections import defaultdict
def dijkstra(start, graph, A):
	dist = [float('inf')*(A+1)]
	dist[start] = 0
	pq = [(0, start)]
	while pq:
		d,u = heapq.heappop(pq)
		if d > dist[u]:
			continue
		for v,w in graph[u]:
			if dist[u] + w < dist[v]:
				dist[v] = dist[u] + w
				heapq.heappush(pq,(dist[v],v))
	return dist
def shortest_path_with_extra_edge(A,B,C,D,E):
	graph = defaultdict(list)
	for u,v,w in B:
		graph[u].append((v,w))
		graph[v].append((u,w)) 
	distFromC = dijkstra(C,graph,A)
	distFromD = dijkstra(D,graph,A)
	best = distFromC[D]
	for u,v,w in E:
		if distFromC[u] + w + distFromD[v] < best:
			best = distFromC[u] + w + distFromD[v]
		if distFromC[v] + w + distFromD[u] < best:
			best = distFromC[v] + w + distFromD[u]
	return best if best != float('inf') else -1
```


## Word Ladder I
Given two words `A` and `B`.
Find shortest transformation from A to B where one letter can be changed at a time. The transformed word must be in the dictionary C.

Return num of words in the shortest path.
Return 0 if no such path exists.

### How
BFS. 

Add all dict words in a set for $O(1)$ lookup

BFS starting from A. Each level of BFS represents one transformation.

For each word, try single letter mutations and push valid transformations into queue.

Stop when you reach B.

```python
from collections import deque
def wordLadder(A,B,C):
	if A == B: return 1
	word_set = set(C)
	if B not in word_set:
		return 0
	queue = deque([(A,1)])
	visited = set([A])
	L = len(A)
	while queue:
		word, steps = queue.popleft()
		for i in range(L):
			for c in 'abcdefghijklmnopqrstuvwxyz':
				if word[i] == c:
					continue
				nxt_word = word[:i] + c + word[i+1:]
				if nxt_word == B:
					return steps + 1
				if nxt_word in word_set and nxt_word not in visited:
					visited.add(nxt_word)
					queue.append((nxt_word,steps + 1))
	return 0
```

## Word Ladder II
Given two words, $start$ and $end$ , and a dict of words C, return all shortest transformation sequences from start to end with the same rules as Word Ladder I

Example

start = hit, end = cog, dict = hot dot dog lot log

Output : hit hot dot dog cog

hit hot lot log cog

### How
Advanced BFS tbh

Track all words at each level, and remove them once fully processed. 
Just keep storing the traversals at each level.

```python
from collections import deque, collections
def word_ladder_ii(beginWord, endWord, wordList):
	wordSet = set(wordList)
	if endWord not in wordSet:
		return []
	
	level = {beginWord}
	parents = defautdict(set)
	found = False

	while level and not found:
		next_level = defaultdict(set)
		for word in level:
			wordSet.discard(word)
		for word in level:
			for i in range(len(word)):
				for c in range 'abcdefghijklmnopqrstuvwxyz':
					newWord = word[:i] + c + word[i+1:]
					if newWord in wordSet:
						next_level[newWord].add(word)
						if newWord == endWord:
							found = True
		level = next_level
		for child, parSet in next_level.items():
			parents[child].update(parSet)
	# back track to get the solution
	res = []
	def backtrack(word, path):
		if word == beginWord:
			res.append([beginWord] + path[::-1])
			return
		for par in parents[word]:
			backtrack(par, path + [word])
	if found:
		backtrack(endWord,[])
	return res
```


## Clone Graph
Clone an undirected graph.

Each node in the graph contains a label and a list of its neighbors.

Given pointer to the node in the graph, return a deep copy.

### How
BFS traversal while cloning each node and its neigbors.

Use a hash map to maintain the mapping from orignal node to its clone

For each node visited, clone it, and clone its neighbors and link them

Return the cloned version of the original starting node.

```python
from collections import deque

class Node:
	def __init__(self, val, neighbors = None):
		self.val = val
		self.neighbors = neighbors if neighbors is not None else []

def cloneGraph(node):
	if not node:
		return None
	# map the starting node
	clones = {node : Node(node.val)}

	queue = deque([node])

	while queue:
		current = queue.popleft()
		for neighbor in current.neighbors:
			if neighbor not in clones:
				clones[neighbor] = Node(neighbor.val)
				queue.append(neighbor)
			clones[current].neighbors.append(clones[neighbor])
	return clones[node]
```
