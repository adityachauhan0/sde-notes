
## Next Greater Number in BST

Given a BST Node, return the node which has val greater than given node.

### How
This is basically finding inorder sucessor.

For finding node with val just > B,

1. While searching for node with val B, whenever we move to left child, curr node is candidate for successor. We record that candidate and move ahead.
2. Once we find the node with val B:
	1. If N has a right subtree, then the sucessor is the leftmost node in the right subtree.
	2. Else sucessor is the latest recorded ancestor (candidate)

```python
class TreeNode:
	def __init__(self,val):
		self.val = val
		self.left = None
		self.right = None
def inorder_sucessor(root,target):
	successor = None
	#search for node while tracking sucessors
	while root:
		if target.val < root.val:
			successor = root
			root = root.left
		elif target.val > root.val:
			root = root.right
		else:
			break # found root
	if target.right:
		successor = target.right
		while successor.left:
			successor = successor.left
	return successor
```

## Valid BST from PreOrder
Given and int arr A representing a preorder traversal of a BST.

Determine if A can correspond to the preorder traversal of some valid BST.

$$
(\text{left subtree keys}) < (\text{node key}) < (\text{right subtree keys})
$$
Return if its a valid preorder of BST. 

### How

A valid preorder of BST would have {root, left subtree, right subtree} order.

Each new preorder value should:
1. Be strictly greater than some lower bound
2. Fit into a position consistent with BST constraints implied previously

We can simulate building the BST without constructing it by
1. A stack to track the chain of ancestors for which we have not yet assigned a right child.
2. A lower_bound variable that tracks the smallest permissible value for the current node (once we pop from the stack, that popped value becomes the new lower bound)
```python
def is_valid_bst_preorder(preorder):
	stack = []
	lower_bound = float('-inf')
	for value in preorder:
		#val must be greater than the lowest allowed val (left subtree needs a right)
		if value < lower_bound:
			return False
		#nodes popped mean we are in the right subtree now
		while stack and value > stack[-1]:
			lower_bound = stack.pop() #pop all right subtree val
		stack.append(value)
	return True
```


## Kth Smallest Element in Tree
Given a BST, write a func to find the Kth smallest element in the tree.

### How
The Kth smallest element is just the Kth element in the inorder traversal
1. Traverse BST iteratively using a stack to simulate recursion.
2. Always push left children onto the stack until we reach a null
3. Pop one node at a time, each pop node corresponds to the next smallest element in the sorted order.
4. Maintain a counter, when counter reaches k, return the popped value

```python
class TreeNode:
	def __init__(self,val):
		self.val = val
		self.left = None
		self.right = None
def kth_smallest(root,k):
	stack = []
	current = root
	while True:
		#go as left as possible
		while current:
			stack.append(current)
			current = current.left
		#pop from stack
		current = stack.pop()
		k -= 1
		if k == 0: return current.val
		current = current.right
```

## 2-Sum Binary Tree

Given a BST, and an int B. Determine whether there are 2 distinct nodes X and Y in the tree such that
$$
X.val + Y.val = B
$$
### How
Normally we use 2 pointer in a sorted array, we need to simulate this in BST.
1. An inorder iterator (left to right) that yields val in asc order.
2. A reverse-inorder iterator (right to left) that yields val in des order.

Both iterators can be implemented with a stack
1. Init $next-smallest$ stack by pushing all left descendents from root down to leftmost node.
2. Init $next-largest$ stack by pushing all right descendents from root down to rightmost node.
3. getNext(): Pop from $s_1$ call that node's val $v$ . Then if that node has a right child, push its right child and all of that child's left descendents onto $s_1$ . Return v.
4. getPrev(): Pop from $s_2$ , Call that node's val $u$. Then if that node has a left child, push its left child and all of its right descendents into $s_2$ Return u. 

Then have a two-pointer loop. leftVal would be getNext(), rightVal would be getPrev(). Then normal 2 pointer approach.

```python
class TreeNode:
	def __init__(self,val):
		self.val = val
		self.left = None
		self.right = None
class BSTIterator:
	def __init__(self,root,forward):
		self.stack = []
		self.forward = forward #true is next val, false for prev
		self.__push_all(root)
	def __push_all(self,node):
		while node:
			self.stack.append(node)
			node = node.left if self.forward else node.right
	def next(self):
		node = self.stack.pop()
		val = node.val
		if self.forward:
			self.__push_all(node.right)
		else:
			self.__push_all(node.left)
		return val
	def has_next(self):
		return len(stack) > 0
def find_target(root,k):
	if not root: return False
	left_iter = BSTIterator(root,True)
	right_iter = BSTIterator(root,False)
	left_val = left_iter.next()
	right_val = right_iter.next()
	while left_val < right_val:
		cur_sum = left_val + right_val
		if cur_sum == k:
			return True
		elif cur_sum < k:
			if left_iter.has_next():
				left_val = left_iter.next()
			else: 
				break
		else:
			if right_iter.has_next():
				right_val = right_iter.next()
			else:
				break
	return False
```

## BST Iterator
Implement an iterator over a BST. Iterator is init with root of BST.
1. next() return the next smallest number in the BST
2. hasNext() return true if there is a next smallest number, otherwise false

### How
Abhi just toh kiya upar

```python
class TreeNode:
	def __init__(self,val):
		self.val = val
		self.left = None
		self.right = None
class BSTIterator:
	def __init__(self,root):
		self.stack = []
		self._push_left(root)
	def _push_left(self,node):
		while node:
			self.stack.append(node)
			node = node.left
	def next(self):
		node = self.stack.pop()
		val = node.val
		if node.right:
			self._push_left(node.right)
		return val
	def hasNext(self):
		return len(stack) > 0		
```


## Recover Binary Search Tree
Two nodes of a BST are swapped by mistake. Identify which, and swap them back.
Return a two-element list $[v_1, v_2]$ in asc order, wherte swapping $v_1$ and $v_2$ corrects the tree.

### How

A correct BST inorder traversal produces a strictly increasing seq of node values.
If  2 nodes are swapped, their inorder will have exactly one or two inversions.

1. Case 1 (Non-adjacent swap): 2 inversions occur. First pair identifies the first wrong node, and second identifies the second wrong node.
2. Case 2 (adjacent swap): Exactly one inversion occurs; just swap the 2 nodes in the inversion.

Perform a Morris inorder traversal, with $O(1)$ space. Keep track of prev pointer. Whenever $prev.val > curr.val$, we found an inversion.

If first inversion, first = prev, second = curr, if second, second = curr

```python
class TreeNode:
	def __init__(self,val):
		self.val = val
		self.left = None
		self.right = None
def recover_bst(root):
	first = second = prev = None
	current = root
	while current:
		if not current.left:
			# check for inversion
			if prev and prev.val > current.val:
				if not first:
					first = prev
				second = current.right
		else:
			# morris traversal setup
			pred = current.left
			while pred.right and pred.right != current:
				pred = pred.right
			if not pred.right:
				pred.right = current
				current = current.left
			else:
				pred.right = None
				if prev and prev.val > current.val:
					if not first:
						first = prev
					second = current
				prev = current
				current = current.right
	return sorted([first.val,second.val])
```

## Xor Between Two Arrays

Given two Int arrays A and B. Pick one element $a \in$ A and one element $b\in$ B so to maximie the val of $a \oplus b$  Where $\oplus$ is bitwise XOR. 

Return this max XOR value.

Example: A = 1 2 3, B = 4 1 2. Max is $3 \oplus 4$ which is 7

### How
We can insert all elements of A into a Binary Trie, instead of manually checking all the values.

Insert all element $a \in A$  into binary trie (a bitwise trie), and then, for each b $\in$ B, walk the trie greedily to pick bits that maximise $b \oplus a$ .

Represent each int in 31 bits, and build a binary trie of depth 31, where each node has two children, $$child[0] \space \text{  bit = 0}, \text{ child[1]     
bit = 1}$$
Inserting an int x simply means descending from root, examining bits and creating child pointers as needed.

To query $b \in B$ , we walk the root, and at bit index $i$ , we know b's ith bit is $b_i$ . To maximise $b \oplus a$ at that bit, we would like to pick $a_i = 1 - b_i$ if such a branch exists. Otherwise we must follow $a_i = b_i$ 

Accumulating these chosen bits builds the best possible partner $a$ from trie, and we compute ($b \oplus a$).

```python
class TrieNode:
	def __init__(self):
		self.child = [None,None] #child[0] for bit 0, child[1] for bit 1
def max_xor_between_arrays(A,B):
	#build trie from elements of A
	root = TrieNode()
	for x in A:
		node = root
		for i in range(30,-1,-1):
			bit = (x >> i) & 1
			if not node.child[bit]:
				node.child[bit] = TrieNode()
			node = node.child[bit]
	#step 2
	max_xor = 0
	for b in B:
		node = root
		curr_xor = 0
		for i in range(30,-1,-1):
			bit = (b >> i) & 1
			desired = bit ^ 1 # opp bit for maximising xor
			if node.child[desired]:
				curr_xor |= (1 << i)
				node = node.child[desired]
			else:
				node = node.child[desired]
		max_xor = max(max_xor, curr_xor)
	return max_xor
```

## Hotel Reviews
Given a string `A` of good words, separated by `_` and a vector B of review strings, where sequence of words are also separated by underscore.

Define the goodness value of a review, as the num of words in the review which match one of the good words. Return a vec of original indices of reviews in B, sorted in descending value of Goodness.
If two reviews have same goodness, their relative order must be stable. (preserving original order)

Example: A = cool_ice_wifi, B = {water_is_cool, cold_ice_drink, cool_wifi_speed}
Ans: 2 0 1

### How
Because num of good words and length of review can be large, we cannot compare every word, so we have to build a trie (prefix tree).

Build a trie of all good words:
1. Split string A on '_' to extract good word. (length $\leq$  4 in this problem).
2. Insert each good word into a 26-ary trie (one child for each letter 'a'-'z'). And mark the end of a good word with a boolean flag.
3. For each review in B split on '_', traverse trie to check if its marked good or not.
4. Count how many tokens appear in the review and that is the goodness val.
5. Finally perform the stable sort by comparing goodness in descending order.

```python
class TrieNode:
	def __init__(self):
		self.children = {}
		self.is_end = True
class Trie:
	def __init__(self):
		self.root = TrieNode()
	def insert(self,word):
		 cur = self.root
		 for char in word:
			 if char not in cur.children:
				 cur.children[char] = TrieNode()
				cur = cur.children[char]
		cur.is_end = True
	def search(self,word);
		cur = self.root
		for char in word:
			if char not in cur.children:
				return False
			cur = cur.children[char]
		return cur.is_end
def hotel_reviews(A,B):
	trie = Trie()
	good_words = A.split('_')
	for word in good_words:
		trie.insert(word)
	review_scores = []
	for index, review in enumerate(B):
		words = review.split('_')
		score = sum(1 for word in words if trie.search(word))
		review_scores.append((score,index))
	# sort by desc goodness
	review_scores.sort(key = lambda x: (-x[0],x[1]))
	#ret indices
	return [idx for _,idx in review_scores]
```



## Shortest Unique Prefix
Given a list of words (all lowercase with no word being a prefix of the other), find the shortest unique prefix for each word that distinguishes it from all other words in the list.

Example: zebra,dog,duck, dove. -> output z,dog,dy,dov

Simply prefix that no one else has.

### How
We build a prefix tree, of all inputs and store at each node, the number of words that pass through the node (`count`) . Then:
1. Insert each word into the trie, incrementing `count` at every node along its path.
2. To find the shortest unique prefix of a word, traverse its path from root, char by char, appending to the prefix string. As soon as we reach a node where the count is 1, the prefix is going to be unique.
3. Because no word is a prefix of other, ans is guaranteed.

```python
class TrieNode:
	def __init__(self):
		self.children = {}
		self.count = 0 #num of words passing through
class Trie:
	def __init__(self):
		self.root = TrieNode()
	def insert(self, word):
		cur = self.root
		for char in word:
			if char not in cur.childrem:
				cur.children[char] = TrieNode()
			cur = cur.children[char]
			cur.count += 1
	def find_prefix(self,word):
		cur = self.root
		prefix = ""
		for char in word:
			prefix += char
			cur = cur.children[char]
			if cur.count == 1:
				return prefix
		return prefix #fallback, full word
def shortest_unique_prefix(words):
	trie = Trie()
	for word in words: trie.insert(word)
	return [trie.find_prefix(word) for word in words]
```

## Path to Given Node
Given a binary tree `A` with N nodes. Each node has unique int value. And a target `B`.
Find the path from `root` to the node whose value is `B`.

Given a root pointer, return a 1D array with the path from root to B.

### How
A common approach for this is DFS and keeping track of node's parent. Once we discover node `B`, we can reconstruct the path by walking backwards to the root using `parent` pointers.
The just reverse the sequence.

1. Init an empty map parent, with $nodeValue -> parentValue$
2. Use an explicit stack to DFS the tree. When we visit a child, record $parent[child.val] = current.val$ 
3. As soon as we pop a node who value equals B, stops the DFS, and start reconstructing using the parent chain.

```python
class TreeNode:
	def __init__(self,val = 0, left= None, right= None):
		self.val = val
		self.left = left
		self.right = right
def path_to_node(root,B):
	if not root: return []
	parent = {root.val : None}
	stack = [root]
	target_node = None

	while stack:
		node = stack.pop()
		if node.val == B:
			target_node= node
			break
		if node.left:
			parent[node.left.val] = node.val
			stack.append(node.left)
		if node.right:
			parent[node.right.val] = node.val
			stack.append(node.right)
	if target_node is None:
		return []
	path = []
	while B is not None:
		path.append(B)
		B = parent[B]
	return path[::-1]
```

## Remove Half Nodes
Given a binary tree A with N nodes, remove all half nodes - nodes that have exactly one child- and return the root of the resulting tree. 

A leaf should not be removed.

### How
If a node has 2 children, keep it, else if it has one child, bypass it by linking it directly with its non-null child.

A bottom-up traversal (post order?)
1. Recursively process left and right subtrees, so all the half nodes below are already removed.
2. After recursion, examine node $u$ :
	1. if u is a leaf, keep it
	2. if it has 2 children, keep it
	3. if it has one child, return `u.left` or `u.right` whichever is non-null
3. Recursive call returns the root of the new pruned tree.

```python
class TreeNode:
	def __init__(self, val = 0, left= None, right = None):
		self.val = val
		self.left = left
		self.right = right
def remove_half_nodes(root):
	if not root: return None
	root.left = remove_half_nodes(root.left)
	root.right = remove_half_nodes(root.right)
	if root.left and not root.right:
		return root.left
	if root.right and not root.left:
		return root.right
	return root
```


## Nodes at distance K
Given the root of a binary tree `A`, and a target node value `B`, and an integer `C`
Return the array of all nodes that are exactly at distance `C` from the node with value `B`. 
You can return ans in any order.

### How
We treat binary tree as an undirected graph. Then we perform a BFS starting from target node, expanding outwards in all three directions (left, right, parent).
After `C` BFS levels, all nodes in the queue are at distance C.

```python
from collections import defaultdict, deque
class TreeNode:
	def __init__(self,val = 0, left = None, right = None):
		self.val = val
		self.left=  left
		self.right = right
def distance_k(root, target_val, k):
	graph = defaultdict(list)
	def build_graph(node, parent = None):
		if node:
			if parent:
				graph[node.val].append(parent.val)
				graph[parent.val].append(node.val)
			build_graph(node.left,node)
			build_graph(node.right, node)
	build_graph(root)
	visited = set()
	queue = deque([target_val])
	visited.add(target_val)
	distance = 0
	while queue and distance < k:
		for _ in range(len(queue)):
			node = queue.popleft()
			for neighbor in graph[node]:
				if neighbor not in visited:
					visited.add(neighbor)
					queue.append(neighbor)
		distance += 1
	return list(queue)
```

## Last Node in a complete binary tree.
Given the root of a complete binary tree A. Return the value of the rightmost node, in the last level of the tree. Aim for better than $O(N)$ time.

### How
In a complete binary tree of height `h`, the last level has indices 0 to $2^k - 1$. 

Define $exists(idx)$ that checks if a node at index `idx` exists. To do this, we start at root, and examine the bits of `idx` from (`h-1`) floor down to 0. A bit of `0` means go left, `1` means go right
. If you never reach `NULL`, then that `index` exists. This costs $O(h)$ 

This is like binary search on `idx`. 

```python
class TreeNode:
	def __init__(self,val = 0, left = None, right = None):
		self.val = val
		self.left = left
		self.right = right
def compute_height(node):
	height = 0
	while node.left:
		height += 1
		node = node.left
	return height
def exists(idx, height, node):
	left = 0
	right = (1 << height) - 1
	for i in range(height):
		mid = (left + right)//2
		if idx <= mid:
			node = node.left
			right = mid
		else:
			node = node.right
			left = mid + 1
		if not node:
			return False
	return True
def last_node_value(root):
	if not root: return None
	height = compute_height(root)
	if height == 0: return root.val
	left = 0
	right = (1 << height) - 1 #max pos nodes at last level
	# binary search for last existing node index
	while left <= right:
		mid = (left + right) // 2
		if exists(mid,height,root):
			left = mid + 1
		else:
			right = mid - 1
	# traverse the node at index 'right' to get its value
	idx = right
	node= root
	left=  0
	right = (1 << height) - 1
	for _ in range(height):
		mid = (left + right) // 2
		if idx <= mid:
			node = node.left
			right = mid
		else:
			node = node.right
			left = mid + 1
	return node.val if node else None
```

## Consecutive Parent-Child
Given root of binary tree A, count number of parent-child pais such that their values differ by exactly 1. 
$$
|parent.val - child.val| = 1
$$
### How
Simple tree traversal, (BFS or DFS), For each node:
1. If it has a left child, check if $|node.val - node.left.val| = 1,$  if yes increment the count.
2. Same for the right.

```python
class TreeNode:
	def __init__(self, val = 0, left= None, right = None):
		self.val = val
		self.left = left
		self.right = right
def count_consec_pairs(root):
	if not root: return 0
	count = 0
	stack = [root]
	while stack:
		node= stack.pop()
		if node.left:
			if abs(node.val - node.left.val) == 1:
				count += 1
			stack.append(node.left)
		if node.right:
			if abs(node.val - node.right.val) == 1L
				count += 1
			stack.append(node.right)
	return count
```

## Balanced Binary Tree
Given root of binary tree A, determine if its height balanced.
A binary tree is height balanced. 

Height balanced means $|depth(A.left) - depth(A.right)| \leq 1$

Return the boolean if its height balanced.

### How
Naive way would be computing height of its left and right subtree and checking the difference.

But we can use a single post-order traversal:
1. Recursively compute the height of each subtree
2. If any tree if already unbalanced, propogate a sentinel (ex: $-1$ ) upward immediately.
3. At each node, obtain the leftH and rightH, and if either is $-1$ otherwise do the usual check.
```python
class TreeNode:
	def __init__(self, val= 0 , left = None, right = None):
		self.val = val
		self.left = left
		self.right= right
def is_balanced(root):
	def check_height(node):
		if not node: return 0
		left_height = check_height(node.left)
		if left_height == -1: return -1
		right_height = check_height(node.right)
		if right_height == -1: return -1
	
		if abs(left_height - right_height) > 1:
			return -1
		return max(left_height, right_height) + 1
	return 0 if check_height(root) == -1 else 1
```

## Maximum Edge Removal
Given an undirected tree, with an even number of nodes A. You may remove as many edges as possible so that each resulting connected component (subtree) has an even number of nodes.

Return the maximum number of edges that can be removed.

### How
Root the tree at node 1, we want to cut as many parent-child edges such that resulting connected components have even size.
1. If a subtree is rooted at u and has even number of nodes, we may cut te edge connecting it to its parent. 
2. If a subtree has odd size, we cannot cut its root to parent edge.

So
1. Compute, for every node `u`, the size of its subtree.
2. Process nodes in post order, (children first). Whenever a node $u\neq$ 1 has an even subtree size, increment the answer by 1, and do not add u's size to its parent. 
3. If it has odd size, add u's size to its parent's running total.
4. Root's final collected size is A, which is even, but we can never cut above the root.

```python
from collections import defaultdict
def max_edge_removal_even_tree(A,B):
	#build adj
	adj = defaultdict(list)
	for u,v in B:
		adj[u].append(v)
		adj[v].append(u)
	#post order setup
	parent = [0]*(A+1)
	parent[1] = 0
	stack = [(1,0)]
	postorder = []

	while stack:
		u,p = stack.pop()
		postorder.append(u)
		parent[u] = p
		for v in adj[u]:
			if v != p:
				stack.append((v,u))
	subsize = [1]*(A+1) #subtree size, default itself (1)
	answer = 0
	for u in reversed(postorder):
		if u != 1 and subsize[u] % 2 == 0:
			answer += 1
		else:
			p = parent[u]
			if p != 0:
				subsize[p] += u
	return answer
```


## Merge Binary Trees
Given two binary trees `A` and `B`. merge them into a single binary tree according to this rule
1. if two nodes overlap (both non-null at same position). sum their value to form a new node.
2. otherwise use the non-null node as in the merged tree.
Return a pointer to the root of the merged tree.

Input, ptr to A and B, Output the root ptr to merged binary tree.

### How
We perform a simultaneous pre-order traversal of both trees:
1. if both current nodes u (from A) and v (from B) are non-null, create (or reuse) a node with value $u.val + v.val$ 
	1. Recursively merge left children
	2. Recursively merge right children
	
```python
class TreeNode:
	def __init__(self,val = 0, left= None, right = None):
		self.val = val
		self.left = left
		self.right = right
def mergeTrees(u,v):
	if not u: return v
	if not v: return u
	u.val += v.val
	u.left = mergeTrees(u.left,v.left)
	u.right = mergeTrees(u.right,v.right)
	return u
```

## Symmetric Binary Tree
Given the root of binary tree A, determine whether it is symmetric around its center (i.e a mirror of itself). In other words, left and right subtree should be mirror images.

### How
1. If both nodes are null, they match
2. if one is null and other is not, they dont
3. otherwise we repeat the top 2 for pair (left.left,right.right) and (left.right, right.left)

```python
from collections import deque
class TreeNode:
	def __init__(self, val = 0, left =None, right = None):
		self.val = val
		self.left = left
		self.right = right
def is_symmetric(root):
	if not root: return 1
	queue = deque()
	queue.append((root.left,root.right))
	while queue:
		u,v = queue.popleft()
		if not u and not v: continue
		if not u or not v: return 0
		if u.val != v.val: return 0
		queue.append((u.left,v.right))
		queue.append((u.right,v.right))
	return 1 
```


## Identical Binary Tree
Given two trees A and B, return if they are identical, structurally and value vise.

### How
```python
class TreeNode:
	def __init__(self,val = 0, left= None, right = None):
		self.val = val
		self.left = left
		self.right = right
def is_same_tree(u,v):
	if not u and not v: return 1
	if not u or not v: return 0
	if u.val != v.val: return 0
	return (is_same_tree(u.left,v.left) and is_same_tree(u.right,v.right)):
```

## Construct BST from PreOrder
Given an int arr of distinct elements representing the preorder traversal of a BST, construct the corresponding BST, and return its root pointer.

### How
In a BST, for a node with val `v`, all subsequent preorder values less than `v` belong to the left subtree. Values greater than `v` belong to the right subtree or higher up.

Simulate insertion using a stack:
1. First element $A[0]$ becomes the root.
2. Maintain a stack of nodes representing the path from the root down to the most recently inserted node.
3. For each new value $A[i]$ :
	1. If $A[i]$ is less than the value at the top of the stack, it must be the left child of that top value. Create a new node and attach it as top.left and push it on to the stack.
	2. Else: pop from stack until you find a node that is greater than $A[i]$. The last popped node is the parent of the new node in its right subtree.  Create the new node as parent.right, then push it onto the stack.
4. Continue until all elements are processed.

```python
class TreeNode:
	def __init__(self,val = 0, left = None, right = None):
		self.val = val
		self.left = left
		self.right = right
def bst_from_preorder(preorder):
	if not preorder: return Node
	root = TreeNode(preorder[0])
	stack = [root]
	for i in range(1,len(preorder)):
		curr = TreeNode(preorder[i])
		#smaller than stack top? left child
		if preorder[i] < stack[-1].val:
			stack[-1].left = curr
		else: #find parent in right chain
			parent = None
			while stack and preorder[i] > stack[-1].val:
				parent = stack.pop()
			parent.right = curr
		stack.append(curr)
	return root
```

## Inorder Traversal Of a Cartesian Tree
Given arr `A` on distinct integers, representing inorder traversal of a **cartesian tree.** Return the cartesian tree and its root. 

Cartesian Tree:
1. **Heap Property**: every node's value is greater than all values in its subtree.
2. **Inorder property**: an inorder traversal of the tree yields exactly the orignal array `A`.

### How
To build the cartesian tree in $O(n)$ time, we process A from left to right using a stack.
1. Init an empty stack `st` of TreeNode pointers.
2. For each value x = `A[i]`
	1. Create a new node `curr` = TreeNode(x)
	2. Pop nodes from the top of the stack while they have value less than x. Let last be the last node popped (or `nullptr` if none). Attach last as `curr.left`. This would maintain the inorder property: everything popped lies to the left of `x`.
	3. If the stack is nonempty aftert popping, the new top's value is > x, so we attach `curr` as `st.top().right`. This would ensure `curr` becomes the right child of the nearest larger node to its left.
	4. Push `curr` onto the stack.
3. After processing all elements, bottom of the stack (first el pushed) is the root of cartesian tree.

```python
class TreeNode:
	def __init__(self,val):
		self.val = val
		self.left = None
		self.right = None
def build_cartesian_tree(A):
	st = []
	for i in range(len(A)):
		curr = TreeNode(A[i])
		last = None
		while st and st[-1].val < A[i]:
			last = st.pop()
		curr.left = last
		if st:
			st[-1].right = curr
		st.append(cur)
	return st[0] if st else None
```

## Sorted Array to Balanced BST

Given an array `A` of length `n` whose elements are sorted in strictly ascending order, convert it into a height balanced BST. 

Basically the depth of right and left subtree differ by atmost 1.

### How

1. In a BST, in-order traversal yields the sorted sequence.
2. To keep it balanced, choose the mid element of the array as root,  and half of the elements go to the left subtree and half to the right.
3. Recursively apply the same procedure to the left subarray's midpoint and right subarray's midpoint.

```python
class TreeNode:
	def __init__(self,val):
		self.val = val
		self.left = None
		self.right = None
def sorted_arr_to_bst(A):
	def build_bst(l,r):
		if l > r:
			return None
		mid = l + (r- l) //2
		node = TreeNode(A[mid])
		node.left = build_bst(l,mid - 1)
		node.right = build_bst(mid + 1, r)
		return node
	return build_bst(0, len(A) - 1)
```

## Construct Binary Tree from Inorder and preorder

Given 2 int arrays A and B having pre-order and in-order traversal. Construct the binary tree and return its root pointer.

### How
In preorder, first element is always the root. In inorder, the elements to the lest of that root are in left subtree, and els to the right are in right subtree.

1. Maintain a global preIndex into the preorder array.
2. Define a recursive function $build(inL,inR)$ that constructs the subtree whose in-order indices range from $inL$ to $inR$ .
3. In $build(inL,inR)$:
	1. root would be $A[preIndex]$ , increment this ptr
	2. look for this value in inOrder array. call this `mid`
	3. Now left subtree would be $B[inL...mid-1]$  and right subtree would be $build(mid+1 ... inR)$ 
	4. return root
```python
class TreeNode:
	def __init__(self,val):
		self.val = val
		self.left=  None
		self.right = None
def build_tree(preorder, inorder):
	idx_map = {val : idx for idx,val in enumerate(inorder)}
	pre_index = [0] #to pass by index
	def build(left,right):
		if left > right: return None
		root_val = preorder[pre_index[0]]
		pre_index[0] += 1
		root = TreeNode(root_val)
		mid = idx_map[root_val]
		root.left = build(left, mid - 1)
		root.right = build(mid + 1, right)
		return root
	return build(0, len(inorder) - 1)
```

## Binary Tree from Inorder and PostOrder

Given inorder and post-order traversal in an array, construct the binary tree and return the root pointer.

### How
In the post order traversal, last element is always the root of the sub tree.
Like above we use it but backwards.

```python
class TreeNode:
	def __init__(self,val):
		self.val = val
		self.left = None
		self.right = None
def build_tree(inorder, postorder):
	idx_map = {val,idx for idx,val in enumerate(inorder)}
	post_index = [len(postorder) - 1] #use list for mutable int
	def build(left,right):
		if left > right: return None
		root_val = postorder[post_index[0]]
		post_index[0] -= 1
		root = TreeNode(root_val)
		mid = idx_map[root_val]
		root.right = build(mid + 1, right) #make the right first
		root.left = build(left, mid - 1)
		return root
	return build(0,len(inorder) - 1)
```

## Vertical Order Traversal of Binary Tree
Given a binary tree of `N` nodes. Return a 2d array denoting its vertical order traversal.
Label the root's column index as `0`; for any node at column `c`, its left child at column `c-1`, and its right child at column `c+1`.

Group nodes by column. Basically column index is the array index. Give all elements column wise.

### How
Perform a BFS but with node, also carry the current column index. When sending in left child, bas do `c-1`, and when going right, do `c+1`.

```python
from collections import defaultdict, deque
class TreeNode:
	def __init__(self,val = 0, left = None, right = None):
		self.val = val
		self.left = left
		self.right = right
def vertical_order_traversal(root):
	result = []
	if root is None: return result
	col_map = defaultdict(list)
	min_col = max_col = 0
	q = deque([(root,0)])
	while q:
		node , col = q.popleft()
		col_map[col].append(node.val)
		min_col = min(min_col,col)
		max_col = max(max_col,col)
		if node.left: q.append((node.left, col - 1))
		if node.right: q.append((node.right, col + 1))
	total_cols = max_col - min_col + 1
	result = [[] for _ in range(total_cols)]
	for col in range(min_col,max_col + 1):
		result[col - min_col] = col_map[col]
	return result
```

## Diagonal Traversal Of Binary Tree

Given a binary tree A with N nodes, output all nodes in a diagonal order. Where nodes lying on the same line of slope - 1 belong to the same diagonal.
Label the node's diagonal as 0.

Within each diagonal, node must be in preorder. Finally concatenate the diagonals from smallest index to largest. (leftmost to rightmost)

Input: Root of binary tree.
Output: 1D array

### How
A node's diagonal index d is defined as:
$$
d(root) = 0, \space d(node.left) = d(node) + 1, \space d(node.right) = d(node)
$$
Bas bhai map ke saath banate reh.

```python
class TreeNode:
	def __init__(self, val = 0, left = None, right = None):
		self.val = val
		self.left = left
		self.right = right
def diagonal_traversal(root):
	if not root: return []
	diag_nodes = [] #list of lists
	max_diag = 0
	stack = [(root,0)]
	while stack:
		node, d = stack.pop() #dfs style
		if len(diag_nodes) <= d: #extend if diagonal too big
			diag_nodes.extend([[] for _ in range(d - len(diag_nodes) + 1)])
		diag_nodes[d].append(node.val)
		max_diag = max(max_diag,d)
		if node.right:
			stack.append((node.right, d))
		if node.left:
			stack.append((node.left, d+1))
	for i in range(max_diag + 1):
		result.extend(diag_nodes[i])
	return result
```

## Vertical Sum of a Binary Tree
Given the root pointer of a binary tree. Comput the *vertical sum* for each vertical line of the tree. Label the root's column as 0; for any node at column c, its left child is at column `c-1` and right at `c+1` . The vertical sum for a column is the sum of all node values that lie in that column. Return an array of these sums, ordered from the leftmost column to the rightmost column.

### How
Perform a BFS of the tree while tracking each node's column index. Keep a hasmap as column's sum metric. 
Similar to what we solved above.

```python
from collections import defaultdict, deque
class TreeNode:
	def __init__(self,val = 0, left = None, right = None):
		self.val = val
		self.left = left
		self.right = right
def vertical_sum(root):
	if not root: return []
	col_sum = defaultdict(int)
	min_col = max_col = 0
	q = deque([(root,0)])
	while q:
		node,col = q.popleft()
		col_sum[col] += node.val
		min_col = min(min_col,col)
		max_col = max(max_col,col)
		if node.left:
			q.append((node.left,col - 1))
		if node.right:
			q.append((node.right, col + 1))
	total_cols = max_col - min_col + 1
	result = [0]*total_cols
	for col in range(min_cols, max_cols + 1):
		result[col - min_col] = col_sum[col]
	return result
```


## Covered / Uncovered Nodes
Given root of a binary tree A.
A node is :
- **Uncovered** if it appears as either the first or the last node on its level.
- Covered otherwise
Compute absolute difference of:
$$
|\text{(sum of covered values) - (sum of uncovered values)}|
$$
### how
Perform BFS, to identfy first or last at each level.
```python
from collections import deque
class TreeNode:
	def __init__(self,val = 0, left= None, right = None):
		self.val = val
		self.left = left
		self.right = right
def covered_minus_uncovered_sum(root):
	if not root:
		return 0
	covered_sum = 0
	uncovered_sum = 0
	q = deque([root])
	while q:
		sz = len(q)
		for i in range(sz):
			node = q.popleft()
			if i == 0 or i == sz - 1:
				uncovered_sum += node.val
			else:
				covered_sum += node.val
			if node.left:
				q.append(node.left)
			if node.right:
				q.append(node.right)
	return covered_sum - uncovered_sum
```


## Inorder Traversal of A Binary Tree
Given `root` pointer of a binary tree, return its nodes in inOrder travesal.

```python
class TreeNode:
	def __init__(self,val = 0, left = None, right = None):
		self.val = val
		self.left = left
		self.right = right
def inorder_traversal(root):
	res, stack = [], []
	while root or stack:
		#go as far left as possible
		while root:
			stack.append(root)
			root = root.left
		#visit node (mid)
		root = stack.pop()
		res.append(root.val)
		# then go right
		root = root.right
	return res
```

## Preorder Traversal Of a Binary Tree
Given `root` pointer of a binary tree, return its nodes in preorder traversal.
```python
class TreeNode:
	def __init__(self, val = 0, left = None, right = None):
		self.val = val
		self.left = left
		self.right = right
def preorder_traversal(root):
	res, stack = [],[]
	if root: stack.append(root)
	while stack:
		node = stack.pop()
		res.append(node.val)
		if node.right: 
			stack.append(node.right) #right first
		if node.left:
			stack.append(node.left) #then left, so left is processed first
	return res
```

## PostOrder Traversal Of a binary Tree
Given `root` pointer of a binary tree, return its nodes in postorder traversal
```python
class TreeNode:
	def __init__(self, val = 0, left = None, right = None):
		self.val = val
		self.left = left
		self.right = right
def postorder_traversal(root):
	if not root: return []
	toVisit, visited = [root], []
	while s1:
		node = toVisit.pop()
		visited.append(node)
		if node.left: toVisit.append(node.left)
		if node.right: toVisit.append(node.right)
	return [n.val for n in reversed(visited)]
```

## Right view of a binary tree
Given `root`, return an `array` of int representing the right view of the tree.
Basically the nodes that are visible when looked from the right.

Basically the last element in every level order.

```python
from collections import deque
class TreeNode:
	def __init__(self,val = 0, left = 0, right = 0):
		self.val = val
		self.left =left
		self.right = right
def rightView(root):
	if not root: return []
	right_view = []
	q = deque([root])
	while q:
		sz = len(q)
		for i in range(sz):
			node = q.popleft()
			if i == sz - 1:
				right_view.append(node.val)
			if node.left: q.append(node.left)
			if node.right: q.append(node.right)
	return right_view
```

## Cousins in a binary tree
Given `root` pointer of a binary tree with `N` nodes, and a target value `B` that exists in the tree. Return an `array` of all the cousins of node whole value is `B`. 

Two nodes are cousins, if they are on the same depth, but have different parents. 
Sibling nodes are not cousins.

### How
Perform a single BFS (level order), that keeps track of each node's parent
1. Record the nodes at the level `levelNodes`
2. Check if `B` is in `levelNodes` along with its parent pointer.
3. Return all other nodes on the level with different parent.

```python
from collections import deque
class TreeNode:
	def __init__(self,val = 0,left = None,right = None):
		self.val = val
		self.left = left
		self.right = right
def find_cousins(root,B):
	if not root: return []
	cousins = []
	q = deque()
	q.append((root,None)) # (node,parent)
	while q: 
		sz = len(q)
		level_nodes = []
		target_parent = None
		for _ in range(sz):
			node, parent = q.popleft()
			level_nodes.append((node,parent))
			if node.val == B:
				target_parent = parent
		if target_parent:
			for node,parent in level_nodes:
				if parent != target_parent and node.val != B:
					cousins.append(node.val)
		#enqueue children for the next level
		for node,parent in level_nodes:
			if node.left: q.append((node.left,node))
			if node.right: q.append((node.right,node))
	return cousins
```

## Reverse Level Order Traversal of a Binary Tree
Given `root`, return nodes in reverse level order. Like from bottom-most to the top.
### How
Just do a normal BFS (level order) from top to bottom. But store each level's value in a list then prepend it to a `deque`. 
1. Record current level, and push it to the front of the `deque`.
2. Now it will have from bottom to top coz we pushed to the front.
```python
from collections import deque
class TreeNode:
	def __init__(self,val = 0, left = None, right = None):
		self.val = val
		self.left = left
		self.right = right
def reverseLevel(root):
	res = []
	if not root: return res
	q = deque([root])
	levels = deque()
	while q:
		sz = len(q)
		level = []
		for _ in range(sz):
			node = q.popleft()
			level.append(node.val)
			if node.left: q.append(node.left)
			if node.right: q.append(node.right)
		levels.appendleft(level)
	for lvl in levels: res.extend(lvl)
	return res
```

## Zigzag Level Order
Given `root` of binary tree, return nodes in zigzag level order.
- level 0, L to R
- level 1, R to L
basically alternating direction every level

just reverse the even level bro i dont even know

```python
from collections import deque
class TreeNode:
	def __init__(self,val=0,left = None, right = None):
		self.val = val
		self.left = left
		self.right = right
def zigzag(root):
	res = []
	if not root: return res
	q = deque([root])
	left_to_right = True
	while q:
		sz = len(q)
		level = []
		for _ in range(sz):
			node = q.popleft()
			level.append(node)
			if  node.left: q.append(node.left)
			if node.right: q.append(node.right)
		if not left_to_right: level = reversed(level)
		left_to_right = !left_to_right
		res.extend(level)
	return res
```

## Populate Next Right Pointers in a Binary Tree
Given the root pointer of a binary tree. The struct also has a next pointer. Populate each node's `next` pointer so that ir points to the node immediately to its right on the same level. If there is no such node, let `next` be null.

```python
class TreeLinkNode:
	def __init__(self, val = 0, left = None, right = None, next =None):
		self.val = val
		self.left= left
		self.right = right
		self.next = next
def connect(root):
	head = root #head of current level
	while head:
		dummy = TreeLinkNode(0)
		tail = dummy
		curr = head

		while cur:
			if cur.left:
				tail.next = cur.left
				tail = tail.next
			if cur.right:
				tail.next = cur.right
				tail = tail.next
			cur = cur.next
		head = dummy.next

```


## Burn a Tree
Given `root` of a binary tree A, and a target leaf `B`, a fire starts at node B at time = 0. Each second, the fire spreads from any burning node to its directly connected neighbors (left child, right child and parent). Compute minimum time required to burn the entire tree.

### How
Run BFS, and treat the tree like an undirected graph. In BFS, also pass the time, and then check the min time.

```python
from collections import deque, defaultdict
class TreeNode:
	def __init__(self, val = 0, left = None, right = None):
		self.val = val
		self.left= left
		self.right = right
def build_parent_map(root,parent_map):
	queue = deque([root])
	while queue:
		node = queue.popleft()
		if node.left:
			parent_map[node.left] = node
			queue.append(node.left)
		if node.right:
			parent_map[node.right] = node
			queue.append(node.right)
def find_target_node(root, target):
	if not root: return None
	if root.val == target: return root
	left = find_target_node(root.left, target)
	if left: return left
	return find_target_node(root.right, target)
def burn_tree(root, target_val):
	if not root: return 0
	parent_map = {}
	build_parent_map(root,parent_map)
	target_node = find_target_node(root,target_val)
	visited = set()
	queue = deque([target_node])
	visited.add(target_node)
	time = -1 #first level at t = 0
	while queue:
		sz = len(queue)
		for _ in range(sz):
			node = queue.popleft()
			for neighbor in [node.left, node.right, parent_map.get(node)]:
				if neighbor and neighbor not in visited:
					visited.add(neighbor)
					queue.append(neighbor)	
		time += 1
	return time
```

## Max Depth of a Binary Tree
Given the `root` pointer of a binary tree, find its maximum depth.
```python
from collections import deque
class TreeNode:
	def __init__(self, val = 0, left = None, right = None):
		self.val = val
		self.left = left
		self.right= right
def maxDepth(root):
	if not root: return 0
	q = deque([root])
	depth = 0
	while q:
		sz = len(q)
		depth ++
		for _ in range(sz):
			node = q.popleft()
			if node.left: q.append(node.left)
			if node.right: q.append(node.right)
	return depth
```

## Sum Root to Leaf Numbers
Given `root` to a binary tree whose node contain 0-9, each root to leaf path represents a number concatenating the digits along the way. Return sum of all sum numbers modulo 1003.

```python
class TreeNode:
	def __init__(self,val = 0, left = None, right = None):
		self.val = val
		self.left = left
		self.right = right
def sum_root_to_leaf_numbers(root):
	if not root: return 0
	MOD = 1003
	result = 0
	stack = [(root,root.val % MOD)] #(node, cur_sum modulo)
	while stack:
		node, curr = stack.pop()
		if not node.left and not node.right:
			result = (result + curr) % MOD
		if node.right:
			next_val = (curr*10 + node.right.val) % MOD
			stack.append((node.right, next_val))
		if node.left:
			next_val = (curr*10 + node.left.val) % MOD
			stack.append((node.left, next_val))
	return result
```

## Path Sum
Given `root` of a binary tree, and an int `B`, determine whether there exists a root-to-leaf path in A such that sum of node values along that path equals B.

Just DFS and pass cur sum bro

```python
class TreeNode:
	def __init__(self, val = 0, left=  None, right = None):
		self.val = val
		self.left = left
		self.right = right
def has_path_sum(root, target_sum):
	if not root: return 0
	stack = [(root, root.val)] #(node, cur sum)
	while stack:
		node, cur_sum = stack.pop()
		if not node.left and not node.right:
			if cur_sum == target_sum: return 1
		if node.right:
			stack.append((node.right, cur_sum + node.right.val))
		if node.left:
			stack.append((node.left, cur_sum + node.left.val))
	return 0
```

## Min Depth of a Binary Tree
Given the `root` of a binary tree. Find its minimum depth. Basically min root-to-leaf path.

```python
from collections import deque
class TreeNode:
	def __init__(self,val = 0, left = None, right = None):
		self.val = val
		self.left = left
		self.right = right
def min_depth(root):
	if not root: return 0
	queue= deque([(root,1)]) #(node, curr depth)
	while queue:
		node, depth = queue.popleft()
		if not node.left and not node.right: #return the earliest you reach leaf
			return depth
		if node.left: 
			queue.append((node.left, depth + 1))
		if node.right:
			queue.append((node.right, depth + 1))
	return 0	
```
