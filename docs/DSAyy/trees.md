
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