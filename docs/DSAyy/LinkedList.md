## Reverse LinkedList
Reverse a linkedlist in place

1. Store the curr's next so we can go ahead in next iteration.
2. just assign the next ptr to prev
3. then prev = curr, and curr is that temp value


```python
class ListNode:
	def __init__(self, val = 0, nxt = None):
		self.val = val
		self.next = nxt
def reverseLL(head):
	prev = None
	curr = head
	while curr:
		next_temp = curr.next
		curr.next = prev
		prev = curr
		curr = next_temp
	return prev
```

## Intersection Of LinkedList
Find the node where two linkedlists intersect.
```python
class ListNode:
	def __init__(self, val = 0, nxt = None):
		self.val = val
		self.next = nxt
def getIntersectNode(headA, headB):
	if not headA or not headB: return None
	pA, pB = headA, headB
	while pA != pB:
		pA = pA.next if pA else headB
		pB = pB.next if pB else headA
	return pA
```

## Sort Binary LinkedList
Do it inplace with $O(1)$ extra space.
```python
class ListNode:
	def __init__(self, val = 0, nxt = None):
		self.val = val
		self.next = nxt
def sortBinaryLL(head):
	zeroDummy, oneDummy = ListNode(0), ListNode(0)
	zeroTail, oneTail = zeroDummy, oneDummy
	curr = head #link 2 chains for one and zero, then link them at the end
	while curr:
		if curr.val == 0:
			zeroTail.next = curr
			zeroTail = curr
		else:
			oneTail.next = curr
			oneTail = curr
		curr = curr.next
	oneTail.next = None #end the list
	zeroTail.next = oneDummy.next #link 0chain to 1 chain
	return zeroDummy.next #first was dummy
```

## Partition List
Given a LL and a value `B`. Partition it so that all nodes with val less than B come before B, and all nodes with val greater than B come after it. 

Preserve the relative order homie.
```python
class ListNode:
	def __init__(self, val = 0, nxt = None):
		self.val = val
		self.next = nxt
def partition(A,B):
	leDummy, geDummy = ListNode(0), ListNode(0) #dummy pointers
	leTail,geTail = leDummy, geDummy
	curr = A
	while curr:
		if curr.val < B:
			leTail.next = curr
			leTail = curr
		else:
			geTail.next = curr
			geTail = curr
		curr = curr.next
	geTail.next = None
	leTail.next = geDummy.next
	return leDummy.next
```

## Insertion Sort List
Sort a LL with insertion sort.

Just insert new node in the sorted portion in the correct order.

```python
class ListNode:
	def __init__(self,val = 0, nxt = None):
		self.val = val
		self.next = nxt
def insertionSortList(head):
	if not head or head.next:
		return head
	dummy = ListNode(0)
	dummy.next = jead
	lastSorted = head
	curr = head.next
	while curr:
		if lastSorted.val <= curr.val:
			lastSorted = curr
		else:
			prev = dummy
			while prev.next.val <= curr.val:
				prev = prev.next
			#re-link the nodes
			lastSorted.next = curr.next #so we know where to cont from next iteration
			curr.next = prev.next
			prev.next = curr
		curr = lastSorted.next
	return dummy.next
```

## Sort List (Merge Sort on LinkedList)

you already know what it is.

```python
class ListNode:
	def __init__(self, val= 0, nxt = None):
		self.val = val
		self.next = nxt
def length(head):
	count = 0
	while head:
		count += 1
		head = head.next
	return count
#split first n nodes, return head of the rest
def split(head,n):
	for _ in range(n-1):
		if head: head = head.next
		else: return None
	if not head:
		return None
	second = head.next
	head.next = None
	return second
#merge l1, l2 after tail, return new tail
def merge(l1,l2,tail):
	curr = tail
	while l1 and l2:
		if l1.val < l2.val:
			curr.next = l1
			l1 = l1.next
		else:
			curr.next= l2
			l2 = l2.next
		curr = curr.next
	curr.next = l1 if l1 else l2
	while curr.next:
		curr = curr.next
	return curr
def mergeSortLL(head):
	dummy = ListNode(0)
	dummy.next = head
	n = length(head)
	step = 1
	while step < n:
		prev,curr = dummy, dummy.next
		while curr:
			left = curr
			right = split(left,step)
			curr = split(right,step)
			prev = merge(left, right, prev)
		step <<= 1
	return dummy.next
```

## Palindrome List
Determine if the LL is a palindrome.

```python
class ListNode:
	def __init__(self,val = 0, nxt = None):
		self.val = val
		self.next = nxt
def isPalindromeLL(head):
	if not head and head.next: return 1
	#find the mid point
	slow, fast = head, head
	while fast.next and fast.next.next:
		slow = fast.next
		fast = fast.next.next
	#reverse the second half
	prev,curr = None, slow.next
	while curr:
		nxt = curr.next
		curr.next = prev
		prev = curr
		curr = nxt
	p1, p2 = head, prev
	isPal = True
	while p2:
		if p1.val != p2.val:
			isPal = False
			break
		p1 = p1.next
		p2 = p2.next
	#we can restore it too
	curr,prev = slow.next = None
	while curr:
		nxt = curr.next
		curr.next = prev
		prev = curr
		curr = nxt
	slow.next =prev
	return 1 is isPal else 0
```


## Remove Duplicates from Sorted List II
Delete all nodes that have duplicate numbers, leaving only distinct numbers from LL.

```python
class ListNode:
	def __init__(self,val = 0, nxt = None):
		self.val = val
		self.next = nxt
def removeDuplicatesII(head):
	dummy = ListNode(0)
	dummy.next = head
	prev,curr = dummy,head
	while curr:
		if curr.next and curr.val == curr.next.val:
			dupVal = curr.val
			#skip all nodes with this value
			while curr and curr.val == dupVal:
				curr = curr.next
			prev.next = curr
		else:
			prev = curr
			curr = curr.next
	return dummy.next
```

## Merge Two Sorted Lists
Merge two sorted linked lists, and return the merged sorted list.

```python
class ListNode:
	def __init__(self,val = 0, nxt = None):
		self.val = val
		self.next = nxt
def merge(A,B):
	dummy = ListNode(0)
	tail = dummy
	while A and B:
		if A.val < B.val:
			tail.next = A
			A= A.next
		else:
			tail.next = B
			B = B.next
		tail = tail.next
	tail.next = A if A else B
	return dummy.next
```

## Remove Duplicates from Sorted List
Remove all duplicates so that each element appears only once.

```python
class ListNode:
	def __init__(self,val = 0, nxt = None):
		self.val = val
		self.next = nxt
def removeDuplicates(A):
	curr = A
	while curr and curr.next:
		if curr.val == curr.next.val:
			curr.next = curr.next.next
		else:
			curr = curr.next
	return A
```


## Remove Nth Node from List End
Given a singly LL, remove `B`th node from the end and return the head of the linked list.

If B is greater than the length, remove the first node.

### How
Advance first ptr at Bth Node.
Then advance both ptr until the first ptr reaches the end. The second ptr would be at the node just before the one we need to delete.

```python
class ListNode:
	def __init__(self,val = 0, nxt = 0):
		self.val = val
		self.next=  nxt
def removeEnd(head, B):
	dummy = ListNode()
	dummy.next = head
	first,second = dummy, dummy
	for _ in range(B):
		if first.next: first = first.next
		else:
			return head.next
	while first.next:
		first = first.next
		second = second.next
	second.next = second.next.next
	return dummy.next
```

## K-Reverse Linked List
Given a singly linked list, and an int K, reverse the nodes of the list, K at a time, and return the modified list.

Note: the length is divisible by K.

```python
class ListNode:
	def __init__(self, val = 0, nxt =None):
		self.val = val
		self.next = nxt
def reverseKGroup(head, K):
	if not head and K <= 1:
		return head
	dummy = ListNode()
	dummy.next = head
	prevGroup = dummy
	while True:
		#find the Kth node
		kth = prevGroup
		for _ in range(K):
			kth = kth.next
			if not kth:
				return dummy.next
		groupStart = prevGroup.next
		nextGroup = kth.next
		#rev the group
		prev, curr = nextGroup, groupStart
		while curr != nextGroup:
			tmp = curr.next
			curr.next = prev
			prev = curr
			curr = tmp
		#connect the prev group
		prevGroup.next = kth
		prevGroup = groupStart
	return dummy.next
```

## Even Reverse
Given a LL, reverse the order of all nodes at even position.

Before: 1 -> 2 -> 3 -> 4 Output: 1 -> 4 -> 3 -> 2

1. Extract even pos nodes into a separate list.
2. Reverse this even list
3. Merge the reverse even list into the orignal list.

```python
class ListNode:
	def __init__(self, val = 0, nxt = None):
		self.val = val
		self.next = nxt
def evenReverse(head):
	if not head or not head.next:
		return head
	odd, even = head, head.next
	evenHead = even
	while even and even.next: #extracting even list
		odd.next = even.next
		odd = odd.next
		even.next = odd.next
		even = even.next
	#rev the even list
	prev, curr = None, evenHead
	while curr:
		nxt = curr.next
		curr.next = prev
		prev = curr
		curr = nxt
	revEven = prev
	#merge back even nodes
	odd,even = head,revEven
	while even:
		tmp = odd.next
		odd.next= even
		even = even.next
		odd.next.next = tmp
		odd = tmp
```

## Swap List Nodes in Pair
Given a LL A, swap every two adjacent nodes and return its head.

```python
class ListNode:
	def __init__(self, val = 0, nxt =  None):
		self.val = val
		self.next = nxt
def swapPair(head):
	dummy = ListNode()
	dummy.next = head
	prev = dummy
	while prev.next and prev.next.next:
		first = prev.next
		second = first.next
		#swap pair
		first.next = second.next
		second.next = first
		prev.next = second
		#move prev two steps ahead
		prev = first
	return dummy.next
```

## Rotate List
Given a linked list, rotate it by k places.

Just connect the linked list in a cycle. Then break at the right spot.

Find the length, then move to the new tail ($n-B$ th node)

```python
class ListNode:
	def __init__(self, val = 0, nxt = None):
		self.val = val
		self.next = nxt
def rotateLL(head, k):
	if not head or not head.next or k == 0:
		return head
	#find len and tail
	length = 1
	tail = head
	while tail.next:
		tail = tail.next
		length += 1
	#normalize k
	k = k % length
	tail.next  = head #make it circular
	#find new tail (len - k)th node
	newTail = head
	for _ in range(length - k - 1):
		newTail = newTail.next
	newHead = newTail.next #break at the len-k - 1th node
	newTail.next = None
	return newHead# len - kth node is the new head
```

## Kth node from the middle
Given a LL, find the value of kth node from the middle towards the head of the linkedlist.

If no such node exists, return -1

### How
Find length, find the middle. To get the Bth node from middle, move B steps backwards from middle.

```python
class ListNode:
	def __init__(self, val = 0, nxt = None):
		self.val = val
		self.next = nxt
def kFromMid(head, k):
	N = 0 #find len to get the mid
	p = head
	while p:
		N += 1
		p = p.next
	#find mid
	mid = N//2 + 1
	target = mid - B
	if target < 1: return -1
	# go to the target
	p = head
	for i in range(1, target):
		p = p.next
	return p
```

## Reverse Alternate K nodes
Given a singly LL, reverse every alternate B nodes in the list.

3->4->7->8->10->12, output: 4->3->7->8->12->10

### How
1. reverse first `B` nodes.
2. skip the next `B` nodes.
3. Continue reversing and skipping

```python
class ListNode:
	def __init__(self,val = 0, nxt = None):
		self.val = val
		self.next = nxt
def revAltK(head,k):
	if not head or k <= 1: return head
	dummy = ListNode()
	dummy.next = head
	prevGroup = dummy
	curr = head
	doReverse = True
	while curr:
		#check if there are B nodes ahead
		node = curr
		count = 0
		while node and count < k:
			node = node.next
			count += 1
		if count < B: break
		if doReverse:
			prev,p = node, curr
			for _ in range(k):
				nxt = p.next
				p.next = prev
				prev = p
				p = nxt
			prevGroup.next = prev
			prevGroup = curr
			curr = node
		else:
			for _ in range(k):
				prevGroup = curr
				curr = curr.next
		doReverse = not doReverse
	return dummy.next
```

## Reverse LinkedList II
Given singly LL, reverse nodes from pos `m` to `n` in one pass and in-place.

### How
1. Traverse to the node just before m (call it `prev`)
2. Reverse the next $(n - m + 1)$ nodes.
3. Carefully reconnect: before m, reversed seg, after n

```python
class ListNode:
	def __init__(self, val = 0, nxt = None):
		self.val = val
		self.next = nxt
def revII(head, m, n):
	if not head or m == n: return head
	dummy = new ListNode()
	dummy.next = head
	prev = dummy
	for _ in range(1,m): #move to node just before m
		prev = prev.next
	reverse_start = prev.next
	curr = reverse_start.next
	#reverse nodes [m,n]
	for _ in range(n - m):
		reverse_start.next = curr.next
		curr.next = prev.next
		prev.next = curr
		curr = reverse_start.next
	return dummy.next
```

## Reorder List
Given a singly LL, re order it to
$$
L_0 L_nL_1L_{n-1} ... L_xL_{n-x}
$$
1 ->2 -> 3 -> 4, output is 1->4->2->3

### How
1. Reverse the second half.
2. Zip first and reversed second half

```python
class ListNode:
	def __init__(self, val = 0, nxt = None):
		self.val = val
		self.next = nxt
def reorder(head):
	if not head or not head.next: return head
	#find the middle
	slow, fast = head, head
	while fast.next and fast.next.next:
		slow = slow.next
		fast = fast.next.next
	#rev the second half
	prev,curr = None, slow.next
	while curr:
		nxt = curr.next
		curr.next = prev
		prev = curr
		curr = nxt
	slow.next = None
	#merge the two halves
	p1,p2 = head, prev
	while p2:
		n1,n2 = p1.next, p2.next
		p1.next = p2
		p2.next = n1
		p1,p2 = n1,n2
	return head
```

## Add two numbers as Lists
Given two non-neg numbers as linkedlists, add them. Each list stores digits in reverse order.

### How
1. Trav both digit by digit
2. maintain the carry

```python
class ListNode:
	def __init__(self,val = 0, nxt = None):
		self.val = val
		self.next = nxt
def addLL(A,B):
	carry = 0
	dummy = ListNode()
	dummy.next = head
	tail = dummy
	while A or B or carry:
		sum = carry
		if A:
			sum += A.val
			A = A.next
		if B:
			sum += B.val
			B = B.next
		carry = sum//10
		tail.next = ListNode(sum % 10)
		tail = tail.next
	return dummy.next
```

## List Cycle
Given a LL, return the node where the cycle begins. If there is no cycle, return null.

### How
Tortoise Hare.
1. To find cycle's entry point, move one pointer to head, and advance both pointers one at a time, the node where they meet is the cycle's start.

```python
class ListNode:
	def __init__(self, val = 0, nxt =None):
		self.val = val
		self.next = None
def cycle(head):
	if not head: return None
	slow,fast = head,head
	while fast and fast.next:
		slow = slow.next
		fast = fast.next.next
		if slow == fast:
			#cycle mil gaya
			ptr = head
			while ptr != slow:
				ptr = ptr.next
				slow = slow.next
			return ptr
	return None
```





