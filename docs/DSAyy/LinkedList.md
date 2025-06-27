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
