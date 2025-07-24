### 146. LRU Cache

**Description:** Design a cache that supports `get(key)` and `put(key,value)` in O(1). When capacity is reached, evict the least‐recently used item.

```cpp
class LRUCache {
    int cap;
    list<pair<int,int>> dq;                                // front = most recent
    unordered_map<int, list<pair<int,int>>::iterator> mp; // key → iterator in dq
public:
    LRUCache(int capacity): cap(capacity) {}

    int get(int key) {
        auto it = mp.find(key);
        if (it == mp.end()) return -1;
        // move accessed node to front
        dq.splice(dq.begin(), dq, it->second);
        return it->second->second;
    }

    void put(int key, int value) {
        if (mp.count(key)) {
            // update existing
            dq.splice(dq.begin(), dq, mp[key]);
            dq.begin()->second = value;
        } else {
            if (dq.size() == cap) {
                // evict LRU at back
                mp.erase(dq.back().first);
                dq.pop_back();
            }
            // insert new at front
            dq.emplace_front(key, value);
            mp[key] = dq.begin();
        }
    }
};
```

**Explanation:**  
We keep a doubly‐linked list (`dq`) ordered by recency and a hash map to find nodes in O(1). On access or update we move the node to the front; on overflow we pop from the back.

---

### 155. Min Stack

**Description:** Support `push`, `pop`, `top`, and retrieving the minimum in O(1).

```cpp
class MinStack {
    stack<pair<int,int>> st; // {value, current_min}
public:
    void push(int x) {
        int mn = st.empty() ? x : min(x, st.top().second);
        st.emplace(x, mn);
    }
    void pop() { st.pop(); }
    int top() { return st.top().first; }
    int getMin() { return st.top().second; }
};
```

_(basic – no extra explanation needed)_

---

### 173. Binary Search Tree Iterator

**Description:** Implement an iterator over a BST that returns nodes in ascending order (`O(1)` average time).

```cpp
class BSTIterator {
    stack<TreeNode*> st;
    void pushLeft(TreeNode* node) {
        while (node) { st.push(node); node = node->left; }
    }
public:
    BSTIterator(TreeNode* root) { pushLeft(root); }

    /** @return the next smallest number */
    int next() {
        TreeNode* node = st.top(); st.pop();
        int val = node->val;
        pushLeft(node->right);
        return val;
    }

    /** @return whether we have a next smallest number */
    bool hasNext() {
        return !st.empty();
    }
};
```

**Explanation:**  
Maintain a stack of the left‐spine; on `next()`, pop, then push all left children of its right subtree.

---

### 208. Implement Trie (Prefix Tree)

**Description:** Build a Trie supporting `insert`, `search`, and `startsWith`.

```cpp
class Trie {
    struct Node {
        bool end = false;
        Node* nxt[26] = {};
    };
    Node* root;
public:
    Trie() { root = new Node(); }

    void insert(const string& word) {
        Node* cur = root;
        for (char c : word) {
            if (!cur->nxt[c-'a']) cur->nxt[c-'a'] = new Node();
            cur = cur->nxt[c-'a'];
        }
        cur->end = true;
    }

    bool search(const string& word) {
        Node* cur = root;
        for (char c : word) {
            if (!cur->nxt[c-'a']) return false;
            cur = cur->nxt[c-'a'];
        }
        return cur->end;
    }

    bool startsWith(const string& prefix) {
        Node* cur = root;
        for (char c : prefix) {
            if (!cur->nxt[c-'a']) return false;
            cur = cur->nxt[c-'a'];
        }
        return true;
    }
};
```

**Explanation:**  
Each node has 26 pointers plus a flag. Traversal/insertion is O(L) for length L.

---

### 211. Add and Search Word – Data structure design

**Description:** Extend a Trie to support wildcard `'.'` which matches any single letter in `search()`.

```cpp
class WordDictionary {
    struct Node {
        bool end = false;
        Node* nxt[26] = {};
    };
    Node* root;

    bool dfs(const string& w, int i, Node* node) {
        if (i == w.size()) return node->end;
        char c = w[i];
        if (c != '.') {
            Node* nxt = node->nxt[c-'a'];
            return nxt ? dfs(w, i+1, nxt) : false;
        }
        for (int j = 0; j < 26; ++j) {
            if (node->nxt[j] && dfs(w, i+1, node->nxt[j])) return true;
        }
        return false;
    }

public:
    WordDictionary() { root = new Node(); }

    void addWord(const string& word) {
        Node* cur = root;
        for (char c : word) {
            if (!cur->nxt[c-'a']) cur->nxt[c-'a'] = new Node();
            cur = cur->nxt[c-'a'];
        }
        cur->end = true;
    }

    bool search(const string& word) {
        return dfs(word, 0, root);
    }
};
```

_(complexity explained by the recursive wildcard DFS)_

---

### 225. Implement Stack using Queues

**Description:** Use only FIFO queues to mimic LIFO stack methods.

```cpp
class MyStack {
    queue<int> q;
public:
    void push(int x) {
        q.push(x);
        int n = q.size();
        while (--n) {
            q.push(q.front());
            q.pop();
        }
    }
    int pop() {
        int x = q.front(); q.pop();
        return x;
    }
    int top() { return q.front(); }
    bool empty() { return q.empty(); }
};
```

_(basic)_

---

### 232. Implement Queue using Stacks

**Description:** Use two stacks to mimic FIFO queue.

```cpp
class MyQueue {
    stack<int> in, out;
    void pour() {
        while (!in.empty()) {
            out.push(in.top());
            in.pop();
        }
    }
public:
    void push(int x) { in.push(x); }
    int pop() {
        if (out.empty()) pour();
        int x = out.top(); out.pop();
        return x;
    }
    int peek() {
        if (out.empty()) pour();
        return out.top();
    }
    bool empty() { return in.empty() && out.empty(); }
};
```

_(basic)_

---

### 284. Peeking Iterator

**Description:** Given an existing `Iterator`, add a `peek()` API without advancing.

```cpp
class PeekingIterator : public Iterator {
    int nextVal;
    bool hasVal;
public:
    PeekingIterator(const vector<int>& nums) : Iterator(nums) {
        hasVal = Iterator::hasNext();
        if (hasVal) nextVal = Iterator::next();
    }

    int peek() { return nextVal; }

    int next() {
        int cur = nextVal;
        hasVal = Iterator::hasNext();
        if (hasVal) nextVal = Iterator::next();
        return cur;
    }

    bool hasNext() const { return hasVal; }
};
```

_(straightforward wrapper – code is self‑explanatory)_

---

### 295. Find Median from Data Stream

**Description:** Continuously add numbers and get median in O(log n) per insertion.

```cpp
class MedianFinder {
    priority_queue<int> lo;                              // max-heap
    priority_queue<int, vector<int>, greater<int>> hi;   // min-heap

public:
    void addNum(int num) {
        lo.push(num);
        hi.push(lo.top()); lo.pop();
        if (lo.size() < hi.size()) {
            lo.push(hi.top()); hi.pop();
        }
    }

    double findMedian() {
        if (lo.size() > hi.size()) return lo.top();
        return (lo.top() + hi.top()) / 2.0;
    }
};
```

**Explanation:**  
Keep the lower half in a max‐heap and upper half in a min‐heap so medians come from one or both tops.

---

### 297. Serialize and Deserialize Binary Tree

**Description:** Convert a binary tree to a string and back.

```cpp
class Codec {
    const string SEP = ",";
    const string NIL = "#";

    void dfsSerialize(TreeNode* node, ostringstream& out) {
        if (!node) {
            out << NIL << SEP;
            return;
        }
        out << node->val << SEP;
        dfsSerialize(node->left, out);
        dfsSerialize(node->right, out);
    }

    TreeNode* dfsDeserialize(istringstream& in) {
        string token;
        if (!getline(in, token, ',')) return nullptr;
        if (token == NIL) return nullptr;
        TreeNode* node = new TreeNode(stoi(token));
        node->left  = dfsDeserialize(in);
        node->right = dfsDeserialize(in);
        return node;
    }

public:
    string serialize(TreeNode* root) {
        ostringstream out;
        dfsSerialize(root, out);
        return out.str();
    }

    TreeNode* deserialize(const string& data) {
        istringstream in(data);
        return dfsDeserialize(in);
    }
};
```

**Explanation:**  
Use preorder traversal with a null‐marker (“#”) so structure is uniquely encoded and decoded.

---

### 303. Range Sum Query – Immutable

**Description:** Given an array, answer sum‐range queries in O(1).

```cpp
class NumArray {
    vector<int> prefix;
public:
    NumArray(vector<int>& nums) {
        prefix.resize(nums.size()+1, 0);
        for (int i = 0; i < nums.size(); ++i)
            prefix[i+1] = prefix[i] + nums[i];
    }
    int sumRange(int i, int j) {
        return prefix[j+1] - prefix[i];
    }
};
```

_(basic)_

---

### 307. Range Sum Query – Mutable

**Description:** Support element updates and range‐sum queries.

```cpp
class NumArray {
    int n;
    vector<int> bit, nums;
    void bitUpdate(int i, int v) {
        for (++i; i <= n; i += i & -i)
            bit[i] += v;
    }
    int bitQuery(int i) {
        int s = 0;
        for (++i; i > 0; i -= i & -i)
            s += bit[i];
        return s;
    }
public:
    NumArray(vector<int>& arr) {
        n = arr.size();
        bit.assign(n+1, 0);
        nums = arr;
        for (int i = 0; i < n; ++i)
            bitUpdate(i, nums[i]);
    }

    void update(int i, int val) {
        int delta = val - nums[i];
        nums[i] = val;
        bitUpdate(i, delta);
    }

    int sumRange(int i, int j) {
        return bitQuery(j) - (i ? bitQuery(i-1) : 0);
    }
};
```

**Explanation:**  
Fenwick Tree (BIT) lets us update in O(log n) and get prefix sums in O(log n).

---

### 341. Flatten Nested List Iterator

**Description:** Given a nested list of integers, implement an iterator to flatten it.

```cpp
class NestedIterator {
    stack<NestedInteger> st;
    void pushList(const vector<NestedInteger>& lst) {
        for (auto it = lst.rbegin(); it != lst.rend(); ++it)
            st.push(*it);
    }

public:
    NestedIterator(vector<NestedInteger>& nestedList) {
        pushList(nestedList);
    }

    int next() {
        int v = st.top().getInteger();
        st.pop();
        return v;
    }

    bool hasNext() {
        while (!st.empty() && !st.top().isInteger()) {
            auto lst = st.top().getList();
            st.pop();
            pushList(lst);
        }
        return !st.empty();
    }
};
```

**Explanation:**  
On `hasNext()`, we unfold any nested list on top until an integer sits at the stack’s top.

---

### 352. Data Stream as Disjoint Intervals

**Description:** As numbers are added one‑by‑one, return the current set of disjoint intervals.

```cpp
class SummaryRanges {
    map<int,int> ivals; // start → end
public:
    SummaryRanges() {}

    void addNum(int v) {
        if (ivals.count(v)) return;
        int l = v, r = v;
        // merge with next
        auto it = ivals.lower_bound(v+1);
        if (it != ivals.end() && it->first == v+1) {
            r = it->second;
            ivals.erase(it);
        }
        // merge with previous
        it = ivals.upper_bound(v);
        if (it != ivals.begin()) {
            auto p = prev(it);
            if (p->second + 1 >= v) {
                l = p->first;
                r = max(r, p->second);
                ivals.erase(p);
            }
        }
        ivals[l] = r;
    }

    vector<vector<int>> getIntervals() {
        vector<vector<int>> res;
        for (auto& [a,b] : ivals)
            res.push_back({a,b});
        return res;
    }
};
```

**Explanation:**  
Use a sorted map of intervals. On insertion, check adjacent intervals to merge in O(log n).

---

### 355. Design Twitter

**Description:** Simulate posting tweets, following/unfollowing, and retrieving the 10 most recent tweets in a user’s news feed.

```cpp
class Twitter {
    int timeCnt = 0;
    unordered_map<int, vector<pair<int,int>>> tw;    // userId → [(time, tweetId), ...]
    unordered_map<int, unordered_set<int>> flw;      // userId → set of followees

public:
    Twitter() {}

    void postTweet(int userId, int tweetId) {
        tw[userId].push_back({timeCnt++, tweetId});
    }

    vector<int> getNewsFeed(int userId) {
        // include self in follow set
        flw[userId].insert(userId);
        vector<pair<int,int>> feed;
        for (int u : flw[userId]) {
            for (auto& p : tw[u]) feed.push_back(p);
        }
        sort(feed.begin(), feed.end(),
             [](auto &a, auto &b){ return a.first > b.first; });
        vector<int> res;
        for (int i = 0; i < feed.size() && i < 10; ++i)
            res.push_back(feed[i].second);
        return res;
    }

    void follow(int followerId, int followeeId) {
        flw[followerId].insert(followeeId);
    }

    void unfollow(int followerId, int followeeId) {
        if (followerId != followeeId)
            flw[followerId].erase(followeeId);
    }
};
```

**Explanation:**  
Maintain a global timestamp for ordering, store each user’s tweets in a vector, and a follow‐graph. On feed retrieval, merge all followees’ tweets and pick the 10 latest.

---

### 380. Insert Delete GetRandom O(1)

**Description:**  
Design a data structure supporting `insert(val)`, `remove(val)`, and `getRandom()` in average O(1) time.

```cpp
class RandomizedSet {
    vector<int> nums;
    unordered_map<int,int> idx; // val → index in nums
public:
    bool insert(int val) {
        if (idx.count(val)) return false;
        idx[val] = nums.size();
        nums.push_back(val);
        return true;
    }
    bool remove(int val) {
        if (!idx.count(val)) return false;
        int i = idx[val], last = nums.back();
        nums[i] = last;
        idx[last] = i;
        nums.pop_back();
        idx.erase(val);
        return true;
    }
    int getRandom() {
        return nums[rand() % nums.size()];
    }
};
```

---

### 381. Insert Delete GetRandom O(1) – Duplicates allowed

**Description:**  
Like 380 but allows duplicates. `remove(val)` removes one occurrence.

```cpp
class RandomizedCollection {
    vector<int> nums;
    unordered_map<int, unordered_set<int>> idx; // val → set of indices
public:
    bool insert(int val) {
        nums.push_back(val);
        idx[val].insert(nums.size()-1);
        return idx[val].size() == 1;
    }
    bool remove(int val) {
        if (!idx.count(val) || idx[val].empty()) return false;
        int i = *idx[val].begin(), last = nums.back();
        idx[val].erase(i);
        nums[i] = last;
        idx[last].erase(nums.size()-1);
        idx[last].insert(i);
        nums.pop_back();
        return true;
    }
    int getRandom() {
        return nums[rand() % nums.size()];
    }
};
```

---

### 384. Shuffle an Array

**Description:**  
Implement `reset()` to return original array and `shuffle()` to return a random permutation.

```cpp
class Solution {
    vector<int> orig;
public:
    Solution(vector<int>& nums): orig(nums) {}

    vector<int> reset() {
        return orig;
    }

    vector<int> shuffle() {
        vector<int> a = orig;
        for (int i = a.size()-1; i > 0; --i)
            swap(a[i], a[rand() % (i+1)]);
        return a;
    }
};
```

---

### 432. All O`1` Data Structure

**Description:**  
Design a structure with `inc(key)`, `dec(key)`, `getMaxKey()`, `getMinKey()` all in O(1).

```cpp
class AllOne {
    struct Node {
        int cnt;
        unordered_set<string> keys;
        Node *prev, *next;
        Node(int c): cnt(c), prev(this), next(this) {}
    };
    unordered_map<string, Node*> m;
    Node *head; // dummy circular list

    void insertAfter(Node* p, Node* node) {
        node->next = p->next;
        node->prev = p;
        p->next->prev = node;
        p->next = node;
    }
    void remove(Node* node) {
        node->prev->next = node->next;
        node->next->prev = node->prev;
        delete node;
    }

public:
    AllOne() { head = new Node(0); }

    void inc(string key) {
        Node* cur = m.count(key) ? m[key] : head;
        Node* nxt = cur->next;
        if (nxt == head || nxt->cnt > cur->cnt+1)
            insertAfter(cur, nxt = new Node(cur->cnt+1));
        nxt->keys.insert(key);
        m[key] = nxt;
        if (cur != head) {
            cur->keys.erase(key);
            if (cur->keys.empty()) remove(cur);
        }
    }

    void dec(string key) {
        Node* cur = m[key];
        if (!cur) return;
        if (cur->cnt == 1) {
            m.erase(key);
        } else {
            Node* prv = cur->prev;
            if (prv == head || prv->cnt < cur->cnt-1)
                insertAfter(prv, prv = new Node(cur->cnt-1));
            prv->keys.insert(key);
            m[key] = prv;
        }
        cur->keys.erase(key);
        if (cur->keys.empty()) remove(cur);
    }

    string getMaxKey() {
        return head->prev == head ? "" : *head->prev->keys.begin();
    }

    string getMinKey() {
        return head->next == head ? "" : *head->next->keys.begin();
    }
};
```

**Explanation:**  
We maintain a doubly‑linked list of count‑nodes in ascending order; each node stores all keys with that count. A hash map points each key to its node. Increment/decrement adjust counts by moving keys to neighbor nodes, creating or removing nodes as needed.

---

### 449. Serialize and Deserialize BST

**Description:**  
Serialize a BST to a string and deserialize back, leveraging BST property (no null markers).

```cpp
class Codec {
    void dfs(ostringstream& out, TreeNode* root) {
        if (!root) return;
        out << root->val << ' ';
        dfs(out, root->left);
        dfs(out, root->right);
    }
    TreeNode* build(istringstream& in, int lo, int hi) {
        int val; 
        if (!(in >> val) || val < lo || val > hi) {
            if (in) in.seekg(-to_string(val).size(), ios::cur);
            return nullptr;
        }
        TreeNode* node = new TreeNode(val);
        node->left  = build(in, lo, val);
        node->right = build(in, val, hi);
        return node;
    }
public:
    string serialize(TreeNode* root) {
        ostringstream out;
        dfs(out, root);
        return out.str();
    }
    TreeNode* deserialize(const string& data) {
        istringstream in(data);
        return build(in, INT_MIN, INT_MAX);
    }
};
```

---

### 460. LFU Cache

**Description:**  
Design a cache with `get` and `put` in O(1), evicting least‑frequently used. Tie‑break by recency.

```cpp
class LFUCache {
    int cap, minf;
    unordered_map<int, pair<int,int>> val_cnt;    // key → {value, freq}
    unordered_map<int, list<int>> freq_list;      // freq → keys in recency order
    unordered_map<int, list<int>::iterator> iter; // key → iterator in its freq list

    void touch(int key) {
        auto [v, f] = val_cnt[key];
        freq_list[f].erase(iter[key]);
        if (freq_list[f].empty() && f == minf) minf++;
        f++;
        freq_list[f].push_front(key);
        iter[key] = freq_list[f].begin();
        val_cnt[key].second = f;
    }

public:
    LFUCache(int capacity): cap(capacity), minf(0) {}

    int get(int key) {
        if (!val_cnt.count(key)) return -1;
        touch(key);
        return val_cnt[key].first;
    }

    void put(int key, int value) {
        if (cap <= 0) return;
        if (val_cnt.count(key)) {
            val_cnt[key].first = value;
            touch(key);
            return;
        }
        if (val_cnt.size() == cap) {
            int old = freq_list[minf].back();
            freq_list[minf].pop_back();
            val_cnt.erase(old);
            iter.erase(old);
        }
        val_cnt[key] = {value,1};
        minf = 1;
        freq_list[1].push_front(key);
        iter[key] = freq_list[1].begin();
    }
};
```

---

### 535. Encode and Decode TinyURL

**Description:**  
Design a TinyURL service: `encode(longUrl)` → short, `decode(shortUrl)` → long.

```cpp
class Codec {
    unordered_map<string,string> d1, d2;
    const string base = "http://tinyurl.com/";
    int id = 0;
public:
    string encode(string longUrl) {
        string s = to_string(id++);
        d1[s] = longUrl;
        return base + s;
    }
    string decode(string shortUrl) {
        string key = shortUrl.substr(base.size());
        return d1.count(key) ? d1[key] : "";
    }
};
```

---

### 622. Design Circular Queue

**Description:**  
Implement a circular queue with fixed capacity, supporting `enQueue`, `deQueue`, `Front`, `Rear`, `isEmpty`, `isFull`.

```cpp
class MyCircularQueue {
    vector<int> q;
    int head=0, tail=0, cnt=0, cap;
public:
    MyCircularQueue(int k): q(k), cap(k) {}

    bool enQueue(int v) {
        if (cnt==cap) return false;
        q[tail] = v;
        tail = (tail+1)%cap;
        cnt++;
        return true;
    }
    bool deQueue() {
        if (!cnt) return false;
        head = (head+1)%cap;
        cnt--;
        return true;
    }
    int Front() { return cnt? q[head] : -1; }
    int Rear()  { return cnt? q[(tail-1+cap)%cap] : -1; }
    bool isEmpty() { return cnt==0; }
    bool isFull()  { return cnt==cap; }
};
```

---

### 641. Design Circular Deque

**Description:**  
Extend 622 to allow insert/delete at both front and rear.

```cpp
class MyCircularDeque {
    vector<int> dq;
    int front=0, back=1, cnt=0, cap;
public:
    MyCircularDeque(int k): dq(k+2), cap(k+2) {}

    bool insertFront(int v) {
        if (cnt==cap-2) return false;
        dq[front] = v;
        front = (front-1+cap)%cap;
        cnt++;
        return true;
    }
    bool insertLast(int v) {
        if (cnt==cap-2) return false;
        dq[back] = v;
        back = (back+1)%cap;
        cnt++;
        return true;
    }
    bool deleteFront() {
        if (!cnt) return false;
        front = (front+1)%cap;
        cnt--;
        return true;
    }
    bool deleteLast() {
        if (!cnt) return false;
        back = (back-1+cap)%cap;
        cnt--;
        return true;
    }
    int getFront() { return cnt? dq[(front+1)%cap] : -1; }
    int getRear()  { return cnt? dq[(back-1+cap)%cap] : -1; }
    bool isEmpty() { return cnt==0; }
    bool isFull()  { return cnt==cap-2; }
};
```

---

### 676. Implement Magic Dictionary

**Description:**  
Build a dictionary, then search if any word in it differs by exactly one character.

```cpp
class MagicDictionary {
    unordered_set<string> dict;
public:
    void buildDict(vector<string> dict_) {
        for (auto& w : dict_) dict.insert(w);
    }
    bool search(string w) {
        for (int i = 0; i < w.size(); ++i) {
            char orig = w[i];
            for (char c = 'a'; c <= 'z'; ++c) if (c!=orig) {
                w[i] = c;
                if (dict.count(w)) return true;
            }
            w[i] = orig;
        }
        return false;
    }
};
```

---

### 677. Map Sum Pairs

**Description:**  
Implement `insert(key,val)` and `sum(prefix)` returning sum of all keys starting with prefix.

```cpp
class MapSum {
    struct Node {
        int val = 0;
        Node* nxt[26] = {};
    };
    Node* root = new Node();
public:
    void insert(string key, int v) {
        Node* p = root;
        for (char c : key) {
            if (!p->nxt[c-'a']) p->nxt[c-'a'] = new Node();
            p = p->nxt[c-'a'];
        }
        p->val = v;
    }
    int sum(string prefix) {
        Node* p = root;
        for (char c : prefix) {
            if (!p->nxt[c-'a']) return 0;
            p = p->nxt[c-'a'];
        }
        function<int(Node*)> dfs = [&](Node* u) {
            if (!u) return 0;
            int s = u->val;
            for (auto nxt : u->nxt) s += dfs(nxt);
            return s;
        };
        return dfs(p);
    }
};
```

---

### 703. Kth Largest Element in a Stream

**Description:**  
Stream numbers with `add(val)`, always return the k-th largest.

```cpp
class KthLargest {
    priority_queue<int, vector<int>, greater<int>> pq;
    int k;
public:
    KthLargest(int k_, vector<int>& a): k(k_) {
        for (int v : a) {
            pq.push(v);
            if (pq.size() > k) pq.pop();
        }
    }
    int add(int v) {
        pq.push(v);
        if (pq.size() > k) pq.pop();
        return pq.top();
    }
};
```

---

### 705. Design HashSet

**Description:**  
Implement `add`, `remove`, and `contains` with buckets.

```cpp
class MyHashSet {
    static const int B = 1000;
    vector<list<int>> buckets{B};
    int hash(int key) { return key % B; }
public:
    void add(int key) {
        if (!contains(key)) buckets[hash(key)].push_back(key);
    }
    void remove(int key) {
        auto &b = buckets[hash(key)];
        b.remove(key);
    }
    bool contains(int key) {
        for (int x : buckets[hash(key)]) if (x==key) return true;
        return false;
    }
};
```

---

### 706. Design HashMap

**Description:**  
Implement `put`, `get`, `remove`.

```cpp
class MyHashMap {
    static const int B = 1000;
    vector<list<pair<int,int>>> buckets{B};
    int hash(int key) { return key % B; }
public:
    void put(int key, int val) {
        auto &b = buckets[hash(key)];
        for (auto &p : b) if (p.first==key) {
            p.second = val; return;
        }
        b.emplace_back(key,val);
    }
    int get(int key) {
        for (auto &p : buckets[hash(key)]) if (p.first==key)
            return p.second;
        return -1;
    }
    void remove(int key) {
        auto &b = buckets[hash(key)];
        b.remove_if([&](auto &p){ return p.first==key; });
    }
};
```

---

### 707. Design Linked List

**Description:**  
Implement a singly linked list with `get`, `addAtHead`, `addAtTail`, `addAtIndex`, and `deleteAtIndex`.

```cpp
class MyLinkedList {
    struct Node { int v; Node* nxt; Node(int x):v(x),nxt(nullptr){} };
    Node *head; int sz;
public:
    MyLinkedList(): head(new Node(0)), sz(0) {} // dummy
    int get(int idx) {
        if (idx<0||idx>=sz) return -1;
        Node* p = head->nxt;
        while (idx--) p=p->nxt;
        return p->v;
    }
    void addAtHead(int v) { addAtIndex(0,v); }
    void addAtTail(int v)  { addAtIndex(sz,v); }
    void addAtIndex(int i,int v) {
        if (i<0) i=0;
        if (i>sz) return;
        Node* p = head;
        for (int k=0;k<i;++k) p=p->nxt;
        Node* u = new Node(v);
        u->nxt = p->nxt;
        p->nxt = u;
        sz++;
    }
    void deleteAtIndex(int i) {
        if (i<0||i>=sz) return;
        Node* p = head;
        for (int k=0;k<i;++k) p=p->nxt;
        Node* del = p->nxt;
        p->nxt = del->nxt;
        delete del;
        sz--;
    }
};
```

---

### 715. Range Module

**Description:**  
Track added intervals; support `addRange`, `queryRange`, `removeRange`.

```cpp
class RangeModule {
    map<int,int> iv; // start → end (exclusive)
public:
    void addRange(int l, int r) {
        if (l>=r) return;
        auto it = iv.lower_bound(l);
        if (it!=iv.begin()) {
            auto p = prev(it);
            if (p->second < l) ;
            else { l = min(l, p->first); r = max(r, p->second); it = iv.erase(p); }
        }
        while (it!=iv.end() && it->first <= r) {
            r = max(r, it->second);
            it = iv.erase(it);
        }
        iv[l] = r;
    }

    bool queryRange(int l, int r) {
        auto it = iv.upper_bound(l);
        if (it==iv.begin()) return false;
        --it;
        return it->second >= r;
    }

    void removeRange(int l, int r) {
        if (l>=r) return;
        auto it = iv.lower_bound(l);
        if (it!=iv.begin()) {
            auto p = prev(it);
            if (p->second > l) {
                int a = p->first, b = p->second;
                p->second = l;
                if (b > r) iv[r] = b;
            }
        }
        while (it!=iv.end() && it->first < r) {
            if (it->second > r) iv[r] = it->second;
            it = iv.erase(it);
        }
    }
};
```

**Explanation:**  
Maintain a sorted map of disjoint intervals. On add/remove, merge or split overlapping segments in O(log n + k).

---

### 729. My Calendar I

**Description:**  
Book non‑overlapping events: `book(start,end)` returns false if overlap.

```cpp
class MyCalendar {
    map<int,int> cal;
public:
    bool book(int s, int e) {
        auto it = cal.lower_bound(s);
        if (it!=cal.end() && it->first < e) return false;
        if (it!=cal.begin() && prev(it)->second > s) return false;
        cal[s]=e;
        return true;
    }
};
```

---

### 731. My Calendar II

**Description:**  
Allow double bookings but not triple: `book` returns false if any time would be booked 3+ times.

```cpp
class MyCalendarTwo {
    map<int,int> d;
public:
    bool book(int s, int e) {
        d[s]++; d[e]--;
        int active=0;
        for (auto& [_,v] : d) {
            active += v;
            if (active > 2) {
                d[s]--; d[e]++;
                if (!d[s]) d.erase(s);
                if (!d[e]) d.erase(e);
                return false;
            }
        }
        return true;
    }
};
```

---

### 732. My Calendar III

**Description:**  
Return maximum concurrent bookings after each `book(start,end)`.

```cpp
class MyCalendarThree {
    map<int,int> d;
    int ans=0;
public:
    int book(int s, int e) {
        d[s]++; d[e]--;
        int cur=0;
        for (auto& [_,v] : d) {
            cur += v;
            ans = max(ans, cur);
        }
        return ans;
    }
};
```

---

### 745. Prefix and Suffix Search

**Description:**  
Given a list of words, support `f(pref, suff)` returning highest index of a word with that prefix and suffix.

```cpp
class WordFilter {
    struct Trie {
        int idx;
        Trie* nxt[27];
        Trie():idx(-1){ memset(nxt,0,sizeof nxt); }
    } *root = new Trie();

    void insert(string s, int i) {
        Trie* p = root;
        for (char c : s) {
            int k = (c == '{'?26:c-'a');
            if (!p->nxt[k]) p->nxt[k] = new Trie();
            p = p->nxt[k];
            p->idx = i;
        }
    }

public:
    WordFilter(vector<string>& words) {
        for (int i = 0; i < words.size(); ++i) {
            string t = words[i] + '{' + words[i];
            for (int j = 0; j <= words[i].size(); ++j)
                insert(t.substr(j), i);
        }
    }

    int f(string pref, string suff) {
        Trie* p = root;
        for (char c : suff+'{'+pref) {
            int k = (c=='{'?26:c-'a');
            if (!p->nxt[k]) return -1;
            p = p->nxt[k];
        }
        return p->idx;
    }
};
```

**Explanation:**  
Build a combined trie of all suffix+‘{’+prefix strings so a single walk answers any `(pref,suff)` query in O(|pref|+|suff|).

---

### 855. Exam Room

**Description:**  
Students pick seats to maximize distance to nearest person; support `seat()` and `leave(p)`.

```cpp
class ExamRoom {
    int N;
    set<pair<int,int>> pq; // (distance, start), custom order
public:
    ExamRoom(int n): N(n) {
        pq.insert({N-1, 0});
    }

    int seat() {
        auto it = prev(pq.end());
        int dist = it->first, x = it->second;
        pq.erase(it);
        int pos = (x==0? 0 : (x + dist/2));
        if (pos > x)
            pq.insert({pos-x, x});
        if (x+dist < N-1)
            pq.insert({x+dist - pos, pos+1});
        return pos;
    }

    void leave(int p) {
        int l = p, r = p;
        for (auto it = pq.lower_bound({0,p}); it!=pq.end(); ++it) {
            if (it->second == p+1) { r = p + it->first; pq.erase(it); break; }
        }
        for (auto it = prev(pq.upper_bound({0,p})); ; --it) {
            int start = it->second, d = it->first;
            if (start + d == p) { l = start; pq.erase(it); break; }
            if (it==pq.begin()) break;
        }
        pq.insert({r-l, l});
    }
};
```

---

### 895. Maximum Frequency Stack

**Description:**  
Implement a stack-like structure with `push(x)` and `pop()` that removes the most frequent element (and most recent among ties).

```cpp
class FreqStack {
    unordered_map<int,int> freq;
    unordered_map<int, stack<int>> group;
    int maxf = 0;
public:
    void push(int x) {
        int f = ++freq[x];
        maxf = max(maxf, f);
        group[f].push(x);
    }
    int pop() {
        int x = group[maxf].top();
        group[maxf].pop();
        if (group[maxf].empty()) maxf--;
        if (--freq[x] == 0) freq.erase(x);
        return x;
    }
};
```

---

### 900. RLE Iterator

**Description:**  
Given run‑length encoding, support `next(n)` returning the next element after consuming n entries, or –1 if exhausted.

```cpp
class RLEIterator {
    vector<pair<int,int>> A;
    int i = 0;
    long long used = 0;
public:
    RLEIterator(vector<int>& arr) {
        for (int j = 0; j+1 < arr.size(); j += 2)
            A.emplace_back(arr[j+1], arr[j]);
    }
    int next(int n) {
        while (i < A.size() && n > A[i].first - used) {
            n -= (A[i].first - used);
            used = 0;
            i++;
        }
        if (i == A.size()) return -1;
        used += n;
        return A[i].second;
    }
};
```

---

### 901. Online Stock Span

**Description:**  
Given daily stock prices, `next(price)` returns span (# of consecutive days ≤ today).

```cpp
class StockSpanner {
    stack<pair<int,int>> st; // {price, span}
public:
    int next(int p) {
        int s = 1;
        while (!st.empty() && st.top().first <= p) {
            s += st.top().second;
            st.pop();
        }
        st.push({p,s});
        return s;
    }
};
```

---

### 911. Online Election

**Description:**  
Given arrays `persons` and `times`, support `q(t)` returning the person leading at time t.

```cpp
class TopVotedCandidate {
    vector<int> lead, T;
public:
    TopVotedCandidate(vector<int>& persons, vector<int>& times) {
        unordered_map<int,int> cnt;
        int curLead = -1, curMax = 0;
        for (int i = 0; i < persons.size(); ++i) {
            int p = ++cnt[persons[i]];
            if (p >= curMax) {
                curLead = persons[i];
                curMax = p;
            }
            lead.push_back(curLead);
        }
        T = times;
    }

    int q(int t) {
        int i = upper_bound(T.begin(), T.end(), t) - T.begin() - 1;
        return i >= 0 ? lead[i] : -1;
    }
};
```

---

### 919. Complete Binary Tree Inserter

**Description:**  
Initialize with a complete binary tree `root`. `insert(val)` adds a node to keep it complete, returning its parent’s value. `get_root()` returns the root.

```cpp
class CBTInserter {
    TreeNode* root;
    queue<TreeNode*> q;
public:
    CBTInserter(TreeNode* r): root(r) {
        queue<TreeNode*> bfs;
        bfs.push(r);
        while (!bfs.empty()) {
            auto n = bfs.front(); bfs.pop();
            if (!n->left || !n->right) q.push(n);
            if (n->left) bfs.push(n->left);
            if (n->right) bfs.push(n->right);
        }
    }

    int insert(int v) {
        TreeNode* p = q.front();
        TreeNode* node = new TreeNode(v);
        if (!p->left) p->left = node;
        else { p->right = node; q.pop(); }
        q.push(node);
        return p->val;
    }

    TreeNode* get_root() {
        return root;
    }
};
```

---
### 933. Number of Recent Calls

**Description:** Implement `RecentCounter` with `ping(t)` returning the count of calls in the past 3000ms.

```cpp
class RecentCounter {
    queue<int> q;
public:
    RecentCounter() {}
    int ping(int t) {
        q.push(t);
        while (q.front() < t - 3000) q.pop();
        return q.size();
    }
};
```

_(basic)_

---

### 981. Time Based Key-Value Store

**Description:** Support `set(key,value,timestamp)` and `get(key,timestamp)` returning the value with the largest timestamp ≤ given.

```cpp
class TimeMap {
    unordered_map<string, vector<pair<int,string>>> m;
public:
    TimeMap() {}
    void set(const string& key, const string& value, int t) {
        m[key].emplace_back(t, value);
    }
    string get(const string& key, int t) {
        auto &v = m[key];
        int l = 0, r = v.size() - 1, idx = -1;
        while (l <= r) {
            int mid = (l + r) / 2;
            if (v[mid].first <= t) idx = mid, l = mid + 1;
            else r = mid - 1;
        }
        return idx == -1 ? "" : v[idx].second;
    }
};
```

_(binary search over time‑sorted vector)_

---

### 1032. Stream of Characters

**Description:** Implement `StreamChecker` that on each `query(letter)` returns true if any suffix of the queried letters matches a word in the given list.

```cpp
class StreamChecker {
    struct Node { 
        Node* nxt[26] = {};
        bool end = false;
    };
    Node* root;
    string buf;
    int maxLen = 0;

    void insert(const string& w) {
        Node* p = root;
        for (int i = w.size()-1; i >= 0; --i) {
            int c = w[i] - 'a';
            if (!p->nxt[c]) p->nxt[c] = new Node();
            p = p->nxt[c];
        }
        p->end = true;
    }

public:
    StreamChecker(vector<string>& words) {
        root = new Node();
        for (auto& w : words) {
            insert(w);
            maxLen = max(maxLen, (int)w.size());
        }
    }

    bool query(char letter) {
        buf.push_back(letter);
        if (buf.size() > maxLen) buf.erase(buf.begin());
        Node* p = root;
        for (int i = buf.size()-1; i >= 0; --i) {
            int c = buf[i] - 'a';
            if (!p->nxt[c]) return false;
            p = p->nxt[c];
            if (p->end) return true;
        }
        return false;
    }
};
```

**Explanation:**  
Store all words in a trie of their reversals. Keep a rolling buffer up to `maxLen`; on each query, walk the buffer backwards in the trie to check for any matching suffix.

---

### 1146. Snapshot Array

**Description:** Design `SnapshotArray` supporting `set(i,val)`, `snap()` → snap_id, and `get(i,snap_id)` retrieving the value at that snapshot.

```cpp
class SnapshotArray {
    int snap_id = 0;
    vector<map<int,int>> A;
public:
    SnapshotArray(int length): A(length) {
        for (auto& m : A) m[0] = 0;
    }

    void set(int idx, int val) {
        A[idx][snap_id] = val;
    }

    int snap() {
        return snap_id++;
    }

    int get(int idx, int s) {
        auto it = A[idx].upper_bound(s);
        return prev(it)->second;
    }
};
```

**Explanation:**  
For each index keep a map of `{snap_id → value}`. `set` writes at current `snap_id`; `get` finds the latest entry ≤ requested snapshot via `upper_bound`.

---

### 1157. Online Majority Element In Subarray

**Description:** Given an array, answer queries `(left,right,target)` asking if `target` is majority (occurs > half-length) in that subarray.

```cpp
class MajorityChecker {
    unordered_map<int, vector<int>> pos;
public:
    MajorityChecker(vector<int>& arr) {
        for (int i = 0; i < arr.size(); ++i)
            pos[arr[i]].push_back(i);
    }

    int query(int left, int right, int target) {
        auto& v = pos[target];
        int cnt = upper_bound(v.begin(), v.end(), right)
                - lower_bound(v.begin(), v.end(), left);
        return cnt > (right - left + 1) / 2;
    }
};
```

_(stores positions and counts with binary search; self‑explanatory)_

---

### 1172. Dinner Plate Stacks

**Description:** Implement stacks of fixed capacity; `push(val)` goes to leftmost non‑full stack, `pop()` from rightmost non‑empty, and `popAtStack(i)` from that stack.

```cpp
class DinnerPlates {
    int cap;
    vector<stack<int>> S;
    set<int> nonFull, nonEmpty;
public:
    DinnerPlates(int capacity): cap(capacity) {}

    void push(int val) {
        if (nonFull.empty()) {
            S.emplace_back();
            nonFull.insert(S.size()-1);
        }
        int i = *nonFull.begin();
        S[i].push(val);
        nonEmpty.insert(i);
        if (S[i].size() == cap) nonFull.erase(i);
    }

    int pop() {
        if (nonEmpty.empty()) return -1;
        int i = *prev(nonEmpty.end());
        int v = popAtStack(i);
        return v;
    }

    int popAtStack(int index) {
        if (index < 0 || index >= S.size() || S[index].empty()) return -1;
        int v = S[index].top(); S[index].pop();
        nonFull.insert(index);
        if (S[index].empty()) nonEmpty.erase(index);
        return v;
    }
};
```

**Explanation:**  
Track two sets: indices of non‑full stacks (for `push`) and non‑empty stacks (for `pop`). Create new stacks on demand.

---

### 1261. Find Elements in a Contaminated Binary Tree

**Description:** Tree nodes initially have value –1; the root is set to 0, and for any node `v`, left child → `2*v+1`, right → `2*v+2`. Support `find(target)`.

```cpp
class FindElements {
    unordered_set<int> S;
    void dfs(TreeNode* node, int v) {
        if (!node) return;
        node->val = v;
        S.insert(v);
        dfs(node->left, 2*v+1);
        dfs(node->right, 2*v+2);
    }
public:
    FindElements(TreeNode* root) {
        dfs(root, 0);
    }
    bool find(int target) {
        return S.count(target);
    }
};
```

_(basic DFS recovery and hash‑set lookup)_

---

### 1286. Iterator for Combination

**Description:** Create an iterator over all combinations of a given string’s characters of length `k`, in lex order.

```cpp
class CombinationIterator {
    string chars;
    string curr;
    int n, k;
public:
    CombinationIterator(string letters, int k_): chars(letters), n(letters.size()), k(k_) {
        curr = letters.substr(0, k);
    }

    string next() {
        string ans = curr;
        int i = k - 1;
        while (i >= 0 && curr[i] == chars[n - k + i]) --i;
        if (i >= 0) {
            int pos = chars.find(curr[i]);
            curr[i] = chars[pos+1];
            for (int j = i+1; j < k; ++j)
                curr[j] = chars[pos+1 + (j - i)];
        } else curr = "";
        return ans;
    }

    bool hasNext() {
        return !curr.empty();
    }
};
```

_(pre‑initialized to first combo; next() uses “next permutation” logic on indices)_

---

### 1348. Tweet Counts Per Frequency

**Description:** Log tweets at timestamps; support `getTweetCountsPerFrequency(freq,name,start,end)` returning counts per interval of length `freq`.

```cpp
class TweetCounts {
    unordered_map<string, vector<int>> m;
public:
    void recordTweet(string name, int t) {
        m[name].push_back(t);
    }

    vector<int> getTweetCountsPerFrequency(string freq, string name, int s, int e) {
        int delta = (freq=="minute"?60:(freq=="hour"?3600:86400));
        int sz = ((e - s) / delta) + 1;
        vector<int> ans(sz);
        auto& v = m[name];
        for (int t : v) {
            if (t < s || t > e) continue;
            ans[(t - s) / delta]++;
        }
        return ans;
    }
};
```

_(basic bucketing by interval index)_

---

### 1352. Product of the Last K Numbers

**Description:** Stream of numbers, support `add(num)` and `getProduct(k)` of last k numbers.

```cpp
class ProductOfNumbers {
    vector<long long> p{1};
public:
    void add(int num) {
        if (num == 0) p = {1};
        else p.push_back(p.back() * num);
    }
    int getProduct(int k) {
        if (k >= p.size()) return 0;
        return p.back() / p[p.size() - 1 - k];
    }
};
```

**Explanation:**  
Maintain prefix products; on zero reset. `getProduct(k)` divides to get the product of the last k.

---

### 1357. Apply Discount Every n Orders

**Description:** For every `n`th order of a customer, apply a given discount rate.

```cpp
class Cashier {
    int n, cnt = 0, discount;
    vector<int> prices;
public:
    Cashier(int n_, int discount_, vector<int>& products, vector<int>& prices_)
      : n(n_), discount(discount_), prices(prices_) {}

    double getBill(vector<int>& product, vector<int>& amount) {
        cnt = (cnt + 1) % n;
        double total = 0;
        for (int i = 0; i < product.size(); ++i)
            total += prices[product[i]-1] * amount[i];
        if (cnt == 0) total *= (100 - discount) / 100.0;
        return total;
    }
};
```

_(basic counter and discount application)_

---

### 1381. Design a Stack With Increment Operation

**Description:** Stack with `push(x)`, `pop()`, and `increment(k,val)` adding `val` to bottom `k` elements.

```cpp
class CustomStack {
    int cap;
    vector<int> st, inc;
public:
    CustomStack(int maxSize): cap(maxSize) {}

    void push(int x) {
        if (st.size() < cap) {
            st.push_back(x);
            inc.push_back(0);
        }
    }

    int pop() {
        if (st.empty()) return -1;
        int i = st.size() - 1;
        int res = st[i] + inc[i];
        if (i > 0) inc[i-1] += inc[i];
        st.pop_back();
        inc.pop_back();
        return res;
    }

    void increment(int k, int val) {
        int i = min(k, (int)st.size()) - 1;
        if (i >= 0) inc[i] += val;
    }
};
```

**Explanation:**  
Use an auxiliary `inc` array: `inc[i]` stores pending increment for all elements ≤ `i`. On pop, propagate `inc` downward.

---

### 1396. Design Underground System

**Description:** Track check‑in and check‑out events to return average travel time between two stations.

```cpp
class UndergroundSystem {
    unordered_map<int, pair<string,int>> in;
    unordered_map<string, pair<long long,int>> sumCnt;
public:
    void checkIn(int id, string s, int t) {
        in[id] = {s, t};
    }

    void checkOut(int id, string s2, int t2) {
        auto [s1, t1] = in[id];
        string key = s1 + "->" + s2;
        auto &sc = sumCnt[key];
        sc.first += t2 - t1;
        sc.second += 1;
    }

    double getAverageTime(string s, string s2) {
        auto &sc = sumCnt[s + "->" + s2];
        return (double)sc.first / sc.second;
    }
};
```

_(basic hash maps for ongoing trips and aggregates)_

---

### 1472. Design Browser History

**Description:** Maintain browser history with `visit(url)`, `back(steps)`, `forward(steps)`.

```cpp
class BrowserHistory {
    vector<string> h;
    int idx = 0;
public:
    BrowserHistory(string homepage) {
        h.push_back(homepage);
    }
    void visit(string url) {
        h.resize(idx+1);
        h.push_back(url);
        idx = h.size()-1;
    }
    string back(int steps) {
        idx = max(0, idx - steps);
        return h[idx];
    }
    string forward(int steps) {
        idx = min((int)h.size()-1, idx + steps);
        return h[idx];
    }
};
```

_(basic vector + index pointer)_

---

### 1476. Subrectangle Queries

**Description:** Initialize with a 2D matrix; support `updateSubrectangle(r1,c1,r2,c2,val)` and `getValue(r,c)`.

```cpp
class SubrectangleQueries {
    vector<vector<int>> rect;
public:
    SubrectangleQueries(vector<vector<int>>& rectangle) : rect(rectangle) {}

    void updateSubrectangle(int r1, int c1, int r2, int c2, int val) {
        for (int i = r1; i <= r2; ++i)
            for (int j = c1; j <= c2; ++j)
                rect[i][j] = val;
    }

    int getValue(int r, int c) {
        return rect[r][c];
    }
};
```

_(basic direct updates)_

---

### 1483. Kth Ancestor of a Tree Node

**Description:** Preprocess a rooted tree to answer `getKthAncestor(node,k)` in O(log n).

```cpp
class TreeAncestor {
    vector<vector<int>> up;
    int LOG;
public:
    TreeAncestor(int n, vector<int>& parent) {
        LOG = 1;
        while ((1<<LOG) <= n) ++LOG;
        up.assign(LOG, vector<int>(n));
        up[0] = parent;
        for (int i = 1; i < LOG; ++i)
            for (int v = 0; v < n; ++v)
                up[i][v] = up[i-1][v] < 0 ? -1 : up[i-1][ up[i-1][v] ];
    }

    int getKthAncestor(int node, int k) {
        for (int i = 0; i < LOG && node >= 0; ++i)
            if (k & (1<<i))
                node = up[i][node];
        return node;
    }
};
```

_(binary lifting on parent pointers)_

---

### 1600. Throne Inheritance

**Description:** Model royal inheritance with births and deaths; return current order of living heirs.

```cpp
class ThroneInheritance {
    string king;
    unordered_map<string, vector<string>> children;
    unordered_set<string> dead;
public:
    ThroneInheritance(string kingName): king(kingName) {}

    void birth(string parent, string child) {
        children[parent].push_back(child);
    }

    void death(string name) {
        dead.insert(name);
    }

    void dfs(const string& name, vector<string>& order) {
        if (!dead.count(name)) order.push_back(name);
        for (auto& c : children[name])
            dfs(c, order);
    }

    vector<string> getInheritanceOrder() {
        vector<string> order;
        dfs(king, order);
        return order;
    }
};
```

_(basic DFS over birth tree, skipping dead nodes)_

---

### 1603. Design Parking System

**Description:** Parking lot with three spots (big, medium, small); `addCar(carType)` returns whether space is available.

```cpp
class ParkingSystem {
    int cap[4];
public:
    ParkingSystem(int b, int m, int s) {
        cap[1]=b; cap[2]=m; cap[3]=s;
    }
    bool addCar(int t) {
        return cap[t]-- > 0;
    }
};
```

_(basic counters)_

---

### 1622. Fancy Sequence

**Description:** Support a sequence with `append(val)`, `addAll(inc)`, `multAll(m)`, and `getIndex(idx)`, all mod 1e9+7.

```cpp
class Fancy {
    static const int MOD = 1e9+7;
    vector<pair<long long,long long>> A; // {a_i, b_i} so val = a_i*x + b_i
    long long a=1, b=0;
public:
    Fancy() {}

    void append(int val) {
        A.emplace_back(val, ((val - b) * modInv(a)) % MOD);
    }

    void addAll(int inc) {
        b = (b + inc) % MOD;
    }

    void multAll(int m) {
        a = (a * m) % MOD;
        b = (b * m) % MOD;
    }

    int getIndex(int i) {
        if (i >= A.size()) return -1;
        auto [ai, bi] = A[i];
        return (ai * a + b) % MOD;
    }

private:
    long long modPow(long long x, long long p=MOD-2) {
        long long r=1;
        while (p) {
            if (p&1) r = r*x % MOD;
            x = x*x % MOD;
            p >>= 1;
        }
        return r;
    }
    long long modInv(long long x) { return modPow(x); }
};
```

**Explanation:**  
Maintain affine transform `f(x)=a*x+b`. Each append stores the inverse‑transformed original so that later `get` applies current `(a,b)` in O(1).

---

### 1656. Design an Ordered Stream

**Description:** Stream of `(id,value)` pairs; `insert(id,value)` returns the longest consecutive stream from the current pointer.

```cpp
class OrderedStream {
    vector<string> A;
    int ptr = 1, n;
public:
    OrderedStream(int n_): A(n_+1), n(n_) {}

    vector<string> insert(int id, string val) {
        A[id] = val;
        vector<string> res;
        while (ptr <= n && !A[ptr].empty()) {
            res.push_back(A[ptr++]);
        }
        return res;
    }
};
```

_(basic array + pointer)_

---

### 1670. Design Front Middle Back Queue

**Description:** Deque supporting `pushFront`, `pushMiddle`, `pushBack`, and corresponding pops, all in O(1).

```cpp
class FrontMiddleBackQueue {
    deque<int> left, right;
    void rebalance() {
        if (left.size() > right.size()+1) {
            right.push_front(left.back());
            left.pop_back();
        } else if (right.size() > left.size()) {
            left.push_back(right.front());
            right.pop_front();
        }
    }
public:
    FrontMiddleBackQueue() {}

    void pushFront(int v) {
        left.push_front(v);
        rebalance();
    }

    void pushMiddle(int v) {
        if (left.size() > right.size())
            right.push_front(left.back()), left.pop_back();
        left.push_back(v);
    }

    void pushBack(int v) {
        right.push_back(v);
        rebalance();
    }

    int popFront() {
        if (left.empty() && right.empty()) return -1;
        int v = left.empty() ? right.front() : left.front();
        if (left.empty()) right.pop_front();
        else left.pop_front();
        rebalance();
        return v;
    }

    int popMiddle() {
        if (left.empty() && right.empty()) return -1;
        int v = left.size() == right.size() ? left.back() : left.back();
        left.pop_back();
        rebalance();
        return v;
    }

    int popBack() {
        if (left.empty() && right.empty()) return -1;
        int v = right.empty() ? left.back() : right.back();
        if (right.empty()) left.pop_back();
        else right.pop_back();
        rebalance();
        return v;
    }
};
```

**Explanation:**  
Maintain two deques (`left` and `right`) of nearly equal size. The middle is the back of `left`.

---

### 1797. Design Authentication Manager

**Description:** Implement token manager with `generate(tokenId,currentTime)`, `renew(tokenId,currentTime)`, and `countUnexpiredTokens(currentTime)` given a TTL.

```cpp
class AuthenticationManager {
    int ttl;
    unordered_map<string,int> expiry;
public:
    AuthenticationManager(int timeToLive): ttl(timeToLive) {}

    void generate(string tokenId, int t) {
        expiry[tokenId] = t + ttl;
    }

    void renew(string tokenId, int t) {
        if (expiry.count(tokenId) && expiry[tokenId] > t)
            expiry[tokenId] = t + ttl;
    }

    int countUnexpiredTokens(int t) {
        int cnt = 0;
        for (auto& [id, exp] : expiry)
            if (exp > t) ++cnt;
        return cnt;
    }
};
```

_(basic hash‑map of expiration times; linear scan on count)_

---
