

## Operating Systems Interview Prep

### Index

1. [Process & Thread Management](#1-process--thread-management)
    
2. [CPU Scheduling](#2-cpu-scheduling)
    
3. [Memory Management](#3-memory-management)
    
4. [Process Synchronization & Concurrency](#4-process-synchronization--concurrency)
    
5. [Deadlocks](#5-deadlocks)
    
6. [File Systems](#6-file-systems)
    
7. [I/O & Device Management](#7-io--device-management)
    
8. [OS Structures & System Calls](#8-os-structures--system-calls)
    
9. [Security & Protection](#9-security--protection)
    
10. [Virtualization & Distributed OS](#10-virtualization--distributed-os)
    

---

## 1. Process & Thread Management

### What Is a Process?

- **Definition**: An instance of a program in execution, with its own address space, resources, and execution context.
    
- **Process Control Block (PCB)**: OS data structure containing PID, state, CPU registers, memory pointers, I/O info, accounting.
    
- **Process States**:
    
    1. **New** → 2. **Ready** → 3. **Running** → 4. **Blocked** → back to **Ready** → **Terminated**.
        
- **Process Creation** (Unix example):
    
    ```c
    pid_t pid = fork();
    if (pid == 0) {
      // child process
      execve(...);
    } else {
      // parent continues
    }
    ```
    

### Threads vs. Processes

|Aspect|Process|Thread|
|---|---|---|
|Address Space|Own memory space|Shared within process|
|Overhead|High (context switch, memory)|Low|
|Communication|IPC (pipes, sockets, shared mem)|Direct via shared vars|

- **Types of Threads**:
    
    - **Kernel Threads** (managed by OS).
        
    - **User Threads** (managed by a runtime library).
        
    - **Hybrid** (both).
        

### Context Switch

- **What Happens**: Save CPU registers to PCB of old process; load registers from PCB of next process.
    
- **Cost**: Hundreds to thousands of CPU cycles.
    

### Common Interview Queries

- “Describe the contents of a PCB.”
    
- “Explain the difference between preemptive and non‑preemptive context switch.”
    
- “How does `fork()` differ from `exec()` in Unix?”
    
- “When would you use threads over processes?”
    
- **Whiteboard Exercise**: Draw the process state diagram and show transitions on I/O request and completion.
    

---

## 2. CPU Scheduling

### Scheduling Criteria

- **CPU Utilization**: Keep CPU busy ≥ 90%.
    
- **Throughput**: Processes completed per unit time.
    
- **Turnaround Time**: Finish − Submit time.
    
- **Waiting Time**: Time spent in the ready queue.
    
- **Response Time**: Time from submission to first response.
    

### Common Scheduling Algorithms

1. **FCFS (First‑Come, First‑Served)**
    
    - Simple queue.
        
    - **Drawback**: Convoy effect; long jobs delay short ones.
        
2. **SJF (Shortest Job First)**
    
    - Chooses process with smallest next CPU burst.
        
    - **Optimal** for average waiting time but needs burst prediction.
        
3. **Priority Scheduling**
    
    - Each process has priority; lower priority may starve.
        
    - **Solution**: Aging (increase priority over time).
        
4. **Round Robin (RR)**
    
    - Time quantum **q**; processes cycle through ready queue.
        
    - **q** too small → too many context switches; too large → approximates FCFS.
        
5. **Multilevel Queue**
    
    - Separate queues for foreground (RR) vs. background (FCFS).
        
6. **Multilevel Feedback Queue**
    
    - Processes can move between queues based on their behavior (I/O‑bound vs. CPU‑bound).
        

### Example: RR Calculation

Given processes P1–P3 with burst times (5, 3, 8) and quantum = 2:

|Time|Running|Ready Queue|
|---|---|---|
|0–2|P1|P2, P3|
|2–4|P2|P3, P1(3 remains)|
|4–6|P3|P1(3), P2(1)|
|…|…|…|

Compute waiting and turnaround times from the Gantt chart.

### Common Interview Queries

- “How do you compute average waiting time for these algorithms?”
    
- “Explain how aging prevents starvation in priority scheduling.”
    
- “What’s the effect of quantum size in Round Robin?”
    
- “Design a hybrid scheduler for an interactive system.”
    

---


---

### 1. CPU Scheduling: Round‑Robin Metrics

**Q:** Given four processes with arrival time & CPU burst (in ms):

```
P1: (AT=0, BT=5)  
P2: (AT=1, BT=3)  
P3: (AT=2, BT=8)  
P4: (AT=3, BT=6)
```

Using Round‑Robin with time quantum = 4 ms, draw the Gantt chart, then compute each process’s **waiting time** and **turnaround time**, and the **average** of each.

1. **Gantt Chart Construction**
    
    - At t=0: only P1→ runs 4 ms → [0–4]
        
    - t=4: P1 has 1 ms remaining; ready queue = {P2, P3, P4, P1}
        
    - t=4–7: P2 (entire 3 ms) → completes at 7
        
    - t=7–11: P3 runs 4 ms (4 ms remaining)
        
    - t=11–15: P4 runs 4 ms (2 ms remaining)
        
    - t=15–16: P1 runs final 1 ms → completes at 16
        
    - t=16–20: P3 runs final 4 ms → completes at 20
        
    - t=20–22: P4 runs final 2 ms → completes at 22
        
    
    **Gantt**:
    
    ```
    |P1|P2|P3|P4|P1|P3|P4|
     0  4  7 11 15 16 20 22
    ```
    
2. **Turnaround Time (TAT = completion − arrival)**
    
    - TAT₁ = 16 − 0 = 16
        
    - TAT₂ = 7 − 1 = 6
        
    - TAT₃ = 20 − 2 = 18
        
    - TAT₄ = 22 − 3 = 19
        
3. **Waiting Time (WT = TAT − burst)**
    
    - WT₁ = 16 − 5 = 11
        
    - WT₂ = 6 − 3 = 3
        
    - WT₃ = 18 − 8 = 10
        
    - WT₄ = 19 − 6 = 13
        
4. **Averages**
    
    - **Avg TAT** = (16 + 6 + 18 + 19) / 4 = 59 / 4 = 14.75 ms
        
    - **Avg WT** = (11 + 3 + 10 + 13) / 4 = 37 / 4 = 9.25 ms
        

> **Why it matters:** Demonstrates handling of time‑slice preemption and how RR balances fairness vs. context‑switch overhead.

---

### 2. Page Replacement: FIFO, LRU & OPT

**Q:** For the reference string

```
7, 0, 1, 2, 0, 3, 0, 4, 2, 3, 0, 3, 2
```

and 3 page frames, calculate the total **page faults** under:

1. **FIFO**
    
2. **LRU**
    
3. **OPT** (Optimal).
    

|Ref|Frame State (FIFO)|Fault?|Frame State (LRU)|Fault?|Frame State (OPT)|Fault?|
|---|---|---|---|---|---|---|
|7|[7,–,–]|Y|[7,–,–]|Y|[7,–,–]|Y|
|0|[7,0,–]|Y|[7,0,–]|Y|[7,0,–]|Y|
|1|[7,0,1]|Y|[7,0,1]|Y|[7,0,1]|Y|
|2|[2,0,1] (evict 7)|Y|[2,0,1] (evict 7)|Y|[2,0,1] (evict 7)|Y|
|0|[2,0,1]|N|[2,0,1]|N|[2,0,1]|N|
|3|[3,0,1] (evict 2)|Y|[3,0,1] (evict 2)|Y|[3,0,1] (evict 2)|Y|
|0|[3,0,1]|N|[3,0,1]|N|[3,0,1]|N|
|4|[4,0,1] (evict 3)|Y|[4,0,1] (evict 3)|Y|[4,0,1] (evict 1)|Y|
|2|[4,2,1] (evict 0)|Y|[4,2,1] (evict 0)|Y|[4,2,1] (evict 0)|Y|
|3|[4,2,3] (evict 1)|Y|[4,2,3] (evict 1)|Y|[4,2,3] (evict 4)|Y|
|0|[0,2,3] (evict 4)|Y|[0,2,3] (evict 4)|Y|[0,2,3] (evict 2)|Y|
|3|[0,2,3]|N|[0,2,3]|N|[0,2,3]|N|
|2|[0,2,3]|N|[0,2,3]|N|[0,2,3]|N|

- **FIFO faults**: 9
    
- **LRU faults**: 9
    
- **OPT faults**: 8
    

> **Why it matters:** Shows trade‑offs between implementability (FIFO), recency awareness (LRU), and theoretical optimality (OPT).

---

### 3. Producer–Consumer with Semaphores

**Q:** Write pseudocode using semaphores to solve the bounded buffer Producer–Consumer problem (one producer, one consumer, buffer size = N).

```c
semaphore empty = N;        // counts empty slots
semaphore full  = 0;        // counts filled slots
semaphore mutex = 1;        // binary mutual exclusion
buffer B[N];
int in = 0, out = 0;

// Producer
void produce(item x) {
  empty.wait();             // wait for empty slot
  mutex.wait();             // enter critical section
    B[in] = x;
    in = (in + 1) % N;
  mutex.signal();           // exit critical section
  full.signal();            // increment full count
}

// Consumer
item consume() {
  full.wait();              // wait for filled slot
  mutex.wait();             // enter critical section
    x = B[out];
    out = (out + 1) % N;
  mutex.signal();           // exit critical section
  empty.signal();           // increment empty count
  return x;
}
```

**Explanation:**

- **`empty`** ensures the producer blocks when buffer is full.
    
- **`full`** ensures the consumer blocks when buffer is empty.
    
- **`mutex`** provides mutual exclusion on buffer indices.
    

> **Why it matters:** Core concurrency problem—tests understanding of semaphores, mutual exclusion, and deadlock avoidance.

---

### 4. Readers–Writers (Second Problem)

**Q:** Provide a solution to the “second” Readers–Writers problem (no writer should starve).

```c
semaphore rw_mutex = 1;       // for writers
semaphore mutex   = 1;       // for reader count
semaphore queue    = 1;       // FIFO queue for fairness
int reader_count = 0;

// Reader
void reader() {
  queue.wait();               // join the queue
  mutex.wait();               // protect reader_count
    if (++reader_count == 1)
      rw_mutex.wait();        // first reader locks out writers
  mutex.signal();
  queue.signal();

  // --- reading happens ---
  
  mutex.wait();
    if (--reader_count == 0)
      rw_mutex.signal();      // last reader releases writers
  mutex.signal();
}

// Writer
void writer() {
  queue.wait();               // join the queue
  rw_mutex.wait();            // exclusive access
  queue.signal();

  // --- writing happens ---
  
  rw_mutex.signal();
}
```

**Why it works:**

- **`queue`** semaphore enforces arrival order, preventing writer starvation.
    
- First/last reader logic allows multiple readers concurrently while blocking writers.
    

> **Why it matters:** Demonstrates handling fairness and starvation in concurrent access patterns.

---

### 5. Deadlock Detection: Banker's Algorithm

**Q:** Given 3 resource types A, B, C and 5 processes with:

- **Available** = [3, 3, 2]
    
- **Max** and **Allocation** matrices:
    

|Proc|Max|Alloc|
|---|---|---|
|P0|[7,5,3]|[0,1,0]|
|P1|[3,2,2]|[2,0,0]|
|P2|[9,0,2]|[3,0,2]|
|P3|[2,2,2]|[2,1,1]|
|P4|[4,3,3]|[0,0,2]|

Determine if the system is in a **safe state**, and if so, give one **safe sequence**.

1. **Compute Need = Max − Alloc**:
    

|Proc|Need|
|---|---|
|P0|[7,4,3]|
|P1|[1,2,2]|
|P2|[6,0,0]|
|P3|[0,1,1]|
|P4|[4,3,1]|

2. **Find safe sequence via “work & finish”**
    
    - **Work** = Available = [3,3,2]
        
    - Look for a process whose Need ≤ Work:
        
        - P1: [1,2,2] ≤ [3,3,2]? Yes → Work += Alloc₁ = [3+2,3+0,2+0] = [5,3,2]; Finish₁= true
            
        - P3: [0,1,1] ≤ [5,3,2]? Yes → Work += Alloc₃ = [5+2,3+1,2+1] = [7,4,3]; Finish₃= true
            
        - P0: [7,4,3] ≤ [7,4,3]? Yes → Work += Alloc₀ = [7+0,4+1,3+0] = [7,5,3]; Finish₀= true
            
        - P2: [6,0,0] ≤ [7,5,3]? Yes → Work += Alloc₂ = [7+3,5+0,3+2] = [10,5,5]; Finish₂= true
            
        - P4: [4,3,1] ≤ [10,5,5]? Yes → Work += Alloc₄ = [10+0,5+0,5+2] = [10,5,7]; Finish₄= true
            
3. **Safe Sequence**: ⟨P1, P3, P0, P2, P4⟩ (one valid example)
    

> **Why it matters:** Shows ability to detect safe states and prevent deadlocks in resource‑allocation systems.

---

### 6. Two‑Level Paging Address Translation

**Q:** A system uses 32‑bit virtual addresses, 4 KB pages, and a two‑level page table. Both levels use 10 bits to index (so 12 bit offset). Given a virtual address `0xCAFEBABE`, show how you’d extract:

1. Level‑1 index
    
2. Level‑2 index
    
3. Offset  
    And, assuming the L1 entry for that index points to a second‑level table at physical frame `0x00123`, and the L2 entry for the L2 index contains physical frame `0x00456`, compute the final physical address.
    

4. **Break the 32‑bit VA**
    
    - **Offset** = low 12 bits = `0xABE` (`0xCAFEBABE & 0xFFF = 0xABE`)
        
    - **L2 index** = next 10 bits = bits [12..21]
        
        ```
        0xCAFEBABE >> 12 = 0xCAFEB (20 bits)  
        0xCAFEB & 0x3FF = 0x2EB  
        ```
        
    - **L1 index** = bits [22..31] = top 10 bits of VA
        
        ```
        0xCAFEBABE >> 22 = 0x32  (decimal 50)  
        ```
        
5. **Walk page tables**
    
    - L1[0x32] → pointer to second‑level at PFN `0x00123`
        
    - L2 table base = `0x00123 << 12` = `0x00123000`
        
    - L2[0x2EB] → PFN `0x00456`
        
6. **Final physical address** = `(0x00456 << 12) | 0xABE`
    
    ```
    0x00456 << 12 = 0x456000
    + 0x00ABE      = 0x456ABE
    ```
    

So VA `0xCAFEBABE` maps to **0x00456ABE** in physical memory.

> **Why it matters:** Two‑level (or multi‑level) page tables scale to large address spaces without huge contiguous tables, and VA→PA translation is fundamental to virtual memory.

---

### 7. Clock (Second‑Chance) Page Replacement

**Q:** Simulate the Clock algorithm for the reference string

```
1, 2, 3, 2, 4, 1, 5, 2, 1, 2, 3, 4
```

with 3 frames. Show pointer movements, reference bits, and count page faults.

- **Initialize** frames = [–,–,–], ref‑bits = [0,0,0], hand at frame 0.
    
- **Step‑by‑step**:
    
    1. **1** → fault; place in f0; bit0=1; hand→f1
        
    2. **2** → fault; place in f1; bit1=1; hand→f2
        
    3. **3** → fault; place in f2; bit2=1; hand→f0
        
    4. **2** → hit (bit1=1→refresh bit1=1); no fault; hand stays
        
    5. **4** → fault
        
        - hand@f0: bit0=1→clear bit0, advance→f1
            
        - f1: bit1=1→clear, advance→f2
            
        - f2: bit2=1→clear, advance→f0
            
        - f0: bit0=0→evict 1, place 4, bit0=1; advance→f1
            
    6. **1** → fault
        
        - hand@f1: bit1=0→evict 2, place 1, bit1=1; hand→f2
            
    7. **5** → fault
        
        - hand@f2: bit2=0→evict 3, place 5, bit2=1; hand→f0
            
    8. **2** → fault
        
        - f0: bit0=1→clear→f1
            
        - f1: bit1=1→clear→f2
            
        - f2: bit2=1→clear→f0
            
        - f0: bit0=0→evict 4; place 2; bit0=1; hand→f1
            
    9. **1** → hit (frame 1), bit1=1
        

10. **2** → hit (frame 0), bit0=1
    
11. **3** → fault
    
    - f1: bit1=1→clear→f2
        
    - f2: bit2=0→evict 5; place 3; bit2=1; hand→f0
        
12. **4** → fault
    
    - f0: bit0=1→clear→f1
        
    - f1: bit1=0→evict 1; place 4; bit1=1; hand→f2
        

- **Total page faults:** 9
    

> **Why it matters:** Clock approximates LRU with O(1) overhead, essential for OSes to manage memory efficiently without full recency lists.

---

### 8. Working‑Set & Thrashing Detection

**Q:** Explain how the Working‑Set model can be used to detect and avoid thrashing. Given a process’s page‑reference string and window Δ=4, show how you’d compute the working set over time and decide if it exceeds available frames (M=3).

**Reference**: `A B C A B D A C B E`

1. **Working‑Set WS(t, Δ)** = set of distinct pages referenced in the last Δ references.
    
2. **Slide window** (Δ=4) at each reference:  
    | t (ref) | Last 4 refs | WS | |WS| |  
    |---------|------------------|----|------|  
    | 1 (A) | [A] | {A} | 1 |  
    | 2 (B) | [A,B] | {A,B}| 2 |  
    | 3 (C) | [A,B,C] | {A,B,C}|3 |  
    | 4 (A) | [A,B,C,A] | {A,B,C}|3 |  
    | 5 (B) | [B,C,A,B] | {A,B,C}|3 |  
    | 6 (D) | [C,A,B,D] | {A,B,C,D}|4 > M ⇒ thrashing risk |  
    | 7 (A) | [A,B,D,A] | {A,B,D,A}|3 |  
    | 8 (C) | [B,D,A,C] | {A,B,C,D}|4 > M ⇒ thrashing risk |  
    | 9 (B) | [D,A,C,B] | {A,B,C,D}|4 > M ⇒ thrashing risk |  
    |10 (E) | [A,C,B,E] | {A,B,C,E}|4 > M ⇒ thrashing risk |
    
3. **Decision**: Whenever |WS| > M, the process’s demand exceeds its allocated frames → risk of thrashing. OS can respond by:
    
    - **Reduce multiprogramming** (swap out some processes).
        
    - **Allocate more frames** if available.
        

> **Why it matters:** The working‑set algorithm formalizes locality; OS can preemptively adjust to avoid the drastic performance drop of thrashing.

---

### 9. Disk Scheduling: SCAN vs. C‑SCAN

**Q:** The disk head is at track 50 on a 0–199 track disk. Pending requests:

```
95, 180, 34, 119, 11, 123, 62, 64
```

Compute total head movement for:

1. **Elevator (SCAN)** moving initially toward track 0
    
2. **Circular SCAN (C‑SCAN)** toward 0
    

3. **SCAN (toward 0 then back to 199)**
    
    - Order when moving ↓: 50 → 34 → 11 → 0 (hit end) → then ↑: 62 → 64 → 95 → 119 → 123 → 180
        
    - Movements:
        
        ```
        |50–34| + |34–11| + |11–0| + |0–62| + |62–64| + |64–95| + |95–119| + |119–123| + |123–180|
        = 16 + 23 + 11 + 62 + 2 + 31 + 24 + 4 + 57
        = 230 tracks
        ```
        
4. **C‑SCAN (toward 0, jump to 199, then toward 0 again)**
    
    - ↓: 50 → 34 → 11 → 0
        
    - Jump: 0 → 199 (no count)
        
    - ↓ again: 199 → 180 → 123 → 119 → 95 → 64 → 62
        
    - Movements:
        
        ```
        |50–34| + |34–11| + |11–0| + (jump) + |199–180| + |180–123| + |123–119|
        + |119–95| + |95–64| + |64–62|
        = 16 + 23 + 11 + 19 + 57 + 4 + 24 + 31 + 2
        = 187 tracks
        ```
        

> **Why it matters:** SCAN reduces starvation and provides more uniform wait times; C‑SCAN gives even better fairness by treating the disk as a circular list.

---

### 10. Inode Structure & Maximum File Size

**Q:** A Unix‑style inode has 12 direct block pointers, 1 single‑indirect, 1 double‑indirect, and 1 triple‑indirect. Block size = 4 KB, pointer size = 4 bytes. What is the **maximum file size** supported?

1. **Direct blocks**: 12 × 4 KB = 48 KB
    
2. **Single‑indirect**:
    
    - One block holds 4 KB/4 B = 1024 pointers → 1024 × 4 KB = 4 MB
        
3. **Double‑indirect**:
    
    - 1024 pointers to single‑indirect blocks, each 1024 pointers to data → 1024² × 4 KB ≈ 4 GB
        
4. **Triple‑indirect**:
    
    - 1024³ × 4 KB ≈ 4 TB
        

**Total max size ≈**

```
48 KB + 4 MB + 4 GB + 4 TB
≈ 4 TB + 4 GB + 4 MB + 48 KB
```

Roughly **4 terabytes** (plus a few‑gig overhead).

> **Why it matters:** Shows how multi‑level indirection scales file size, and demonstrates understanding of pointer arithmetic and FS metadata design.

---

### 11. Dining Philosophers with Monitors

**Q:** Five philosophers sit around a table with one fork between each pair. Each must pick up both forks to eat, then put them down. Provide a monitor‑based solution that avoids deadlock and starvation.

```c
monitor DiningPhilosophers {
  enum { THINKING, HUNGRY, EATING } state[5];
  condition self[5];
  
  // Helper to test if philosopher i can eat
  void test(int i) {
    if (state[i] == HUNGRY &&
        state[(i+4)%5] != EATING &&
        state[(i+1)%5] != EATING) {
      state[i] = EATING;
      self[i].signal();
    }
  }

  // Called by philosopher i to pick up forks
  void pickup(int i) {
    state[i] = HUNGRY;            
    test(i);                       // try to go to EATING
    if (state[i] != EATING)       
      self[i].wait();             // block until signaled
  }

  // Called by philosopher i to put down forks
  void putdown(int i) {
    state[i] = THINKING;
    // allow hungry neighbors to eat
    test((i+4)%5);
    test((i+1)%5);
  }
}
```

- **Why it works**:
    
    1. Only a philosopher with both neighbors not eating moves to EATING.
        
    2. No circular wait: a waiter (the monitor) ensures at most one transition per test.
        
    3. Starvation is prevented because putting down always tests both neighbors, so waiting philosophers eventually proceed.
        

> **Why it matters:** Demonstrates monitor use, condition synchronization, and starvation freedom in classic concurrency.

---

### 12. Buddy Memory Allocation Simulation

**Q:** A system has 1 MB of RAM managed by a buddy allocator with minimum block size 16 KB. Show the free lists after these operations:

1. Allocate 100 KB (A)
    
2. Allocate 200 KB (B)
    
3. Allocate 32 KB (C)
    
4. Free B
    
5. Allocate 128 KB (D)
    

6. **Initial**: one free block of 1 MB
    
7. **Alloc A (100 KB)** → round up to next power of two = 128 KB.
    
    - Split 1 MB → two 512 KB; split one 512 KB → two 256 KB; split one 256 KB → two 128 KB; allocate one 128 KB to A.
        
    - Free lists: 128 KB: 1 free; 256 KB: 1; 512 KB: 1; 1 MB: 0.
        
8. **Alloc B (200 KB)** → round up = 256 KB.
    
    - Take the free 256 KB block.
        
    - Free lists: 128 KB: 1; 256 KB: 0; 512 KB: 1.
        
9. **Alloc C (32 KB)** → round up = 32 KB (fits exactly).
    
    - Split 128 KB free → two 64 KB; split one 64 KB → two 32 KB; allocate one 32 KB to C.
        
    - Free lists: 32 KB: 1; 64 KB: 1; 128 KB: 0; 256 KB: 0; 512 KB: 1.
        
10. **Free B (256 KB)**
    
    - Return 256 KB; its buddy is the other 256 KB? No, that was split earlier. So 256 KB has no buddy free → just added.
        
    - Free lists: 32 KB: 1; 64 KB: 1; 128 KB: 0; 256 KB: 1; 512 KB: 1.
        
11. **Alloc D (128 KB)** → need 128 KB.
    
    - No free 128 KB block: split free 256 KB (the one returned) → two 128 KB; allocate one to D.
        
    - Free lists end up: 32 KB: 1; 64 KB: 1; 128 KB: 1; 256 KB: 0; 512 KB: 1.
        

> **Why it matters:** Shows how splitting & coalescing maintain power‑of‑two free lists for fast allocation/deallocation with minimal fragmentation.

---

### 13. Slab Allocator Design

**Q:** Describe how a slab allocator works in a kernel to efficiently allocate frequently used object types (e.g., process structs).

1. **Caches per Object Type**
    
    - For each type (e.g., `task_struct`), maintain a **cache** with multiple **slabs**.
        
2. **Slab Structure**
    
    - A slab is one or more contiguous pages, subdivided into equal‑sized **objects**, with a freelist.
        
3. **Allocation Path**
    
    - To allocate an object:
        
        - Look in cache’s **partial** slabs (some free objects).
            
        - If none, grab a **new slab** (from buddy allocator), carve objects, add to partial.
            
        - Pop one object from freelist; serve it.
            
4. **Deallocation Path**
    
    - Return object to its slab’s freelist; if slab becomes fully free, move to **empty** list.
        
    - Optionally, free empty slabs back to buddy allocator when under memory pressure.
        
5. **Benefits**
    
    - **No per‑allocation metadata** (object size known).
        
    - **Cache coloring** and alignment reduce cache‑line conflicts.
        
    - **Constructor/Destructor** callbacks allow object initialization and cleanup.
        

> **Why it matters:** Kernel‑level allocator optimized for speed, space, and low fragmentation when allocating many same‑sized objects.

---

### 14. TLB Shootdown in SMP Systems

**Q:** On an SMP machine, when one CPU updates a page table entry (e.g., unmaps a page), explain how the OS ensures other CPUs’ TLBs don’t hold stale entries.

1. **Local TLB Flush**
    
    - CPU writing its page tables executes an architecture‑specific instruction (e.g., `invlpg` on x86) to flush its own TLB entry for that VA.
        
2. **Inter‑Processor Interrupt (IPI)**
    
    - OS sends a **TLB shootdown IPI** to all other CPUs that might have cached that mapping.
        
3. **Remote Flush Handlers**
    
    - Each target CPU’s interrupt handler executes its own `invlpg` (or global TLB flush) to invalidate the stale entry.
        
4. **Synchronization**
    
    - Originating CPU waits for acknowledgments (IPI completion) before proceeding—ensures all TLBs are coherent.
        
5. **Optimizations**
    
    - **Batching**: coalesce multiple unmaps and send a single IPI for a range.
        
    - **Per‑CPU freelists**: avoid frequent shootdowns by reusing address space per CPU when possible.
        

> **Why it matters:** Critical for correctness in virtual memory on multicore—stale TLB entries can lead to use‑after‑free or security breaches.

---

### 15. Journaling File Systems & Crash Consistency

**Q:** Explain how a journaling file system like ext4 uses its journal to ensure consistency, and compare its three journaling modes.

1. **Journal Structure**
    
    - A circular log where filesystem **metadata** (and optionally **data**) updates are recorded before being applied.
        
2. **Two‑Phase Commit**
    
    - **Write intent**: log a transaction header + associated blocks.
        
    - **Commit record**: mark transaction as committed.
        
    - **Apply**: background thread flushes logged changes to their final on‑disk locations.
        
    - **Checkpoint**: mark journal entries as free for reuse.
        
3. **Modes**
    
    - **Writeback**:
        
        - Only metadata is journaled; data blocks may be written **after** metadata commit → highest performance, risk of stale data.
            
    - **Ordered** (default):
        
        - Metadata journaled; data blocks **must** be written before metadata commit → no stale data pointers, balanced performance.
            
    - **Journal**:
        
        - Both metadata **and data** are journaled → strongest consistency, safest for data, but highest overhead.
            
4. **Crash Recovery**
    
    - On mount after crash, FS scans the journal:
        
        - **Committed** transactions are replayed (redo).
            
        - **Uncommitted** or partially committed are discarded.
            
    - Ensures on‑disk metadata is always in a consistent state.
        

> **Why it matters:** Journaling avoids lengthy fsck at boot, guarantees atomic updates, and lets you tune performance vs. safety for a high‑volume service like Zomato.

---

