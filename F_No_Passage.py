# ----------------------------------------
#  squirtle squirtle squirtle (Py3)
#  Author: squirt1e
# ----------------------------------------

import sys
import math
import random
from bisect import bisect_left, bisect_right
from collections import defaultdict, Counter, deque
from heapq import heappush, heappop, heapify
from itertools import combinations, permutations, accumulate
from fractions import Fraction
from decimal import Decimal, getcontext
from array import array

# ---------- squirtle ----------
data = sys.stdin.buffer.read().split()
it = iter(data)
def inp()  -> str:      return next(it).decode()
def inint() -> int:     return int(inp())
def inmap() -> list:    return list(map(int, (inp() for _ in range(k))))  # use k locally
def inlist(n:int) -> list[int]:   # read n ints
    return [int(inp()) for _ in range(n)]

# ---------- squirtle ----------
INF  = 10**18
MOD  = 1_000_000_007  # or 998244353
dirs4 = ((1,0),(-1,0),(0,1),(0,-1))
dirs8 = ((1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1))
#----------- squirtle ---------
def ceil_div(a:int,b:int)->int: return -(-a//b)
def chmin(a:list, idx:int, v):  # inplace relax
    if v < a[idx]: a[idx] = v; return True
    return False
def chmax(a:list, idx:int, v):
    if v > a[idx]: a[idx] = v; return True
    return False
def rnd_shuffle(a:list):
    random.shuffle(a)

def gcd(a:int,b:int)->int:
    while b: a,b=b,a%b
    return a
def lcm(a:int,b:int)->int:
    return a//gcd(a,b)*b

# ---------- squirtle ----------
DEBUG = False   # flip to True locally
def dbg(*args, **kw):
    if DEBUG:
        print(*args, **kw, file=sys.stderr)

# ---------- squirtle ----------
def squirtle() -> None:
    H, W, K = inint(), inint(), inint()
    N   = H * W               
    BIG = 1_000_000_000          
    dist = array('I', [BIG]) * N
    mn1  = array('I', [BIG]) * N
    mn2  = array('I', [BIG]) * N
    buckets = [[]]              
    for _ in range(K):
        r = inint() - 1
        c = inint() - 1
        v = r * W + c
        dist[v] = 0
        buckets[0].append(v)
    cur = 0
    while cur < len(buckets):       
        if not buckets[cur]:
            cur += 1
            continue
        v = buckets[cur].pop()
        if dist[v] != cur:            
            continue
        vr, vc = divmod(v, W)
        for dr, dc in dirs4:
            nr, nc = vr + dr, vc + dc
            if nr < 0 or nr >= H or nc < 0 or nc >= W:
                continue
            u   = nr * W + nc
            val = dist[v]
            if val < mn1[u]:
                mn2[u] = mn1[u]
                mn1[u] = val
            elif val < mn2[u]:
                mn2[u] = val
            else:
                continue
            if mn2[u] == BIG:         
                continue
            newd = mn2[u] + 1         
            if newd >= len(buckets):  
                buckets.extend([] for _ in range(newd - len(buckets) + 1))
            if newd < dist[u]:
                dist[u] = newd
                buckets[newd].append(u)
    total = 0
    for d in dist:
        if d != BIG:
            total += d
    print(total)

# ---------- squirtle ---------
def main() -> None:
    t = 1        
    # t = inint() 
    for _ in range(t):
        squirtle()

if __name__ == "__main__":
    main()
