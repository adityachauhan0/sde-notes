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
from functools import lru_cache

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
    n = inint()
    size = 1 << n
    perm = inlist(size)
    def best(a,l,r):
        if r - l == 1:
            return [a[l]]
        m = (l + r) >> 1
        left = best(a,l,m)
        right = best(a,m,r)
        if left + right <= right + left:
            res = left + right
        else:
            res = right + left
        a[l:r] = res
        return res
    best(perm,0,size)
    print(" ".join(map(str,perm)))
    return
# ---------- squirtle ---------
def main() -> None:
    t = 1
    t = inint()
    for _ in range(t):
        squirtle()

if __name__ == "__main__":
    main()
