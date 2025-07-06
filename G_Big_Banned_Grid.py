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
    H = inint()
    W = inint()
    K = inint()
    row_obs = defaultdict(list)
    col_obs = defaultdict(list)
    for _ in range(K):
        r = inint()
        c = inint()
        row_obs[r].append(c)
        col_obs[c].append(r)
    for r, lst in row_obs.items():
        lst.sort()
        lst.insert(0, 0)
        lst.append(W + 1)
    for c, lst in col_obs.items():
        lst.sort()
        lst.insert(0, 0)
        lst.append(H + 1)
    start = (1, 1)
    goal = (H, W)
    if start == goal:
        print("Yes")
        return
    q = deque([start])
    visited = {start}
    while q:
        r, c = q.popleft()
        if (r, c) == goal:
            print("Yes")
            return
        row = row_obs.get(r, [0, W + 1])
        i = bisect_left(row, c)
        left = row[i - 1] + 1
        if left < c and (r, left) not in visited:
            visited.add((r, left))
            q.append((r, left))
        right = row[i] - 1
        if right > c and (r, right) not in visited:
            visited.add((r, right))
            q.append((r, right))
        col = col_obs.get(c, [0, H + 1])
        i = bisect_left(col, r)
        up = col[i - 1] + 1
        if up < r and (up, c) not in visited:
            visited.add((up, c))
            q.append((up, c))
        down = col[i] - 1
        if down > r and (down, c) not in visited:
            visited.add((down, c))
            q.append((down, c))
    print("No")

# ---------- squirtle ---------
def main() -> None:
    t = 1
    # t = inint()
    for _ in range(t):
        squirtle()

if __name__ == "__main__":
    main()
