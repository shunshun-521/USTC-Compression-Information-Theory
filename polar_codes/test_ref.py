"""Quick test for polarcodes-style decoder"""
import numpy as np
from itertools import product
from construction import ga_construction
from channel import bpsk_modulate, compute_llr

def bit_reversed(x, n):
    result = 0
    for i in range(n):
        if x & (1 << i):
            result |= (1 << (n - 1 - i))
    return result

def logdomain_sum(x, y):
    if x > y:
        return x + np.log1p(np.exp(y - x))
    return y + np.log1p(np.exp(x - y))

def upper_llr(l1, l2):
    return logdomain_sum(l1 + l2, 0) - logdomain_sum(l1, l2)

def lower_llr(l1, l2, b):
    return l1 + l2 if b == 0 else l1 - l2

def active_llr_level(i, n):
    mask = 2 ** (n - 1)
    count = 1
    for _ in range(n):
        if (mask & i) == 0:
            count += 1
            mask >>= 1
        else:
            break
    return min(count, n)

def active_bit_level(i, n):
    mask = 2 ** (n - 1)
    count = 1
    for _ in range(n):
        if (mask & i) > 0:
            count += 1
            mask >>= 1
        else:
            break
    return min(count, n)

def scd_decode(llrs, frozen_set, n):
    N = 2 ** n
    L = np.full((N, n + 1), np.nan, dtype=np.float64)
    B = np.full((N, n + 1), np.nan)
    L[:, 0] = llrs
    frozen_set = set(frozen_set)

    def update_llrs(l):
        for s in range(n - active_llr_level(l, n), n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = upper_llr(L[j, s], L[j + branch_size, s])
                else:
                    L[j, s + 1] = lower_llr(
                        L[j, s], L[j - branch_size, s], B[j - branch_size, s + 1]
                    )

    def update_bits(l):
        if l < N / 2:
            return
        for s in range(n, n - active_bit_level(l, n), -1):
            block_size = 2 ** s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                    B[j, s - 1] = B[j, s]

    for l in [bit_reversed(i, n) for i in range(N)]:
        update_llrs(l)
        B[l, n] = 0 if l in frozen_set else (0 if L[l, n] >= 0 else 1)
        update_bits(l)
    return B[:, n].astype(int)

def polar_encode_ref(u):
    u = np.array(u, dtype=int).copy()
    N = len(u)
    n = N
    while n > 1:
        n_split = n // 2
        for p in range(0, N, n):
            for k in range(n_split):
                l = p + k
                u[l] ^= u[l + n_split]
        n = n_split
    return u

N = 8
K = 4
n = 3
info_idx, frozen_idx, _ = ga_construction(N, K, 2.5)
errors = 0
for bits in product([0, 1], repeat=K):
    u = np.zeros(N, dtype=int)
    u[info_idx] = bits
    llr = compute_llr(bpsk_modulate(polar_encode_ref(u)), 1e-9)
    u_hat = scd_decode(llr, frozen_idx, n)
    if tuple(u_hat[info_idx]) != bits:
        errors += 1
print(f"N=8 errors: {errors}/16")
