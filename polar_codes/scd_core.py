import numpy as np
from construction import ga_construction


def bit_reversed(x, n):
    r = 0
    for i in range(n):
        if x & (1 << i):
            r |= 1 << (n - 1 - i)
    return r


def logdomain_sum(x, y):
    return x + np.log1p(np.exp(y - x)) if x > y else y + np.log1p(np.exp(x - y))


def upper_ref(l1, l2):
    if l1 == np.inf and l2 != np.inf:
        return l2
    if l2 == np.inf and l1 != np.inf:
        return l1
    if l1 == np.inf and l2 == np.inf:
        return np.inf
    return logdomain_sum(l1 + l2, 0) - logdomain_sum(l1, l2)


def lower_llr(l1, l2, b):
    if b == 0:
        if l1 == np.inf or l2 == np.inf:
            return np.inf
        return l1 + l2
    return l1 - l2


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


def _b(v):
    return 0 if np.isnan(v) else int(v)


def scd(llr, frozen, n):
    N = len(llr)
    L = np.full((N, n + 1), np.nan)
    B = np.full((N, n + 1), np.nan)
    L[:, 0] = llr
    for i in range(N):
        l = bit_reversed(i, n)
        for s in range(n - active_llr_level(l, n), n):
            bs = 2 ** (s + 1)
            brs = bs // 2
            for j in range(l, N, bs):
                if j % bs < brs:
                    L[j, s + 1] = upper_ref(L[j, s], L[j + brs, s])
                else:
                    L[j, s + 1] = lower_llr(L[j, s], L[j - brs, s], _b(B[j - brs, s + 1]))
        if l in frozen:
            B[l, n] = 0
        else:
            B[l, n] = 0 if L[l, n] >= 0 else 1
        if l < N // 2:
            continue
        for s in range(n, n - active_bit_level(l, n), -1):
            bs = 2 ** s
            brs = bs // 2
            for j in range(l, -1, -bs):
                if j % bs >= brs:
                    B[j - brs, s - 1] = _b(B[j, s]) ^ _b(B[j - brs, s])
                    B[j, s - 1] = B[j, s]
    return B[:, n].astype(int)


def enc_xor(u):
    u = u.copy()
    N = len(u)
    step = 1
    while step < N:
        for i in range(0, N, 2 * step):
            u[i : i + step] ^= u[i + step : i + 2 * step]
        step *= 2
    return u


if __name__ == "__main__":
    for N in [8, 64]:
        n = int(np.log2(N))
        K = N // 2
        info, frozen_idx, _ = ga_construction(N, K, 2.5)
        frozen = set(frozen_idx)
        rng = np.random.default_rng(0)
        fail = 0
        tests = 16 if N == 8 else 100
        for t in range(tests):
            u = np.zeros(N, dtype=int)
            if N == 8:
                u[info] = [(t >> i) & 1 for i in range(K)]
            else:
                u[info] = rng.integers(0, 2, K)
            codeword = enc_xor(u)
            tx = 2 * (codeword - 0.5)
            llr = -2 * tx
            uh = scd(llr, frozen, n)
            if not np.array_equal(uh, u):
                fail += 1
        print(N, fail, tests)
