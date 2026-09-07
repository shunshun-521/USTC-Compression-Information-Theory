"""Verified SC decoder - direct port of SCD algorithm."""
import numpy as np


def bit_reversed(x, n):
    result = 0
    for i in range(n):
        if (x & (1 << i)):
            result |= (1 << (n - 1 - i))
    return result


def logdomain_sum(x, y):
    if x > y:
        return x + np.log1p(np.exp(y - x))
    return y + np.log1p(np.exp(x - y))


def upper_llr(l1, l2):
    return logdomain_sum(l1 + l2, 0) - logdomain_sum(l1, l2)


def lower_llr(l1, l2, b):
    if l1 == np.inf and l2 != np.inf:
        return l2
    if l1 != np.inf and l2 == np.inf:
        return l1
    if l1 == np.inf and l2 == np.inf:
        return np.inf
    if b == 0:
        return l1 + l2
    if b == 1:
        return l1 - l2
    return np.nan


def hard_decision(y):
    if y >= 0:
        return 0
    return 1


def active_llr_level(i, n):
    mask = 2 ** (n - 1)
    count = 1
    for k in range(n):
        if (mask & i) == 0:
            count += 1
            mask >>= 1
        else:
            break
    return min(count, n)


def active_bit_level(i, n):
    mask = 2 ** (n - 1)
    count = 1
    for k in range(n):
        if (mask & i) > 0:
            count += 1
            mask >>= 1
        else:
            break
    return min(count, n)


class SCDecoder:
    def __init__(self, N, frozen_set):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_set = frozen_set
        self.L = np.full((self.N, self.n + 1), np.nan, dtype=np.float64)
        self.B = np.full((self.N, self.n + 1), np.nan)

    def decode(self, llr):
        self.L[:, 0] = llr
        u_hat = np.zeros(self.N, dtype=int)

        for l in [bit_reversed(i, self.n) for i in range(self.N)]:
            self._update_llrs(l)
            if l in self.frozen_set:
                self.B[l, self.n] = 0
            else:
                self.B[l, self.n] = hard_decision(self.L[l, self.n])
            u_hat[l] = int(self.B[l, self.n])
            self._update_bits(l)

        return u_hat

    def _update_llrs(self, l):
        for s in range(self.n - active_llr_level(l, self.n), self.n):
            block_size = int(2 ** (s + 1))
            branch_size = int(block_size / 2)
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    self.L[j, s + 1] = upper_llr(self.L[j, s], self.L[j + branch_size, s])
                else:
                    btm_llr = self.L[j, s]
                    top_llr = self.L[j - branch_size, s]
                    top_bit = self.B[j - branch_size, s + 1]
                    self.L[j, s + 1] = lower_llr(btm_llr, top_llr, top_bit)

    def _update_bits(self, l):
        if l < self.N / 2:
            return
        for s in range(self.n, self.n - active_bit_level(l, self.n), -1):
            block_size = int(2 ** s)
            branch_size = int(block_size / 2)
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    self.B[j - branch_size, s - 1] = int(self.B[j, s]) ^ int(self.B[j - branch_size, s])
                    self.B[j, s - 1] = self.B[j, s]


def scd_decode_natural(llr, frozen_bits):
    N = len(llr)
    frozen_set = set(np.where(np.asarray(frozen_bits, dtype=int) == 1)[0])
    return SCDecoder(N, frozen_set).decode(llr)
