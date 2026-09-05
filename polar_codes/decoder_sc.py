"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math

import numpy as np


def bit_reversed_index(x, n):
    result = 0
    for i in range(n):
        if x & (1 << i):
            result |= 1 << (n - 1 - i)
    return result


def logdomain_sum(x, y):
    if x > y:
        return x + np.log1p(np.exp(y - x))
    return y + np.log1p(np.exp(x - y))


def upper_llr(l1, l2):
    if l1 == np.inf and l2 != np.inf:
        return l2
    if l1 != np.inf and l2 == np.inf:
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


def hard_decision(y):
    return 0 if y >= 0 else 1


def f_operation(La, Lb):
    return upper_llr(float(La), float(Lb))


def g_operation(La, Lb, u_hat):
    return lower_llr(float(Lb), float(La), int(u_hat))


def active_llr_level(i, n):
    mask = 1 << (n - 1)
    count = 1
    for _ in range(n):
        if (mask & i) == 0:
            count += 1
            mask >>= 1
        else:
            break
    return min(count, n)


def active_bit_level(i, n):
    mask = 1 << (n - 1)
    count = 1
    for _ in range(n):
        if (mask & i) > 0:
            count += 1
            mask >>= 1
        else:
            break
    return min(count, n)


def _frozen_indices(frozen_bits):
    frozen_bits = np.asarray(frozen_bits)
    if frozen_bits.dtype == bool:
        return np.where(frozen_bits)[0]
    return np.where(frozen_bits != 0)[0]


class _SCDecoderCore:
    """非递归 SC 译码核心（与标准矩阵实现一致）"""

    def __init__(self, llr_ch, frozen_indices):
        self.N = len(llr_ch)
        self.n = int(math.log2(self.N))
        self.frozen = np.asarray(frozen_indices, dtype=int)
        self.L = np.full((self.N, self.n + 1), np.nan, dtype=np.float64)
        self.B = np.full((self.N, self.n + 1), np.nan)
        self.L[:, 0] = llr_ch

    def decode(self):
        for i in range(self.N):
            l = bit_reversed_index(i, self.n)
            self._update_llrs(l)
            if l in self.frozen:
                self.B[l, self.n] = 0
            else:
                self.B[l, self.n] = hard_decision(self.L[l, self.n])
            self._update_bits(l)
        return self.B[:, self.n].astype(np.int8)

    def _update_llrs(self, l):
        for s in range(self.n - active_llr_level(l, self.n), self.n):
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    self.L[j, s + 1] = upper_llr(self.L[j, s], self.L[j + branch_size, s])
                else:
                    self.L[j, s + 1] = lower_llr(
                        self.L[j, s],
                        self.L[j - branch_size, s],
                        self.B[j - branch_size, s + 1],
                    )

    def _update_bits(self, l):
        if l < self.N / 2:
            return
        for s in range(self.n, self.n - active_bit_level(l, self.n), -1):
            block_size = 1 << s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    self.B[j - branch_size, s - 1] = (
                        int(self.B[j, s]) ^ int(self.B[j - branch_size, s])
                    )
                    self.B[j, s - 1] = self.B[j, s]


def sc_decode_recursive(llr, frozen_bits):
    frozen_set = set(_frozen_indices(frozen_bits))
    llr = np.asarray(llr, dtype=np.float64)
    N = len(llr)
    u_hat = np.zeros(N, dtype=np.int8)

    def decode_node(llr_node, bit_offset):
        n = len(llr_node)
        if n == 1:
            idx = bit_offset
            if idx in frozen_set:
                u_hat[idx] = 0
            else:
                u_hat[idx] = 0 if llr_node[0] >= 0 else 1
            return

        half = n // 2
        llr_left = np.array([
            f_operation(llr_node[i], llr_node[i + half]) for i in range(half)
        ])
        decode_node(llr_left, bit_offset)

        u_left = u_hat[bit_offset:bit_offset + half]
        llr_right = np.array([
            g_operation(llr_node[i], llr_node[i + half], u_left[i]) for i in range(half)
        ])
        decode_node(llr_right, bit_offset + half)

    decode_node(llr, 0)
    return u_hat


def precompute_sc_indices(N):
    n = int(math.log2(N))
    decode_order = [bit_reversed_index(i, n) for i in range(N)]
    return n, decode_order


def sc_decode(llr_ch, frozen_bits):
    frozen_idx = _frozen_indices(frozen_bits)
    return _SCDecoderCore(np.asarray(llr_ch, dtype=np.float64), frozen_idx).decode()


def compute_sc_llr_at_phase(L, B, l, n):
    for s in range(n - active_llr_level(l, n), n):
        block_size = 1 << (s + 1)
        branch_size = block_size // 2
        for j in range(l, L.shape[0], block_size):
            if j % block_size < branch_size:
                L[j, s + 1] = upper_llr(L[j, s], L[j + branch_size, s])
            else:
                L[j, s + 1] = lower_llr(
                    L[j, s],
                    L[j - branch_size, s],
                    B[j - branch_size, s + 1],
                )
    return L[l, n]


def propagate_bit_sc(B, l, u_bit, n, N):
    B[l, n] = u_bit
    if l < N / 2:
        return
    for s in range(n, n - active_bit_level(l, n), -1):
        block_size = 1 << s
        branch_size = block_size // 2
        for j in range(l, -1, -block_size):
            if j % block_size >= branch_size:
                B[j - branch_size, s - 1] = (
                    int(B[j, s]) ^ int(B[j - branch_size, s])
                )
                B[j, s - 1] = B[j, s]
