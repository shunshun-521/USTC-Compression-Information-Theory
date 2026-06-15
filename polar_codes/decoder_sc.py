"""
极化码 SC（串行抵消）译码器
PSC（Permuted Successive Cancellation, Vangala 2014）实现
"""
import math

import numpy as np

from encoder import bit_reversed_index


def f_operation(La, Lb):
    """min-sum f（供 SCL/BP 使用）。"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def _logdomain_sum(x, y):
    if x > y:
        return x + np.log1p(np.exp(y - x))
    return y + np.log1p(np.exp(x - y))


def _upper_llr(l1, l2):
    if np.isinf(l1) and not np.isinf(l2):
        return l2
    if not np.isinf(l1) and np.isinf(l2):
        return l1
    if np.isinf(l1) and np.isinf(l2):
        return np.inf
    return _logdomain_sum(l1 + l2, 0) - _logdomain_sum(l1, l2)


def g_operation(La, Lb, u_hat):
    """g(top, btm, u): u=0 -> top+btm; u=1 -> btm-top"""
    if np.isscalar(u_hat):
        return La + Lb if u_hat == 0 else Lb - La
    return np.where(u_hat == 0, La + Lb, Lb - La)


def _lower_llr(l1, l2, b):
    if b == 0:
        if l1 == np.inf or l2 == np.inf:
            return np.inf
        return l1 + l2
    if b == 1:
        return l1 - l2
    return np.nan


def _active_llr_level(i, n):
    mask = 2 ** (n - 1)
    count = 1
    for _ in range(n):
        if (mask & i) == 0:
            count += 1
            mask >>= 1
        else:
            break
    return min(count, n)


def _active_bit_level(i, n):
    mask = 2 ** (n - 1)
    count = 1
    for _ in range(n):
        if (mask & i) > 0:
            count += 1
            mask >>= 1
        else:
            break
    return min(count, n)


def precompute_sc_indices(N):
    n = int(math.log2(N))
    decode_order = [bit_reversed_index(i, n) for i in range(N)]
    llr_layers = [
        list(range(n - _active_llr_level(l, n), n)) for l in decode_order
    ]
    bit_layers = [
        list(range(n, n - _active_bit_level(l, n), -1)) for l in decode_order
    ]
    return decode_order, llr_layers, bit_layers


class _PSCState:
    """内部 PSC 状态（与 polarcodes.SCD 等价）。"""

    def __init__(self, llr_ch, frozen_bits):
        self.llr_ch = np.asarray(llr_ch, dtype=np.float64)
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.N = len(llr_ch)
        self.n = int(math.log2(self.N))
        self.L = np.full((self.N, self.n + 1), np.nan, dtype=np.float64)
        self.B = np.full((self.N, self.n + 1), np.nan)
        self.L[:, 0] = self.llr_ch

    def update_llrs(self, l):
        for s in range(self.n - _active_llr_level(l, self.n), self.n):
            block_size = int(2 ** (s + 1))
            branch_size = int(block_size / 2)
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    self.L[j, s + 1] = _upper_llr(
                        self.L[j, s], self.L[j + branch_size, s]
                    )
                else:
                    self.L[j, s + 1] = _lower_llr(
                        self.L[j, s],
                        self.L[j - branch_size, s],
                        self.B[j - branch_size, s + 1],
                    )

    def update_bits(self, l):
        if l < self.N / 2:
            return
        for s in range(self.n, self.n - _active_bit_level(l, self.n), -1):
            block_size = int(2 ** s)
            branch_size = int(block_size / 2)
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    self.B[j - branch_size, s - 1] = int(self.B[j, s]) ^ int(
                        self.B[j - branch_size, s]
                    )
                    self.B[j, s - 1] = self.B[j, s]

    def decode(self):
        for l in [bit_reversed_index(i, self.n) for i in range(self.N)]:
            self.update_llrs(l)
            if self.frozen_bits[l]:
                self.B[l, self.n] = 0
            else:
                self.B[l, self.n] = 0 if self.L[l, self.n] >= 0 else 1
            self.update_bits(l)
        return self.B[:, self.n].astype(int)


def sc_decode(llr_ch, frozen_bits):
    return _PSCState(llr_ch, frozen_bits).decode()


def sc_decode_recursive(llr_ch, frozen_bits):
    return sc_decode(llr_ch, frozen_bits)


sc_decode_nonrecursive = sc_decode
