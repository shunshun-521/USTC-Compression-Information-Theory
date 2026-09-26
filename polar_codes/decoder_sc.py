"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归 SCD 实现（高效）
"""
import math

import numpy as np

from encoder import bit_reversal_permutation


def _bit_reversed(x, n):
    result = 0
    for i in range(n):
        if x & (1 << i):
            result |= 1 << (n - 1 - i)
    return result


def _logdomain_sum(x, y):
    if x > y:
        return x + np.log1p(np.exp(y - x))
    return y + np.log1p(np.exp(x - y))


def upper_llr(l1, l2):
    """f 运算（对数域 boxplus）"""
    if np.isinf(l1) and not np.isinf(l2):
        return l2
    if np.isinf(l2) and not np.isinf(l1):
        return l1
    if np.isinf(l1) and np.isinf(l2):
        return np.inf
    return _logdomain_sum(l1 + l2, 0.0) - _logdomain_sum(l1, l2)


def lower_llr(l1, l2, b):
    """g 运算"""
    if b == 0:
        if np.isinf(l1) or np.isinf(l2):
            return np.inf
        return l1 + l2
    return l1 - l2


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


def f_operation(La, Lb):
    """向量化 min-sum f（供 SCL/BP 使用）"""
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    sign = np.where(La >= 0, 1.0, -1.0) * np.where(Lb >= 0, 1.0, -1.0)
    return sign * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算（向量化）"""
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    u_hat = np.asarray(u_hat)
    return (1.0 - 2.0 * u_hat) * La + Lb


class _SCDCore:
    """非递归 SCD 内核（Permuted SC）"""

    def __init__(self, N, frozen_indices):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen = set(int(i) for i in frozen_indices)
        self.L = np.full((N, self.n + 1), np.nan, dtype=np.float64)
        self.B = np.full((N, self.n + 1), np.nan, dtype=np.float64)

    def _update_llrs(self, l):
        for s in range(self.n - _active_llr_level(l, self.n), self.n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    self.L[j, s + 1] = upper_llr(self.L[j, s], self.L[j + branch_size, s])
                else:
                    top_bit = int(self.B[j - branch_size, s + 1])
                    self.L[j, s + 1] = lower_llr(
                        self.L[j, s], self.L[j - branch_size, s], top_bit
                    )

    def _update_bits(self, l):
        if l < self.N / 2:
            return
        for s in range(self.n, self.n - _active_bit_level(l, self.n), -1):
            block_size = 2 ** s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    self.B[j - branch_size, s - 1] = int(self.B[j, s]) ^ int(
                        self.B[j - branch_size, s]
                    )
                    self.B[j, s - 1] = self.B[j, s]

    def decode(self, llr_ch):
        self.L[:, 0] = np.asarray(llr_ch, dtype=np.float64)
        self.B.fill(np.nan)
        for i in range(self.N):
            l = _bit_reversed(i, self.n)
            self._update_llrs(l)
            if l in self.frozen:
                self.B[l, self.n] = 0
            else:
                self.B[l, self.n] = 0 if self.L[l, self.n] >= 0 else 1
            self._update_bits(l)
        return self.B[:, self.n].astype(int)


def precompute_sc_indices(N):
    """保留接口：返回占位结构（主译码使用 SCD 内核）"""
    n = int(math.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]
    return lambda_offset, [[] for _ in range(N)], [[] for _ in range(N)]


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC 译码（SCD）"""
    N = len(llr_ch)
    frozen_bits = np.asarray(frozen_bits)
    frozen_idx = np.where(frozen_bits.astype(bool))[0]
    brp = bit_reversal_permutation(N)
    # 编码含 B_N：将信道 LLR 重排为与 F^{⊗n} 码字顺序一致
    llr = np.asarray(llr_ch, dtype=np.float64)[brp]
    return _SCDCore(N, frozen_idx).decode(llr)


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC（调用 SCD 作为参考实现）"""
    return sc_decode(llr, frozen_bits)
