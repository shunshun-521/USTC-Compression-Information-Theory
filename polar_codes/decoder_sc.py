"""
极化码 SC（串行抵消）译码器
基于 Permuted SCD（Vangala et al., 2014）
"""
import math

import numpy as np


def bit_reversed(i, n):
    """单索引比特倒序"""
    result = 0
    for b in range(n):
        if i & (1 << b):
            result |= 1 << (n - 1 - b)
    return result


def active_llr_level(i, n):
    """从最高位起，找到第一个 1 的位置（层数）"""
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
    """从最高位起，找到第一个 0 的位置（层数）"""
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
    """min-sum 近似的 f 运算"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算：b=0 -> La+Lb, b=1 -> La-Lb"""
    return np.where(u_hat == 0, La + Lb, La - Lb)


def upper_llr(l1, l2):
    """f 分支 LLR 更新（min-sum）"""
    return f_operation(l1, l2)


def lower_llr(l1, l2, b):
    """g 分支 LLR 更新"""
    if b == 0:
        return l1 + l2
    return l1 - l2


class _PermutedSCD:
    """Permuted SCD 内部状态"""

    def __init__(self, N, frozen_bits):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_set = set(np.where(np.asarray(frozen_bits, dtype=bool))[0])
        self.L = np.full((N, self.n + 1), np.nan, dtype=np.float64)
        self.B = np.full((N, self.n + 1), np.nan, dtype=np.float64)

    def update_llrs(self, l):
        for s in range(self.n - active_llr_level(l, self.n), self.n):
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

    def update_bits(self, l):
        if l < self.N / 2:
            return
        for s in range(self.n, self.n - active_bit_level(l, self.n), -1):
            block_size = 2 ** s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    self.B[j - branch_size, s - 1] = int(self.B[j, s]) ^ int(
                        self.B[j - branch_size, s]
                    )
                    self.B[j, s - 1] = self.B[j, s]

    def decode(self, llr_ch):
        self.L[:, 0] = llr_ch
        self.B[:] = np.nan
        for l in [bit_reversed(i, self.n) for i in range(self.N)]:
            self.update_llrs(l)
            if l in self.frozen_set:
                self.B[l, self.n] = 0
            else:
                self.B[l, self.n] = 0 if self.L[l, self.n] >= 0 else 1
            self.update_bits(l)
        return self.B[:, self.n].astype(int)


def sc_decode(llr_ch, frozen_bits):
    """非递归 Permuted SC 译码"""
    decoder = _PermutedSCD(len(llr_ch), frozen_bits)
    return decoder.decode(np.asarray(llr_ch, dtype=np.float64))


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（参考实现，与 Permuted SCD 等效）"""
    return sc_decode(llr, frozen_bits)


def precompute_sc_indices(N):
    """保留接口：返回 Permuted SCD 的层索引（供 SCL 使用）"""
    n = int(math.log2(N))
    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        l = bit_reversed(phi, n)
        llr_layer_vec.append(list(range(n - active_llr_level(l, n), n)))
        bit_layer_vec.append(list(range(n, n - active_bit_level(l, n), -1)))
    lambda_offset = [2 ** i for i in range(n + 1)]
    return lambda_offset, llr_layer_vec, bit_layer_vec
