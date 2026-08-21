"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math

import numpy as np

from encoder import bit_reversal_permutation, bit_reversed_value


def logdomain_sum(x, y):
    if x > y:
        return x + np.log1p(np.exp(y - x))
    return y + np.log1p(np.exp(x - y))


def hard_decision(y):
    return 0 if y >= 0.0 else 1


def upper_llr(l1, l2):
    if l1 == np.inf and l2 != np.inf:
        return l2
    if l1 != np.inf and l2 == np.inf:
        return l1
    if l1 == np.inf and l2 == np.inf:
        return np.inf
    return logdomain_sum(l1 + l2, 0.0) - logdomain_sum(l1, l2)


def lower_llr(l1, l2, b):
    if b == 0:
        if l1 == np.inf or l2 == np.inf:
            return np.inf
        return l1 + l2
    if b == 1:
        return l1 - l2
    return np.nan


def f_operation(La, Lb):
    """f 运算（box-plus，与 upper_llr 等价）。"""
    return upper_llr(La, Lb)


def g_operation(La, Lb, u_hat):
    """g 运算，与 lower_llr(Lb, La, u) 等价。"""
    return lower_llr(Lb, La, int(u_hat))


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


class SCDecoderCore:
    """非递归 SC 译码核心。"""

    def __init__(self, N, frozen_indices):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_set = set(int(i) for i in frozen_indices)

    def decode_llr(self, llr_ch):
        L = np.full((self.N, self.n + 1), np.nan, dtype=np.float64)
        B = np.full((self.N, self.n + 1), np.nan)
        L[:, 0] = llr_ch

        for l in [bit_reversed_value(i, self.n) for i in range(self.N)]:
            for s in range(self.n - active_llr_level(l, self.n), self.n):
                block_size = 1 << (s + 1)
                branch_size = block_size >> 1
                for j in range(l, self.N, block_size):
                    if j % block_size < branch_size:
                        L[j, s + 1] = upper_llr(L[j, s], L[j + branch_size, s])
                    else:
                        top_bit = B[j - branch_size, s + 1]
                        L[j, s + 1] = lower_llr(L[j, s], L[j - branch_size, s], top_bit)

            if l in self.frozen_set:
                B[l, self.n] = 0
            else:
                B[l, self.n] = hard_decision(L[l, self.n])

            if l >= self.N // 2:
                for s in range(self.n, self.n - active_bit_level(l, self.n), -1):
                    block_size = 1 << s
                    branch_size = block_size >> 1
                    for j in range(l, -1, -block_size):
                        if j % block_size >= branch_size:
                            B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                            B[j, s - 1] = B[j, s]

        return B[:, self.n].astype(int)


def _frozen_bits_to_indices(frozen_bits):
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    return np.where(frozen_bits)[0]


def _align_llr_for_decoder(llr_ch):
    """编码端含比特倒序置换时，对齐 LLR 到译码树顺序。"""
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    br = bit_reversal_permutation(len(llr_ch))
    return llr_ch[br]


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC 译码主函数。"""
    frozen_indices = _frozen_bits_to_indices(frozen_bits)
    llr_aligned = _align_llr_for_decoder(llr_ch)
    return SCDecoderCore(len(llr_ch), frozen_indices).decode_llr(llr_aligned)


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（与 fast-SC 等价）。"""
    return sc_decode(llr, frozen_bits)


def precompute_sc_indices(N):
    n = int(math.log2(N))
    lambda_offset = np.zeros(N, dtype=int)
    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        psi = phi + 1
        lambda_offset[phi] = 1 << (n - int(math.floor(math.log2(psi))))
        layers = []
        tmp = phi
        while tmp & 1:
            layers.append(int(math.log2(tmp & -tmp)))
            tmp >>= 1
        llr_layer_vec.append(layers)
        bit_layer_vec.append(layers.copy())
    return lambda_offset, llr_layer_vec, bit_layer_vec


def _precompute_sc_tables(N):
    return precompute_sc_indices(N)
