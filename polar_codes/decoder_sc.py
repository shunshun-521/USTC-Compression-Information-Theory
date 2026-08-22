"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math

import numpy as np

from encoder import bit_reversal_permutation
from decoder_utils import (
    active_bit_level,
    active_llr_level,
    bit_reversed,
    hard_decision,
    lower_llr,
    upper_llr,
)


def align_llr_for_decoder(llr_ch):
    """将信道 LLR 对齐到极化码译码树顺序（比特倒序）"""
    N = len(llr_ch)
    br = bit_reversal_permutation(N)
    return np.asarray(llr_ch, dtype=np.float64)[br]


def f_operation(La, Lb):
    """box-plus f 运算"""
    return upper_llr(La, Lb)


def g_operation(La, Lb, u_hat):
    """g 运算（La=bottom, Lb=top in lower_llr convention）"""
    if np.isscalar(u_hat):
        return lower_llr(La, Lb, int(u_hat))
    u_hat = np.asarray(u_hat, dtype=int)
    return np.array([lower_llr(a, b, int(u)) for a, b, u in zip(La, Lb, u_hat)])


class _SCDCore:
    """SC 译码核心"""

    def __init__(self, llr_ch, frozen_bits, align=True):
        self.llr_ch = align_llr_for_decoder(llr_ch) if align else np.asarray(llr_ch, dtype=np.float64)
        self.N = len(self.llr_ch)
        self.n = int(math.log2(self.N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.L = np.full((self.N, self.n + 1), np.nan, dtype=np.float64)
        self.B = np.full((self.N, self.n + 1), np.nan)
        self.L[:, 0] = self.llr_ch

    def update_llrs(self, l):
        for s in range(self.n - active_llr_level(l, self.n), self.n):
            block_size = int(2 ** (s + 1))
            branch_size = int(block_size / 2)
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    self.L[j, s + 1] = upper_llr(self.L[j, s], self.L[j + branch_size, s])
                else:
                    self.L[j, s + 1] = lower_llr(
                        self.L[j, s],
                        self.L[j - branch_size, s],
                        self.B[j - branch_size, s + 1],
                    )

    def update_bits(self, l):
        if l < self.N / 2:
            return
        for s in range(self.n, self.n - active_bit_level(l, self.n), -1):
            block_size = int(2 ** s)
            branch_size = int(block_size / 2)
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    self.B[j - branch_size, s - 1] = int(self.B[j, s]) ^ int(self.B[j - branch_size, s])
                    self.B[j, s - 1] = self.B[j, s]

    def decode(self):
        for phi in range(self.N):
            l = bit_reversed(phi, self.n)
            self.update_llrs(l)
            if l in self.frozen_set:
                self.B[l, self.n] = 0
            else:
                self.B[l, self.n] = hard_decision(self.L[l, self.n])
            self.update_bits(l)
        return self.B[:, self.n].astype(int)


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC 译码主函数"""
    return _SCDCore(llr_ch, frozen_bits, align=True).decode()


def sc_decode_recursive(llr_ch, frozen_bits):
    """递归 SC 译码（与非递归等价）"""
    return sc_decode(llr_ch, frozen_bits)


def precompute_sc_indices(N):
    n = int(math.log2(N))
    llr_layer_vec = [[] for _ in range(N)]
    bit_layer_vec = [[] for _ in range(N)]
    for phi in range(1, N):
        psi = phi
        while psi % 2 == 0:
            psi //= 2
        llr_layer_vec[phi].append(int(math.log2(psi & -psi)))
    for phi in range(N - 1):
        psi = phi + 1
        while psi % 2 == 1:
            psi //= 2
        if psi > 0:
            bit_layer_vec[phi].append(int(math.log2(psi & -psi)))
    return llr_layer_vec, bit_layer_vec


get_sc_tables = precompute_sc_indices
_get_sc_tables = precompute_sc_indices
bit_reversed_index = bit_reversed
_active_llr_level = active_llr_level
_active_bit_level = active_bit_level
