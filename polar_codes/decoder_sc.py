"""
极化码 SC（串行抵消）译码器（PSCD + 递归封装）
"""
import math
import numpy as np
from decoder_utils_local import (
    active_bit_level,
    active_llr_level,
    bit_reversed,
    hard_decision,
    lower_llr,
    upper_llr,
)


def f_operation(La, Lb):
    return np.vectorize(upper_llr)(np.asarray(La, float), np.asarray(Lb, float))


def g_operation(La, Lb, u_hat):
    return np.vectorize(lower_llr)(
        np.asarray(La, float), np.asarray(Lb, float), np.asarray(u_hat, int)
    )


class _SCD:
    def __init__(self, llr_ch, frozen_indices):
        self.N = len(llr_ch)
        self.n = int(math.log2(self.N))
        self.frozen = set(int(x) for x in frozen_indices)
        self.L = np.full((self.N, self.n + 1), np.nan, dtype=np.float64)
        self.B = np.full((self.N, self.n + 1), np.nan)
        self.L[:, 0] = llr_ch

    def update_llrs(self, l):
        for s in range(self.n - active_llr_level(l, self.n), self.n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    self.L[j, s + 1] = upper_llr(self.L[j, s], self.L[j + branch_size, s])
                else:
                    self.L[j, s + 1] = lower_llr(
                        self.L[j - branch_size, s],
                        self.L[j, s],
                        int(self.B[j - branch_size, s + 1]),
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

    def decode(self):
        for i in range(self.N):
            l = bit_reversed(i, self.n)
            self.update_llrs(l)
            if i in self.frozen:
                self.B[l, self.n] = 0
            else:
                self.B[l, self.n] = hard_decision(self.L[l, self.n])
            self.update_bits(l)
        u_hat = np.zeros(self.N, dtype=int)
        for i in range(self.N):
            u_hat[i] = int(self.B[bit_reversed(i, self.n), self.n])
        return u_hat


def _frozen_indices_from_mask(frozen_bits):
    fb = np.asarray(frozen_bits)
    if fb.dtype == bool:
        return np.where(fb)[0]
    return np.where(fb.astype(int) == 1)[0]


def sc_decode(llr_ch, frozen_bits):
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_idx = _frozen_indices_from_mask(frozen_bits)
    return _SCD(llr_ch, frozen_idx).decode()


def sc_decode_recursive(llr, frozen_bits):
    return sc_decode(llr, frozen_bits)


def precompute_sc_indices(N):
    n = int(math.log2(N))
    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        llr_layers = []
        p = phi
        while p & 1:
            llr_layers.append(int(math.log2(p & -p)))
            p >>= 1
        llr_layer_vec.append(llr_layers)
        bit_layers = []
        p = phi + 1
        while p < N and (p & 1) == 0:
            bit_layers.append(int(math.log2(p & -p)) - 1)
            p >>= 1
        bit_layer_vec.append(bit_layers)
    return [1 << i for i in range(n + 1)], llr_layer_vec, bit_layer_vec
