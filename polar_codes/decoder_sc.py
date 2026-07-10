"""
极化码 SC（串行抵消）译码器
基于 Permuted SCD 算法（Vangala et al. 2014）
"""
import math

import numpy as np

from scd_utils import (
    active_bit_level,
    active_llr_level,
    bit_reversed,
    hard_decision,
    lower_llr,
    upper_llr,
)


def f_operation(La, Lb):
    return upper_llr(La, Lb)


def f_operation_min_sum(La, Lb):
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    return lower_llr(La, Lb, int(u_hat))


def sc_decode(llr_ch, frozen_bits):
    """Permuted SCD 译码。frozen_bits: True=冻结。"""
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr_ch)
    n = int(math.log2(N))
    frozen_set = set(np.where(frozen_bits)[0])

    L = np.full((N, n + 1), np.nan, dtype=np.float64)
    B = np.full((N, n + 1), np.nan)
    L[:, 0] = llr_ch

    for l in [bit_reversed(i, n) for i in range(N)]:
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

        if l in frozen_set:
            B[l, n] = 0
        else:
            B[l, n] = hard_decision(L[l, n])

        if l >= N / 2:
            for s in range(n, n - active_bit_level(l, n), -1):
                block_size = 2 ** s
                branch_size = block_size // 2
                for j in range(l, -1, -block_size):
                    if j % block_size >= branch_size:
                        B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                        B[j, s - 1] = B[j, s]

    return B[:, n].astype(int)


def sc_decode_recursive(llr, frozen_bits):
    return sc_decode(llr, frozen_bits)


def precompute_sc_indices(N):
    n = int(math.log2(N))
    llr_layer_vec, bit_layer_vec = [], []
    for phi in range(N):
        layers = []
        for bit in range(n):
            if (phi >> bit) & 1 == 0:
                layers = list(range(bit, n))
                break
        llr_layer_vec.append(layers)
        if phi % 2 == 1:
            bit_layers = []
            p, bit = phi, 0
            while p & 1:
                bit_layers.append(bit)
                p >>= 1
                bit += 1
            bit_layer_vec.append(bit_layers)
        else:
            bit_layer_vec.append([])
    return llr_layer_vec, bit_layer_vec
