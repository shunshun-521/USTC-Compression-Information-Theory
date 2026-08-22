"""
极化码 SC（串行抵消）译码器
"""
import math

import numpy as np

from decoder_utils import (
    active_bit_level,
    active_llr_level,
    hard_decision,
    lower_llr,
    upper_llr,
)
from encoder import bit_reversed


def f_operation(La, Lb):
    return upper_llr(float(La), float(Lb))


def g_operation(La, Lb, u_hat):
    return lower_llr(float(La), float(Lb), int(u_hat))


def sc_decode_recursive(llr, frozen_bits):
    llr = np.asarray(llr, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)

    def decode_node(llr_node, frozen_node):
        n = len(llr_node)
        if n == 1:
            if frozen_node[0]:
                return np.array([0], dtype=int)
            return np.array([hard_decision(llr_node[0])], dtype=int)
        half = n // 2
        llr_left = np.array(
            [upper_llr(llr_node[i], llr_node[i + half]) for i in range(half)]
        )
        u_left = decode_node(llr_left, frozen_node[:half])
        llr_right = np.array(
            [lower_llr(llr_node[i], llr_node[i + half], u_left[i]) for i in range(half)]
        )
        u_right = decode_node(llr_right, frozen_node[half:])
        return np.concatenate([u_left, u_right])

    return decode_node(llr, frozen_bits)


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC 译码（SCD）。"""
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    frozen_set = set(np.where(frozen_bits)[0])
    N = len(llr_ch)
    n = int(math.log2(N))

    L = np.full((N, n + 1), np.nan, dtype=np.float64)
    B = np.full((N, n + 1), np.nan)
    L[:, 0] = llr_ch

    for i in range(N):
        l = bit_reversed(i, n)
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

        if l >= N // 2:
            for s in range(n, n - active_bit_level(l, n), -1):
                block_size = 2 ** s
                branch_size = block_size // 2
                for j in range(l, -1, -block_size):
                    if j % block_size >= branch_size:
                        B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                        B[j, s - 1] = B[j, s]

    return B[:, n].astype(int)
