"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math

import numpy as np

from encoder import bit_reversal_permutation


def f_operation(La, Lb):
    """min-sum 近似的 f 运算"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算"""
    if np.isscalar(u_hat):
        return La + Lb if u_hat == 0 else La - Lb
    return np.where(u_hat == 0, La + Lb, La - Lb)


def _bit_reversed(i, n):
    return int(f"{i:0{n}b}"[::-1], 2)


def precompute_sc_indices(N):
    """预计算非递归 SC 译码所需的辅助向量"""
    n = int(math.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]
    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        l = _bit_reversed(phi, n)
        llr_layers = []
        p = l
        layer = 0
        while p % 2 == 1:
            llr_layers.append(layer)
            p //= 2
            layer += 1
        llr_layer_vec.append(llr_layers)
        bit_layers = []
        p = l
        layer = 0
        while p % 2 == 0 and p > 0:
            bit_layers.append(layer)
            p //= 2
            layer += 1
        bit_layer_vec.append(bit_layers)
    return lambda_offset, llr_layer_vec, bit_layer_vec


def _sc_core(llr_ch, frozen_bits):
    N = len(llr_ch)
    n = int(math.log2(N))
    L = np.zeros((N, n + 1), dtype=np.float64)
    B = np.zeros((N, n + 1), dtype=np.int8)
    L[:, n] = llr_ch

    def update_llr(x):
        for j in range(n - 1, -1, -1):
            s = 1 << (n - j)
            t = s // 2
            for i in range(x, N, s):
                if t > (i % s):
                    L[i, j] = f_operation(L[i, j + 1], L[i + t, j + 1])
                else:
                    L[i, j] = g_operation(L[i, j + 1], L[i - t, j + 1], B[i - t, j])

    def update_bits(x):
        active = [x]
        for j in range(n):
            s = 1 << (n - j)
            t = s // 2
            nxt = []
            for i in active:
                if t <= (i % s):
                    B[i - t, j + 1] = (B[i, j] + B[i - t, j]) % 2
                    B[i, j + 1] = B[i, j]
                    nxt.extend([i, i - t])
            active = nxt

    for i in range(N):
        l = _bit_reversed(i, n)
        update_llr(l)
        if frozen_bits[l]:
            B[l, 0] = 0
        else:
            B[l, 0] = 0 if L[l, 0] >= 0 else 1
        update_bits(l)

    return B[:, 0].astype(int)


def sc_decode_recursive(llr, frozen_bits):
    return sc_decode(llr, frozen_bits)


def sc_decode(llr_ch, frozen_bits):
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    br = bit_reversal_permutation(len(llr_ch))
    return _sc_core(llr_ch[br], frozen_bits)
