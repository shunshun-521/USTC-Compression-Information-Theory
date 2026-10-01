"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math

import numpy as np

from encoder import bit_reversal_permutation


def f_operation(La, Lb):
    """f 运算（box-plus，向量化）"""
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    ta = np.tanh(La * 0.5)
    tb = np.tanh(Lb * 0.5)
    prod = np.clip(ta * tb, -1.0 + 1e-12, 1.0 - 1e-12)
    return 2.0 * np.arctanh(prod)


def g_operation(La, Lb, u_hat):
    """g 运算"""
    return (1.0 - 2.0 * u_hat) * La + Lb


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（参考实现）"""
    llr = np.asarray(llr, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)

    def decode_rec(llr_node, frozen_node):
        n = len(llr_node)
        if n == 1:
            return np.array([0 if frozen_node[0] else (0 if llr_node[0] >= 0 else 1)], dtype=np.int8)
        half = n // 2
        llr_left = f_operation(llr_node[:half], llr_node[half:])
        u_left = decode_rec(llr_left, frozen_node[:half])
        llr_right = g_operation(llr_node[:half], llr_node[half:], u_left)
        u_right = decode_rec(llr_right, frozen_node[half:])
        return np.concatenate([u_left, u_right])

    return decode_rec(llr, frozen_bits)


def precompute_sc_indices(N):
    """预计算非递归 SC 译码辅助向量"""
    m = int(math.log2(N))
    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        if phi == 0:
            layers_llr = list(range(m - 1, -1, -1))
        else:
            layers_llr = []
            p = phi
            while (p & 1) == 1:
                layers_llr.append(int(math.log2(p & -p)))
                p >>= 1
        llr_layer_vec.append(layers_llr)

        if phi == 0:
            layers_bit = list(range(m))
        elif (phi & 1) == 0:
            layers_bit = []
            p = phi
            while (p & 1) == 0 and p > 0:
                layers_bit.append(int(math.log2(p & -p)))
                p >>= 1
        else:
            layers_bit = []
        bit_layer_vec.append(layers_bit)

    return llr_layer_vec, bit_layer_vec


def sc_decode_nonrecursive(llr_ch, frozen_bits):
    """非递归 SC 译码（层状 LLR/比特更新）"""
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr_ch)
    m = int(math.log2(N))

    br = bit_reversal_permutation(N)
    llr = llr_ch[br]
    frozen = frozen_bits[br]

    llr_layer_vec, bit_layer_vec = precompute_sc_indices(N)

    P = np.zeros((m + 1, N), dtype=np.float64)
    C = np.zeros((m + 1, N), dtype=np.int8)
    P[m, :] = llr
    u_hat = np.zeros(N, dtype=np.int8)

    for phi in range(N):
        for layer in llr_layer_vec[phi]:
            blk = 1 << layer
            for b in range(blk):
                P[layer, b] = f_operation(P[layer + 1, b], P[layer + 1, b + blk])
                P[layer, b + blk] = g_operation(
                    P[layer + 1, b],
                    P[layer + 1, b + blk],
                    C[layer, b],
                )

        if frozen[phi]:
            u_hat[phi] = 0
        else:
            u_hat[phi] = 0 if P[0, 0] >= 0 else 1
        C[0, 0] = u_hat[phi]

        for layer in bit_layer_vec[phi]:
            blk = 1 << layer
            for b in range(blk):
                C[layer + 1, b] = C[layer, b] ^ C[layer, b + blk]
                C[layer + 1, b + blk] = C[layer, b]

    return u_hat


def sc_decode(llr_ch, frozen_bits):
    """SC 译码主入口（信道 LLR 为码字自然顺序）"""
    from scd_vendor.adapter import vendor_sc_decode

    return vendor_sc_decode(llr_ch, frozen_bits)
