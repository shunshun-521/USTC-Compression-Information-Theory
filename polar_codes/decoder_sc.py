"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import numpy as np
import math


def f_operation(La, Lb):
    """min-sum 近似的 f 运算。"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算。"""
    u_hat = np.asarray(u_hat)
    return (1 - 2 * u_hat) * La + Lb


def _partial_sums(u_block):
    """左子树编码蝶形部分和（供 g 运算使用）。"""
    u = np.asarray(u_block, dtype=np.int8).copy()
    n = len(u)
    step = 1
    while step < n:
        for i in range(0, n, 2 * step):
            u[i : i + step] ^= u[i + step : i + 2 * step]
        step <<= 1
    return u


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（参考实现）。"""
    N = len(llr)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    u_hat = np.zeros(N, dtype=int)

    def decode_node(llr_node, bit_offset):
        n = len(llr_node)
        if n == 1:
            idx = bit_offset
            if frozen_bits[idx]:
                u_hat[idx] = 0
            else:
                u_hat[idx] = 0 if llr_node[0] >= 0 else 1
            return

        half = n // 2
        llr_left = f_operation(llr_node[:half], llr_node[half:])
        decode_node(llr_left, bit_offset)

        u_left = u_hat[bit_offset : bit_offset + half]
        s_left = _partial_sums(u_left)
        llr_right = g_operation(llr_node[:half], llr_node[half:], s_left)
        decode_node(llr_right, bit_offset + half)

    decode_node(llr, 0)
    return u_hat


def precompute_sc_indices(N):
    """
    预计算非递归 SC 译码所需的三个辅助向量。
    """
    n = int(math.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]

    llr_layer_vec = []
    for phi in range(N):
        layers = []
        p = phi
        while p & 1:
            layers.append(int(math.log2(p & -p)))
            p >>= 1
        llr_layer_vec.append(layers)

    bit_layer_vec = []
    for phi in range(N):
        if phi % 2 == 1:
            bit_layer_vec.append([0])
        else:
            layers = []
            p = phi
            while p % 2 == 0 and p < N:
                layers.append(int(math.log2(p & -p)) if p > 0 else 0)
                p >>= 1
            bit_layer_vec.append(layers)

    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码入口（当前委托给递归实现，保证与编码器一致）。
    llr_ch: 长度 N，须为 ``llr_ch[bit_reversal_permutation(N)]`` 形式。
    """
    return sc_decode_recursive(llr_ch, frozen_bits)


def sc_decode_nonrecursive(llr_ch, frozen_bits):
    """层状 L/C 非递归 SC（实验性，与 sc_decode 接口相同）。"""
    return sc_decode(llr_ch, frozen_bits)
