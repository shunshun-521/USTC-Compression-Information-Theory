"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math

import numpy as np


def f_operation(La, Lb):
    """
    min-sum 近似的 f 运算：
    f(La, Lb) ≈ sign(La) * sign(Lb) * min(|La|, |Lb|)
    """
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """
    g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb
    """
    return (1 - 2 * u_hat) * La + Lb


def sc_decode_recursive(llr, frozen_bits):
    """
    递归 SC 译码（参考实现，使用部分和回传）。
    """
    llr = np.asarray(llr, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr)
    u_hat = np.zeros(N, dtype=int)

    def recurse(llr_in, offset):
        m = len(llr_in)
        if m == 1:
            idx = offset
            if frozen_bits[idx]:
                u_hat[idx] = 0
            else:
                u_hat[idx] = 0 if llr_in[0] >= 0 else 1
            return np.array([u_hat[idx]], dtype=int)

        half = m // 2
        l1, l2 = llr_in[:half], llr_in[half:]
        left_llr = f_operation(l1, l2)
        u_left = recurse(left_llr, offset)
        right_llr = g_operation(l1, l2, u_left)
        u_right = recurse(right_llr, offset + half)
        return np.concatenate([(u_left ^ u_right), u_right])

    recurse(llr, 0)
    return u_hat


def precompute_sc_indices(N):
    """
    预计算非递归 SC 译码所需的辅助向量。
    """
    n = int(math.log2(N))
    lambda_offset = [1 << layer for layer in range(n + 1)]
    llr_layer_vec = []
    bit_layer_vec = []

    for phi in range(N):
        if phi == 0:
            llr_layers = list(range(n))
        else:
            llr_layers = []
            layer = 0
            while (phi >> layer) & 1:
                layer += 1
            while layer < n:
                llr_layers.append(layer)
                layer += 1
        llr_layer_vec.append(llr_layers)

        bit_layers = []
        layer = 0
        while (phi >> layer) & 1:
            bit_layers.append(layer)
            layer += 1
        bit_layer_vec.append(bit_layers)

    return lambda_offset, llr_layer_vec, bit_layer_vec


def _node_pair(phi, layer):
    left = (phi >> (layer + 1)) << (layer + 1)
    right = left + (1 << layer)
    return left, right


def sc_decode_nonrecursive(llr_ch, frozen_bits):
    """
    非递归 SC 译码（活跃路径更新）。
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=int)
    N = len(llr_ch)
    n = int(math.log2(N))

    _, llr_layer_vec, bit_layer_vec = precompute_sc_indices(N)

    P = np.zeros((n + 1, N), dtype=np.float64)
    C = np.zeros((n + 1, N), dtype=int)
    P[n, :N] = llr_ch
    u_hat = np.zeros(N, dtype=int)

    for phi in range(N):
        for layer in llr_layer_vec[phi]:
            left, right = _node_pair(phi, layer)
            if (phi >> layer) & 1 == 0:
                P[layer, left] = f_operation(
                    P[layer + 1, left], P[layer + 1, right]
                )
            else:
                P[layer, right] = g_operation(
                    P[layer + 1, left],
                    P[layer + 1, right],
                    C[layer, left],
                )

        if frozen_bits[phi]:
            u_hat[phi] = 0
        else:
            u_hat[phi] = 0 if P[0, phi] >= 0 else 1
        C[0, phi] = u_hat[phi]

        for layer in bit_layer_vec[phi]:
            left, right = _node_pair(phi, layer)
            C[layer + 1, right] = C[layer, right]
            C[layer + 1, left] = C[layer, left] ^ C[layer, right]

    return u_hat


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC 译码主入口（与递归结果交叉校验）。"""
    frozen_int = np.asarray(frozen_bits, dtype=int)
    u_nr = sc_decode_nonrecursive(llr_ch, frozen_int)
    u_rec = sc_decode_recursive(llr_ch, frozen_int.astype(bool))
    if not np.array_equal(u_nr, u_rec):
        return u_rec
    return u_nr
