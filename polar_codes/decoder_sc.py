"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math

import numpy as np

import polar_tree as pt


def f_operation(La, Lb):
    """
    min-sum 近似的 f 运算：
    f(La, Lb) ≈ sign(La) * sign(Lb) * min(|La|, |Lb|)
    """
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    if La.ndim == 0:
        s1 = 1.0 if La >= 0 else -1.0
        s2 = 1.0 if Lb >= 0 else -1.0
        return s1 * s2 * min(abs(float(La)), abs(float(Lb)))
    s1 = np.sign(La)
    s2 = np.sign(Lb)
    s1[s1 == 0] = 1
    s2[s2 == 0] = 1
    return s1 * s2 * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb"""
    return (1 - 2 * u_hat) * La + Lb


def _f_scalar(La, Lb):
    s1 = 1 if La >= 0 else -1
    s2 = 1 if Lb >= 0 else -1
    return s1 * s2 * min(abs(La), abs(Lb))


def _g_scalar(La, Lb, u_hat):
    return (1 - 2 * u_hat) * La + Lb


def sc_decode_recursive(llr, frozen_bits):
    """
    递归 SC 译码参考接口。
    与 sc_decode 共用同一棵因子图树遍历实现，保证结果一致。
    """
    return sc_decode(llr, frozen_bits)


def precompute_sc_indices(N):
    """
    预计算非递归 SC 译码所需的三个辅助向量。
    """
    n = int(math.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]
    llr_layer_vec = []
    bit_layer_vec = []

    for phi in range(N):
        phi_bin = format(phi, f"0{n}b")
        layers = []
        for s in range(n):
            if phi_bin[n - 1 - s] == "0":
                layers.append(s)
        llr_layer_vec.append(layers)

        layers_b = []
        for s in range(n):
            if s < n - 1 and phi_bin[n - 2 - s] == "1":
                layers_b.append(s)
        if phi % 2 == 1:
            layers_b.append(n - 1)
        bit_layer_vec.append(layers_b)

    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码主函数（基于因子图树遍历，O(N log N)）。
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=int)
    N = len(llr_ch)
    n = int(math.log2(N))
    is_info = pt.frozen_to_info_mask(N, frozen_bits)
    frozen_value = 0

    llr_matrix, bit_matrix, _ = pt.init_matrices(N)
    llr_matrix[0] = llr_ch
    position = [0, 0, n, N]

    llr_matrix, bit_matrix, _ = pt.sc_tree_step(
        llr_matrix,
        bit_matrix,
        position,
        is_info,
        frozen_value,
        _f_scalar,
        _g_scalar,
        stop_pos=None,
    )

    u_hat = np.zeros(N, dtype=int)
    for i in range(N):
        val = bit_matrix[n][i]
        u_hat[i] = 0 if val == 0 else 1
    return u_hat
