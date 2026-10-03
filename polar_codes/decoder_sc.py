"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）与非递归入口（委托递归树遍历）
"""
import numpy as np


def f_operation(La, Lb):
    """精确 log-domain f（check-node）运算。"""
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    return np.logaddexp(0.0, La + Lb) - np.logaddexp(La, Lb)


def g_operation(La, Lb, u_beta):
    """g 运算；u_beta 为子树返回的 beta 部分和向量（与 La 等长）。"""
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    u = np.asarray(u_beta, dtype=np.float64)
    return Lb + (1.0 - 2.0 * u) * La


def _sc_node(llr, frozen_bits, base, length):
    """
    递归 SC 子树译码。
    返回 (u_hat_segment, beta) 其中 beta 为 g 运算所需的部分和结构。
    """
    if length == 1:
        idx = base
        if frozen_bits[idx]:
            bit = 0
        else:
            bit = 0 if llr[0] >= 0 else 1
        u_seg = np.array([bit], dtype=int)
        beta = np.array([bit], dtype=int)
        return u_seg, beta

    half = length // 2
    llr_upper = f_operation(llr[:half], llr[half:])
    u_left, beta_left = _sc_node(llr_upper, frozen_bits, base, half)
    llr_lower = g_operation(llr[:half], llr[half:], beta_left)
    u_right, beta_right = _sc_node(llr_lower, frozen_bits, base + half, half)
    u_seg = np.concatenate([u_left, u_right])
    beta = np.concatenate([beta_left ^ beta_right, beta_right])
    return u_seg, beta


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码。"""
    llr = np.asarray(llr, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    u_hat, _ = _sc_node(llr, frozen_bits, 0, len(llr))
    return u_hat


def precompute_sc_indices(N):
    """保留接口：非递归 SCD 预计算（供 SCL 层索引参考）。"""
    import math

    n = int(math.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]
    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        llr_layers = []
        pp = phi
        while pp % 2 == 1:
            llr_layers.append(int(math.log2(pp & -pp)) - 1)
            pp >>= 1
        llr_layer_vec.append(llr_layers)
        bit_layers = []
        if phi % 2 == 1:
            pp = phi
            while pp % 2 == 1:
                bit_layers.append(int(math.log2(pp & -pp)) - 1)
                pp >>= 1
        bit_layer_vec.append(bit_layers)
    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode_nonrecursive(llr, frozen_bits):
    """非递归入口（当前与递归实现等价，便于对照验证）。"""
    return sc_decode_recursive(llr, frozen_bits)


def sc_decode(llr_ch, frozen_bits):
    """SC 译码主入口。"""
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    return sc_decode_recursive(np.asarray(llr_ch, dtype=np.float64), frozen_bits)


def sc_decode_recursive_natural(llr_ch, frozen_bits):
    return sc_decode(llr_ch, frozen_bits)
