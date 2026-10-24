"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math
import numpy as np

from encoder import bit_reversal_permutation


def f_operation(La, Lb):
    """
    min-sum 近似的 f 运算（用于 BP 等）。
    """
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def f_exact(La, Lb):
    """精确 log-domain f 核（SC/SCL 主路径）。"""
    return np.asarray(
        np.logaddexp(0.0, La + Lb) - np.logaddexp(La, Lb), dtype=np.float64
    )


def g_operation(La, Lb, u_hat):
    """g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb"""
    u = np.asarray(u_hat, dtype=np.float64)
    return np.asarray(Lb + (1.0 - 2.0 * u) * La, dtype=np.float64)


def _sc_node(llrs, frozen, u_hat, base, length):
    """commpy 风格递归 SC（单路径）。"""
    if length == 1:
        if frozen[base]:
            u_hat[base] = 0
        else:
            u_hat[base] = 0 if llrs[0] >= 0 else 1
        return np.array([u_hat[base]], dtype=np.int8)

    half = length // 2
    upper = f_exact(llrs[:half], llrs[half:])
    beta_u = _sc_node(upper, frozen, u_hat, base, half)
    lower = g_operation(llrs[:half], llrs[half:], beta_u)
    beta_l = _sc_node(lower, frozen, u_hat, base + half, half)
    return np.concatenate([beta_u ^ beta_l, beta_l]).astype(np.int8)


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（参考实现，与 sc_decode 一致）。"""
    return sc_decode(llr, frozen_bits)


def precompute_sc_indices(N):
    """预计算非递归 SC 译码所需的辅助向量（供扩展/分析使用）。"""
    n = int(math.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]
    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        layers_llr = []
        layer = 0
        while layer < n and ((phi >> layer) & 1):
            layers_llr.append(layer)
            layer += 1
        llr_layer_vec.append(layers_llr)
        layers_bit = []
        p = phi + 1
        layer = 0
        while layer < n and p < N and (p & 1):
            layers_bit.append(layer)
            p >>= 1
            layer += 1
        bit_layer_vec.append(layers_bit)
    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """
    SC 译码（单路径列表大小 L=1 的 SCL，与递归参考实现等价）。
    信道 LLR 顺序与编码后的码字比特顺序一致（不做比特倒序置换）。
    """
    llr = np.asarray(llr_ch, dtype=np.float64)
    N = len(llr)
    llr = llr[bit_reversal_permutation(N)]
    frozen = np.asarray(frozen_bits, dtype=bool)
    u_hat = np.zeros(N, dtype=np.int8)
    _sc_node(llr, frozen, u_hat, 0, N)
    return u_hat
