"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math
import numpy as np

from encoder import channel_llr_to_decoder


def f_operation(La, Lb, min_sum=False):
    """
    f 运算：默认精确 log-domain box-plus；可选 min-sum 近似
    """
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    if min_sum:
        return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))
    return np.logaddexp(0.0, La + Lb) - np.logaddexp(La, Lb)


def g_operation(La, Lb, u_hat):
    """g 运算"""
    u_hat = np.asarray(u_hat, dtype=np.float64)
    return np.asarray(Lb + (1.0 - 2.0 * u_hat) * La, dtype=np.float64)


def _sc_node(llr, frozen_bits, u_hat, base, length):
    """CommPy 风格递归 SC 子树译码（单路径）"""
    if length == 1:
        idx = base
        if frozen_bits[idx]:
            u_hat[idx] = 0
        else:
            u_hat[idx] = 0 if llr[0] >= 0.0 else 1
        return np.array([u_hat[idx]], dtype=np.int8)

    half = length // 2
    upper = f_operation(llr[:half], llr[half:])
    beta_u = _sc_node(upper, frozen_bits, u_hat, base, half)
    lower = g_operation(llr[:half], llr[half:], beta_u)
    beta_l = _sc_node(lower, frozen_bits, u_hat, base + half, half)
    return np.concatenate([beta_u ^ beta_l, beta_l])


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（参考实现）"""
    llr = np.asarray(llr, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    u_hat = np.zeros(len(llr), dtype=int)
    _sc_node(llr, frozen_bits, u_hat, 0, len(llr))
    return u_hat


def precompute_sc_indices(N):
    """预计算非递归 SC 辅助索引"""
    n = int(math.log2(N))
    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        layers_llr = []
        for layer in range(n - 1, -1, -1):
            if (phi >> layer) & 1 == 0:
                layers_llr.append(layer)
        llr_layer_vec.append(layers_llr)
        layers_bit = []
        for layer in range(n):
            if (phi >> layer) & 1 == 0:
                layers_bit.append(layer)
        bit_layer_vec.append(layers_bit)
    lambda_offset = [1 << i for i in range(n + 1)]
    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """
    SC 译码主入口：信道 LLR 重排 + 递归 SC（与编码 B_N F^{⊗n} 一致）
    """
    llr = channel_llr_to_decoder(llr_ch)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    return sc_decode_recursive(llr, frozen_bits)
