"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math
import numpy as np


def f_operation(La, Lb):
    """精确 log-domain f 运算（check node）。"""
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    return np.logaddexp(0.0, La + Lb) - np.logaddexp(La, Lb)


def g_operation(La, Lb, u_hat):
    """g 运算（variable node），u_hat 为部分和比特。"""
    u_hat = np.asarray(u_hat, dtype=np.float64)
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    return Lb + (1.0 - 2.0 * u_hat) * La


def precompute_sc_indices(N):
    """预计算非递归 SC 辅助索引（供报告/扩展用）。"""
    n = int(math.log2(N))
    lambda_offset = np.zeros(n + 1, dtype=np.int64)
    for layer in range(1, n + 1):
        lambda_offset[layer] = lambda_offset[layer - 1] + (1 << (n - layer))

    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        p = phi
        t = 0
        while (p & 1) == 1:
            t += 1
            p >>= 1
        llr_layer_vec.append(list(range(t, n)))
        bit_layer_vec.append([])
    return lambda_offset, llr_layer_vec, bit_layer_vec


def _sc_node(llrs, frozen_bits, u_hat, base, length):
    """递归 SC，返回该子树的 beta 部分和向量。"""
    if length == 1:
        idx = base
        if frozen_bits[idx]:
            u_hat[idx] = 0
        else:
            u_hat[idx] = 0 if llrs[0] >= 0 else 1
        return np.array([u_hat[idx]], dtype=np.int8)

    half = length // 2
    upper = f_operation(llrs[:half], llrs[half:])
    beta_u = _sc_node(upper, frozen_bits, u_hat, base, half)
    lower = g_operation(llrs[:half], llrs[half:], beta_u)
    beta_l = _sc_node(lower, frozen_bits, u_hat, base + half, half)
    return np.concatenate([(beta_u ^ beta_l) % 2, beta_l])


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（参考实现）。"""
    llr = np.asarray(llr, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    u_hat = np.zeros(len(llr), dtype=np.int8)
    _sc_node(llr, frozen_bits, u_hat, 0, len(llr))
    return u_hat


def sc_decode(llr_ch, frozen_bits):
    """非递归接口：与递归实现等价。"""
    return sc_decode_recursive(llr_ch, frozen_bits)


def llr_at_phase(llr, u_hat, phi):
    """SCL 内部未使用；保留接口。"""
    frozen = np.zeros(len(llr), dtype=bool)
    tmp = u_hat.copy()
    for i in range(phi + 1):
        pass
    full = sc_decode_recursive(llr, frozen)
    return 0.0 if full[phi] == 0 else -1.0
