"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现，与 SCL L=1 等价）
"""
import numpy as np
import math


from utils_llr import llr_to_decoder_order as _llr_to_decoder_order


def _sgn(x):
    return np.where(x >= 0, 1.0, -1.0)


def f_operation(La, Lb):
    """min-sum 近似的 f 运算（BP 等模块复用）"""
    return _sgn(La) * _sgn(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def f_exact(La, Lb):
    """精确 log-domain f（SC/SCL 主路径）"""
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    return np.logaddexp(0.0, La + Lb) - np.logaddexp(La, Lb)


def g_operation(La, Lb, u_hat):
    """g 运算"""
    u = np.asarray(u_hat, dtype=np.float64)
    return np.asarray(Lb, dtype=np.float64) + (1.0 - 2.0 * u) * np.asarray(La, dtype=np.float64)


def _penalty(llr: float, bit: int) -> float:
    return float(np.logaddexp(0.0, -(1.0 - 2.0 * bit) * llr))


def sc_decode_recursive(llr_ch, frozen_bits):
    """递归 SC 译码（参考实现，与 CommPy 树遍历一致）"""
    llr = _llr_to_decoder_order(llr_ch)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr)
    u_hat = np.zeros(N, dtype=int)

    def leaf(llr_vec, index):
        if frozen_bits[index]:
            u_hat[index] = 0
        else:
            u_hat[index] = 0 if llr_vec[0] >= 0 else 1

    def node(llr_vec, base, length):
        if length == 1:
            leaf(llr_vec, base)
            return np.array([u_hat[base]], dtype=np.int8)
        half = length // 2
        upper = f_exact(llr_vec[:half], llr_vec[half:])
        beta_u = node(upper, base, half)
        lower = g_operation(llr_vec[:half], llr_vec[half:], beta_u)
        beta_l = node(lower, base + half, half)
        return np.concatenate([beta_u ^ beta_l, beta_l])

    node(llr, 0, N)
    return u_hat


def precompute_sc_indices(N):
    """预计算非递归 SC 辅助向量（Tal 快速 SC，与递归版本等价）"""
    n = int(math.log2(N))
    lambda_offset = np.zeros(n + 1, dtype=np.int32)
    for i in range(1, n + 1):
        lambda_offset[i] = lambda_offset[i - 1] + (1 << (i - 1))

    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        llr_layers = []
        p = phi
        while p & 1:
            llr_layers.append(int(math.log2(p & -p)))
            p >>= 1
        llr_layer_vec.append(llr_layers)

        bit_layers = []
        if (phi & 1) == 0:
            p = phi
            while (p & 1) == 0 and p > 0:
                bit_layers.append(int(math.log2(p & -p)))
                p >>= 1
        bit_layer_vec.append(bit_layers)

    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC：委托 SCL L=1（高效且已验证）"""
    from decoder_scl import SCLDecoder

    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    dec = SCLDecoder(len(llr_ch), frozen_bits, list_size=1, crc_length=0)
    u_hat, _ = dec.decode(llr_ch)
    return u_hat
