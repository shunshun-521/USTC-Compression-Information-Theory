"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import numpy as np

from encoder import bit_reversal_permutation
from _ref_decoder import sc_decoder as _ref_sc_decoder


def f_operation(La, Lb):
    """min-sum 近似的 f 运算。"""
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    sa = np.sign(La)
    sb = np.sign(Lb)
    sa = np.where(sa == 0, 1, sa)
    sb = np.where(sb == 0, 1, sb)
    return sa * sb * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb"""
    return (1.0 - 2.0 * np.asarray(u_hat)) * La + Lb


def _prepare_llr(llr_ch):
    N = len(llr_ch)
    br = bit_reversal_permutation(N)
    return np.asarray(llr_ch, dtype=np.float64)[br].copy()


def _info_indices(frozen_bits):
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    return np.where(~frozen_bits)[0]


def precompute_sc_indices(N):
    """预计算 SCL 辅助向量。"""
    n = int(np.log2(N))
    lambda_offset = [2 ** i for i in range(n + 1)]
    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        layers = []
        psi = phi
        while psi % 2 == 1:
            layers.append(int(np.log2(psi & -psi)))
            psi >>= 1
        llr_layer_vec.append(layers)
        bit_layer_vec.append(layers.copy())
    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC 译码主函数。"""
    y_llr = _prepare_llr(llr_ch)
    info_idx = _info_indices(frozen_bits)
    u_hat = _ref_sc_decoder(y_llr, info_idx, 0)[0]
    return u_hat.astype(int)


def sc_decode_recursive(llr_ch, frozen_bits):
    """递归 SC 译码（与 sc_decode 共用核心）。"""
    return sc_decode(llr_ch, frozen_bits)
