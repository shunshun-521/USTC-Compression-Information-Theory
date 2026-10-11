"""
极化码 SC（串行抵消）译码器
"""
import numpy as np

from sc_backend import sc_decoder as _sc_decoder_matrix


def f_operation(La, Lb):
    """min-sum 近似的 f 运算。"""
    s = np.sign(La)
    s[s == 0] = 1
    sb = np.sign(Lb)
    sb[sb == 0] = 1
    return s * sb * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    return (1.0 - 2.0 * np.asarray(u_hat)) * La + Lb


def _frozen_to_info_list(frozen_bits):
    fb = np.asarray(frozen_bits)
    if fb.dtype != bool:
        fb = fb.astype(bool)
    return np.where(~fb)[0].tolist()


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC 译码（因子图矩阵遍历实现）。"""
    info_pos = _frozen_to_info_list(frozen_bits)
    u_hat, _ = _sc_decoder_matrix(np.asarray(llr_ch, dtype=np.float64), info_pos, 0)
    return np.asarray(u_hat, dtype=int)


def sc_decode_recursive(llr, frozen_bits):
    """递归版本接口（与 sc_decode 等价）。"""
    return sc_decode(llr, frozen_bits)


def precompute_sc_indices(N):
    n = int(np.log2(N))
    return [1 << i for i in range(n + 1)], [[] for _ in range(N)], [[] for _ in range(N)]


def _precompute_sc_indices_v2(N):
    return precompute_sc_indices(N)
