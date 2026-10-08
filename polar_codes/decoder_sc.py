"""
极化码 SC（串行抵消）译码器
"""
import math
import numpy as np


def f_operation(La, Lb):
    """精确 log 域 f 运算（check node）"""
    return np.logaddexp(0.0, La + Lb) - np.logaddexp(La, Lb)


def g_operation(La, Lb, u_hat):
    """g 运算（variable node）"""
    u_hat = np.asarray(u_hat, dtype=np.float64)
    return Lb + (1.0 - 2.0 * u_hat) * La


def precompute_sc_indices(N):
    """预计算非递归 SC 辅助结构（接口兼容）"""
    n = int(math.log2(N))
    lambda_offset = [2 ** i for i in range(n + 1)]
    llr_layer_vec = [list(range(n)) if phi == 0 else [] for phi in range(N)]
    bit_layer_vec = [[] for _ in range(N)]
    return lambda_offset, llr_layer_vec, bit_layer_vec


def _penalty(llr, bit):
    return float(np.logaddexp(0.0, -(1.0 - 2.0 * bit) * llr))


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC（与 SCL L=1 等价）"""
    from decoder_scl import SCLDecoder

    u_hat, _ = SCLDecoder(len(llr), frozen_bits, list_size=1, crc_length=0).decode(llr)
    return u_hat


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC 主入口"""
    return sc_decode_recursive(llr_ch, frozen_bits)
