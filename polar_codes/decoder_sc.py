"""
极化码 SC（串行抵消）译码器
"""
import numpy as np
from decoder_scl import SCLDecoder, _f_min_sum, _f_exact, _g


def f_operation(La, Lb):
    """min-sum 近似的 f 运算"""
    return _f_min_sum(La, Lb)


def g_operation(La, Lb, u_hat):
    """g 运算"""
    return _g(La, Lb, u_hat)


def sc_decode_recursive(llr, frozen_bits, use_min_sum=False):
    """递归 SC（与 SCL L=1 等价，用于对照）"""
    dec = SCLDecoder(
        len(llr), frozen_bits, list_size=1, use_min_sum=use_min_sum
    )
    u_hat, _ = dec.decode(llr)
    return u_hat


def precompute_sc_indices(N):
    """预计算非递归 SC 索引（接口保留；当前 SC 基于 SCL L=1 实现）"""
    import math

    n = int(math.log2(N))
    lambda_offset = np.array([(1 << i) - 1 for i in range(n + 1)], dtype=int)
    llr_layer_vec = [list(range(n)) for _ in range(N)]
    bit_layer_vec = [list(range(n)) for _ in range(N)]
    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits, use_min_sum=True):
    """非递归 SC：SCL 列表大小 1"""
    dec = SCLDecoder(
        len(llr_ch), frozen_bits, list_size=1, use_min_sum=use_min_sum
    )
    u_hat, _ = dec.decode(llr_ch)
    return u_hat
