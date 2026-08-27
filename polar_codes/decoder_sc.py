"""
极化码 SC（串行抵消）译码器
"""
import math
import numpy as np
from polar_core import SCDecoder as _SCDecoder


def f_operation(La, Lb):
    """min-sum f 运算"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算"""
    return (1.0 - 2.0 * u_hat) * La + Lb


def sc_decode(llr_ch, frozen_bits):
    """
    SC 译码。
    frozen_bits: 1=冻结位, 0=信息位
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float32)
    frozen_bits = np.asarray(frozen_bits, dtype=int)
    if_info = np.zeros(len(llr_ch), dtype=np.int32)
    if_info[np.where(frozen_bits == 0)[0]] = 1
    return _SCDecoder(llr_ch, if_info).astype(int)


def sc_decode_recursive(llr, frozen_bits):
    return sc_decode(llr, frozen_bits)


def precompute_sc_indices(N):
    n = int(math.log2(N))
    lambda_offset = [(1 << layer) - 1 for layer in range(n + 1)]
    llr_layer_vec = [list(range(n)) for _ in range(N)]
    bit_layer_vec = [list(range(n)) for _ in range(N)]
    return lambda_offset, llr_layer_vec, bit_layer_vec
