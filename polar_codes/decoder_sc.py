"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math

import numpy as np


def f_operation(La, Lb):
    """
    min-sum 近似的 f 运算：
    f(La, Lb) ≈ sign(La) * sign(Lb) * min(|La|, |Lb|)
    """
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb"""
    u_hat = np.asarray(u_hat)
    return (1 - 2 * u_hat) * La + Lb


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（通过 SCL L=1 实现，与列表译码一致）。"""
    from decoder_scl import SCLDecoder

    N = len(llr)
    u_hat, _ = SCLDecoder(N, frozen_bits, list_size=1, crc_length=0).decode(llr)
    return u_hat


def precompute_sc_indices(N):
    """
    预计算非递归 SC 译码所需的三个辅助向量。
    """
    n = int(math.log2(N))
    lambda_offset = list(range(N))
    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        layers_llr = []
        tmp = phi
        l = 0
        while (tmp & 1) == 1 and l < n:
            layers_llr.append(l)
            tmp >>= 1
            l += 1
        if l < n:
            layers_llr.append(l)
        llr_layer_vec.append(layers_llr)

        layers_bit = []
        tmp = phi
        l = 0
        while (tmp & 1) == 1 and l < n:
            layers_bit.append(l)
            tmp >>= 1
            l += 1
        bit_layer_vec.append(layers_bit)

    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码主函数（调用 SCL L=1，等价于高效 SC）。
    """
    from decoder_scl import SCLDecoder

    N = len(llr_ch)
    u_hat, _ = SCLDecoder(N, frozen_bits, list_size=1, crc_length=0).decode(llr_ch)
    return u_hat


def sc_decode_nonrecursive_precomputed(llr_ch, frozen_bits):
    return sc_decode(llr_ch, frozen_bits)
