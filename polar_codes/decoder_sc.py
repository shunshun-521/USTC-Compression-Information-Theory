"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math

import numpy as np


def f_operation(La, Lb):
    """
    精确 log 域 f 运算（check node）：
    f(a,b) = logaddexp(0, a+b) - logaddexp(a, b)
    """
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    return np.logaddexp(0.0, La + Lb) - np.logaddexp(La, Lb)


def g_operation(La, Lb, u_hat):
    """g 运算：g(La, Lb, u_hat) = Lb + (1 - 2*u_hat) * La"""
    u_hat = np.asarray(u_hat, dtype=np.float64)
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    return Lb + (1.0 - 2.0 * u_hat) * La


def _channel_llr_to_decode_order(llr_ch):
    """信道 LLR 与编码器输出一一对应，无需重排。"""
    return np.asarray(llr_ch, dtype=np.float64)


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（与 sc_decode 等价，保留接口）。"""
    return sc_decode(llr, frozen_bits)


def precompute_sc_indices(N):
    """预计算 SCL 辅助索引。"""
    n = int(math.log2(N))
    lambda_offset = [(1 << layer) - 1 for layer in range(n + 1)]

    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        layers_llr = []
        p = phi
        for layer in range(n):
            if (p & 1) == 0:
                layers_llr.append(layer)
            p >>= 1
        llr_layer_vec.append(layers_llr)

        layers_bit = []
        p = phi
        for layer in range(n):
            if (p & 1) == 1:
                layers_bit.append(layer)
            p >>= 1
        bit_layer_vec.append(layers_bit)

    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC 译码（SCL 列表大小为 1 的高效树实现）。"""
    from decoder_scl import SCLDecoder

    N = len(llr_ch)
    u_hat, _ = SCLDecoder(N, frozen_bits, list_size=1, crc_length=0).decode(llr_ch)
    return u_hat
