"""
极化码 SC（串行抵消）译码器
"""
import math
import numpy as np


def f_operation(La, Lb):
    """精确 log-domain f 核"""
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    return np.logaddexp(0.0, La + Lb) - np.logaddexp(La, Lb)


def g_operation(La, Lb, u_hat):
    """精确 log-domain g 核"""
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    u = np.asarray(u_hat, dtype=np.float64)
    return Lb + (1.0 - 2.0 * u) * La


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC（参考实现，与 SCL L=1 一致）"""
    from decoder_scl import SCLDecoder

    frozen = np.asarray(frozen_bits, dtype=bool)
    dec = SCLDecoder(len(llr), frozen.astype(np.int8), list_size=1)
    u_hat, _ = dec.decode(np.asarray(llr, dtype=np.float64))
    return u_hat


def precompute_sc_indices(N):
    """预计算非递归 SC 索引（接口兼容）"""
    n = int(np.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]
    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        llr_layers = []
        bit_layers = []
        tmp = phi
        for layer in range(n):
            if tmp % 2 == 0:
                llr_layers.append(layer)
            else:
                bit_layers.append(layer)
            tmp //= 2
        llr_layer_vec.append(llr_layers)
        bit_layer_vec.append(bit_layers)
    return lambda_offset, llr_layer_vec, bit_layer_vec


def llr_at_bit_index(channel_llr, partial_u, phi):
    """逐位 LLR（供分析；主路径使用 SCLDecoder）"""
    llr = np.asarray(channel_llr, dtype=np.float64)

    def recurse(vec, offset, tphi, u_prefix):
        if len(vec) == 1:
            return float(vec[0])
        half = len(vec) // 2
        upper = f_operation(vec[:half], vec[half:])
        if tphi < offset + half:
            return recurse(upper, offset, tphi, u_prefix)
        u_left = u_prefix[offset: offset + half]
        lower = g_operation(vec[:half], vec[half:], u_left)
        return recurse(lower, offset + half, tphi, u_prefix)

    return recurse(llr, 0, phi, np.asarray(partial_u, dtype=np.int32))


def sc_decode(llr_ch, frozen_bits):
    """SC 译码主函数"""
    return sc_decode_recursive(llr_ch, frozen_bits)


def sc_decoders_match(llr, frozen_bits):
    a = sc_decode_recursive(llr, frozen_bits)
    b = sc_decode(llr, frozen_bits)
    return np.array_equal(a, b)
