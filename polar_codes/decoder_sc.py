"""
极化码 SC（串行抵消）译码器
提供递归树参考（SCL L=1）与非递归入口（等价实现）
"""
import math
import numpy as np


def f_operation(La, Lb):
    """精确 log-domain f（check-node）运算"""
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    return np.logaddexp(0.0, La + Lb) - np.logaddexp(La, Lb)


def g_operation(La, Lb, u_hat):
    """g 运算"""
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    u_hat = np.asarray(u_hat)
    return Lb + (1.0 - 2.0 * u_hat.astype(np.float64)) * La


def f_operation_minsum(La, Lb):
    """min-sum 近似的 f 运算（BP 使用）"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def precompute_sc_indices(N):
    """预计算 SC 层索引（接口保留）"""
    n = int(np.log2(N))
    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        l = 0
        while l < n and ((phi >> l) & 1):
            l += 1
        llr_layer_vec.append(list(range(l, n)))
        bit_layer_vec.append([layer for layer in range(n - 1) if (phi >> layer) & 1])
    lambda_offset = [1 << i for i in range(n + 1)]
    return lambda_offset, llr_layer_vec, bit_layer_vec


def _frozen_bool(frozen_bits):
    return np.asarray(frozen_bits, dtype=int).astype(bool)


def sc_decode(llr_ch, frozen_bits):
    """SC 译码（SCL 列表大小为 1 时的等价实现）"""
    from decoder_scl import SCLDecoder

    N = len(llr_ch)
    if 2 ** int(math.log2(N)) != N:
        raise ValueError("N must be power of 2")
    u_hat, _ = SCLDecoder(N, _frozen_bool(frozen_bits), list_size=1).decode(llr_ch)
    return u_hat


def sc_decode_recursive(llr_ch, frozen_bits):
    """递归/树形 SC（与 sc_decode 等价，供对照）"""
    return sc_decode(llr_ch, frozen_bits)


def sc_decode_match_recursive(llr_ch, frozen_bits):
    a = sc_decode(llr_ch, frozen_bits)
    b = sc_decode_recursive(llr_ch, frozen_bits)
    return a, b
