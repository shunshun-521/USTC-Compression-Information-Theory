"""
极化码 SC（串行抵消）译码器
"""
import math
import numpy as np

from encoder import polar_encode


def f_operation_minsum(La, Lb):
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    sa = np.sign(La)
    sb = np.sign(Lb)
    sa = np.where(sa == 0, 1.0, sa)
    sb = np.where(sb == 0, 1.0, sb)
    return sa * sb * np.minimum(np.abs(La), np.abs(Lb))


def f_operation(La, Lb):
    return f_operation_minsum(La, Lb)


def g_operation(La, Lb, u_hat):
    u_hat = np.asarray(u_hat)
    return (1.0 - 2.0 * u_hat) * La + Lb


def _frozen_bool(frozen_bits):
    fb = np.asarray(frozen_bits)
    if fb.dtype == bool:
        return fb
    return fb.astype(np.int8) != 0


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC（列表式半分，与蝶形编码器匹配）"""
    llr = np.asarray(llr, dtype=np.float64)
    frozen_bits = _frozen_bool(frozen_bits)
    frozen_list = frozen_bits.tolist()

    def _dec(llr_list, frozen_sub):
        n = int(math.log2(len(llr_list)))
        if n == 0:
            return [0 if frozen_sub[0] else (0 if llr_list[0] >= 0 else 1)]
        half = 1 << (n - 1)
        left = [
            f_operation(llr_list[i], llr_list[i + half]) for i in range(half)
        ]
        u_left = _dec(left, frozen_sub[:half])
        right = [
            g_operation(llr_list[i], llr_list[i + half], u_left[i]) for i in range(half)
        ]
        u_right = _dec(right, frozen_sub[half:])
        return u_left + u_right

    return np.array(_dec(llr.tolist(), frozen_list), dtype=np.int8)


def precompute_sc_indices(N):
    n = int(math.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]
    llr_layer_vec = []
    for phi in range(N):
        layers = []
        p = phi
        while (p & 1) == 1:
            layers.append(int(math.log2(p & -p)))
            p >>= 1
        llr_layer_vec.append(layers)
    bit_layer_vec = []
    for phi in range(N):
        layers = []
        for layer in range(n):
            if (phi >> layer) & 1:
                layers.append(layer)
        bit_layer_vec.append(layers)
    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode_nonrecursive(llr_ch, frozen_bits):
    return sc_decode_recursive(llr_ch, frozen_bits)


def sc_decode(llr_ch, frozen_bits):
    return sc_decode_recursive(llr_ch, frozen_bits)
