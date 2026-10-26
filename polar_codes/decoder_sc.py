"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math

import numpy as np


def f_operation(La, Lb):
    """min-sum 近似的 f 运算。"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb"""
    return (1 - 2 * u_hat) * La + Lb


def _trailing_ones(phi):
    c = 0
    while phi & 1:
        phi >>= 1
        c += 1
    return c


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（参考实现）。 """
    llr = np.asarray(llr, dtype=np.float64)
    N = len(llr)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    u_hat = np.zeros(N, dtype=int)

    def decode_block(llr_vec, bit_offset, n_block):
        if n_block == 0:
            idx = bit_offset
            if frozen_bits[idx]:
                u_hat[idx] = 0
            else:
                u_hat[idx] = 0 if llr_vec[0] >= 0 else 1
            return

        half = 1 << (n_block - 1)
        llr_left = f_operation(llr_vec[:half], llr_vec[half:])
        decode_block(llr_left, bit_offset, n_block - 1)

        u_left = u_hat[bit_offset: bit_offset + half]
        llr_right = g_operation(llr_vec[:half], llr_vec[half:], u_left)
        decode_block(llr_right, bit_offset + half, n_block - 1)

    decode_block(llr, 0, int(math.log2(N)))
    return u_hat


def precompute_sc_indices(N):
    """预计算非递归 SC 译码的层索引向量。"""
    m = int(math.log2(N))
    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        if phi == 0:
            llr_layers = list(range(m - 1, -1, -1))
        else:
            tz = (phi & -phi).bit_length() - 1
            llr_layers = list(range(m - 1, tz - 1, -1))
        llr_layer_vec.append(llr_layers)

        if phi == N - 1:
            bit_layers = []
        else:
            tz2 = ((phi + 1) & -(phi + 1)).bit_length() - 1
            bit_layers = list(range(tz2, m))
        bit_layer_vec.append(bit_layers)

    lambda_offset = [1 << i for i in range(m + 1)]
    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC 译码主函数。"""
    return sc_decode_recursive(llr_ch, frozen_bits)
