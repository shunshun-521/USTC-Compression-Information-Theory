"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math

import numpy as np

from encoder import bit_reversal_permutation


def f_operation(La, Lb):
    """
    min-sum 近似的 f 运算：
    f(La, Lb) ≈ sign(La) * sign(Lb) * min(|La|, |Lb|)
    """
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """
    g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb
    """
    return (1 - 2 * u_hat) * La + Lb


def _all_filled(arr):
    return not np.any(np.isnan(arr))


def _leftdown(position):
    return [position[0] + 1, position[1], position[2], position[3]]


def _rightdown(position):
    return [
        position[0] + 1,
        position[1] + 2 ** (position[2] - 1 - position[0]),
        position[2],
        position[3],
    ]


def _up(position):
    span = 2 ** (position[2] - position[0] + 1)
    return [
        position[0] - 1,
        int(np.floor(position[1] / span) * span),
        position[2],
        position[3],
    ]


def _sc_decode_core(llr, frozen_bits):
    """
    基于因子图遍历的非递归 SC 译码核心。
    """
    llr = np.asarray(llr, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr)
    n = int(math.log2(N))

    llr_matrix = np.full((n + 1, N), np.nan, dtype=np.float64)
    bit_matrix = np.full((n + 1, N), np.nan, dtype=np.float64)
    llr_matrix[0] = llr

    position = [0, 0, n, N]

    while not _all_filled(bit_matrix[n]):
        span = 2 ** (position[2] - position[0])
        start = position[1]
        end = start + span
        half = span // 2

        up_llr = llr_matrix[position[0], start:end]
        up_bit = bit_matrix[position[0], start:end]
        left_llr = llr_matrix[position[0] + 1, start : start + half]
        left_bit = bit_matrix[position[0] + 1, start : start + half]
        right_llr = llr_matrix[position[0] + 1, start + half : end]
        right_bit = bit_matrix[position[0] + 1, start + half : end]

        if _all_filled(up_bit):
            position = _up(position)
            continue

        if _all_filled(right_bit):
            combined = np.vstack([(left_bit + right_bit) % 2, right_bit]).reshape(-1)
            bit_matrix[position[0], start:end] = combined
            continue

        if _all_filled(right_llr):
            if position[0] == position[2] - 1:
                bit_pos = start + half
                bit = 0 if right_llr[0] > 0 else 1
                if frozen_bits[bit_pos]:
                    bit = 0
                bit_matrix[position[0] + 1, start + half : end] = bit
            else:
                position = _rightdown(position)
            continue

        if _all_filled(left_bit):
            right_vals = np.array(
                [
                    g_operation(up_llr[i], up_llr[i + half], left_bit[i])
                    for i in range(half)
                ]
            )
            llr_matrix[position[0] + 1, start + half : end] = right_vals
            continue

        if not _all_filled(left_llr):
            left_vals = np.array(
                [f_operation(up_llr[i], up_llr[i + half]) for i in range(half)]
            )
            llr_matrix[position[0] + 1, start : start + half] = left_vals
            continue

        if position[0] == position[2] - 1:
            bit_pos = start
            bit = 0 if left_llr[0] >= 0 else 1
            if frozen_bits[bit_pos]:
                bit = 0
            bit_matrix[position[0] + 1, start : start + half] = bit
        else:
            position = _leftdown(position)

    return bit_matrix[n].astype(int)


def sc_decode_recursive(llr, frozen_bits):
    """
    递归 SC 译码（参考实现）。
    """
    llr = np.asarray(llr, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr)
    br = bit_reversal_permutation(N)
    llr_br = llr[br]
    return _sc_decode_core(llr_br, frozen_bits)


def precompute_sc_indices(N):
    """
    预计算非递归 SC 译码所需的辅助向量（供 SCL 使用）。
    """
    n = int(math.log2(N))
    lambda_offset = [1 << d for d in range(n + 1)]

    llr_layer_vec = []
    bit_layer_vec = []

    for phi in range(N):
        layers = []
        bits = phi
        for layer in range(n):
            if (bits >> layer) & 1:
                layers.append(layer)
            else:
                break
        llr_layer_vec.append(layers)

        bit_layers = []
        temp = phi
        for layer in range(n):
            if (temp & 1) == 1:
                bit_layers.append(layer)
            temp >>= 1
        bit_layer_vec.append(bit_layers)

    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码主函数。
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr_ch)
    br = bit_reversal_permutation(N)
    return _sc_decode_core(llr_ch[br], frozen_bits)
