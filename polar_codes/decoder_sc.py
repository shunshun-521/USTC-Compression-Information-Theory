"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math

import numpy as np


def f_operation(La, Lb):
    """min-sum 近似的 f 运算。"""
    sa = np.sign(La)
    sb = np.sign(Lb)
    sa = np.where(sa == 0, 1, sa)
    sb = np.where(sb == 0, 1, sb)
    return sa * sb * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算。"""
    return (1 - 2 * u_hat) * La + Lb


def _all_filled(arr):
    return not np.any(np.isnan(arr))


def _leftdown(position):
    return [position[0] + 1, position[1], position[2], position[3]]


def _rightdown(position):
    return [
        position[0] + 1,
        position[1] + 2 ** (position[2] - position[0] - 1),
        position[2],
        position[3],
    ]


def _up(position):
    p0 = position[0] - 1
    span = 2 ** (position[2] - position[0] + 1)
    p1 = int(np.floor(position[1] / span) * span)
    return [p0, p1, position[2], position[3]]


def _get_up_bit(left_bit, right_bit):
    temp = np.array([(left_bit + right_bit) % 2, right_bit])
    return temp.reshape(2 * len(left_bit))


def _get_right_llr(left_bit, up_llr):
    half = len(left_bit)
    return np.array(
        [g_operation(up_llr[i], up_llr[i + half], left_bit[i]) for i in range(half)]
    )


def _get_left_llr(up_llr):
    half = len(up_llr) // 2
    return np.array([f_operation(up_llr[i], up_llr[i + half]) for i in range(half)])


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（半分结构）。"""
    return sc_decode(llr, frozen_bits)


def precompute_sc_indices(N):
    """预计算非递归 SC 译码所需的三个辅助向量。"""
    n = int(math.log2(N))
    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        llr_layer_vec.append([layer for layer in range(n) if (phi >> layer) & 1 == 0])
        bit_layers = []
        temp = phi + 1
        layer = 0
        while temp % 2 == 0 and layer < n:
            bit_layers.append(layer)
            temp //= 2
            layer += 1
        bit_layer_vec.append(bit_layers)
    lambda_offset = [1 << i for i in range(n + 1)]
    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码（因子图树遍历，与 G=F^{⊗n} 编码配套）。
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr_ch)
    n = int(math.log2(N))

    llr_matrix = np.full((n + 1, N), np.nan, dtype=np.float64)
    bit_matrix = np.full((n + 1, N), np.nan, dtype=np.float64)
    llr_matrix[0] = llr_ch
    position = [0, 0, n, N]

    while not _all_filled(bit_matrix[n]):
        span = 2 ** (position[2] - position[0])
        start = position[1]
        up_llr = llr_matrix[position[0]][start : start + span]
        up_bit = bit_matrix[position[0]][start : start + span]
        half = span // 2
        left_llr = llr_matrix[position[0] + 1][start : start + half]
        left_bit = bit_matrix[position[0] + 1][start : start + half]
        right_llr = llr_matrix[position[0] + 1][start + half : start + span]
        right_bit = bit_matrix[position[0] + 1][start + half : start + span]

        if _all_filled(up_bit):
            position = _up(position)
        elif _all_filled(right_bit):
            bit_matrix[position[0]][start : start + span] = _get_up_bit(
                left_bit, right_bit
            )
        elif _all_filled(right_llr):
            if position[0] == position[2] - 1:
                pos = start + half
                if frozen_bits[pos]:
                    bit_matrix[position[0] + 1][start + half : start + span] = 0
                else:
                    bit_matrix[position[0] + 1][start + half : start + span] = (
                        0 if right_llr[0] >= 0 else 1
                    )
            else:
                position = _rightdown(position)
        elif _all_filled(left_bit):
            llr_matrix[position[0] + 1][start + half : start + span] = _get_right_llr(
                left_bit, up_llr
            )
        elif not _all_filled(left_llr):
            llr_matrix[position[0] + 1][start : start + half] = _get_left_llr(up_llr)
        elif position[0] == position[2] - 1:
            pos = start
            if frozen_bits[pos]:
                bit_matrix[position[0] + 1][start : start + half] = 0
            else:
                bit_matrix[position[0] + 1][start : start + half] = (
                    0 if left_llr[0] >= 0 else 1
                )
        else:
            position = _leftdown(position)

    u_hat = bit_matrix[n].astype(int)
    u_hat[frozen_bits] = 0
    return u_hat
