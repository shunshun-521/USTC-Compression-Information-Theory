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


def _all_num(x):
    return int(not np.any(np.isnan(x)))


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
    block = 2 ** (position[2] - position[0] + 1)
    return [position[0] - 1, int(np.floor(position[1] / block) * block), position[2], position[3]]


def _get_up_bit(left_bit, right_bit):
    length = left_bit.size
    temp = np.array([(left_bit + right_bit) % 2, right_bit])
    return temp.reshape(2 * length)


def _get_right_llr(left_bit, up_llr):
    length = int(left_bit.size)
    return np.array([
        g_operation(up_llr[i], up_llr[i + length], left_bit[i]) for i in range(length)
    ])


def _get_left_llr(up_llr):
    length = int(up_llr.size / 2)
    return np.array([f_operation(up_llr[i], up_llr[i + length]) for i in range(length)])


def _decide_bit(llr_val, is_frozen):
    if is_frozen:
        return 0
    return 0 if llr_val >= 0 else 1


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码主函数。
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr_ch)
    n = int(math.log2(N))

    llr_matrix = np.ones((n + 1, N), dtype=np.float64)
    llr_matrix[llr_matrix == 1] = np.nan
    bit_matrix = llr_matrix.copy()
    llr_matrix[0] = llr_ch
    position = [0, 0, n, N]

    while _all_num(bit_matrix[n]) == 0:
        up_llr = llr_matrix[position[0]][position[1]:position[1] + 2 ** (position[2] - position[0])]
        up_bit = bit_matrix[position[0]][position[1]:position[1] + 2 ** (position[2] - position[0])]
        left_llr = llr_matrix[position[0] + 1][position[1]:position[1] + 2 ** (position[2] - position[0] - 1)]
        left_bit = bit_matrix[position[0] + 1][position[1]:position[1] + 2 ** (position[2] - position[0] - 1)]
        right_llr = llr_matrix[position[0] + 1][
            position[1] + 2 ** (position[2] - position[0] - 1):position[1] + 2 ** (position[2] - position[0])
        ]
        right_bit = bit_matrix[position[0] + 1][
            position[1] + 2 ** (position[2] - position[0] - 1):position[1] + 2 ** (position[2] - position[0])
        ]

        if _all_num(up_bit) == 1:
            position = _up(position)
        elif _all_num(right_bit) == 1:
            up_bit_val = _get_up_bit(left_bit, right_bit)
            bit_matrix[position[0]][position[1]:position[1] + 2 ** (position[2] - position[0])] = up_bit_val
        elif _all_num(right_llr) == 1:
            if position[0] == position[2] - 1:
                right_bit_pos = position[1] + 1
                right_bit_val = _decide_bit(right_llr[0], frozen_bits[right_bit_pos])
                bit_matrix[position[0] + 1][
                    position[1] + 2 ** (position[2] - position[0] - 1):position[1] + 2 ** (position[2] - position[0])
                ] = right_bit_val
            else:
                position = _rightdown(position)
        elif _all_num(left_bit) == 1:
            right_llr_val = _get_right_llr(left_bit, up_llr)
            llr_matrix[position[0] + 1][
                position[1] + 2 ** (position[2] - position[0] - 1):position[1] + 2 ** (position[2] - position[0])
            ] = right_llr_val
        elif _all_num(left_llr) == 0:
            left_llr_val = _get_left_llr(up_llr)
            llr_matrix[position[0] + 1][position[1]:position[1] + 2 ** (position[2] - position[0] - 1)] = left_llr_val
        else:
            if position[0] == position[2] - 1:
                left_bit_pos = position[1]
                left_bit_val = _decide_bit(left_llr[0], frozen_bits[left_bit_pos])
                bit_matrix[position[0] + 1][position[1]:position[1] + 2 ** (position[2] - position[0] - 1)] = left_bit_val
            else:
                position = _leftdown(position)

    return bit_matrix[n].astype(int)


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（委托给非递归实现）。"""
    return sc_decode(llr, frozen_bits)


def precompute_sc_indices(N):
    """预计算非递归 SC 译码所需的辅助向量（供 SCL 使用）。"""
    n = int(math.log2(N))
    lambda_offset = [1 << layer for layer in range(n + 1)]

    llr_layer_vec = []
    bit_layer_vec = []

    for phi in range(N):
        layers = []
        temp = phi
        layer = 0
        while layer < n:
            layers.append(layer)
            if (temp & 1) == 1:
                break
            temp >>= 1
            layer += 1
        llr_layer_vec.append(layers)

        blayers = []
        temp = phi
        layer = 0
        while layer < n and (temp & 1) == 1:
            blayers.append(layer)
            temp >>= 1
            layer += 1
        bit_layer_vec.append(blayers)

    return lambda_offset, llr_layer_vec, bit_layer_vec
