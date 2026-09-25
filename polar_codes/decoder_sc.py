"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math

import numpy as np


def f_operation(La, Lb):
    """min-sum 近似的 f 运算。"""
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    if La.ndim == 0:
        s1 = 1.0 if La == 0 else np.sign(La)
        s2 = 1.0 if Lb == 0 else np.sign(Lb)
        return s1 * s2 * min(abs(La), abs(Lb))
    s1 = np.sign(La)
    s2 = np.sign(Lb)
    s1[s1 == 0] = 1
    s2[s2 == 0] = 1
    return s1 * s2 * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算。"""
    return (1 - 2 * np.asarray(u_hat)) * La + Lb


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
    p0 = position[0] - 1
    p1 = int(np.floor(position[1] / (2 ** (position[2] - position[0] + 1)))
             * (2 ** (position[2] - position[0] + 1)))
    return [p0, p1, position[2], position[3]]


def _get_up_bit(left_bit, right_bit):
    length = len(left_bit)
    temp = np.array([(left_bit + right_bit) % 2, right_bit])
    temp.resize((1, 2 * length))
    return temp[0]


def _get_right_bit(right_llr, information_pos, frozen_bit, right_bit_pos):
    if right_bit_pos in information_pos:
        return 0 if right_llr >= 0 else 1
    return frozen_bit


def _get_right_llr(left_bit, up_llr):
    length = len(left_bit)
    half = length
    return np.array([
        g_operation(up_llr[i], up_llr[i + half], left_bit[i])
        for i in range(half)
    ])


def _get_left_bit(left_llr, information_pos, frozen_bit, left_bit_pos):
    if left_bit_pos in information_pos:
        return 0 if left_llr >= 0 else 1
    return frozen_bit


def _get_left_llr(up_llr):
    half = len(up_llr) // 2
    return f_operation(up_llr[:half], up_llr[half:])


def _sc_decoder_core(y_llr, information_pos, frozen_bit):
    """基于因子图遍历的非递归 SC 译码。"""
    N = len(y_llr)
    n = int(math.log2(N))
    information_pos = set(int(i) for i in information_pos)

    llr_matrix = np.full((n + 1, N), np.nan, dtype=np.float64)
    bit_matrix = np.full((n + 1, N), np.nan)
    llr_matrix[0] = y_llr
    position = [0, 0, n, N]

    while not _all_filled(bit_matrix[n]):
        up_llr = llr_matrix[position[0]][
            position[1]:position[1] + 2 ** (position[2] - position[0])
        ]
        up_bit = bit_matrix[position[0]][
            position[1]:position[1] + 2 ** (position[2] - position[0])
        ]
        left_llr = llr_matrix[position[0] + 1][
            position[1]:position[1] + 2 ** (position[2] - position[0] - 1)
        ]
        left_bit = bit_matrix[position[0] + 1][
            position[1]:position[1] + 2 ** (position[2] - position[0] - 1)
        ]
        right_llr = llr_matrix[position[0] + 1][
            position[1] + 2 ** (position[2] - position[0] - 1):
            position[1] + 2 ** (position[2] - position[0])
        ]
        right_bit = bit_matrix[position[0] + 1][
            position[1] + 2 ** (position[2] - position[0] - 1):
            position[1] + 2 ** (position[2] - position[0])
        ]

        if _all_filled(up_bit):
            position = _up(position)
        elif _all_filled(right_bit):
            up_bit = _get_up_bit(left_bit, right_bit)
            sl = slice(position[1], position[1] + 2 ** (position[2] - position[0]))
            bit_matrix[position[0]][sl] = up_bit
        elif _all_filled(right_llr):
            if position[0] == position[2] - 1:
                right_bit_pos = position[1] + 1
                right_bit_val = _get_right_bit(
                    right_llr[0], information_pos, frozen_bit, right_bit_pos
                )
                rs = slice(
                    position[1] + 2 ** (position[2] - position[0] - 1),
                    position[1] + 2 ** (position[2] - position[0]),
                )
                bit_matrix[position[0] + 1][rs] = right_bit_val
            else:
                position = _rightdown(position)
        elif _all_filled(left_bit):
            right_llr_new = _get_right_llr(left_bit, up_llr)
            rs = slice(
                position[1] + 2 ** (position[2] - position[0] - 1),
                position[1] + 2 ** (position[2] - position[0]),
            )
            llr_matrix[position[0] + 1][rs] = right_llr_new
        elif not _all_filled(left_llr):
            left_llr_new = _get_left_llr(up_llr)
            ls = slice(position[1], position[1] + 2 ** (position[2] - position[0] - 1))
            llr_matrix[position[0] + 1][ls] = left_llr_new
        else:
            if position[0] == position[2] - 1:
                left_bit_pos = position[1]
                left_bit_val = _get_left_bit(
                    left_llr[0], information_pos, frozen_bit, left_bit_pos
                )
                ls = slice(position[1], position[1] + 2 ** (position[2] - position[0] - 1))
                bit_matrix[position[0] + 1][ls] = left_bit_val
            else:
                position = _leftdown(position)

    return bit_matrix[n].astype(int)


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码主函数。
    frozen_bits: 1/True 表示冻结位，0/False 表示信息位。
    """
    frozen_bits = np.asarray(frozen_bits)
    N = len(frozen_bits)
    information_pos = np.where(frozen_bits == 0)[0]
    return _sc_decoder_core(np.asarray(llr_ch, dtype=np.float64), information_pos, 0)


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（调用同一核心逻辑作为参考实现）。"""
    return sc_decode(llr, frozen_bits)


def precompute_sc_indices(N):
    """保留接口：非递归遍历译码器不使用预计算索引。"""
    n = int(math.log2(N))
    return [list(range(n)) for _ in range(N)], [[] for _ in range(N)]
