"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math

import numpy as np


def f_operation(La, Lb):
    """min-sum 近似的 f 运算"""
    sa = np.sign(La)
    sb = np.sign(Lb)
    sa = np.where(sa == 0, 1, sa)
    sb = np.where(sb == 0, 1, sb)
    return sa * sb * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算"""
    return (1.0 - 2.0 * u_hat) * La + Lb


def _all_decided(arr):
    return not np.any(np.isnan(arr))


def _up_position(pos):
    p0, p1, p2, p3 = pos
    p0 -= 1
    p1 = int(np.floor(p1 / (2 ** (p2 - p0 + 1))) * (2 ** (p2 - p0 + 1)))
    return [p0, p1, p2, p3]


def _leftdown(pos):
    return [pos[0] + 1, pos[1], pos[2], pos[3]]


def _rightdown(pos):
    return [pos[0] + 1, pos[1] + 2 ** (pos[2] - pos[0] - 1), pos[2], pos[3]]


def _get_up_bit(left_bit, right_bit):
    length = len(left_bit)
    temp = np.array([(left_bit + right_bit) % 2, right_bit])
    temp = temp.reshape(2 * length)
    return temp


def _get_left_llr(up_llr):
    half = len(up_llr) // 2
    return f_operation(up_llr[:half], up_llr[half:])


def _get_right_llr(left_bit, up_llr):
    half = len(up_llr) // 2
    return g_operation(up_llr[:half], up_llr[half:], left_bit)


# ==================== 递归 SC 译码（参考实现）====================


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码"""
    llr = np.asarray(llr, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)

    def decode_node(node_llr, frozen):
        if len(node_llr) == 1:
            if frozen[0]:
                return np.array([0])
            return np.array([0 if node_llr[0] >= 0 else 1])

        half = len(node_llr) // 2
        llr_left = f_operation(node_llr[:half], node_llr[half:])
        u_left = decode_node(llr_left, frozen[:half])
        llr_right = g_operation(node_llr[:half], node_llr[half:], u_left)
        u_right = decode_node(llr_right, frozen[half:])
        return np.concatenate([u_left, u_right])

    return decode_node(llr, frozen_bits)


# ==================== 非递归 SC 译码（树遍历实现）====================


def precompute_sc_indices(N):
    """保留接口：返回空结构供 SCL 兼容"""
    n = int(math.log2(N))
    return [1 << i for i in range(n + 1)], [[] for _ in range(N)], [[] for _ in range(N)]


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码（树遍历，与 u@G 蝶形编码器配套）。
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr_ch)
    n = int(math.log2(N))

    llr_matrix = np.full((n + 1, N), np.nan, dtype=np.float64)
    bit_matrix = np.full((n + 1, N), np.nan, dtype=np.float64)
    llr_matrix[0] = llr_ch
    position = [0, 0, n, N]

    while not _all_decided(bit_matrix[n]):
        up_llr = llr_matrix[position[0], position[1] : position[1] + 2 ** (position[2] - position[0])]
        up_bit = bit_matrix[position[0], position[1] : position[1] + 2 ** (position[2] - position[0])]
        left_llr = llr_matrix[
            position[0] + 1, position[1] : position[1] + 2 ** (position[2] - position[0] - 1)
        ]
        left_bit = bit_matrix[
            position[0] + 1, position[1] : position[1] + 2 ** (position[2] - position[0] - 1)
        ]
        right_llr = llr_matrix[
            position[0] + 1,
            position[1] + 2 ** (position[2] - position[0] - 1) : position[1] + 2 ** (position[2] - position[0]),
        ]
        right_bit = bit_matrix[
            position[0] + 1,
            position[1] + 2 ** (position[2] - position[0] - 1) : position[1] + 2 ** (position[2] - position[0]),
        ]

        if _all_decided(up_bit):
            position = _up_position(position)
        elif _all_decided(right_bit):
            merged = _get_up_bit(left_bit.astype(int), right_bit.astype(int))
            bit_matrix[
                position[0], position[1] : position[1] + 2 ** (position[2] - position[0])
            ] = merged
        elif _all_decided(right_llr):
            if position[0] == position[2] - 1:
                idx = position[1] + 2 ** (position[2] - position[0] - 1)
                if frozen_bits[idx]:
                    val = 0
                else:
                    val = 0 if right_llr[0] >= 0 else 1
                bit_matrix[position[0] + 1, idx] = val
            else:
                position = _rightdown(position)
        elif _all_decided(left_bit):
            new_right = _get_right_llr(left_bit.astype(int), up_llr)
            sl = slice(
                position[1] + 2 ** (position[2] - position[0] - 1),
                position[1] + 2 ** (position[2] - position[0]),
            )
            llr_matrix[position[0] + 1, sl] = new_right
        elif np.any(np.isnan(left_llr)):
            new_left = _get_left_llr(up_llr)
            sl = slice(position[1], position[1] + 2 ** (position[2] - position[0] - 1))
            llr_matrix[position[0] + 1, sl] = new_left
        else:
            if position[0] == position[2] - 1:
                idx = position[1]
                if frozen_bits[idx]:
                    val = 0
                else:
                    val = 0 if left_llr[0] >= 0 else 1
                bit_matrix[position[0] + 1, idx] = val
            else:
                position = _leftdown(position)

    return bit_matrix[n].astype(int)
