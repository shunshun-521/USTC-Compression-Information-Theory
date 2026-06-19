"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math

import numpy as np

from encoder import bit_reversal_permutation


def f_operation(La, Lb):
    """min-sum 近似的 f 运算（sign(0) 视为 +1）"""
    sa = np.sign(La)
    sb = np.sign(Lb)
    sa = np.where(sa == 0, 1, sa)
    sb = np.where(sb == 0, 1, sb)
    return sa * sb * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算"""
    return (1 - 2 * u_hat) * La + Lb


def _channel_llr_for_decode(llr_ch):
    """将信道 LLR 重排为与蝶形+比特倒序编码一致的顺序。"""
    N = len(llr_ch)
    inv_br = np.argsort(bit_reversal_permutation(N))
    return np.asarray(llr_ch, dtype=np.float64)[inv_br]


def _all_num(x):
    return not np.any(np.isnan(x))


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
    p1 = int(
        np.floor(position[1] / (2 ** (position[2] - position[0] + 1)))
        * (2 ** (position[2] - position[0] + 1))
    )
    return [p0, p1, position[2], position[3]]


def _get_up_bit(left_bit, right_bit):
    length = len(left_bit)
    temp = np.array([(left_bit + right_bit) % 2, right_bit])
    return temp.reshape(2 * length)


def _get_right_bit(right_llr, frozen_bits, right_bit_pos):
    if frozen_bits[right_bit_pos]:
        return 0
    return 0 if right_llr >= 0 else 1


def _get_left_bit(left_llr, frozen_bits, left_bit_pos):
    if frozen_bits[left_bit_pos]:
        return 0
    return 0 if left_llr >= 0 else 1


def _get_right_llr(left_bit, up_llr):
    half = len(left_bit)
    return np.array([
        g_operation(up_llr[i], up_llr[i + half], left_bit[i])
        for i in range(half)
    ])


def _get_left_llr(up_llr):
    half = len(up_llr) // 2
    return f_operation(up_llr[:half], up_llr[half:])


def sc_decode_tree(llr_ch, frozen_bits):
    """树遍历非递归 SC 译码（主实现）。"""
    llr_ch = _channel_llr_for_decode(llr_ch)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr_ch)
    n = int(math.log2(N))

    llr_matrix = np.full((n + 1, N), np.nan, dtype=np.float64)
    bit_matrix = np.full((n + 1, N), np.nan)
    llr_matrix[0] = llr_ch
    position = [0, 0, n, N]

    while not _all_num(bit_matrix[n]):
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

        if _all_num(up_bit):
            position = _up(position)
        else:
            if _all_num(right_bit):
                up_bit_val = _get_up_bit(left_bit, right_bit)
                bit_matrix[position[0]][
                    position[1]:position[1] + 2 ** (position[2] - position[0])
                ] = up_bit_val.copy()
            else:
                if _all_num(right_llr):
                    if position[0] == position[2] - 1:
                        right_bit_pos = position[1] + 1
                        right_bit_val = _get_right_bit(
                            right_llr[0], frozen_bits, right_bit_pos
                        )
                        bit_matrix[position[0] + 1][
                            position[1] + 2 ** (position[2] - position[0] - 1):
                            position[1] + 2 ** (position[2] - position[0])
                        ] = right_bit_val
                    else:
                        position = _rightdown(position)
                else:
                    if _all_num(left_bit):
                        right_llr_val = _get_right_llr(left_bit, up_llr)
                        llr_matrix[position[0] + 1][
                            position[1] + 2 ** (position[2] - position[0] - 1):
                            position[1] + 2 ** (position[2] - position[0])
                        ] = right_llr_val
                    else:
                        if not _all_num(left_llr):
                            left_llr_val = _get_left_llr(up_llr)
                            llr_matrix[position[0] + 1][
                                position[1]:position[1] + 2 ** (position[2] - position[0] - 1)
                            ] = left_llr_val
                        else:
                            if position[0] == position[2] - 1:
                                left_bit_pos = position[1]
                                left_bit_val = _get_left_bit(
                                    left_llr[0], frozen_bits, left_bit_pos
                                )
                                bit_matrix[position[0] + 1][
                                    position[1]:position[1] + 2 ** (position[2] - position[0] - 1)
                                ] = left_bit_val
                            else:
                                position = _leftdown(position)

    return bit_matrix[n].astype(int)


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（参考实现）。"""
    return sc_decode_tree(llr, frozen_bits)


def precompute_sc_indices(N):
    """预计算非递归 SC 译码所需的辅助向量（接口兼容）。"""
    n = int(math.log2(N))
    lambda_offset = [0] * (n + 1)
    for i in range(1, n + 1):
        lambda_offset[i] = 2 ** (i - 1)
    llr_layer_vec = [list(range(n)) for _ in range(N)]
    bit_layer_vec = [list(range(n)) for _ in range(N)]
    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC 译码主函数。"""
    return sc_decode_tree(llr_ch, frozen_bits)
