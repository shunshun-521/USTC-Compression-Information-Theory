"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math

import numpy as np


def f_operation(La, Lb):
    """
    min-sum 近似的 f 运算：
    f(La, Lb) ≈ sign(La) * sign(Lb) * min(|La|, |Lb|)
    """
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    sa = np.sign(La)
    sb = np.sign(Lb)
    sa = np.where(sa == 0, 1, sa)
    sb = np.where(sb == 0, 1, sb)
    return sa * sb * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """
    g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb
    """
    return (1 - 2 * u_hat) * La + Lb


def _all_filled(x):
    return not np.any(np.isnan(x))


def _leftdown(position):
    p0, p1, p2, p3 = position
    return [p0 + 1, p1, p2, p3]


def _rightdown(position):
    p0, p1, p2, p3 = position
    return [p0 + 1, p1 + 2 ** (p2 - 1 - p0), p2, p3]


def _up(position):
    p0, p1, p2, p3 = position
    p1 = int(np.floor(p1 / (2 ** (p2 - p0 + 1))) * (2 ** (p2 - p0 + 1)))
    return [p0 - 1, p1, p2, p3]


def _get_up_bit(left_bit, right_bit):
    length = left_bit.size
    temp = np.array([(left_bit + right_bit) % 2, right_bit])
    return temp.reshape(1, 2 * length)


def _get_right_llr(left_bit, up_llr):
    length = left_bit.size
    return np.array(
        [g_operation(up_llr[i], up_llr[i + length], left_bit[i]) for i in range(length)]
    )


def _get_left_llr(up_llr):
    length = up_llr.size // 2
    return np.array(
        [f_operation(up_llr[i], up_llr[i + length]) for i in range(length)]
    )


def _get_bit(llr_val, is_info):
    if is_info:
        return 0 if llr_val >= 0 else 1
    return 0


def sc_decode_nonrecursive(llr_ch, frozen_bits):
    """
    非递归 SC 译码（基于因子图遍历的高效实现）。
    """
    y_llr = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=int)
    N = y_llr.size
    n = int(math.log2(N))
    info_positions = set(np.where(frozen_bits == 0)[0])

    llr_matrix = np.full((n + 1, N), np.nan, dtype=np.float64)
    bit_matrix = np.full((n + 1, N), np.nan, dtype=np.float64)
    llr_matrix[0] = y_llr
    position = [0, 0, n, N]

    while not _all_filled(bit_matrix[n]):
        up_llr = llr_matrix[position[0]][
            position[1] : position[1] + 2 ** (position[2] - position[0])
        ]
        up_bit = bit_matrix[position[0]][
            position[1] : position[1] + 2 ** (position[2] - position[0])
        ]
        left_llr = llr_matrix[position[0] + 1][
            position[1] : position[1] + 2 ** (position[2] - position[0] - 1)
        ]
        left_bit = bit_matrix[position[0] + 1][
            position[1] : position[1] + 2 ** (position[2] - position[0] - 1)
        ]
        right_llr = llr_matrix[position[0] + 1][
            position[1]
            + 2 ** (position[2] - position[0] - 1) : position[1]
            + 2 ** (position[2] - position[0])
        ]
        right_bit = bit_matrix[position[0] + 1][
            position[1]
            + 2 ** (position[2] - position[0] - 1) : position[1]
            + 2 ** (position[2] - position[0])
        ]

        if _all_filled(up_bit):
            position = _up(position)
        else:
            if _all_filled(right_bit):
                up_bit_val = _get_up_bit(left_bit, right_bit)
                bit_matrix[position[0]][
                    position[1] : position[1] + 2 ** (position[2] - position[0])
                ] = up_bit_val.copy()
            else:
                if _all_filled(right_llr):
                    if position[0] == position[2] - 1:
                        right_bit_pos = position[1] + 1
                        bit_matrix[position[0] + 1][
                            position[1]
                            + 2 ** (position[2] - position[0] - 1) : position[1]
                            + 2 ** (position[2] - position[0])
                        ] = _get_bit(
                            right_llr[0],
                            right_bit_pos in info_positions,
                        )
                    else:
                        position = _rightdown(position)
                else:
                    if _all_filled(left_bit):
                        right_llr_val = _get_right_llr(left_bit, up_llr)
                        llr_matrix[position[0] + 1][
                            position[1]
                            + 2 ** (position[2] - position[0] - 1) : position[1]
                            + 2 ** (position[2] - position[0])
                        ] = right_llr_val
                    else:
                        if not _all_filled(left_llr):
                            left_llr_val = _get_left_llr(up_llr)
                            llr_matrix[position[0] + 1][
                                position[1] : position[1]
                                + 2 ** (position[2] - position[0] - 1)
                            ] = left_llr_val
                        else:
                            if position[0] == position[2] - 1:
                                left_bit_pos = position[1]
                                bit_matrix[position[0] + 1][
                                    position[1] : position[1]
                                    + 2 ** (position[2] - position[0] - 1)
                                ] = _get_bit(
                                    left_llr[0],
                                    left_bit_pos in info_positions,
                                )
                            else:
                                position = _leftdown(position)

    return np.nan_to_num(bit_matrix[n], nan=0).astype(int)


def sc_decode_recursive(llr, frozen_bits):
    """
    递归 SC 译码（参考实现）。
    """
    llr = np.asarray(llr, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=int)
    N = len(llr)
    u_hat = np.zeros(N, dtype=int)

    def decode_block(llr_block, frozen_block, offset):
        n = len(llr_block)
        if n == 1:
            if frozen_block[0]:
                u_hat[offset] = 0
            else:
                u_hat[offset] = 0 if llr_block[0] >= 0 else 1
            return

        half = n // 2
        llr_left = f_operation(llr_block[:half], llr_block[half:])
        decode_block(llr_left, frozen_block[:half], offset)

        u_left = u_hat[offset : offset + half]
        llr_right = g_operation(llr_block[:half], llr_block[half:], u_left)
        decode_block(llr_right, frozen_block[half:], offset + half)

    decode_block(llr, frozen_bits, 0)
    return u_hat


def precompute_sc_indices(N):
    """
    预计算非递归 SC 译码所需的三个辅助向量。
    """
    n = int(math.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]

    llr_layer_vec = []
    bit_layer_vec = []

    for phi in range(N):
        llr_layers = []
        i = 0
        while i < n and ((phi == 0) or ((phi >> i) & 1)):
            llr_layers.append(i)
            i += 1
        llr_layer_vec.append(llr_layers)

        bit_layers = []
        if phi % 2 == 0:
            j = 0
            while j < n and ((phi >> j) & 1) == 0:
                bit_layers.append(j)
                j += 1
        bit_layer_vec.append(bit_layers)

    return lambda_offset, llr_layer_vec, bit_layer_vec


sc_decode = sc_decode_nonrecursive
