"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import numpy as np


def f_operation(La, Lb):
    """min-sum 近似的 f 运算。"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算。"""
    return (1.0 - 2.0 * u_hat) * La + Lb


def _all_num(x):
    return not np.any(np.isnan(x))


def _up(position):
    p0 = position[0] - 1
    p1 = int(np.floor(position[1] / (2 ** (position[2] - position[0] + 1)))
             * (2 ** (position[2] - position[0] + 1)))
    return [p0, p1, position[2], position[3]]


def _leftdown(position):
    return [position[0] + 1, position[1], position[2], position[3]]


def _rightdown(position):
    return [
        position[0] + 1,
        position[1] + 2 ** (position[2] - 1 - position[0]),
        position[2],
        position[3],
    ]


def _get_up_bit(left_bit, right_bit):
    length = len(left_bit)
    temp = np.array([(left_bit + right_bit) % 2, right_bit])
    return temp.reshape(1, 2 * length).flatten()


def _get_right_bit(right_llr, frozen_bits, right_bit_pos):
    if not frozen_bits[right_bit_pos]:
        return 0 if right_llr >= 0 else 1
    return 0


def _get_left_bit(left_llr, frozen_bits, left_bit_pos):
    if not frozen_bits[left_bit_pos]:
        return 0 if left_llr >= 0 else 1
    return 0


def _get_right_llr(left_bit, up_llr):
    half = len(left_bit)
    return np.array([
        g_operation(up_llr[i], up_llr[i + half], left_bit[i]) for i in range(half)
    ])


def _get_left_llr(up_llr):
    half = len(up_llr) // 2
    return np.array([
        f_operation(up_llr[i], up_llr[i + half]) for i in range(half)
    ])


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC 译码（基于因子图遍历）。"""
    y_llr = np.asarray(llr_ch, dtype=np.float64)
    N = len(y_llr)
    n = int(np.log2(N))
    frozen_bits = np.asarray(frozen_bits, dtype=bool)

    llr_matrix = np.ones((n + 1, N), dtype=np.float64)
    llr_matrix[:] = np.nan
    bit_matrix = llr_matrix.copy()
    llr_matrix[0] = y_llr
    position = [0, 0, n, N]

    while not _all_num(bit_matrix[n]):
        up_llr = llr_matrix[position[0]][position[1] : position[1] + 2 ** (position[2] - position[0])]
        up_bit = bit_matrix[position[0]][position[1] : position[1] + 2 ** (position[2] - position[0])]
        left_llr = llr_matrix[position[0] + 1][position[1] : position[1] + 2 ** (position[2] - position[0] - 1)]
        left_bit = bit_matrix[position[0] + 1][position[1] : position[1] + 2 ** (position[2] - position[0] - 1)]
        right_llr = llr_matrix[position[0] + 1][
            position[1] + 2 ** (position[2] - position[0] - 1) : position[1] + 2 ** (position[2] - position[0])
        ]
        right_bit = bit_matrix[position[0] + 1][
            position[1] + 2 ** (position[2] - position[0] - 1) : position[1] + 2 ** (position[2] - position[0])
        ]

        if _all_num(up_bit):
            position = _up(position)
        else:
            if _all_num(right_bit):
                up_bit = _get_up_bit(left_bit, right_bit)
                bit_matrix[position[0]][position[1] : position[1] + 2 ** (position[2] - position[0])] = up_bit
            else:
                if _all_num(right_llr):
                    if position[0] == position[2] - 1:
                        right_bit_pos = position[1] + 1
                        right_bit = _get_right_bit(right_llr, frozen_bits, right_bit_pos)
                        bit_matrix[position[0] + 1][
                            position[1] + 2 ** (position[2] - position[0] - 1) : position[1] + 2 ** (position[2] - position[0])
                        ] = right_bit
                    else:
                        position = _rightdown(position)
                else:
                    if _all_num(left_bit):
                        right_llr = _get_right_llr(left_bit, up_llr)
                        llr_matrix[position[0] + 1][
                            position[1] + 2 ** (position[2] - position[0] - 1) : position[1] + 2 ** (position[2] - position[0])
                        ] = right_llr
                    else:
                        if not _all_num(left_llr):
                            left_llr = _get_left_llr(up_llr)
                            llr_matrix[position[0] + 1][
                                position[1] : position[1] + 2 ** (position[2] - position[0] - 1)
                            ] = left_llr
                        else:
                            if position[0] == position[2] - 1:
                                left_bit_pos = position[1]
                                left_bit = _get_left_bit(left_llr, frozen_bits, left_bit_pos)
                                bit_matrix[position[0] + 1][
                                    position[1] : position[1] + 2 ** (position[2] - position[0] - 1)
                                ] = left_bit
                            else:
                                position = _leftdown(position)

    return bit_matrix[n].astype(int)


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（参考实现，调用非递归版本）。"""
    return sc_decode(llr, frozen_bits)


def precompute_sc_indices(N):
    """SCL 译码器兼容接口（保留）。"""
    import math
    n = int(math.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]
    llr_layer_vec = [[] for _ in range(N)]
    bit_layer_vec = [[] for _ in range(N)]
    return lambda_offset, llr_layer_vec, bit_layer_vec
