"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math
import numpy as np
from encoder import bit_reversal_permutation


def f_operation(La, Lb):
    sa = np.sign(La)
    sb = np.sign(Lb)
    sa = np.where(sa == 0, 1, sa)
    sb = np.where(sb == 0, 1, sb)
    return sa * sb * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    return (1.0 - 2.0 * u_hat) * La + Lb


def _permute_llr_for_decode(llr_ch):
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    N = len(llr_ch)
    inv_br = np.argsort(bit_reversal_permutation(N))
    return llr_ch[inv_br]


def _frozen_mask_to_info(frozen_bits):
    frozen = np.asarray(frozen_bits, dtype=int).astype(bool)
    return [i for i in range(len(frozen)) if not frozen[i]]


def _all_num(x):
    x = np.asarray(x).ravel()
    for v in x:
        if np.isnan(v):
            return 0
    return 1


def _up(position):
    p0, p1, p2, p3 = position
    block = 2 ** (p2 - p0 + 1)
    return [p0 - 1, int(np.floor(p1 / block) * block), p2, p3]


def _leftdown(position):
    p0, p1, p2, p3 = position
    return [p0 + 1, p1, p2, p3]


def _rightdown(position):
    p0, p1, p2, p3 = position
    return [p0 + 1, p1 + 2 ** (p2 - 1 - p0), p2, p3]


def _get_up_bit(left_bit, right_bit):
    length = left_bit.size
    temp = np.array([(left_bit + right_bit) % 2, right_bit])
    temp.resize((1, 2 * length))
    return temp[0]


def _get_left_llr(up_llr):
    length = int(up_llr.size / 2)
    return np.array([f_operation(up_llr[i], up_llr[i + length]) for i in range(length)])


def _get_right_llr(left_bit, up_llr):
    length = int(left_bit.size)
    return np.array([g_operation(up_llr[i], up_llr[i + length], left_bit[i]) for i in range(length)])


def _get_left_bit(left_llr, information_pos, frozen_bit, left_bit_pos):
    if left_bit_pos in information_pos:
        val = left_llr[0] if np.ndim(left_llr) else left_llr
        return 0 if val >= 0 else 1
    return frozen_bit


def _get_right_bit(right_llr, information_pos, frozen_bit, right_bit_pos):
    if right_bit_pos in information_pos:
        val = right_llr[0] if np.ndim(right_llr) else right_llr
        return 0 if val >= 0 else 1
    return frozen_bit


def _tree_sc_decode(y_llr, information_pos, frozen_bit=0):
    N = y_llr.size
    n = int(math.log2(N))
    llr_matrix = np.ones((n + 1, N), dtype=np.float64)
    llr_matrix[llr_matrix == 1] = np.nan
    bit_matrix = llr_matrix.copy()
    llr_matrix[0] = y_llr
    position = [0, 0, n, N]

    while _all_num(bit_matrix[n]) == 0:
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

        if _all_num(up_bit) == 1:
            position = _up(position)
        elif _all_num(right_bit) == 1:
            up_bit_new = _get_up_bit(left_bit, right_bit)
            bit_matrix[position[0]][
                position[1] : position[1] + 2 ** (position[2] - position[0])
            ] = up_bit_new.copy()
        elif _all_num(right_llr) == 1:
            if position[0] == position[2] - 1:
                right_bit_pos = position[1] + 1
                rb = _get_right_bit(right_llr, information_pos, frozen_bit, right_bit_pos)
                bit_matrix[position[0] + 1][
                    position[1]
                    + 2 ** (position[2] - position[0] - 1) : position[1]
                    + 2 ** (position[2] - position[0])
                ] = rb
            else:
                position = _rightdown(position)
        elif _all_num(left_bit) == 1:
            right_llr_new = _get_right_llr(left_bit, up_llr)
            llr_matrix[position[0] + 1][
                position[1]
                + 2 ** (position[2] - position[0] - 1) : position[1]
                + 2 ** (position[2] - position[0])
            ] = right_llr_new
        elif _all_num(left_llr) == 0:
            left_llr_new = _get_left_llr(up_llr)
            llr_matrix[position[0] + 1][
                position[1] : position[1] + 2 ** (position[2] - position[0] - 1)
            ] = left_llr_new
        else:
            if position[0] == position[2] - 1:
                left_bit_pos = position[1]
                lb = _get_left_bit(left_llr, information_pos, frozen_bit, left_bit_pos)
                bit_matrix[position[0] + 1][
                    position[1] : position[1] + 2 ** (position[2] - position[0] - 1)
                ] = lb
            else:
                position = _leftdown(position)

    return np.array([0 if bit_matrix[n][i] == 0 else 1 for i in range(N)], dtype=int)


def sc_decode_recursive(llr, frozen_bits):
    return sc_decode(llr, frozen_bits)


def precompute_sc_indices(N):
    n = int(math.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]
    llr_layer_vec, bit_layer_vec = [], []
    for phi in range(N):
        layers_llr, tmp = [], phi
        while tmp & 1:
            layers_llr.append(int(math.log2(tmp & -tmp)))
            tmp >>= 1
        layers_llr.append(n if tmp == 0 else int(math.log2(tmp & -tmp)))
        llr_layer_vec.append(layers_llr)
        layers_bit, tmp = [], phi
        while (tmp & 1) == 0 and tmp < N:
            layers_bit.append(0 if tmp == 0 else int(math.log2(tmp & -tmp)))
            tmp >>= 1
        if tmp & 1:
            layers_bit.append(int(math.log2(tmp & -tmp)))
        bit_layer_vec.append(layers_bit)
    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    llr = _permute_llr_for_decode(llr_ch)
    info_pos = _frozen_mask_to_info(frozen_bits)
    return _tree_sc_decode(llr, info_pos, frozen_bit=0)
