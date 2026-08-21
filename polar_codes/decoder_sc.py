"""
极化码 SC（串行抵消）译码器
"""
import math
import numpy as np
from encoder import bit_reversal_permutation


def f_operation(La, Lb):
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    return (1.0 - 2.0 * u_hat) * La + Lb


def align_llr_for_decoder(llr_ch):
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    return llr_ch[bit_reversal_permutation(len(llr_ch))]


def _all_num(x):
    return not np.isnan(x).any()


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
    p1 = int(
        np.floor(position[1] / (2 ** (position[2] - position[0] + 1)))
        * (2 ** (position[2] - position[0] + 1))
    )
    return [position[0] - 1, p1, position[2], position[3]]


def _get_up_bit(left_bit, right_bit):
    temp = np.array([(left_bit + right_bit) % 2, right_bit])
    temp.resize((1, 2 * len(left_bit)))
    return temp


def _get_right_bit(right_llr, information_pos, frozen_bit, right_bit_pos):
    if right_bit_pos in information_pos:
        return 0 if right_llr >= 0 else 1
    return frozen_bit


def _get_right_llr(left_bit, up_llr):
    length = len(left_bit)
    return np.array([g_operation(up_llr[i], up_llr[i + length], left_bit[i]) for i in range(length)])


def _get_left_bit(left_llr, information_pos, frozen_bit, left_bit_pos):
    if left_bit_pos in information_pos:
        return 0 if left_llr >= 0 else 1
    return frozen_bit


def _get_left_llr(up_llr):
    length = len(up_llr) // 2
    return f_operation(up_llr[:length], up_llr[length:])


def _sc_tree_decode(y_llr, information_pos, frozen_bit=0):
    N = len(y_llr)
    n = int(math.log2(N))
    info = set(int(i) for i in information_pos)

    llr_matrix = np.full((n + 1, N), np.nan, dtype=np.float64)
    bit_matrix = np.full((n + 1, N), np.nan, dtype=np.float64)
    llr_matrix[0] = y_llr
    position = [0, 0, n, N]

    while not _all_num(bit_matrix[n]):
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

        if _all_num(up_bit):
            position = _up(position)
        elif _all_num(right_bit):
            up_bit_new = _get_up_bit(left_bit, right_bit)
            bit_matrix[position[0]][
                position[1] : position[1] + 2 ** (position[2] - position[0])
            ] = up_bit_new.copy()
        elif _all_num(right_llr):
            if position[0] == position[2] - 1:
                right_bit_pos = position[1] + 1
                rb = _get_right_bit(right_llr[0], info, frozen_bit, right_bit_pos)
                bit_matrix[position[0] + 1][
                    position[1]
                    + 2 ** (position[2] - position[0] - 1) : position[1]
                    + 2 ** (position[2] - position[0])
                ] = rb
            else:
                position = _rightdown(position)
        elif _all_num(left_bit):
            right_llr_new = _get_right_llr(left_bit, up_llr)
            bit_matrix_slice = llr_matrix[position[0] + 1]
            bit_matrix_slice[
                position[1]
                + 2 ** (position[2] - position[0] - 1) : position[1]
                + 2 ** (position[2] - position[0])
            ] = right_llr_new
        else:
            if not _all_num(left_llr):
                left_llr_new = _get_left_llr(up_llr)
                llr_matrix[position[0] + 1][
                    position[1] : position[1] + 2 ** (position[2] - position[0] - 1)
                ] = left_llr_new
            else:
                if position[0] == position[2] - 1:
                    left_bit_pos = position[1]
                    lb = _get_left_bit(left_llr[0], info, frozen_bit, left_bit_pos)
                    bit_matrix[position[0] + 1][
                        position[1] : position[1] + 2 ** (position[2] - position[0] - 1)
                    ] = lb
                else:
                    position = _leftdown(position)

    return bit_matrix[n].astype(int)


def sc_decode_recursive(llr, frozen_bits):
    return sc_decode(llr, frozen_bits)


def precompute_sc_indices(N):
    n = int(math.log2(N))
    return [1 << i for i in range(n + 1)], [[] for _ in range(N)], [[] for _ in range(N)]


def sc_decode(llr_ch, frozen_bits):
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    info_pos = np.where(~frozen_bits)[0]
    llr = align_llr_for_decoder(llr_ch)
    return _sc_tree_decode(llr, info_pos, frozen_bit=0)
