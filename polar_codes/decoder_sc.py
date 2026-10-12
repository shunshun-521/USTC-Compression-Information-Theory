"""
极化码 SC（串行抵消）译码器
"""
import math
import numpy as np

import ref_function as fn


def f_operation(La, Lb):
    return fn.f_hf(La, Lb)


def g_operation(La, Lb, u_hat):
    return fn.g(La, Lb, u_hat)


def precompute_sc_indices(N):
    n = int(math.log2(N))
    return [1 << i for i in range(n + 1)], [[] for _ in range(N)], [[] for _ in range(N)]


def _sc_decoder_core(y_llr, information_pos, frozen_bit=0):
    N = y_llr.size
    n = int(np.log2(N))
    llr_matrix = np.ones((n + 1, N))
    llr_matrix[llr_matrix == 1] = float("nan")
    bit_matrix = llr_matrix.copy()
    llr_matrix[0] = y_llr
    position = [0, 0, n, N]
    information_pos = list(information_pos)

    while fn.all_num(bit_matrix[n]) == 0:
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

        if fn.all_num(up_bit) == 1:
            position = fn.up(position)
        elif fn.all_num(right_bit) == 1:
            up_bit = fn.get_up_bit(left_bit, right_bit)
            bit_matrix[position[0]][position[1] : position[1] + 2 ** (position[2] - position[0])] = up_bit.copy()
        elif fn.all_num(right_llr) == 1:
            if position[0] == position[2] - 1:
                right_bit_pos = position[1] + 1
                rb = fn.get_right_bit(right_llr, information_pos, frozen_bit, right_bit_pos)
                bit_matrix[position[0] + 1][
                    position[1] + 2 ** (position[2] - position[0] - 1) : position[1] + 2 ** (position[2] - position[0])
                ] = rb
            else:
                position = fn.rightdown(position)
        elif fn.all_num(left_bit) == 1:
            right_llr = fn.get_right_llr(left_bit, up_llr)
            llr_matrix[position[0] + 1][
                position[1] + 2 ** (position[2] - position[0] - 1) : position[1] + 2 ** (position[2] - position[0])
            ] = right_llr
        elif fn.all_num(left_llr) == 0:
            left_llr = fn.get_left_llr(up_llr)
            llr_matrix[position[0] + 1][position[1] : position[1] + 2 ** (position[2] - position[0] - 1)] = left_llr
        else:
            if position[0] == position[2] - 1:
                left_bit_pos = position[1]
                lb = fn.get_left_bit(left_llr, information_pos, frozen_bit, left_bit_pos)
                bit_matrix[position[0] + 1][position[1] : position[1] + 2 ** (position[2] - position[0] - 1)] = lb
            else:
                position = fn.leftdown(position)

    return bit_matrix[n]


def sc_decode(y_llr, frozen_bits):
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    information_pos = np.where(~frozen_bits)[0].tolist()
    u_hat = _sc_decoder_core(np.asarray(y_llr, dtype=np.float64), information_pos, 0)
    u_hat = u_hat.astype(np.int8)
    u_hat[frozen_bits] = 0
    return u_hat


def sc_decode_recursive(llr, frozen_bits):
    return sc_decode(llr, frozen_bits)
