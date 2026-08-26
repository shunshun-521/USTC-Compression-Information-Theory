"""SC 译码核心。"""
import numpy as np
import polar_tree_functions as fn


def sc_tree_decode(y_llr, frozen_bits):
    frozen_bits = np.asarray(frozen_bits)
    information_pos = np.where(frozen_bits == 0)[0]
    frozen_bit_val = 0

    N = y_llr.size
    n = int(np.log2(N))
    llr_matrix = np.ones((n + 1, N))
    llr_matrix[llr_matrix == 1] = np.nan
    bit_matrix = llr_matrix.copy()
    llr_matrix[0] = y_llr
    position = [0, 0, n, N]

    while fn.all_num(bit_matrix[n]) == 0:
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

        if fn.all_num(up_bit) == 1:
            position = fn.up(position)
        elif fn.all_num(right_bit) == 1:
            up_bit_val = fn.get_up_bit(left_bit, right_bit)
            bit_matrix[position[0]][position[1]:position[1] + 2 ** (position[2] - position[0])] = up_bit_val.copy()
        elif fn.all_num(right_llr) == 1:
            if position[0] == position[2] - 1:
                right_bit_pos = position[1] + 1
                right_bit_val = fn.get_right_bit(right_llr[0], information_pos, frozen_bit_val, right_bit_pos)
                bit_matrix[position[0] + 1][
                    position[1] + 2 ** (position[2] - position[0] - 1):position[1] + 2 ** (position[2] - position[0])
                ] = right_bit_val
            else:
                position = fn.rightdown(position)
        elif fn.all_num(left_bit) == 1:
            right_llr_val = fn.get_right_llr(left_bit, up_llr)
            llr_matrix[position[0] + 1][
                position[1] + 2 ** (position[2] - position[0] - 1):position[1] + 2 ** (position[2] - position[0])
            ] = right_llr_val
        elif fn.all_num(left_llr) == 0:
            left_llr_val = fn.get_left_llr(up_llr)
            llr_matrix[position[0] + 1][position[1]:position[1] + 2 ** (position[2] - position[0] - 1)] = left_llr_val
        else:
            if position[0] == position[2] - 1:
                left_bit_pos = position[1]
                left_bit_val = fn.get_left_bit(left_llr[0], information_pos, frozen_bit_val, left_bit_pos)
                bit_matrix[position[0] + 1][position[1]:position[1] + 2 ** (position[2] - position[0] - 1)] = left_bit_val
            else:
                position = fn.leftdown(position)

    return bit_matrix[n].astype(int)
