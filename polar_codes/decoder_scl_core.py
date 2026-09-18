"""SCL 译码核心（改编自验证过的参考实现）。"""
import numpy as np

import function_ref as function
from utils import crc_check


def sc_stepping_decoder(llr_matrix, bit_matrix, information_pos, frozen_bit, split_pos):
    N = int(bit_matrix[0].size)
    n = int(np.log2(N))
    loc = function.get_up_loc(bit_matrix)
    position = [loc[0], loc[1], n, N]

    while bit_matrix[n][split_pos] != 0 and bit_matrix[n][split_pos] != 1:
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

        if function.all_num(up_bit) == 1:
            position = function.up(position)
        else:
            if function.all_num(right_bit) == 1:
                up_bit = function.get_up_bit(left_bit, right_bit)
                bit_matrix[position[0]][
                    position[1] : position[1] + 2 ** (position[2] - position[0])
                ] = up_bit.copy()
            else:
                if function.all_num(right_llr) == 1:
                    if position[0] == position[2] - 1:
                        right_bit_pos = position[1] + 1
                        right_bit = function.get_right_bit(
                            right_llr, information_pos, frozen_bit, right_bit_pos
                        )
                        bit_matrix[position[0] + 1][
                            position[1]
                            + 2 ** (position[2] - position[0] - 1) : position[1]
                            + 2 ** (position[2] - position[0])
                        ] = right_bit
                    else:
                        position = function.rightdown(position)
                else:
                    if function.all_num(left_bit) == 1:
                        right_llr = function.get_right_llr(left_bit, up_llr)
                        llr_matrix[position[0] + 1][
                            position[1]
                            + 2 ** (position[2] - position[0] - 1) : position[1]
                            + 2 ** (position[2] - position[0])
                        ] = right_llr
                    else:
                        if function.all_num(left_llr) == 0:
                            left_llr = function.get_left_llr(up_llr)
                            llr_matrix[position[0] + 1][
                                position[1] : position[1]
                                + 2 ** (position[2] - position[0] - 1)
                            ] = left_llr
                        else:
                            if position[0] == position[2] - 1:
                                left_bit_pos = position[1]
                                left_bit = function.get_left_bit(
                                    left_llr, information_pos, frozen_bit, left_bit_pos
                                )
                                bit_matrix[position[0] + 1][
                                    position[1] : position[1]
                                    + 2 ** (position[2] - position[0] - 1)
                                ] = left_bit
                            else:
                                position = function.leftdown(position)
    return [llr_matrix, bit_matrix]


def scl_decode_core(y_llr, information_pos, frozen_set, list_size, crc_length):
    """SCL 译码入口。"""
    N = y_llr.size
    n = int(np.log2(N))
    information_pos = np.asarray(information_pos, dtype=int)
    frozen_bit = 0
    list_max = list_size
    pm_method = "hf"
    split_pos = information_pos

    llr_matrix = np.ones((n + 1, N))
    llr_matrix[llr_matrix == 1] = float("nan")
    bit_matrix = llr_matrix.copy()
    llr_matrix[0] = y_llr
    llr_list = [llr_matrix]
    bit_list = [bit_matrix]
    pm_list = [0.0]
    split_loc = 0
    split_len = len(split_pos)
    l_now = 1

    while split_len - 1 >= split_loc:
        new_llr, new_bit, new_pm = [], [], []
        for i in range(l_now):
            llr_matrix_temp = llr_list[i]
            bit_matrix_temp = bit_list[i]
            pm_temp = pm_list[i]

            matrix_temp = sc_stepping_decoder(
                llr_matrix_temp.copy(),
                bit_matrix_temp.copy(),
                information_pos,
                frozen_bit,
                split_pos[split_loc],
            )

            start = 0 if split_loc == 0 else split_pos[split_loc - 1] + 1
            end = split_pos[split_loc] + 1
            llr_slice = matrix_temp[0][n][start:end]
            bit_slice = matrix_temp[1][n][start:end]

            right_pm = function.get_pm_update(llr_slice, bit_slice, pm_method)
            new_llr.append(matrix_temp[0])
            new_bit.append(matrix_temp[1])
            new_pm.append(pm_temp + right_pm)

            wrong_bit = matrix_temp[1].copy()
            wrong_bit[n][split_pos[split_loc]] = 1 - wrong_bit[n][split_pos[split_loc]]
            wrong_bit_slice = wrong_bit[n][start:end]
            wrong_pm = function.get_pm_update(llr_slice, wrong_bit_slice, pm_method)
            new_llr.append(matrix_temp[0].copy())
            new_bit.append(wrong_bit)
            new_pm.append(pm_temp + wrong_pm)

        order = np.argsort(new_pm)[:list_max]
        llr_list = [new_llr[i] for i in order]
        bit_list = [new_bit[i] for i in order]
        pm_list = [new_pm[i] for i in order]
        l_now = len(pm_list)
        split_loc += 1

    if split_pos[-1] != N - 1:
        for i in range(l_now):
            matrix_temp = sc_stepping_decoder(
                llr_list[i].copy(),
                bit_list[i].copy(),
                information_pos,
                frozen_bit,
                N - 1,
            )
            llr_list[i] = matrix_temp[0]
            bit_list[i] = matrix_temp[1]
            start = split_pos[split_loc - 1] + 1
            pm_list[i] += function.get_pm_update(
                matrix_temp[0][n][start:N],
                matrix_temp[1][n][start:N],
                pm_method,
            )

    pm_argsort = np.argsort(pm_list)
    best_u = None
    best_pm = pm_list[pm_argsort[0]]

    if crc_length > 0:
        for idx in pm_argsort:
            u_candidate = bit_list[idx][n].astype(int)
            if crc_check(u_candidate, crc_length):
                best_u = u_candidate
                best_pm = pm_list[idx]
                break
        if best_u is None:
            best_u = bit_list[pm_argsort[0]][n].astype(int)
    else:
        best_u = bit_list[pm_argsort[0]][n].astype(int)

    return best_u, best_pm
