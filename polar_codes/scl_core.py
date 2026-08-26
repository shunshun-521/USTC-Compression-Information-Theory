"""SCL 译码核心（基于参考实现）。"""
import numpy as np
import polar_tree_functions as fn


def _pm_update(llr_array, bit_array):
    pm = 0.0
    for i in range(len(llr_array)):
        if np.sign(llr_array[i]) != np.sign(1 - 2 * bit_array[i]):
            pm += abs(llr_array[i])
    return pm


def scl_decode(y_llr, frozen_bits, list_size, crc_length=0, crc_check_fn=None):
    N = y_llr.size
    n = int(np.log2(N))
    information_pos = np.where(np.asarray(frozen_bits) == 0)[0]
    frozen_bit_val = 0

    llr_matrix = np.ones((n + 1, N))
    llr_matrix[llr_matrix == 1] = np.nan
    bit_matrix = llr_matrix.copy()
    llr_matrix[0] = y_llr

    llr_list = [llr_matrix.copy()]
    bit_list = [bit_matrix.copy()]
    pm_list = [0.0]
    split_pos = information_pos.tolist()
    split_loc = 0
    l_now = 1

    while split_loc < len(split_pos):
        for i in range(l_now):
            llr_temp = llr_list[i].copy()
            bit_temp = bit_list[i].copy()
            pm_temp = pm_list[i]
            llr_out, bit_out = _sc_step_to(llr_temp, bit_temp, information_pos, frozen_bit_val, split_pos[split_loc])
            llr_list[i] = llr_out
            bit_list[i] = bit_out
            start = 0 if split_loc == 0 else split_pos[split_loc - 1] + 1
            end = split_pos[split_loc] + 1
            pm_list[i] = pm_temp + _pm_update(llr_out[n, start:end], bit_out[n, start:end])

            llr_list.append(llr_out.copy())
            bit_wrong = bit_out.copy()
            bit_wrong[n, split_pos[split_loc]] = 1 - bit_wrong[n, split_pos[split_loc]]
            bit_list.append(bit_wrong)
            pm_list.append(pm_temp + _pm_update(llr_out[n, start:end], bit_wrong[n, start:end]))

        if l_now > list_size / 2:
            order = np.argsort(pm_list)
            keep = order[:list_size]
            pm_list = [pm_list[i] for i in keep]
            llr_list = [llr_list[i] for i in keep]
            bit_list = [bit_list[i] for i in keep]
        l_now = len(pm_list)
        split_loc += 1

    if split_pos and split_pos[-1] != N - 1:
        for i in range(l_now):
            llr_out, bit_out = _sc_step_to(
                llr_list[i], bit_list[i], information_pos, frozen_bit_val, N - 1
            )
            llr_list[i] = llr_out
            bit_list[i] = bit_out
            start = split_pos[-1] + 1
            pm_list[i] += _pm_update(llr_out[n, start:N], bit_out[n, start:N])

    order = np.argsort(pm_list)
    if crc_length > 0 and crc_check_fn is not None:
        for idx in order:
            u = bit_list[idx][n].astype(int)
            if crc_check_fn(u[information_pos], crc_length):
                return u, pm_list[idx]
    best = order[0]
    return bit_list[best][n].astype(int), pm_list[best]


def _sc_step_to(llr_matrix, bit_matrix, information_pos, frozen_bit, split_pos):
    N = int(bit_matrix[0].size)
    n = int(np.log2(N))
    loc = fn.get_up_loc(bit_matrix)
    position = [loc[0], loc[1], n, N]

    while bit_matrix[n][split_pos] != 0 and bit_matrix[n][split_pos] != 1:
        up_llr = llr_matrix[position[0]][position[1]:position[1] + 2 ** (position[2] - position[0])]
        left_llr = llr_matrix[position[0] + 1][position[1]:position[1] + 2 ** (position[2] - position[0] - 1)]
        left_bit = bit_matrix[position[0] + 1][position[1]:position[1] + 2 ** (position[2] - position[0] - 1)]
        right_llr = llr_matrix[position[0] + 1][
            position[1] + 2 ** (position[2] - position[0] - 1):position[1] + 2 ** (position[2] - position[0])
        ]
        right_bit = bit_matrix[position[0] + 1][
            position[1] + 2 ** (position[2] - position[0] - 1):position[1] + 2 ** (position[2] - position[0])
        ]
        up_bit = bit_matrix[position[0]][position[1]:position[1] + 2 ** (position[2] - position[0])]

        if fn.all_num(up_bit) == 1:
            position = fn.up(position)
        elif fn.all_num(right_bit) == 1:
            up_bit_val = fn.get_up_bit(left_bit, right_bit)
            bit_matrix[position[0]][position[1]:position[1] + 2 ** (position[2] - position[0])] = up_bit_val.copy()
        elif fn.all_num(right_llr) == 1:
            if position[0] == position[2] - 1:
                right_bit_pos = position[1] + 1
                bit_matrix[position[0] + 1][
                    position[1] + 2 ** (position[2] - position[0] - 1):position[1] + 2 ** (position[2] - position[0])
                ] = fn.get_right_bit(right_llr[0], information_pos, frozen_bit, right_bit_pos)
            else:
                position = fn.rightdown(position)
        elif fn.all_num(left_bit) == 1:
            llr_matrix[position[0] + 1][
                position[1] + 2 ** (position[2] - position[0] - 1):position[1] + 2 ** (position[2] - position[0])
            ] = fn.get_right_llr(left_bit, up_llr)
        elif fn.all_num(left_llr) == 0:
            llr_matrix[position[0] + 1][position[1]:position[1] + 2 ** (position[2] - position[0] - 1)] = fn.get_left_llr(up_llr)
        else:
            if position[0] == position[2] - 1:
                left_bit_pos = position[1]
                bit_matrix[position[0] + 1][position[1]:position[1] + 2 ** (position[2] - position[0] - 1)] = fn.get_left_bit(
                    left_llr[0], information_pos, frozen_bit, left_bit_pos
                )
            else:
                position = fn.leftdown(position)
    return llr_matrix, bit_matrix
