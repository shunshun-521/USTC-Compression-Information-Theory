"""参考 SC 译码器封装（树形遍历，已验证正确）"""
import numpy as np

from _ref_function import (
    all_num,
    get_left_bit,
    get_left_llr,
    get_right_bit,
    get_right_llr,
    get_up_bit,
    leftdown,
    rightdown,
    up,
)


def sc_decoder_ref(y_llr, information_pos, frozen_bit=0):
    N = y_llr.size
    n = int(np.log2(N))
    llr_matrix = np.ones((n + 1, N), dtype=np.float64)
    llr_matrix[llr_matrix == 1] = np.nan
    bit_matrix = llr_matrix.copy()
    llr_matrix[0] = y_llr
    position = [0, 0, n, N]

    while all_num(bit_matrix[n]) == 0:
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

        if all_num(up_bit) == 1:
            position = up(position)
        else:
            if all_num(right_bit) == 1:
                up_bit = get_up_bit(left_bit, right_bit)
                bit_matrix[position[0]][
                    position[1] : position[1] + 2 ** (position[2] - position[0])
                ] = up_bit.copy()
            else:
                if all_num(right_llr) == 1:
                    if position[0] == position[2] - 1:
                        right_bit_pos = position[1] + 1
                        right_bit = get_right_bit(
                            right_llr, information_pos, frozen_bit, right_bit_pos
                        )
                        bit_matrix[position[0] + 1][
                            position[1]
                            + 2 ** (position[2] - position[0] - 1) : position[1]
                            + 2 ** (position[2] - position[0])
                        ] = right_bit
                    else:
                        position = rightdown(position)
                else:
                    if all_num(left_bit) == 1:
                        right_llr = get_right_llr(left_bit, up_llr)
                        llr_matrix[position[0] + 1][
                            position[1]
                            + 2 ** (position[2] - position[0] - 1) : position[1]
                            + 2 ** (position[2] - position[0])
                        ] = right_llr
                    else:
                        if all_num(left_llr) == 0:
                            left_llr = get_left_llr(up_llr)
                            llr_matrix[position[0] + 1][
                                position[1] : position[1]
                                + 2 ** (position[2] - position[0] - 1)
                            ] = left_llr
                        else:
                            if position[0] == position[2] - 1:
                                left_bit_pos = position[1]
                                left_bit = get_left_bit(
                                    left_llr,
                                    information_pos,
                                    frozen_bit,
                                    left_bit_pos,
                                )
                                bit_matrix[position[0] + 1][
                                    position[1] : position[1]
                                    + 2 ** (position[2] - position[0] - 1)
                                ] = left_bit
                            else:
                                position = leftdown(position)

    return bit_matrix[n].astype(int)


def sc_stepping_decoder_ref(llr_matrix, bit_matrix, information_pos, frozen_bit, split_pos):
    N = int(bit_matrix[0].size)
    n = int(np.log2(N))
    from _ref_function import get_up_loc

    loc = get_up_loc(bit_matrix)
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

        if all_num(up_bit) == 1:
            position = up(position)
        else:
            if all_num(right_bit) == 1:
                up_bit = get_up_bit(left_bit, right_bit)
                bit_matrix[position[0]][
                    position[1] : position[1] + 2 ** (position[2] - position[0])
                ] = up_bit.copy()
            else:
                if all_num(right_llr) == 1:
                    if position[0] == position[2] - 1:
                        right_bit_pos = position[1] + 1
                        right_bit = get_right_bit(
                            right_llr, information_pos, frozen_bit, right_bit_pos
                        )
                        bit_matrix[position[0] + 1][
                            position[1]
                            + 2 ** (position[2] - position[0] - 1) : position[1]
                            + 2 ** (position[2] - position[0])
                        ] = right_bit
                    else:
                        position = rightdown(position)
                else:
                    if all_num(left_bit) == 1:
                        right_llr = get_right_llr(left_bit, up_llr)
                        llr_matrix[position[0] + 1][
                            position[1]
                            + 2 ** (position[2] - position[0] - 1) : position[1]
                            + 2 ** (position[2] - position[0])
                        ] = right_llr
                    else:
                        if all_num(left_llr) == 0:
                            left_llr = get_left_llr(up_llr)
                            llr_matrix[position[0] + 1][
                                position[1] : position[1]
                                + 2 ** (position[2] - position[0] - 1)
                            ] = left_llr
                        else:
                            if position[0] == position[2] - 1:
                                left_bit_pos = position[1]
                                left_bit = get_left_bit(
                                    left_llr,
                                    information_pos,
                                    frozen_bit,
                                    left_bit_pos,
                                )
                                bit_matrix[position[0] + 1][
                                    position[1] : position[1]
                                    + 2 ** (position[2] - position[0] - 1)
                                ] = left_bit
                            else:
                                position = leftdown(position)
    return llr_matrix, bit_matrix


def scl_decoder_ref(y_llr, information_pos, frozen_bit, list_size, crc_check_fn=None):
    from _ref_function import get_pm_update

    N = y_llr.size
    n = int(np.log2(N))
    llr_matrix = np.ones((n + 1, N), dtype=np.float64)
    llr_matrix[llr_matrix == 1] = np.nan
    bit_matrix = llr_matrix.copy()
    llr_matrix[0] = y_llr
    split_pos = list(information_pos)
    llr_list = [llr_matrix]
    bit_list = [bit_matrix]
    pm_list = [0.0]
    split_loc = 0
    split_len = len(split_pos)
    l_now = 1

    while split_len - 1 >= split_loc:
        new_llr_list = []
        new_bit_list = []
        new_pm_list = []
        for i in range(l_now):
            llr_temp = llr_list[i]
            bit_temp = bit_list[i]
            pm_temp = pm_list[i]
            llr_out, bit_out = sc_stepping_decoder_ref(
                llr_temp.copy(), bit_temp.copy(), information_pos, frozen_bit, split_pos[split_loc]
            )
            prev = split_pos[split_loc - 1] + 1 if split_loc > 0 else 0
            cur = split_pos[split_loc] + 1
            llr_seg = llr_out[n][prev:cur]
            bit_seg = bit_out[n][prev:cur]
            pm0 = pm_temp + get_pm_update(llr_seg, bit_seg, "hf")
            new_llr_list.append(llr_out)
            new_bit_list.append(bit_out)
            new_pm_list.append(pm0)
            bit_wrong = bit_out.copy()
            bit_wrong[n][split_pos[split_loc]] = 1 - bit_wrong[n][split_pos[split_loc]]
            bit_seg_w = bit_wrong[n][prev:cur]
            pm1 = pm_temp + get_pm_update(llr_seg, bit_seg_w, "hf")
            new_llr_list.append(llr_out.copy())
            new_bit_list.append(bit_wrong)
            new_pm_list.append(pm1)

        order = np.argsort(new_pm_list)[:list_size]
        llr_list = [new_llr_list[i] for i in order]
        bit_list = [new_bit_list[i] for i in order]
        pm_list = [new_pm_list[i] for i in order]
        l_now = len(pm_list)
        split_loc += 1

    if split_pos and split_pos[-1] != N - 1:
        for i in range(l_now):
            llr_temp = llr_list[i]
            bit_temp = bit_list[i]
            pm_temp = pm_list[i]
            llr_out, bit_out = sc_stepping_decoder_ref(
                llr_temp.copy(), bit_temp.copy(), information_pos, frozen_bit, N - 1
            )
            prev = split_pos[split_loc - 1] + 1
            pm_temp += get_pm_update(llr_out[n][prev:N], bit_out[n][prev:N], "hf")
            llr_list[i] = llr_out
            bit_list[i] = bit_out
            pm_list[i] = pm_temp

    order = np.argsort(pm_list)
    for idx in order:
        u_hat = bit_list[idx][n].astype(int)
        if crc_check_fn is None or crc_check_fn(u_hat):
            return u_hat, pm_list[idx]
    best = order[0]
    return bit_list[best][n].astype(int), pm_list[best]
