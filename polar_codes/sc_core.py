"""SC/SCL 译码核心（迭代树遍历）"""
import numpy as np


def all_num(x):
    return not np.any(np.isnan(x))


def leftdown(position):
    return [position[0] + 1, position[1], position[2], position[3]]


def rightdown(position):
    return [
        position[0] + 1,
        position[1] + 2 ** (position[2] - 1 - position[0]),
        position[2],
        position[3],
    ]


def up(position):
    p1 = int(
        np.floor(position[1] / (2 ** (position[2] - position[0] + 1)))
        * (2 ** (position[2] - position[0] + 1))
    )
    return [position[0] - 1, p1, position[2], position[3]]


def get_up_bit(left_bit, right_bit):
    length = left_bit.size
    temp = np.array([(left_bit + right_bit) % 2, right_bit])
    temp.resize((1, 2 * length))
    return temp


def get_right_bit(right_llr, information_pos, frozen_bit, right_bit_pos):
    if right_bit_pos in information_pos:
        return 0 if right_llr > 0 else 1
    return frozen_bit


def get_right_llr(left_bit, up_llr):
    length = int(left_bit.size)
    return np.array([g(up_llr[i], up_llr[i + length], left_bit[i]) for i in range(length)])


def get_left_bit(left_llr, information_pos, frozen_bit, left_bit_pos):
    if left_bit_pos in information_pos:
        return 0 if left_llr >= 0 else 1
    return frozen_bit


def get_left_llr(up_llr):
    length = int(up_llr.size / 2)
    return np.array([f_hf(up_llr[i], up_llr[i + length]) for i in range(length)])


def f_hf(L1, L2):
    s1 = np.sign(L1)
    s2 = np.sign(L2)
    if s1 == 0:
        s1 = 1
    if s2 == 0:
        s2 = 1
    return s1 * s2 * min(abs(L1), abs(L2))


def g(L1, L2, U1):
    return (1 - 2 * U1) * L1 + L2


def get_up_loc(bit_matrix):
    N = int(bit_matrix[0].size)
    n = int(np.log2(N))
    detect_array = bit_matrix[n]
    detect = -1
    for i in range(N):
        if detect_array[i] != 0 and detect_array[i] != 1:
            detect = i - 1
            break
    if detect % 2 == 0:
        loc_row = n - 1
        loc_col = detect
    else:
        loc_row = n - 1
        loc_col = detect - 1
    if detect == -1:
        loc_row = 0
        loc_col = 0
    return [loc_row, loc_col]


def get_pm_update(llr_array, bit_array):
    pm = 0.0
    for i in range(llr_array.size):
        hard = 0 if llr_array[i] >= 0 else 1
        if hard != bit_array[i]:
            pm += abs(llr_array[i])
    return pm


def sc_decode_tree(y_llr, information_pos, frozen_bit):
    N = y_llr.size
    n = int(np.log2(N))
    llr_matrix = np.full((n + 1, N), np.nan)
    bit_matrix = np.full((n + 1, N), np.nan)
    llr_matrix[0] = y_llr
    position = [0, 0, n, N]

    while not all_num(bit_matrix[n]):
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

        if all_num(up_bit):
            position = up(position)
        else:
            if all_num(right_bit):
                up_bit = get_up_bit(left_bit, right_bit)
                bit_matrix[position[0]][
                    position[1] : position[1] + 2 ** (position[2] - position[0])
                ] = up_bit.copy()
            else:
                if all_num(right_llr):
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
                    if all_num(left_bit):
                        right_llr = get_right_llr(left_bit, up_llr)
                        llr_matrix[position[0] + 1][
                            position[1]
                            + 2 ** (position[2] - position[0] - 1) : position[1]
                            + 2 ** (position[2] - position[0])
                        ] = right_llr
                    else:
                        if not all_num(left_llr):
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

    return bit_matrix[n], llr_matrix[n]


def sc_step_to_bit(llr_matrix, bit_matrix, information_pos, frozen_bit, target_bit):
    N = int(bit_matrix[0].size)
    n = int(np.log2(N))
    loc = get_up_loc(bit_matrix)
    position = [loc[0], loc[1], n, N]

    while bit_matrix[n][target_bit] != 0 and bit_matrix[n][target_bit] != 1:
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

        if all_num(up_bit):
            position = up(position)
        else:
            if all_num(right_bit):
                up_bit = get_up_bit(left_bit, right_bit)
                bit_matrix[position[0]][
                    position[1] : position[1] + 2 ** (position[2] - position[0])
                ] = up_bit.copy()
            else:
                if all_num(right_llr):
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
                    if all_num(left_bit):
                        right_llr = get_right_llr(left_bit, up_llr)
                        llr_matrix[position[0] + 1][
                            position[1]
                            + 2 ** (position[2] - position[0] - 1) : position[1]
                            + 2 ** (position[2] - position[0])
                        ] = right_llr
                    else:
                        if not all_num(left_llr):
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
