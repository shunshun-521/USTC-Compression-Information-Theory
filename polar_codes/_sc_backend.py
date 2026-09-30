"""SC 译码后端（状态机实现）"""
import numpy as np


def all_num(x):
    for v in x:
        if np.isnan(v):
            return 0
    return 1


def leftdown(position):
    return [position[0] + 1, position[1], position[2], position[3]]


def rightdown(position):
    p0 = position[0] + 1
    p1 = position[1] + 2 ** (position[2] - 1 - position[0])
    return [p0, p1, position[2], position[3]]


def up(position):
    p0 = position[0] - 1
    p1 = int(np.floor(position[1] / (2 ** (position[2] - position[0] + 1))) * (2 ** (position[2] - position[0] + 1)))
    return [p0, p1, position[2], position[3]]


def get_up_bit(left_bit, right_bit):
    length = left_bit.size
    temp = np.array([(left_bit + right_bit) % 2, right_bit])
    temp.resize((1, 2 * length))
    return temp


def get_right_bit(right_llr, information_pos, frozen_bit, right_bit_pos):
    if right_bit_pos in information_pos:
        return 0 if right_llr > 0 else 1
    return frozen_bit


def get_left_bit(left_llr, information_pos, frozen_bit, left_bit_pos):
    if left_bit_pos in information_pos:
        return 0 if left_llr >= 0 else 1
    return frozen_bit


def get_right_llr(left_bit, up_llr):
    length = int(left_bit.size)
    temp = np.array([
        (1 - 2 * left_bit[i]) * up_llr[i] + up_llr[i + length]
        for i in range(length)
    ])
    return temp


def get_left_llr(up_llr):
    length = int(up_llr.size / 2)
    temp = np.array([
        np.sign(up_llr[i]) * np.sign(up_llr[i + length]) * min(abs(up_llr[i]), abs(up_llr[i + length]))
        for i in range(length)
    ])
    return temp


def sc_decoder(y_llr, information_pos, frozen_bit):
    N = y_llr.size
    n = int(np.log2(N))
    llr_matrix = np.ones((n + 1, N))
    llr_matrix[llr_matrix == 1] = float("nan")
    bit_matrix = llr_matrix.copy()
    llr_matrix[0] = y_llr
    position = [0, 0, n, N]
    while all_num(bit_matrix[n]) == 0:
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

        if all_num(up_bit) == 1:
            position = up(position)
        else:
            if all_num(right_bit) == 1:
                up_bit = get_up_bit(left_bit, right_bit)
                bit_matrix[position[0]][position[1]:position[1] + 2 ** (position[2] - position[0])] = up_bit.copy()
            else:
                if all_num(right_llr) == 1:
                    if position[0] == position[2] - 1:
                        right_bit_pos = position[1] + 1
                        right_bit = get_right_bit(right_llr, information_pos, frozen_bit, right_bit_pos)
                        bit_matrix[position[0] + 1][
                            position[1] + 2 ** (position[2] - position[0] - 1):position[1] + 2 ** (position[2] - position[0])
                        ] = right_bit
                    else:
                        position = rightdown(position)
                else:
                    if all_num(left_bit) == 1:
                        right_llr = get_right_llr(left_bit, up_llr)
                        llr_matrix[position[0] + 1][
                            position[1] + 2 ** (position[2] - position[0] - 1):position[1] + 2 ** (position[2] - position[0])
                        ] = right_llr
                    else:
                        if all_num(left_llr) == 0:
                            left_llr = get_left_llr(up_llr)
                            llr_matrix[position[0] + 1][
                                position[1]:position[1] + 2 ** (position[2] - position[0] - 1)
                            ] = left_llr
                        else:
                            if position[0] == position[2] - 1:
                                left_bit_pos = position[1]
                                left_bit = get_left_bit(left_llr, information_pos, frozen_bit, left_bit_pos)
                                bit_matrix[position[0] + 1][
                                    position[1]:position[1] + 2 ** (position[2] - position[0] - 1)
                                ] = left_bit
                            else:
                                position = leftdown(position)
    return bit_matrix[n].astype(int)
