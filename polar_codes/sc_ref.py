"""Reference SC decoder functions (adapted from PolarCodesPython)"""
import numpy as np


def all_num(x):
    for v in x:
        if np.isnan(v):
            return 0
    return 1


def leftdown(position):
    return [position[0] + 1, position[1], position[2], position[3]]


def rightdown(position):
    return [position[0] + 1, position[1] + 2 ** (position[2] - 1 - position[0]), position[2], position[3]]


def up(position):
    p0 = position[0] - 1
    p1 = int(np.floor(position[1] / (2 ** (position[2] - position[0] + 1))) * (2 ** (position[2] - position[0] + 1)))
    return [p0, p1, position[2], position[3]]


def get_up_bit(left_bit, right_bit):
    length = len(left_bit)
    temp = np.array([(left_bit[i] + right_bit[i]) % 2 for i in range(length)] +
                    [right_bit[i] for i in range(length)])
    return temp


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


def get_right_llr(left_bit, up_llr):
    length = len(left_bit)
    return np.array([g(up_llr[i], up_llr[i + length], left_bit[i]) for i in range(length)])


def get_left_llr(up_llr):
    length = len(up_llr) // 2
    return np.array([f_hf(up_llr[i], up_llr[i + length]) for i in range(length)])


def get_right_bit(right_llr, information_pos, frozen_bit, right_bit_pos):
    if right_bit_pos in information_pos:
        return 0 if right_llr > 0 else 1
    return frozen_bit


def get_left_bit(left_llr, information_pos, frozen_bit, left_bit_pos):
    if left_bit_pos in information_pos:
        return 0 if left_llr >= 0 else 1
    return frozen_bit


def sc_decoder_ref(y_llr, information_pos, frozen_bit):
    N = y_llr.size
    n = int(np.log2(N))
    llr_matrix = np.ones((n + 1, N))
    llr_matrix[llr_matrix == 1] = float('nan')
    bit_matrix = llr_matrix.copy()
    llr_matrix[0] = y_llr
    position = [0, 0, n, N]

    while all_num(bit_matrix[n]) == 0:
        span = 2 ** (position[2] - position[0])
        up_llr = llr_matrix[position[0]][position[1]:position[1] + span]
        up_bit = bit_matrix[position[0]][position[1]:position[1] + span]
        half = span // 2
        left_llr = llr_matrix[position[0] + 1][position[1]:position[1] + half]
        left_bit = bit_matrix[position[0] + 1][position[1]:position[1] + half]
        right_llr = llr_matrix[position[0] + 1][position[1] + half:position[1] + span]
        right_bit = bit_matrix[position[0] + 1][position[1] + half:position[1] + span]

        if all_num(up_bit) == 1:
            position = up(position)
        elif all_num(right_bit) == 1:
            up_bit = get_up_bit(left_bit, right_bit)
            bit_matrix[position[0]][position[1]:position[1] + span] = up_bit.copy()
        elif all_num(right_llr) == 1:
            if position[0] == position[2] - 1:
                right_bit_pos = position[1] + 1
                right_bit_val = get_right_bit(right_llr[0], information_pos, frozen_bit, right_bit_pos)
                bit_matrix[position[0] + 1][position[1] + half] = right_bit_val
            else:
                position = rightdown(position)
        elif all_num(left_bit) == 1:
            right_llr_new = get_right_llr(left_bit, up_llr)
            llr_matrix[position[0] + 1][position[1] + half:position[1] + span] = right_llr_new
        elif all_num(left_llr) == 0:
            left_llr_new = get_left_llr(up_llr)
            llr_matrix[position[0] + 1][position[1]:position[1] + half] = left_llr_new
        else:
            if position[0] == position[2] - 1:
                left_bit_pos = position[1]
                left_bit_val = get_left_bit(left_llr[0], information_pos, frozen_bit, left_bit_pos)
                bit_matrix[position[0] + 1][position[1]] = left_bit_val
            else:
                position = leftdown(position)

    return bit_matrix[n].astype(int)
