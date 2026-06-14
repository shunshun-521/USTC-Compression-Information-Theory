"""SC 译码树遍历辅助函数（与标准极化码实现一致）。"""
import numpy as np


def all_num(x):
    for i in range(x.size):
        if np.isnan(x[i]):
            return 0
    return 1


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
    p0 = position[0] - 1
    p1 = int(
        np.floor(position[1] / (2 ** (position[2] - position[0] + 1)))
        * (2 ** (position[2] - position[0] + 1))
    )
    return [p0, p1, position[2], position[3]]


def get_up_bit(left_bit, right_bit):
    length = left_bit.size
    temp = np.array([(left_bit + right_bit) % 2, right_bit])
    temp.resize((1, 2 * length))
    return temp


def g(L1, L2, u):
    return (1 - 2 * u) * L1 + L2


def f_hf(L1, L2):
    s1 = np.sign(L1)
    s2 = np.sign(L2)
    if s1 == 0:
        s1 = 1
    if s2 == 0:
        s2 = 1
    if s1 == s2:
        return min(abs(L1), abs(L2))
    return -min(abs(L1), abs(L2))


def get_right_bit(right_llr, information_pos, frozen_bit, right_bit_pos):
    if right_bit_pos in information_pos:
        if right_llr > 0:
            return 0
        return 1
    return frozen_bit


def get_right_llr(left_bit, up_llr):
    length = int(left_bit.size)
    return np.array(
        [g(up_llr[i], up_llr[i + length], left_bit[i]) for i in range(length)]
    )


def get_left_bit(left_llr, information_pos, frozen_bit, left_bit_pos):
    if left_bit_pos in information_pos:
        if left_llr >= 0:
            return 0
        return 1
    return frozen_bit


def get_left_llr(up_llr):
    length = int(up_llr.size / 2)
    return np.array([f_hf(up_llr[i], up_llr[i + length]) for i in range(length)])


def get_up_loc(bit_matrix):
    N = int(bit_matrix[0].size)
    n = int(np.log2(N))
    detect_array = bit_matrix[n]
    detect = -1
    for i in range(N):
        if detect_array[i] != 0 and detect_array[i] != 1:
            detect = i - 1
            break
    if detect == -1:
        return [0, 0]
    if detect % 2 == 0:
        return [n - 1, detect]
    return [n - 1, detect - 1]


def sc_step_to_position(llr_matrix, bit_matrix, information_pos, frozen_bit, split_pos):
    """推进 SC 树直到 bit_matrix[n][split_pos] 完成判决。"""
    N = int(bit_matrix[0].size)
    n = int(np.log2(N))
    info = list(information_pos)
    loc = get_up_loc(bit_matrix)
    position = [loc[0], loc[1], n, N]

    while bit_matrix[n][split_pos] != 0 and bit_matrix[n][split_pos] != 1:
        span = 2 ** (position[2] - position[0])
        up_llr = llr_matrix[position[0]][position[1] : position[1] + span]
        up_bit = bit_matrix[position[0]][position[1] : position[1] + span]
        half = span // 2
        left_llr = llr_matrix[position[0] + 1][position[1] : position[1] + half]
        left_bit = bit_matrix[position[0] + 1][position[1] : position[1] + half]
        right_llr = llr_matrix[position[0] + 1][position[1] + half : position[1] + span]
        right_bit = bit_matrix[position[0] + 1][position[1] + half : position[1] + span]

        if all_num(up_bit) == 1:
            position = up(position)
        elif all_num(right_bit) == 1:
            up_bit = get_up_bit(left_bit, right_bit)
            bit_matrix[position[0]][position[1] : position[1] + span] = up_bit.copy()
        elif all_num(right_llr) == 1:
            if position[0] == position[2] - 1:
                right_bit_pos = position[1] + 1
                right_bit = get_right_bit(
                    right_llr, info, frozen_bit, right_bit_pos
                )
                bit_matrix[position[0] + 1][position[1] + half : position[1] + span] = (
                    right_bit
                )
            else:
                position = rightdown(position)
        elif all_num(left_bit) == 1:
            right_llr = get_right_llr(left_bit, up_llr)
            llr_matrix[position[0] + 1][position[1] + half : position[1] + span] = (
                right_llr
            )
        elif all_num(left_llr) == 0:
            left_llr = get_left_llr(up_llr)
            llr_matrix[position[0] + 1][position[1] : position[1] + half] = left_llr
        elif position[0] == position[2] - 1:
            left_bit_pos = position[1]
            left_bit = get_left_bit(left_llr, info, frozen_bit, left_bit_pos)
            bit_matrix[position[0] + 1][position[1] : position[1] + half] = left_bit
        else:
            position = leftdown(position)

    return llr_matrix, bit_matrix


def pm_update_hf(llr_array, bit_array):
    pm = 0.0
    for llr, bit in zip(np.atleast_1d(llr_array), np.atleast_1d(bit_array)):
        hard = 0 if llr >= 0 else 1
        if hard != bit:
            pm += abs(llr)
    return pm


def sc_tree_decode(y_llr, information_pos, frozen_bit=0):
    """SC 树遍历译码。"""
    N = y_llr.size
    n = int(np.log2(N))
    info = list(information_pos)
    llr_matrix = np.ones((n + 1, N))
    llr_matrix[llr_matrix == 1] = float("nan")
    bit_matrix = llr_matrix.copy()
    llr_matrix[0] = y_llr
    position = [0, 0, n, N]

    while all_num(bit_matrix[n]) == 0:
        span = 2 ** (position[2] - position[0])
        up_llr = llr_matrix[position[0]][position[1] : position[1] + span]
        up_bit = bit_matrix[position[0]][position[1] : position[1] + span]
        half = span // 2
        left_llr = llr_matrix[position[0] + 1][position[1] : position[1] + half]
        left_bit = bit_matrix[position[0] + 1][position[1] : position[1] + half]
        right_llr = llr_matrix[position[0] + 1][position[1] + half : position[1] + span]
        right_bit = bit_matrix[position[0] + 1][position[1] + half : position[1] + span]

        if all_num(up_bit) == 1:
            position = up(position)
        elif all_num(right_bit) == 1:
            up_bit = get_up_bit(left_bit, right_bit)
            bit_matrix[position[0]][position[1] : position[1] + span] = up_bit.copy()
        elif all_num(right_llr) == 1:
            if position[0] == position[2] - 1:
                right_bit_pos = position[1] + 1
                right_bit = get_right_bit(
                    right_llr, info, frozen_bit, right_bit_pos
                )
                bit_matrix[position[0] + 1][position[1] + half : position[1] + span] = (
                    right_bit
                )
            else:
                position = rightdown(position)
        elif all_num(left_bit) == 1:
            right_llr = get_right_llr(left_bit, up_llr)
            llr_matrix[position[0] + 1][position[1] + half : position[1] + span] = (
                right_llr
            )
        elif all_num(left_llr) == 0:
            left_llr = get_left_llr(up_llr)
            llr_matrix[position[0] + 1][position[1] : position[1] + half] = left_llr
        elif position[0] == position[2] - 1:
            left_bit_pos = position[1]
            left_bit = get_left_bit(left_llr, info, frozen_bit, left_bit_pos)
            bit_matrix[position[0] + 1][position[1] : position[1] + half] = left_bit
        else:
            position = leftdown(position)

    u_hat = np.zeros(N, dtype=int)
    for i, v in enumerate(bit_matrix[n]):
        u_hat[i] = 0 if v == 0 else 1
    return u_hat
