"""SC 译码核心（基于标准矩阵布局 llr_matrix[0]=信道）"""
import numpy as np


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
    p0 = position[0] - 1
    p1 = int(np.floor(position[1] / (2 ** (position[2] - position[0] + 1))) * (2 ** (position[2] - position[0] + 1)))
    return [p0, p1, position[2], position[3]]


def _get_up_bit(left_bit, right_bit):
    length = left_bit.size
    temp = np.array([(left_bit + right_bit) % 2, right_bit])
    temp.resize((1, 2 * length))
    return temp


def _f_hf(L1, L2):
    s1 = 1 if L1 == 0 else np.sign(L1)
    s2 = 1 if L2 == 0 else np.sign(L2)
    return s1 * s2 * min(abs(L1), abs(L2))


def _g(L1, L2, U1):
    return (1 - 2 * U1) * L1 + L2


def _get_right_llr(left_bit, up_llr):
    length = int(left_bit.size)
    return np.array([_g(up_llr[i], up_llr[i + length], left_bit[i]) for i in range(length)])


def _get_left_llr(up_llr):
    length = int(up_llr.size / 2)
    return np.array([_f_hf(up_llr[i], up_llr[i + length]) for i in range(length)])


def _get_right_bit(right_llr, information_pos, frozen_bit, right_bit_pos):
    if right_bit_pos in information_pos:
        return 0 if right_llr > 0 else 1
    return frozen_bit


def _get_left_bit(left_llr, information_pos, frozen_bit, left_bit_pos):
    if left_bit_pos in information_pos:
        return 0 if left_llr >= 0 else 1
    return frozen_bit


def sc_decode_core(y_llr, information_pos, frozen_bit=0):
    N = y_llr.size
    n = int(np.log2(N))
    llr_matrix = np.ones((n + 1, N))
    llr_matrix[llr_matrix == 1] = np.nan
    bit_matrix = llr_matrix.copy()
    llr_matrix[0] = y_llr
    position = [0, 0, n, N]
    info_set = set(int(i) for i in information_pos)

    while not _all_num(bit_matrix[n]):
        span = 2 ** (position[2] - position[0])
        up_llr = llr_matrix[position[0]][position[1]:position[1] + span]
        up_bit = bit_matrix[position[0]][position[1]:position[1] + span]
        half = span // 2
        left_llr = llr_matrix[position[0] + 1][position[1]:position[1] + half]
        left_bit = bit_matrix[position[0] + 1][position[1]:position[1] + half]
        right_llr = llr_matrix[position[0] + 1][position[1] + half:position[1] + span]
        right_bit = bit_matrix[position[0] + 1][position[1] + half:position[1] + span]

        if _all_num(up_bit):
            position = _up(position)
        else:
            if _all_num(right_bit):
                up_bit = _get_up_bit(left_bit, right_bit)
                bit_matrix[position[0]][position[1]:position[1] + span] = up_bit.copy()
            elif _all_num(right_llr):
                if position[0] == position[2] - 1:
                    right_bit_pos = position[1] + 1
                    right_bit = _get_right_bit(right_llr, info_set, frozen_bit, right_bit_pos)
                    bit_matrix[position[0] + 1][position[1] + half:position[1] + span] = right_bit
                else:
                    position = _rightdown(position)
            elif _all_num(left_bit):
                right_llr = _get_right_llr(left_bit, up_llr)
                llr_matrix[position[0] + 1][position[1] + half:position[1] + span] = right_llr
            elif not _all_num(left_llr):
                left_llr = _get_left_llr(up_llr)
                llr_matrix[position[0] + 1][position[1]:position[1] + half] = left_llr
            else:
                if position[0] == position[2] - 1:
                    left_bit_pos = position[1]
                    left_bit = _get_left_bit(left_llr, info_set, frozen_bit, left_bit_pos)
                    bit_matrix[position[0] + 1][position[1]:position[1] + half] = left_bit
                else:
                    position = _leftdown(position)

    return bit_matrix[n].astype(int)


def _get_up_loc(bit_matrix):
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


def _pm_update(llr_array, bit_array):
    pm = 0.0
    for i in range(llr_array.size):
        if np.sign(llr_array[i]) != np.sign(1 - 2 * bit_array[i]):
            pm += abs(llr_array[i])
    return pm


def _sc_stepping(llr_matrix, bit_matrix, information_pos, frozen_bit, split_pos):
    N = int(bit_matrix[0].size)
    n = int(np.log2(N))
    info_set = set(information_pos)
    loc = _get_up_loc(bit_matrix)
    position = [loc[0], loc[1], n, N]
    while bit_matrix[n][split_pos] != 0 and bit_matrix[n][split_pos] != 1:
        span = 2 ** (position[2] - position[0])
        up_llr = llr_matrix[position[0]][position[1]:position[1] + span]
        up_bit = bit_matrix[position[0]][position[1]:position[1] + span]
        half = span // 2
        left_llr = llr_matrix[position[0] + 1][position[1]:position[1] + half]
        left_bit = bit_matrix[position[0] + 1][position[1]:position[1] + half]
        right_llr = llr_matrix[position[0] + 1][position[1] + half:position[1] + span]
        right_bit = bit_matrix[position[0] + 1][position[1] + half:position[1] + span]

        if _all_num(up_bit):
            position = _up(position)
        elif _all_num(right_bit):
            up_bit = _get_up_bit(left_bit, right_bit)
            bit_matrix[position[0]][position[1]:position[1] + span] = up_bit.copy()
        elif _all_num(right_llr):
            if position[0] == position[2] - 1:
                right_bit_pos = position[1] + 1
                right_bit = _get_right_bit(right_llr, info_set, frozen_bit, right_bit_pos)
                bit_matrix[position[0] + 1][position[1] + half:position[1] + span] = right_bit
            else:
                position = _rightdown(position)
        elif _all_num(left_bit):
            right_llr = _get_right_llr(left_bit, up_llr)
            llr_matrix[position[0] + 1][position[1] + half:position[1] + span] = right_llr
        elif not _all_num(left_llr):
            left_llr = _get_left_llr(up_llr)
            llr_matrix[position[0] + 1][position[1]:position[1] + half] = left_llr
        else:
            if position[0] == position[2] - 1:
                left_bit_pos = position[1]
                left_bit = _get_left_bit(left_llr, info_set, frozen_bit, left_bit_pos)
                bit_matrix[position[0] + 1][position[1]:position[1] + half] = left_bit
            else:
                position = _leftdown(position)
    return llr_matrix, bit_matrix


def scl_decode_core(y_llr, information_pos, list_size, frozen_bit=0):
    """SCL 译码（在信息位索引处分裂）"""
    N = y_llr.size
    n = int(np.log2(N))
    split_pos = list(information_pos)
    if not split_pos:
        return sc_decode_core(y_llr, information_pos, frozen_bit)

    llr_matrix = np.ones((n + 1, N))
    llr_matrix[llr_matrix == 1] = np.nan
    bit_matrix = llr_matrix.copy()
    llr_matrix[0] = y_llr
    llr_list = [llr_matrix]
    bit_list = [bit_matrix]
    pm_list = [0.0]
    split_loc = 0
    l_now = 1

    while split_loc < len(split_pos):
        new_llr, new_bit, new_pm = [], [], []
        for i in range(l_now):
            lm, bm, pm = llr_list[i], bit_list[i], pm_list[i]
            lm2, bm2 = _sc_stepping(lm.copy(), bm.copy(), split_pos, frozen_bit, split_pos[split_loc])
            bit_val = int(bm2[n][split_pos[split_loc]])
            llr_bit = lm2[n][split_pos[split_loc]]
            pm_add = _pm_update(np.array([llr_bit]), np.array([bit_val]))
            new_llr.append(lm2)
            new_bit.append(bm2)
            new_pm.append(pm + pm_add)
            bm_wrong = bm2.copy()
            bm_wrong[n][split_pos[split_loc]] = 1 - bit_val
            wrong_val = int(bm_wrong[n][split_pos[split_loc]])
            pm_wrong = pm + _pm_update(np.array([llr_bit]), np.array([wrong_val]))
            new_llr.append(lm2.copy())
            new_bit.append(bm_wrong)
            new_pm.append(pm_wrong)

        order = np.argsort(new_pm)[:list_size]
        llr_list = [new_llr[i] for i in order]
        bit_list = [new_bit[i] for i in order]
        pm_list = [new_pm[i] for i in order]
        l_now = len(pm_list)
        split_loc += 1

    if split_pos[-1] != N - 1:
        for i in range(l_now):
            lm, bm, pm = llr_list[i], bit_list[i], pm_list[i]
            lm2, bm2 = _sc_stepping(lm, bm, split_pos, frozen_bit, N - 1)
            llr_list[i], bit_list[i] = lm2, bm2

    best = int(np.argmin(pm_list))
    return bit_list[best][n].astype(int), float(pm_list[best])
