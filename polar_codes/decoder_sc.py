"""
极化码 SC（串行抵消）译码器
基于树形遍历的非递归实现（对数域 min-sum f，g 为线性组合）
"""
import math

import numpy as np


def _all_num(x):
    return not np.any(np.isnan(x))


def _leftdown(position):
    return [position[0] + 1, position[1], position[2], position[3]]


def _rightdown(position):
    return [position[0] + 1, position[1] + 2 ** (position[2] - 1 - position[0]), position[2], position[3]]


def _up(position):
    p0 = position[0] - 1
    p1 = int(np.floor(position[1] / (2 ** (position[2] - position[0] + 1))) * (2 ** (position[2] - position[0] + 1)))
    return [p0, p1, position[2], position[3]]


def _get_up_bit(left_bit, right_bit):
    length = left_bit.size
    temp = np.array([(left_bit + right_bit) % 2, right_bit])
    temp.resize((1, 2 * length))
    return temp


def f_operation(La, Lb):
    """min-sum 近似 f 运算"""
    s1 = np.sign(La) or 1
    s2 = np.sign(Lb) or 1
    return s1 * s2 * min(abs(La), abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算"""
    return (1 - 2 * int(u_hat)) * La + Lb


def _get_left_llr(up_llr):
    half = len(up_llr) // 2
    return np.array([f_operation(up_llr[i], up_llr[i + half]) for i in range(half)])


def _get_right_llr(left_bit, up_llr):
    half = len(left_bit)
    return np.array([g_operation(up_llr[i], up_llr[i + half], left_bit[i]) for i in range(half)])


def _get_left_bit(left_llr, info_set, frozen_val, pos):
    if pos in info_set:
        return 0 if left_llr >= 0 else 1
    return frozen_val


def _get_right_bit(right_llr, info_set, frozen_val, pos):
    if pos in info_set:
        return 0 if right_llr > 0 else 1
    return frozen_val


def _sc_decode_core(y_llr, info_set, frozen_val):
    N = y_llr.size
    n = int(np.log2(N))
    llr_matrix = np.ones((n + 1, N), dtype=np.float64)
    llr_matrix[:] = np.nan
    bit_matrix = llr_matrix.copy()
    llr_matrix[0] = y_llr
    position = [0, 0, n, N]

    while not _all_num(bit_matrix[n]):
        span = 2 ** (position[2] - position[0])
        up_llr = llr_matrix[position[0]][position[1]:position[1] + span]
        up_bit = bit_matrix[position[0]][position[1]:position[1] + span]
        left_llr = llr_matrix[position[0] + 1][position[1]:position[1] + span // 2]
        left_bit = bit_matrix[position[0] + 1][position[1]:position[1] + span // 2]
        right_llr = llr_matrix[position[0] + 1][position[1] + span // 2:position[1] + span]
        right_bit = bit_matrix[position[0] + 1][position[1] + span // 2:position[1] + span]

        if _all_num(up_bit):
            position = _up(position)
        elif _all_num(right_bit):
            bit_matrix[position[0]][position[1]:position[1] + span] = _get_up_bit(left_bit, right_bit).copy()
        elif _all_num(right_llr):
            if position[0] == position[2] - 1:
                pos = position[1] + 1
                bit_matrix[position[0] + 1][position[1] + span // 2:position[1] + span] = _get_right_bit(
                    right_llr, info_set, frozen_val, pos
                )
            else:
                position = _rightdown(position)
        elif _all_num(left_bit):
            llr_matrix[position[0] + 1][position[1] + span // 2:position[1] + span] = _get_right_llr(left_bit, up_llr)
        elif not _all_num(left_llr):
            llr_matrix[position[0] + 1][position[1]:position[1] + span // 2] = _get_left_llr(up_llr)
        elif position[0] == position[2] - 1:
            bit_matrix[position[0] + 1][position[1]:position[1] + span // 2] = _get_left_bit(
                left_llr, info_set, frozen_val, position[1]
            )
        else:
            position = _leftdown(position)

    return bit_matrix[n].astype(int)


def _get_up_loc(bit_matrix):
    N = int(bit_matrix[0].size)
    n = int(np.log2(N))
    detect_array = bit_matrix[n]
    detect = -1
    for i in range(N):
        if detect_array[i] not in (0, 1):
            detect = i - 1
            break
    if detect == -1:
        return [0, 0]
    if detect % 2 == 0:
        return [n - 1, detect]
    return [n - 1, detect - 1]


def _sc_stepping_decoder(llr_matrix, bit_matrix, info_set, frozen_val, split_pos):
    N = int(bit_matrix[0].size)
    n = int(np.log2(N))
    loc = _get_up_loc(bit_matrix)
    position = [loc[0], loc[1], n, N]

    while bit_matrix[n][split_pos] not in (0, 1):
        span = 2 ** (position[2] - position[0])
        up_llr = llr_matrix[position[0]][position[1]:position[1] + span]
        up_bit = bit_matrix[position[0]][position[1]:position[1] + span]
        left_llr = llr_matrix[position[0] + 1][position[1]:position[1] + span // 2]
        left_bit = bit_matrix[position[0] + 1][position[1]:position[1] + span // 2]
        right_llr = llr_matrix[position[0] + 1][position[1] + span // 2:position[1] + span]
        right_bit = bit_matrix[position[0] + 1][position[1] + span // 2:position[1] + span]

        if _all_num(up_bit):
            position = _up(position)
        elif _all_num(right_bit):
            bit_matrix[position[0]][position[1]:position[1] + span] = _get_up_bit(left_bit, right_bit).copy()
        elif _all_num(right_llr):
            if position[0] == position[2] - 1:
                pos = position[1] + 1
                bit_matrix[position[0] + 1][position[1] + span // 2:position[1] + span] = _get_right_bit(
                    right_llr, info_set, frozen_val, pos
                )
            else:
                position = _rightdown(position)
        elif _all_num(left_bit):
            llr_matrix[position[0] + 1][position[1] + span // 2:position[1] + span] = _get_right_llr(left_bit, up_llr)
        elif not _all_num(left_llr):
            llr_matrix[position[0] + 1][position[1]:position[1] + span // 2] = _get_left_llr(up_llr)
        elif position[0] == position[2] - 1:
            bit_matrix[position[0] + 1][position[1]:position[1] + span // 2] = _get_left_bit(
                left_llr, info_set, frozen_val, position[1]
            )
        else:
            position = _leftdown(position)

    return llr_matrix, bit_matrix


def _pm_update_segment(llr_array, bit_array):
    pm = 0.0
    for llr, bit in zip(llr_array, bit_array):
        hard = 0 if llr >= 0 else 1
        if bit != hard:
            pm += abs(llr)
    return pm
    """预计算辅助向量（接口兼容）"""
    n = int(math.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]
    llr_layer_vec = [list(range(n)) for _ in range(N)]
    bit_layer_vec = [list(range(n)) for _ in range(N)]
    return lambda_offset, llr_layer_vec, bit_layer_vec


def precompute_sc_indices(N):
    """预计算辅助向量（接口兼容，当前译码器使用树形遍历）"""
    n = int(math.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]
    llr_layer_vec = [[] for _ in range(N)]
    bit_layer_vec = [[] for _ in range(N)]
    return lambda_offset, llr_layer_vec, bit_layer_vec


def _frozen_to_info_set(frozen_bits):
    frozen_bits = np.asarray(frozen_bits, dtype=int)
    info_set = set(np.where(frozen_bits == 0)[0])
    return info_set, 0


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC 译码"""
    info_set, frozen_val = _frozen_to_info_set(frozen_bits)
    return _sc_decode_core(np.asarray(llr_ch, dtype=np.float64), info_set, frozen_val)


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 参考实现（调用非递归核心）"""
    return sc_decode(llr, frozen_bits)
