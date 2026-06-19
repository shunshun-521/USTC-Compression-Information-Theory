"""
极化码 SC（串行抵消）译码器
树形遍历实现，信道 LLR 经比特倒序逆置换后与编码器对齐
"""
import math

import numpy as np

from encoder import bit_reversal_permutation


def f_operation(La, Lb):
    """min-sum 近似的 f 运算。"""
    s1 = np.sign(La)
    s2 = np.sign(Lb)
    s1 = np.where(s1 == 0, 1, s1)
    s2 = np.where(s2 == 0, 1, s2)
    return s1 * s2 * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算。"""
    return (1 - 2 * u_hat) * La + Lb


def reorder_channel_llr(llr_ch):
    """比特倒序编码下，将信道 LLR 逆置换到译码树自然序。"""
    N = len(llr_ch)
    inv_br = np.argsort(bit_reversal_permutation(N))
    return np.asarray(llr_ch, dtype=np.float64)[inv_br]


def _all_filled(arr):
    return not np.isnan(arr).any()


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
    p1 = int(np.floor(position[1] / (2 ** (position[2] - position[0] + 1)))
             * (2 ** (position[2] - position[0] + 1)))
    return [position[0] - 1, p1, position[2], position[3]]


def _get_up_bit(left_bit, right_bit):
    length = left_bit.size
    temp = np.array([(left_bit + right_bit) % 2, right_bit])
    temp.resize((1, 2 * length))
    return temp[0]


def _get_left_llr(up_llr):
    length = up_llr.size // 2
    return np.array([f_operation(up_llr[i], up_llr[i + length]) for i in range(length)])


def _get_right_llr(left_bit, up_llr):
    length = left_bit.size
    return np.array([g_operation(up_llr[i], up_llr[i + length], left_bit[i]) for i in range(length)])


def _sc_decode_tree(y_llr, information_pos, frozen_value=0):
    """SC 树形译码核心。"""
    N = y_llr.size
    n = int(math.log2(N))
    info_set = set(information_pos)
    llr_matrix = np.full((n + 1, N), np.nan)
    bit_matrix = np.full((n + 1, N), np.nan)
    llr_matrix[0] = y_llr
    position = [0, 0, n, N]

    while not _all_filled(bit_matrix[n]):
        span = 2 ** (position[2] - position[0])
        up_llr = llr_matrix[position[0]][position[1]:position[1] + span]
        up_bit = bit_matrix[position[0]][position[1]:position[1] + span]
        half = span // 2
        left_llr = llr_matrix[position[0] + 1][position[1]:position[1] + half]
        left_bit = bit_matrix[position[0] + 1][position[1]:position[1] + half]
        right_llr = llr_matrix[position[0] + 1][position[1] + half:position[1] + span]
        right_bit = bit_matrix[position[0] + 1][position[1] + half:position[1] + span]

        if _all_filled(up_bit):
            position = _up(position)
        elif _all_filled(right_bit):
            up_bit = _get_up_bit(left_bit, right_bit)
            bit_matrix[position[0]][position[1]:position[1] + span] = up_bit.copy()
        elif _all_filled(right_llr):
            if position[0] == position[2] - 1:
                right_bit_pos = position[1] + 1
                if right_bit_pos in info_set:
                    bit = 0 if right_llr[0] > 0 else 1
                else:
                    bit = frozen_value
                bit_matrix[position[0] + 1][position[1] + half:position[1] + span] = bit
            else:
                position = _rightdown(position)
        elif _all_filled(left_bit):
            right_llr = _get_right_llr(left_bit, up_llr)
            bit_matrix_slice = llr_matrix[position[0] + 1]
            bit_matrix_slice[position[1] + half:position[1] + span] = right_llr
        elif not _all_filled(left_llr):
            left_llr = _get_left_llr(up_llr)
            llr_matrix[position[0] + 1][position[1]:position[1] + half] = left_llr
        elif position[0] == position[2] - 1:
            left_bit_pos = position[1]
            if left_bit_pos in info_set:
                bit = 0 if left_llr[0] >= 0 else 1
            else:
                bit = frozen_value
            bit_matrix[position[0] + 1][position[1]:position[1] + half] = bit
        else:
            position = _leftdown(position)

    return bit_matrix[n].astype(int)


def _get_up_loc(bit_row):
    """定位树中下一个待译码位置。"""
    n = int(math.log2(len(bit_row)))
    detect = -1
    for i in range(len(bit_row)):
        if bit_row[i] not in (0, 1):
            detect = i - 1
            break
    if detect == -1:
        return [0, 0]
    if detect % 2 == 0:
        return [n - 1, detect]
    return [n - 1, detect - 1]


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC 译码主函数。"""
    frozen_bits = np.asarray(frozen_bits, dtype=int)
    information_pos = np.where(frozen_bits == 0)[0]
    llr = reorder_channel_llr(llr_ch)
    return _sc_decode_tree(llr, information_pos, frozen_value=0)


def sc_decode_recursive(llr_ch, frozen_bits):
    """递归 SC 译码（与树形实现等价）。"""
    return sc_decode(llr_ch, frozen_bits)


def precompute_sc_indices(N):
    """预计算辅助向量（接口保留）。"""
    n = int(math.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]
    llr_layer_vec = [[] for _ in range(N)]
    bit_layer_vec = [[] for _ in range(N)]
    for phi in range(N):
        temp = phi
        for layer in range(n):
            if temp % 2 == 0:
                llr_layer_vec[phi].append(layer)
            temp //= 2
        temp = phi
        for layer in range(n):
            if temp % 2 == 1:
                bit_layer_vec[phi].append(layer)
            temp //= 2
    return lambda_offset, llr_layer_vec, bit_layer_vec
