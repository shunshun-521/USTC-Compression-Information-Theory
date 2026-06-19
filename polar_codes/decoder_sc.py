"""
极化码 SC（串行抵消）译码器
提供树形遍历实现（与 PolarCodesPython 一致）及非递归接口
"""
import math

import numpy as np


def f_operation(La, Lb):
    """min-sum 近似的 f 运算（sign(0) 视为 +1）"""
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    s1 = np.sign(La)
    s2 = np.sign(Lb)
    s1 = np.where(s1 == 0, 1, s1)
    s2 = np.where(s2 == 0, 1, s2)
    return s1 * s2 * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算"""
    u_hat = np.asarray(u_hat)
    return (1 - 2 * u_hat) * La + Lb


def _all_num(arr):
    return not np.any(np.isnan(arr))


def _leftdown(position):
    p0, p1, p2, p3 = position
    return [p0 + 1, p1, p2, p3]


def _rightdown(position):
    p0, p1, p2, p3 = position
    return [p0 + 1, p1 + 2 ** (p2 - p0 - 1), p2, p3]


def _up(position):
    p0, p1, p2, p3 = position
    block = 2 ** (p2 - p0 + 1)
    return [p0 - 1, int(np.floor(p1 / block) * block), p2, p3]


def _get_up_bit(left_bit, right_bit):
    length = left_bit.size
    temp = np.array([(left_bit + right_bit) % 2, right_bit])
    temp = temp.reshape(1, 2 * length)
    return temp


def _get_left_llr(up_llr):
    half = up_llr.size // 2
    return np.array([f_operation(up_llr[i], up_llr[i + half]) for i in range(half)])


def _get_right_llr(left_bit, up_llr):
    half = left_bit.size
    return np.array(
        [g_operation(up_llr[i], up_llr[i + half], left_bit[i]) for i in range(half)]
    )


def _get_left_bit(left_llr, information_pos, frozen_bit, left_bit_pos):
    if left_bit_pos in information_pos:
        return 0 if left_llr[0] >= 0 else 1
    return frozen_bit


def _get_right_bit(right_llr, information_pos, frozen_bit, right_bit_pos):
    if right_bit_pos in information_pos:
        return 0 if right_llr[0] > 0 else 1
    return frozen_bit


def _sc_tree_decode(llr_ch, information_pos, frozen_bit=0):
    """树形遍历 SC 译码"""
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    information_pos = np.asarray(information_pos, dtype=int)
    N = llr_ch.size
    n = int(math.log2(N))

    llr_matrix = np.ones((n + 1, N), dtype=np.float64)
    llr_matrix[:] = np.nan
    bit_matrix = np.ones((n + 1, N), dtype=np.float64)
    bit_matrix[:] = np.nan
    llr_matrix[0] = llr_ch
    position = [0, 0, n, N]

    while not _all_num(bit_matrix[n]):
        span = 2 ** (position[2] - position[0])
        half = span // 2
        p0, p1 = position[0], position[1]

        up_llr = llr_matrix[p0][p1 : p1 + span]
        up_bit = bit_matrix[p0][p1 : p1 + span]
        left_llr = llr_matrix[p0 + 1][p1 : p1 + half]
        left_bit = bit_matrix[p0 + 1][p1 : p1 + half]
        right_llr = llr_matrix[p0 + 1][p1 + half : p1 + span]
        right_bit = bit_matrix[p0 + 1][p1 + half : p1 + span]

        if _all_num(up_bit):
            position = _up(position)
        elif _all_num(right_bit):
            up = _get_up_bit(left_bit, right_bit)
            bit_matrix[p0][p1 : p1 + span] = up.copy()
        elif _all_num(right_llr):
            if position[0] == position[2] - 1:
                right_val = _get_right_bit(
                    right_llr, information_pos, frozen_bit, p1 + 1
                )
                bit_matrix[p0 + 1][p1 + half : p1 + span] = right_val
            else:
                position = _rightdown(position)
        elif _all_num(left_bit):
            right_llr_new = _get_right_llr(left_bit, up_llr)
            llr_matrix[p0 + 1][p1 + half : p1 + span] = right_llr_new
        elif not _all_num(left_llr):
            left_llr_new = _get_left_llr(up_llr)
            llr_matrix[p0 + 1][p1 : p1 + half] = left_llr_new
        else:
            if position[0] == position[2] - 1:
                left_val = _get_left_bit(left_llr, information_pos, frozen_bit, p1)
                bit_matrix[p0 + 1][p1 : p1 + half] = left_val
            else:
                position = _leftdown(position)

    return bit_matrix[n].astype(int)


def precompute_sc_indices(N):
    """预计算非递归 SC 译码所需的辅助向量（保留接口）"""
    n = int(math.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]
    llr_layer_vec = []
    bit_layer_vec = []

    for phi in range(N):
        llr_layers = []
        bit_layers = []
        temp = phi
        for layer in range(n):
            if temp % 2 == 0:
                llr_layers.append(layer)
            else:
                bit_layers.append(layer)
            temp //= 2
        llr_layer_vec.append(llr_layers)
        bit_layer_vec.append(bit_layers)

    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """SC 译码主函数"""
    frozen_bits = np.asarray(frozen_bits, dtype=int)
    information_pos = np.where(frozen_bits == 0)[0]
    frozen_val = int(frozen_bits[information_pos[0] if len(information_pos) else 0])
    if len(information_pos) == 0:
        frozen_val = 0
    else:
        frozen_val = 0
    return _sc_tree_decode(llr_ch, information_pos, frozen_val)


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（委托树形实现）"""
    return sc_decode(llr, frozen_bits)
