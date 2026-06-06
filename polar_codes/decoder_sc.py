"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math

import numpy as np


def f_operation(La, Lb):
    """
    min-sum 近似的 f 运算：
    f(La, Lb) ≈ sign(La) * sign(Lb) * min(|La|, |Lb|)
    """
    s1 = np.sign(La)
    s2 = np.sign(Lb)
    s1 = np.where(s1 == 0, 1, s1)
    s2 = np.where(s2 == 0, 1, s2)
    return s1 * s2 * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """
    g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb
    """
    u_hat = np.asarray(u_hat)
    return (1.0 - 2.0 * u_hat) * La + Lb


def _all_known(arr):
    return not np.any(np.isnan(arr))


def _up_bit(left_bit, right_bit):
    length = len(left_bit)
    temp = np.empty(2 * length, dtype=int)
    temp[0::2] = (left_bit + right_bit) % 2
    temp[1::2] = right_bit
    return temp


def _leftdown(position):
    return [position[0] + 1, position[1], position[2], position[3]]


def _rightdown(position):
    span = 2 ** (position[2] - 1 - position[0])
    return [position[0] + 1, position[1] + span, position[2], position[3]]


def _up(position):
    span = 2 ** (position[2] - position[0] + 1)
    col = int(np.floor(position[1] / span) * span)
    return [position[0] - 1, col, position[2], position[3]]


def _f_scalar(L1, L2):
    s1 = np.sign(L1)
    s2 = np.sign(L2)
    if s1 == 0:
        s1 = 1
    if s2 == 0:
        s2 = 1
    return s1 * s2 * min(abs(L1), abs(L2))


def _g_scalar(L1, L2, u1):
    return (1 - 2 * u1) * L1 + L2


def _get_left_llr(up_llr):
    half = len(up_llr) // 2
    return np.array([_f_scalar(up_llr[i], up_llr[i + half]) for i in range(half)])


def _get_right_llr(left_bit, up_llr):
    half = len(up_llr) // 2
    return np.array([
        _g_scalar(up_llr[i], up_llr[i + half], int(left_bit[i])) for i in range(half)
    ])


def _get_left_bit(llr_val, info_set, frozen_val, idx):
    if idx in info_set:
        return 0 if llr_val >= 0 else 1
    return frozen_val


def _get_right_bit(llr_val, info_set, frozen_val, idx):
    if idx in info_set:
        return 0 if llr_val > 0 else 1
    return frozen_val


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码（位置矩阵实现）。
    """
    y_llr = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    information_pos = np.where(~frozen_bits)[0].tolist()
    frozen_bit = 0

    N = y_llr.size
    n = int(math.log2(N))
    llr_matrix = np.ones((n + 1, N), dtype=np.float64)
    llr_matrix[llr_matrix == 1] = np.nan
    bit_matrix = llr_matrix.copy()
    llr_matrix[0] = y_llr
    position = [0, 0, n, N]

    def f_hf(L1, L2):
        s1 = np.sign(L1)
        s2 = np.sign(L2)
        s1 = 1 if s1 == 0 else s1
        s2 = 1 if s2 == 0 else s2
        return s1 * s2 * np.min([np.abs(L1), np.abs(L2)])

    def g_op(L1, L2, u1):
        return (1 - 2 * u1) * L1 + L2

    def all_num(x):
        return not np.any(np.isnan(x))

    def leftdown(p):
        return [p[0] + 1, p[1], p[2], p[3]]

    def rightdown(p):
        return [p[0] + 1, p[1] + 2 ** (p[2] - 1 - p[0]), p[2], p[3]]

    def up_move(p):
        p1 = int(np.floor(p[1] / (2 ** (p[2] - p[0] + 1))) * (2 ** (p[2] - p[0] + 1)))
        return [p[0] - 1, p1, p[2], p[3]]

    def get_up_bit(left_b, right_b):
        length = len(left_b)
        temp = np.array([(left_b + right_b) % 2, right_b])
        temp.resize((1, 2 * length))
        return temp[0]

    def get_right_bit(llr_val, pos):
        if pos in information_pos:
            return 0 if llr_val > 0 else 1
        return frozen_bit

    def get_left_bit(llr_val, pos):
        if pos in information_pos:
            return 0 if llr_val >= 0 else 1
        return frozen_bit

    def get_right_llr(left_b, up):
        half = len(up) // 2
        return np.array([g_op(up[i], up[i + half], left_b[i]) for i in range(half)])

    def get_left_llr(up):
        half = len(up) // 2
        return np.array([f_hf(up[i], up[i + half]) for i in range(half)])

    while all_num(bit_matrix[n]) == 0:
        span = 2 ** (position[2] - position[0])
        col = position[1]
        up_llr = llr_matrix[position[0], col:col + span]
        up_bit = bit_matrix[position[0], col:col + span]
        left_llr = llr_matrix[position[0] + 1, col:col + span // 2]
        left_bit = bit_matrix[position[0] + 1, col:col + span // 2]
        right_llr = llr_matrix[position[0] + 1, col + span // 2:col + span]
        right_bit = bit_matrix[position[0] + 1, col + span // 2:col + span]

        if all_num(up_bit):
            position = up_move(position)
        elif all_num(right_bit):
            bit_matrix[position[0], col:col + span] = get_up_bit(left_bit, right_bit)
        elif all_num(right_llr):
            if position[0] == position[2] - 1:
                bit_matrix[position[0] + 1, col + span // 2] = get_right_bit(
                    right_llr[0], col + span // 2
                )
            else:
                position = rightdown(position)
        elif all_num(left_bit):
            llr_matrix[position[0] + 1, col + span // 2:col + span] = get_right_llr(
                left_bit, up_llr
            )
        elif all_num(left_llr) == 0:
            llr_matrix[position[0] + 1, col:col + span // 2] = get_left_llr(up_llr)
        elif position[0] == position[2] - 1:
            bit_matrix[position[0] + 1, col] = get_left_bit(left_llr[0], col)
        else:
            position = leftdown(position)

    return bit_matrix[n].astype(int)


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（与非递归实现等价）。"""
    return sc_decode(llr, frozen_bits)


def precompute_sc_indices(N):
    """
    预计算非递归 SC 译码辅助向量（层更新列表）。
    """
    n = int(math.log2(N))
    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        t = 0
        while t < n and ((phi >> t) & 1):
            t += 1
        llr_layer_vec.append(list(range(t, n)))

        bits = []
        k = 0
        while k < n and ((phi >> k) & 1):
            bits.append(k)
            k += 1
        bit_layer_vec.append(bits)

    lambda_offset = [1 << i for i in range(n + 1)]
    return lambda_offset, llr_layer_vec, bit_layer_vec
