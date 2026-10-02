"""
极化码 SC（串行抵消）译码器
非递归树遍历实现（与蝶形 ``polar_encode`` 配套）
"""
import math

import numpy as np


def f_operation(La, Lb):
    """min-sum 近似 f（用于 BP 等模块）。"""
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    s1 = np.sign(La)
    s2 = np.sign(Lb)
    s1 = np.where(s1 == 0, 1, s1)
    s2 = np.where(s2 == 0, 1, s2)
    return s1 * s2 * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb"""
    return (1 - 2 * np.asarray(u_hat)) * La + Lb


def bit_reversed_index(i, n):
    result = 0
    for k in range(n):
        if i & (1 << k):
            result |= 1 << (n - 1 - k)
    return result


def precompute_sc_indices(N):
    """兼容接口：返回译码顺序。"""
    n = int(math.log2(N))
    return [bit_reversed_index(i, n) for i in range(N)], None, None


# ---------- SC 树遍历辅助 ----------


def _all_computed(x):
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


def _f_hf(l1, l2):
    s1 = 1 if np.sign(l1) == 0 else np.sign(l1)
    s2 = 1 if np.sign(l2) == 0 else np.sign(l2)
    return s1 * s2 * min(abs(l1), abs(l2))


def _g(l1, l2, u1):
    return (1 - 2 * int(u1)) * l1 + l2


def _get_left_llr(up_llr):
    length = len(up_llr) // 2
    return np.array([_f_hf(up_llr[i], up_llr[i + length]) for i in range(length)], dtype=np.float64)


def _get_right_llr(left_bit, up_llr):
    length = len(left_bit)
    return np.array([_g(up_llr[i], up_llr[i + length], left_bit[i]) for i in range(length)], dtype=np.float64)


def _get_up_bit(left_bit, right_bit):
    length = len(left_bit)
    temp = np.array([(left_bit[i] + right_bit[i]) % 2 for i in range(length)] + list(right_bit))
    return temp.reshape(1, 2 * length)


def _get_left_bit(left_llr, info_set, left_bit_pos):
    if left_bit_pos in info_set:
        return 0 if left_llr >= 0 else 1
    return 0


def _get_right_bit(right_llr, info_set, right_bit_pos):
    if right_bit_pos in info_set:
        return 0 if right_llr > 0 else 1
    return 0


def _sc_tree_decode(llr_ch, info_set):
    """核心 SC 树遍历，返回 (llr_matrix, bit_matrix)。"""
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    N = len(llr_ch)
    n = int(math.log2(N))
    info_set = set(int(i) for i in info_set)

    llr_matrix = np.ones((n + 1, N), dtype=np.float64)
    llr_matrix[llr_matrix == 1] = np.nan
    bit_matrix = llr_matrix.copy()
    llr_matrix[0] = llr_ch
    position = [0, 0, n, N]

    while not _all_computed(bit_matrix[n]):
        p0, p1, p2, _p3 = position
        span = 2 ** (p2 - p0)
        up_llr = llr_matrix[p0][p1 : p1 + span]
        up_bit = bit_matrix[p0][p1 : p1 + span]
        left_llr = llr_matrix[p0 + 1][p1 : p1 + span // 2]
        left_bit = bit_matrix[p0 + 1][p1 : p1 + span // 2]
        right_llr = llr_matrix[p0 + 1][p1 + span // 2 : p1 + span]
        right_bit = bit_matrix[p0 + 1][p1 + span // 2 : p1 + span]

        if _all_computed(up_bit):
            position = _up(position)
            continue

        if _all_computed(right_bit):
            up_res = _get_up_bit(left_bit, right_bit)
            bit_matrix[p0][p1 : p1 + span] = up_res.copy()
            continue

        if _all_computed(right_llr):
            if p0 == p2 - 1:
                right_bit_pos = p1 + 1
                rb = _get_right_bit(float(right_llr[0]), info_set, right_bit_pos)
                bit_matrix[p0 + 1][p1 + span // 2 : p1 + span] = rb
            else:
                position = _rightdown(position)
            continue

        if _all_computed(left_bit):
            rr = _get_right_llr(left_bit, up_llr)
            llr_matrix[p0 + 1][p1 + span // 2 : p1 + span] = rr
            continue

        if not _all_computed(left_llr):
            ll = _get_left_llr(up_llr)
            llr_matrix[p0 + 1][p1 : p1 + span // 2] = ll
            continue

        if p0 == p2 - 1:
            lb = _get_left_bit(float(left_llr[0]), info_set, p1)
            bit_matrix[p0 + 1][p1 : p1 + span // 2] = lb
        else:
            position = _leftdown(position)

    return llr_matrix, bit_matrix


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC 译码。"""
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    info_set = np.where(~frozen_bits)[0]
    _, bit_matrix = _sc_tree_decode(llr_ch, info_set)
    n = int(math.log2(len(llr_ch)))
    return bit_matrix[n].astype(int)


def sc_decode_recursive(llr, frozen_bits):
    """与 ``sc_decode`` 相同（保留递归别名供验证脚本调用）。"""
    return sc_decode(llr, frozen_bits)
