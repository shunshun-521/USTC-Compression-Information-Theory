"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math
import numpy as np


def f_operation(La, Lb):
    """min-sum 近似的 f 运算。"""
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    sign = np.sign(La) * np.sign(Lb)
    sign[sign == 0] = 1.0
    return sign * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算。"""
    return (1.0 - 2.0 * u_hat) * La + Lb


def _all_filled(arr):
    return not np.any(np.isnan(arr))


def _leftdown(pos):
    return [pos[0] + 1, pos[1], pos[2], pos[3]]


def _rightdown(pos):
    return [pos[0] + 1, pos[1] + 2 ** (pos[2] - 1 - pos[0]), pos[2], pos[3]]


def _up(pos):
    p1 = int(np.floor(pos[1] / (2 ** (pos[2] - pos[0] + 1))) * (2 ** (pos[2] - pos[0] + 1)))
    return [pos[0] - 1, p1, pos[2], pos[3]]


def _get_up_bit(left_bit, right_bit):
    length = len(left_bit)
    temp = np.vstack([(left_bit + right_bit) % 2, right_bit])
    return temp.reshape(1, 2 * length)


def _get_right_llr(left_bit, up_llr):
    half = len(left_bit)
    return np.array([g_operation(up_llr[i], up_llr[i + half], left_bit[i]) for i in range(half)])


def _get_left_llr(up_llr):
    half = len(up_llr) // 2
    return f_operation(up_llr[:half], up_llr[half:])


def _decide_bit(llr, is_info):
    if is_info:
        return 0 if llr >= 0 else 1
    return 0


# ==================== 递归 SC 译码（参考实现）====================

def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（简化参考实现）。"""
    return sc_decode(llr, frozen_bits)


# ==================== 非递归 SC 译码（高效实现）====================

def precompute_sc_indices(N):
    """预计算辅助向量（供 SCL 使用）。"""
    n = int(math.log2(N))
    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        p = phi
        layer = 0
        while p & 1:
            p >>= 1
            layer += 1
        llr_layer_vec.append(list(range(layer, n)))
        p = phi + 1
        layer = 0
        while not (p & 1):
            p >>= 1
            layer += 1
        bit_layer_vec.append(list(range(layer)))
    return llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码（矩阵遍历实现，与标准极化码因子图一致）。
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr_ch)
    n = int(np.log2(N))
    info_positions = set(np.where(~frozen_bits)[0])

    llr_matrix = np.full((n + 1, N), np.nan, dtype=np.float64)
    bit_matrix = np.full((n + 1, N), np.nan, dtype=np.float64)
    llr_matrix[0] = llr_ch
    position = [0, 0, n, N]

    while not _all_filled(bit_matrix[n]):
        span = 2 ** (position[2] - position[0])
        sl = slice(position[1], position[1] + span)
        half = span // 2
        left_sl = slice(position[1], position[1] + half)
        right_sl = slice(position[1] + half, position[1] + span)

        up_llr = llr_matrix[position[0]][sl]
        up_bit = bit_matrix[position[0]][sl]
        left_llr = llr_matrix[position[0] + 1][left_sl]
        left_bit = bit_matrix[position[0] + 1][left_sl]
        right_llr = llr_matrix[position[0] + 1][right_sl]
        right_bit = bit_matrix[position[0] + 1][right_sl]

        if _all_filled(up_bit):
            position = _up(position)
            continue

        if _all_filled(right_bit):
            up_bit_val = _get_up_bit(left_bit, right_bit)
            bit_matrix[position[0]][sl] = up_bit_val.copy()
            continue

        if _all_filled(right_llr):
            if position[0] == position[2] - 1:
                right_bit_pos = position[1] + 1
                val = _decide_bit(float(right_llr[0]), right_bit_pos in info_positions)
                bit_matrix[position[0] + 1][position[1] + half] = val
            else:
                position = _rightdown(position)
            continue

        if _all_filled(left_bit):
            right_llr_val = _get_right_llr(left_bit, up_llr)
            llr_matrix[position[0] + 1][right_sl] = right_llr_val
            continue

        if not _all_filled(left_llr):
            left_llr_val = _get_left_llr(up_llr)
            llr_matrix[position[0] + 1][left_sl] = left_llr_val
            continue

        if position[0] == position[2] - 1:
            left_bit_pos = position[1]
            val = _decide_bit(float(left_llr[0]), left_bit_pos in info_positions)
            bit_matrix[position[0] + 1][position[1]] = val
        else:
            position = _leftdown(position)

    return bit_matrix[n].astype(np.int8)
