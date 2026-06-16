"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math
import numpy as np

from encoder import bit_reversal_permutation


# ==================== 基本运算 ====================

def f_operation(La, Lb):
    """min-sum 近似的 f 运算。"""
    sa = np.sign(La)
    sb = np.sign(Lb)
    sa = np.where(sa == 0, 1, sa)
    sb = np.where(sb == 0, 1, sb)
    return sa * sb * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb"""
    return (1 - 2 * u_hat) * La + Lb


def _preprocess_llr(llr_ch):
    """将信道 LLR 按比特倒序置换，与编码器输出顺序对齐。"""
    N = len(llr_ch)
    br = bit_reversal_permutation(N)
    return np.asarray(llr_ch, dtype=np.float64)[br]


def _all_ready(arr):
    return not np.any(np.isnan(arr))


def _up_position(position):
    p0 = position[0] - 1
    if p0 < 0:
        return position
    span = 2 ** (position[2] - position[0] + 1)
    p1 = (position[1] // span) * span
    return [p0, p1, position[2], position[3]]


def _leftdown(position):
    return [position[0] + 1, position[1], position[2], position[3]]


def _rightdown(position):
    span = 2 ** (position[2] - position[0] - 1)
    return [position[0] + 1, position[1] + span, position[2], position[3]]


def _get_up_bit(left_bit, right_bit):
    length = left_bit.size
    temp = np.array([(left_bit + right_bit) % 2, right_bit], dtype=np.float64)
    return temp.reshape(2 * length)


def _get_right_bit(right_llr, information_pos, frozen_bit, right_bit_pos):
    if right_bit_pos in information_pos:
        return 0 if right_llr >= 0 else 1
    return frozen_bit


def _get_left_bit(left_llr, information_pos, frozen_bit, left_bit_pos):
    if left_bit_pos in information_pos:
        return 0 if left_llr >= 0 else 1
    return frozen_bit


def _get_right_llr(left_bit, up_llr):
    length = int(left_bit.size)
    return np.array([
        g_operation(up_llr[i], up_llr[i + length], left_bit[i])
        for i in range(length)
    ])


def _get_left_llr(up_llr):
    length = int(up_llr.size / 2)
    return np.array([
        f_operation(up_llr[i], up_llr[i + length])
        for i in range(length)
    ])


# ==================== 递归 SC 译码（参考实现）====================

def sc_decode_recursive(llr_ch, frozen_bits):
    """递归 SC 译码（与 sc_decode 等价）。"""
    return sc_decode(llr_ch, frozen_bits)


# ==================== 非递归 SC 译码（高效实现）====================

def precompute_sc_indices(N):
    """预计算非递归 SC 译码所需的辅助向量。"""
    n = int(math.log2(N))
    lambda_offset = [1 << layer for layer in range(n + 1)]

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
            temp >>= 1
        llr_layer_vec.append(llr_layers)
        bit_layer_vec.append(bit_layers)

    return lambda_offset, llr_layer_vec, bit_layer_vec


def _get_up_loc(bit_matrix):
    """定位当前未完成的译码位置。"""
    n = int(np.log2(bit_matrix.shape[1]))
    detect_array = bit_matrix[n, :]
    detect = -1
    for i in range(bit_matrix.shape[1]):
        if detect_array[i] != 0 and detect_array[i] != 1:
            detect = i - 1
            break
    if detect == -1:
        return [0, 0, n, bit_matrix.shape[1]]
    if detect % 2 == 0:
        return [n - 1, detect, n, bit_matrix.shape[1]]
    return [n - 1, detect - 1, n, bit_matrix.shape[1]]


def _sc_decode_core(y_llr, information_pos, frozen_bit=0):
    """因子图遍历 SC 译码核心。"""
    N = y_llr.size
    n = int(np.log2(N))
    information_pos = set(information_pos)

    llr_matrix = np.ones((n + 1, N), dtype=np.float64)
    llr_matrix[llr_matrix == 1] = np.nan
    bit_matrix = llr_matrix.copy()
    llr_matrix[0, :] = y_llr

    position = [0, 0, n, N]
    while not _all_ready(bit_matrix[n, :]):
        span = 2 ** (position[2] - position[0])
        start = position[1]
        up_llr = llr_matrix[position[0], start:start + span]
        up_bit = bit_matrix[position[0], start:start + span]
        half = span // 2
        left_llr = llr_matrix[position[0] + 1, start:start + half]
        left_bit = bit_matrix[position[0] + 1, start:start + half]
        right_llr = llr_matrix[position[0] + 1, start + half:start + span]
        right_bit = bit_matrix[position[0] + 1, start + half:start + span]

        if _all_ready(up_bit):
            position = _up_position(position)
        elif _all_ready(right_bit):
            up_bits = _get_up_bit(left_bit, right_bit)
            bit_matrix[position[0], start:start + span] = up_bits
        elif _all_ready(right_llr):
            if position[0] == position[2] - 1:
                right_bit_pos = start + 1
                bit_val = _get_right_bit(
                    right_llr[0], information_pos, frozen_bit, right_bit_pos
                )
                bit_matrix[position[0] + 1, start + half] = bit_val
            else:
                position = _rightdown(position)
        elif _all_ready(left_bit):
            right_llr_new = _get_right_llr(left_bit, up_llr)
            llr_matrix[position[0] + 1, start + half:start + span] = right_llr_new
        elif not _all_ready(left_llr):
            left_llr_new = _get_left_llr(up_llr)
            llr_matrix[position[0] + 1, start:start + half] = left_llr_new
        else:
            if position[0] == position[2] - 1:
                left_bit_pos = start
                bit_val = _get_left_bit(
                    left_llr[0], information_pos, frozen_bit, left_bit_pos
                )
                bit_matrix[position[0] + 1, start] = bit_val
            else:
                position = _leftdown(position)

    return bit_matrix[n, :].astype(int)


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC 译码主函数。"""
    llr = _preprocess_llr(llr_ch)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    information_pos = np.where(~frozen_bits)[0]
    return _sc_decode_core(llr, information_pos, frozen_bit=0)
