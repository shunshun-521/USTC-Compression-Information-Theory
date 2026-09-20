"""
极化码 SC（串行抵消）译码器
基于树遍历的高效 SC 实现（与 u@G 蝶形编码配套）
"""
import numpy as np
import math


def f_operation(La, Lb):
    """min-sum 近似的 f 运算"""
    s1 = np.sign(La) if La != 0 else 1.0
    s2 = np.sign(Lb) if Lb != 0 else 1.0
    return s1 * s2 * min(abs(La), abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算"""
    return (1.0 - 2.0 * u_hat) * La + Lb


def _all_filled(row):
    return not np.any(np.isnan(row))


def _leftdown(pos):
    return [pos[0] + 1, pos[1], pos[2], pos[3]]


def _rightdown(pos):
    return [pos[0] + 1, pos[1] + 2 ** (pos[2] - 1 - pos[0]), pos[2], pos[3]]


def _up(pos):
    p0 = pos[0] - 1
    p1 = int(np.floor(pos[1] / (2 ** (pos[2] - pos[0] + 1))) * (2 ** (pos[2] - pos[0] + 1)))
    return [p0, p1, pos[2], pos[3]]


def _get_up_bit(left_bit, right_bit):
    length = len(left_bit)
    temp = np.array([(left_bit[i] + right_bit[i]) % 2 for i in range(length)] + list(right_bit))
    return temp


def _get_right_llr(left_bit, up_llr):
    length = len(left_bit)
    return np.array([g_operation(up_llr[i], up_llr[i + length], left_bit[i]) for i in range(length)])


def _get_left_llr(up_llr):
    length = len(up_llr) // 2
    return np.array([f_operation(up_llr[i], up_llr[i + length]) for i in range(length)])


def _get_bit(llr_val, is_info):
    if not is_info:
        return 0
    return 0 if llr_val >= 0 else 1


def sc_decode(llr_ch, frozen_bits):
    """
    SC 译码主函数。
    frozen_bits: bool 数组，True 表示冻结位
    """
    y_llr = np.asarray(llr_ch, dtype=np.float64)
    N = len(y_llr)
    n = int(math.log2(N))
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    info_pos = set(np.where(~frozen_bits)[0])

    llr_matrix = np.full((n + 1, N), np.nan, dtype=np.float64)
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
            bit_matrix[position[0]][position[1]:position[1] + span] = up_bit
        elif not _all_filled(right_llr):
            if _all_filled(left_bit):
                right_llr = _get_right_llr(left_bit, up_llr)
                llr_matrix[position[0] + 1][position[1] + half:position[1] + span] = right_llr
            elif not _all_filled(left_llr):
                left_llr = _get_left_llr(up_llr)
                llr_matrix[position[0] + 1][position[1]:position[1] + half] = left_llr
            elif position[0] == position[2] - 1:
                left_bit_pos = position[1]
                left_bit_val = _get_bit(left_llr[0], left_bit_pos in info_pos)
                bit_matrix[position[0] + 1][position[1]] = left_bit_val
            else:
                position = _leftdown(position)
        elif position[0] == position[2] - 1:
            right_bit_pos = position[1] + 1
            right_bit_val = _get_bit(right_llr[0], right_bit_pos in info_pos)
            bit_matrix[position[0] + 1][position[1] + half] = right_bit_val
        else:
            position = _rightdown(position)

    return bit_matrix[n].astype(int)


def sc_decode_recursive(llr_ch, frozen_bits):
    """递归版本（调用非递归实现）"""
    return sc_decode(llr_ch, frozen_bits)


def precompute_sc_indices(N):
    """兼容接口（非递归 SC 使用树遍历，无需预计算）"""
    n = int(math.log2(N))
    return [1 << d for d in range(n + 1)], [], []
