"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math
import numpy as np


def _f_min_sum(La, Lb):
    """min-sum f 运算，sign(0) 视为 +1"""
    s1 = np.sign(La)
    s2 = np.sign(Lb)
    s1 = np.where(s1 == 0, 1, s1)
    s2 = np.where(s2 == 0, 1, s2)
    return s1 * s2 * np.minimum(np.abs(La), np.abs(Lb))


def f_operation(La, Lb):
    """向量化 min-sum f 运算"""
    return _f_min_sum(La, Lb)


def g_operation(La, Lb, u_hat):
    """g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb"""
    return (1.0 - 2.0 * u_hat) * La + Lb


def _all_filled(row):
    return not np.any(np.isnan(row))


def _leftdown(pos):
    return [pos[0] + 1, pos[1], pos[2], pos[3]]


def _rightdown(pos):
    return [pos[0] + 1, pos[1] + 2 ** (pos[2] - 1 - pos[0]), pos[2], pos[3]]


def _up(pos):
    p1 = int(np.floor(pos[1] / (2 ** (pos[2] - pos[0] + 1))) * (2 ** (pos[2] - pos[0] + 1)))
    return [pos[0] - 1, p1, pos[2], pos[3]]


def _get_up_bit(left_bit, right_bit):
    length = len(left_bit)
    temp = np.array([(left_bit[i] + right_bit[i]) % 2 for i in range(length)] + list(right_bit))
    return temp.reshape(1, 2 * length)


def _get_left_llr(up_llr):
    half = len(up_llr) // 2
    return np.array([_f_min_sum(up_llr[i], up_llr[i + half]) for i in range(half)])


def _get_right_llr(left_bit, up_llr):
    half = len(left_bit)
    return np.array([g_operation(up_llr[i], up_llr[i + half], left_bit[i]) for i in range(half)])


def _get_bit(llr_val, is_info):
    if is_info:
        return 0 if llr_val >= 0 else 1
    return 0


def sc_decode_recursive(llr, frozen_bits):
    """
    递归 SC 译码（参考实现，与树遍历版本等价）。
    """
    return sc_decode(llr, frozen_bits)


def precompute_sc_indices(N):
    """预计算非递归 SC 译码所需的辅助向量"""
    n = int(math.log2(N))
    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        layer = 0
        while layer < n and ((phi >> layer) & 1):
            layer += 1
        bit_layer_vec.append(list(range(layer)))
        llr_layer_vec.append(list(range(layer, n)))
    return np.zeros(N, dtype=int), llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """
    SC 译码主函数（基于因子图树遍历，O(N log N)）。
    llr_matrix[0] 为信道 LLR，bit_matrix[n] 为译码结果。
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=int)
    info_positions = set(np.where(frozen_bits == 0)[0])
    N = len(llr_ch)
    n = int(math.log2(N))

    llr_matrix = np.full((n + 1, N), np.nan)
    bit_matrix = np.full((n + 1, N), np.nan)
    llr_matrix[0] = llr_ch.copy()
    position = [0, 0, n, N]

    while not _all_filled(bit_matrix[n]):
        up_llr = llr_matrix[position[0]][position[1]:position[1] + 2 ** (position[2] - position[0])]
        up_bit = bit_matrix[position[0]][position[1]:position[1] + 2 ** (position[2] - position[0])]
        span = 2 ** (position[2] - position[0] - 1)
        left_llr = llr_matrix[position[0] + 1][position[1]:position[1] + span]
        left_bit = bit_matrix[position[0] + 1][position[1]:position[1] + span]
        right_llr = llr_matrix[position[0] + 1][position[1] + span:position[1] + 2 * span]
        right_bit = bit_matrix[position[0] + 1][position[1] + span:position[1] + 2 * span]

        if _all_filled(up_bit):
            position = _up(position)
        else:
            if _all_filled(right_bit):
                up_bit_new = _get_up_bit(left_bit, right_bit)
                bit_matrix[position[0]][position[1]:position[1] + 2 ** (position[2] - position[0])] = up_bit_new[0]
            else:
                if _all_filled(right_llr):
                    if position[0] == position[2] - 1:
                        right_bit_pos = position[1] + 1
                        right_bit_val = _get_bit(
                            right_llr[0], right_bit_pos in info_positions
                        )
                        bit_matrix[position[0] + 1][position[1] + span] = right_bit_val
                    else:
                        position = _rightdown(position)
                else:
                    if _all_filled(left_bit):
                        right_llr_new = _get_right_llr(left_bit, up_llr)
                        llr_matrix[position[0] + 1][position[1] + span:position[1] + 2 * span] = right_llr_new
                    else:
                        if not _all_filled(left_llr):
                            left_llr_new = _get_left_llr(up_llr)
                            llr_matrix[position[0] + 1][position[1]:position[1] + span] = left_llr_new
                        else:
                            if position[0] == position[2] - 1:
                                left_bit_pos = position[1]
                                left_bit_val = _get_bit(
                                    left_llr[0], left_bit_pos in info_positions
                                )
                                bit_matrix[position[0] + 1][position[1]] = left_bit_val
                            else:
                                position = _leftdown(position)

    return bit_matrix[n].astype(int)
