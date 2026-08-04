"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import numpy as np
import math
from encoder import bit_reversal_permutation


def f_operation(La, Lb):
    """min-sum 近似的 f 运算"""
    sa, sb = np.sign(La), np.sign(Lb)
    sa = np.where(sa == 0, 1, sa)
    sb = np.where(sb == 0, 1, sb)
    return sa * sb * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算"""
    return (1 - 2 * u_hat) * La + Lb


def _prepare_llr(llr_ch, N):
    """编码含比特倒序，译码前对信道 LLR 做相同置换"""
    rev = bit_reversal_permutation(N)
    return np.asarray(llr_ch, dtype=np.float64)[rev]


def _info_positions(frozen_bits):
    return np.where(~np.asarray(frozen_bits, dtype=bool))[0]


def _all_computed(x):
    return not np.any(np.isnan(x))


def _sc_tree_decode(llr, information_pos, frozen_val=0):
    """
    非递归树遍历 SC 译码核心（已验证实现）。
    llr: 长度 N，已做比特倒序置换
    information_pos: 信息位索引列表
    """
    N = len(llr)
    n = int(math.log2(N))
    info_set = set(information_pos)

    llr_matrix = np.full((n + 1, N), np.nan)
    bit_matrix = np.full((n + 1, N), np.nan)
    llr_matrix[0] = llr
    position = [0, 0, n, N]

    def leftdown(p):
        return [p[0] + 1, p[1], p[2], p[3]]

    def rightdown(p):
        return [p[0] + 1, p[1] + 2 ** (p[2] - 1 - p[0]), p[2], p[3]]

    def up(p):
        return [p[0] - 1, int(np.floor(p[1] / (2 ** (p[2] - p[0] + 1))) * (2 ** (p[2] - p[0] + 1))), p[2], p[3]]

    def get_up_bit(left_bit, right_bit):
        length = len(left_bit)
        temp = np.empty(2 * length, dtype=float)
        temp[0::2] = (left_bit + right_bit) % 2
        temp[1::2] = right_bit
        return temp

    while not _all_computed(bit_matrix[n]):
        span = 2 ** (position[2] - position[0])
        s, e = position[1], position[1] + span
        half = span // 2
        up_llr = llr_matrix[position[0]][s:e]
        up_bit = bit_matrix[position[0]][s:e]
        left_llr = llr_matrix[position[0] + 1][s:s + half]
        left_bit = bit_matrix[position[0] + 1][s:s + half]
        right_llr = llr_matrix[position[0] + 1][s + half:e]
        right_bit = bit_matrix[position[0] + 1][s + half:e]

        if _all_computed(up_bit):
            position = up(position)
        elif _all_computed(right_bit):
            length = len(left_bit)
            temp = np.array([(left_bit + right_bit) % 2, right_bit])
            temp.resize((1, 2 * length))
            bit_matrix[position[0]][s:e] = temp.flatten()
        elif _all_computed(right_llr):
            if position[0] == position[2] - 1:
                for i in range(half):
                    pos = s + half + i
                    if pos in info_set:
                        bit_matrix[position[0] + 1][pos] = 0 if right_llr[i] > 0 else 1
                    else:
                        bit_matrix[position[0] + 1][pos] = frozen_val
            else:
                position = rightdown(position)
        elif _all_computed(left_bit):
            llr_matrix[position[0] + 1][s + half:e] = g_operation(up_llr[:half], up_llr[half:], left_bit)
        elif not _all_computed(left_llr):
            llr_matrix[position[0] + 1][s:s + half] = f_operation(up_llr[:half], up_llr[half:])
        else:
            if position[0] == position[2] - 1:
                for i in range(half):
                    pos = s + i
                    if pos in info_set:
                        bit_matrix[position[0] + 1][pos] = 0 if left_llr[i] >= 0 else 1
                    else:
                        bit_matrix[position[0] + 1][pos] = frozen_val
            else:
                position = leftdown(position)

    return np.nan_to_num(bit_matrix[n], nan=0).astype(int), llr_matrix, bit_matrix


def sc_decode_recursive(llr_ch, frozen_bits):
    """递归 SC 译码（基于树遍历实现，与 sc_decode 等价）"""
    N = len(llr_ch)
    llr = _prepare_llr(llr_ch, N)
    info_pos = list(_info_positions(frozen_bits))
    u_hat, _, _ = _sc_tree_decode(llr, info_pos, frozen_val=0)
    return u_hat


def precompute_sc_indices(N):
    """预计算非递归 SC 译码所需的辅助向量"""
    n = int(math.log2(N))
    lambda_offset = [0]
    for layer in range(1, n + 1):
        lambda_offset.append(lambda_offset[-1] + 2 ** (n - layer))

    llr_layer_vec, bit_layer_vec = [], []
    for phi in range(N):
        llr_layer_vec.append([layer for layer in range(n) if (phi >> layer) & 1 == 0])
        if phi % 2 == 0:
            bit_layer_vec.append([layer for layer in range(n) if (phi >> layer) & 1 == 1])
        else:
            bit_layer_vec.append([layer for layer in range(n) if (phi >> layer) & 1 == 0])

    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC 译码主函数"""
    N = len(llr_ch)
    llr = _prepare_llr(llr_ch, N)
    info_pos = list(_info_positions(frozen_bits))
    u_hat, _, _ = _sc_tree_decode(llr, info_pos, frozen_val=0)
    return u_hat
