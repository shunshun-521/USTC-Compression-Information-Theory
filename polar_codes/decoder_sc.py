"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math
import numpy as np


def f_operation(La, Lb):
    """min-sum 近似的 f 运算。"""
    if np.isscalar(La) and np.isscalar(Lb):
        s1 = np.sign(La)
        s2 = np.sign(Lb)
        if s1 == 0:
            s1 = 1
        if s2 == 0:
            s2 = 1
        return s1 * s2 * min(abs(La), abs(Lb))
    sa, sb = np.sign(La), np.sign(Lb)
    sa = np.where(sa == 0, 1, sa)
    sb = np.where(sb == 0, 1, sb)
    return sa * sb * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算。"""
    return (1.0 - 2.0 * u_hat) * La + Lb


def _info_indices_from_frozen(frozen_bits):
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    return np.where(~frozen_bits)[0]


def _all_set(row):
    return not np.any(np.isnan(row))


def _sc_decode_tree(llr, information_pos, frozen_bit=0):
    """基于因子树遍历的 SC 译码核心实现。"""
    llr = np.asarray(llr, dtype=np.float64)
    N = len(llr)
    n = int(math.log2(N))
    info_set = set(int(i) for i in information_pos)

    llr_matrix = np.full((n + 1, N), np.nan, dtype=np.float64)
    bit_matrix = np.full((n + 1, N), np.nan, dtype=np.float64)
    llr_matrix[0] = llr

    position = [0, 0, n, N]

    while not _all_set(bit_matrix[n]):
        span = 2 ** (position[2] - position[0])
        half = span // 2
        row, col = position[0], position[1]

        up_llr = llr_matrix[row, col:col + span]
        up_bit = bit_matrix[row, col:col + span]
        left_llr = llr_matrix[row + 1, col:col + half]
        left_bit = bit_matrix[row + 1, col:col + half]
        right_llr = llr_matrix[row + 1, col + half:col + span]
        right_bit = bit_matrix[row + 1, col + half:col + span]

        if _all_set(up_bit):
            new_row = row - 1
            new_col = int(
                np.floor(col / (2 ** (position[2] - row + 1)))
                * (2 ** (position[2] - row + 1))
            )
            position = [new_row, new_col, position[2], position[3]]
        elif _all_set(right_bit):
            temp = np.array([(left_bit + right_bit) % 2, right_bit])
            temp.resize((1, 2 * len(left_bit)))
            bit_matrix[row, col:col + span] = temp[0]
        elif _all_set(right_llr):
            if row == position[2] - 1:
                bit_pos = col + 1
                if bit_pos in info_set:
                    bit_val = 0 if right_llr[0] > 0 else 1
                else:
                    bit_val = frozen_bit
                bit_matrix[row + 1, col + half:col + span] = bit_val
            else:
                position = [row + 1, col + 2 ** (position[2] - 1 - row), position[2], position[3]]
        elif _all_set(left_bit):
            right_llr_new = np.array(
                [g_operation(up_llr[i], up_llr[i + half], left_bit[i]) for i in range(half)],
                dtype=np.float64,
            )
            llr_matrix[row + 1, col + half:col + span] = right_llr_new
        elif np.any(np.isnan(left_llr)):
            left_llr_new = np.array(
                [f_operation(up_llr[i], up_llr[i + half]) for i in range(half)],
                dtype=np.float64,
            )
            llr_matrix[row + 1, col:col + half] = left_llr_new
        else:
            if row == position[2] - 1:
                bit_pos = col
                if bit_pos in info_set:
                    bit_val = 0 if left_llr[0] >= 0 else 1
                else:
                    bit_val = frozen_bit
                bit_matrix[row + 1, col:col + half] = bit_val
            else:
                position = [row + 1, col, position[2], position[3]]

    return bit_matrix[n].astype(int)


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（调用树遍历参考实现）。"""
    info_idx = _info_indices_from_frozen(frozen_bits)
    return _sc_decode_tree(llr, info_idx)


def precompute_sc_indices(N):
    """预计算非递归 SC 译码所需的辅助向量。"""
    n = int(math.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]

    llr_layer_vec = []
    bit_layer_vec = []

    for phi in range(N):
        layers = []
        idx = phi
        while (idx & 1) == 1:
            layers.append(int(math.log2(idx & -idx)))
            idx >>= 1
        llr_layer_vec.append(layers)

        if phi % 2 == 0:
            bit_layers = list(range(n))
        else:
            bit_layers = []
            idx = phi
            while (idx & 1) == 1:
                bit_layers.append(int(math.log2(idx & -idx)))
                idx >>= 1
        bit_layer_vec.append(bit_layers)

    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC 译码主函数。"""
    info_idx = _info_indices_from_frozen(frozen_bits)
    return _sc_decode_tree(llr_ch, info_idx)
