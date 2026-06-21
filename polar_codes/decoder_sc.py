"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math
import numpy as np
from encoder import bit_reversal_permutation


def f_operation(La, Lb):
    """min-sum 近似的 f 运算"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算"""
    return (1.0 - 2.0 * u_hat) * La + Lb


def _all_filled(x):
    return not np.any(np.isnan(x))


def _leftdown(p):
    return [p[0] + 1, p[1], p[2], p[3]]


def _rightdown(p):
    return [p[0] + 1, p[1] + 2 ** (p[2] - 1 - p[0]), p[2], p[3]]


def _up(p):
    return [
        p[0] - 1,
        int(np.floor(p[1] / (2 ** (p[2] - p[0] + 1))) * (2 ** (p[2] - p[0] + 1))),
        p[2],
        p[3],
    ]


def _get_up_bit(left_bit, right_bit):
    length = len(left_bit)
    temp = np.array([(left_bit + right_bit) % 2, right_bit])
    return temp.reshape(1, 2 * length)


def _sc_tree_decode(y_llr, frozen_bits, prefix=None):
    """树遍历 SC 译码核心（层 0 为信道 LLR）。"""
    N = y_llr.size
    n = int(math.log2(N))
    llr_matrix = np.full((n + 1, N), np.nan, dtype=np.float64)
    bit_matrix = np.full((n + 1, N), np.nan, dtype=np.float64)
    llr_matrix[0] = y_llr
    if prefix is not None:
        for i, b in prefix.items():
            bit_matrix[n, i] = b
    position = [0, 0, n, N]

    while not _all_filled(bit_matrix[n]):
        span = 2 ** (position[2] - position[0])
        start = position[1]
        up_llr = llr_matrix[position[0]][start : start + span]
        up_bit = bit_matrix[position[0]][start : start + span]
        half = span // 2
        left_llr = llr_matrix[position[0] + 1][start : start + half]
        left_bit = bit_matrix[position[0] + 1][start : start + half]
        right_llr = llr_matrix[position[0] + 1][start + half : start + span]
        right_bit = bit_matrix[position[0] + 1][start + half : start + span]

        if _all_filled(up_bit):
            position = _up(position)
        elif _all_filled(right_bit):
            up_bit = _get_up_bit(left_bit, right_bit)
            bit_matrix[position[0]][start : start + span] = up_bit.copy()
        elif _all_filled(right_llr):
            if position[0] == position[2] - 1:
                pos = position[1] + 1
                if prefix is not None and pos in prefix:
                    bit_val = prefix[pos]
                else:
                    bit_val = 0 if frozen_bits[pos] else (0 if right_llr[0] > 0 else 1)
                bit_matrix[position[0] + 1][start + half : start + span] = bit_val
            else:
                position = _rightdown(position)
        elif _all_filled(left_bit):
            right_llr = g_operation(up_llr[:half], up_llr[half:], left_bit)
            llr_matrix[position[0] + 1][start + half : start + span] = right_llr
        elif not _all_filled(left_llr):
            left_llr = f_operation(up_llr[:half], up_llr[half:])
            llr_matrix[position[0] + 1][start : start + half] = left_llr
        else:
            if position[0] == position[2] - 1:
                pos = position[1]
                if prefix is not None and pos in prefix:
                    bit_val = prefix[pos]
                else:
                    bit_val = 0 if frozen_bits[pos] else (0 if left_llr[0] >= 0 else 1)
                bit_matrix[position[0] + 1][start : start + half] = bit_val
            else:
                position = _leftdown(position)

    return bit_matrix[n].astype(int)


def precompute_sc_indices(N):
    """预计算非递归 SC 译码辅助向量。"""
    n = int(math.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]
    llr_layer_vec, bit_layer_vec = [], []
    for phi in range(N):
        llr_layers, bit_layers = [], []
        temp = phi
        for layer in range(n):
            if (temp & 1) == 0:
                llr_layers.append(layer)
            if (temp & 1) == 1:
                bit_layers.append(layer)
            temp >>= 1
        llr_layer_vec.append(llr_layers)
        bit_layer_vec.append(bit_layers)
    return lambda_offset, llr_layer_vec, bit_layer_vec


def _new_pc_arrays(n, N, llr_perm):
    P = np.zeros((n + 1, N), dtype=np.float64)
    C = np.zeros((n + 1, N), dtype=np.int8)
    P[n, :] = llr_perm
    return P, C


def _path_metric_update(pm, llr, bit):
    hard = 0 if llr >= 0 else 1
    return pm if bit == hard else pm + abs(llr)


def _leaf_llr(y_llr, frozen_bits, prefix, phi):
    """获取第 phi 个比特的 LLR（通过树译码至该叶节点）。"""
    N = len(y_llr)
    n = int(math.log2(N))
    llr_matrix = np.full((n + 1, N), np.nan, dtype=np.float64)
    bit_matrix = np.full((n + 1, N), np.nan, dtype=np.float64)
    llr_matrix[0] = y_llr
    for i, b in prefix.items():
        bit_matrix[n, i] = b
    position = [0, 0, n, N]

    while True:
        if not np.isnan(bit_matrix[n, phi]):
            if position[0] == n and position[1] == phi:
                return llr_matrix[n, phi]
        span = 2 ** (position[2] - position[0])
        start = position[1]
        up_llr = llr_matrix[position[0]][start : start + span]
        up_bit = bit_matrix[position[0]][start : start + span]
        half = span // 2
        left_llr = llr_matrix[position[0] + 1][start : start + half]
        left_bit = bit_matrix[position[0] + 1][start : start + half]
        right_llr = llr_matrix[position[0] + 1][start + half : start + span]
        right_bit = bit_matrix[position[0] + 1][start + half : start + span]

        if _all_filled(up_bit):
            position = _up(position)
        elif _all_filled(right_bit):
            up_bit = _get_up_bit(left_bit, right_bit)
            bit_matrix[position[0]][start : start + span] = up_bit.copy()
        elif _all_filled(right_llr):
            if position[0] == position[2] - 1:
                pos = position[1] + 1
                if pos in prefix:
                    bit_val = prefix[pos]
                else:
                    bit_val = 0 if frozen_bits[pos] else (0 if right_llr[0] > 0 else 1)
                bit_matrix[position[0] + 1][start + half : start + span] = bit_val
                if pos == phi:
                    return right_llr[0]
            else:
                position = _rightdown(position)
        elif _all_filled(left_bit):
            right_llr = g_operation(up_llr[:half], up_llr[half:], left_bit)
            llr_matrix[position[0] + 1][start + half : start + span] = right_llr
        elif not _all_filled(left_llr):
            left_llr = f_operation(up_llr[:half], up_llr[half:])
            llr_matrix[position[0] + 1][start : start + half] = left_llr
        else:
            if position[0] == position[2] - 1:
                pos = position[1]
                if pos in prefix:
                    bit_val = prefix[pos]
                else:
                    bit_val = 0 if frozen_bits[pos] else (0 if left_llr[0] >= 0 else 1)
                bit_matrix[position[0] + 1][start : start + half] = bit_val
                if pos == phi:
                    return left_llr[0]
            else:
                position = _leftdown(position)


def _permute_llr(llr_ch, N):
    rev = bit_reversal_permutation(N)
    return np.asarray(llr_ch, dtype=np.float64)[rev]


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC 译码主函数（树遍历实现）。"""
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    llr_perm = _permute_llr(llr_ch, len(llr_ch))
    return _sc_tree_decode(llr_perm, frozen_bits)


def sc_decode_recursive(llr_ch, frozen_bits):
    """树遍历 SC 译码（参考实现）。"""
    return sc_decode(llr_ch, frozen_bits)


def sc_decode_nonrecursive(llr_perm, frozen_bits):
    """非递归接口（与树遍历等价）。"""
    return _sc_tree_decode(np.asarray(llr_perm, dtype=np.float64), np.asarray(frozen_bits, dtype=bool))
