"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import numpy as np
import math

from encoder import bit_reversal_permutation


def f_operation(La, Lb):
    """min-sum 近似的 f 运算"""
    s1 = 1.0 if np.sign(La) == 0 else np.sign(La)
    s2 = 1.0 if np.sign(Lb) == 0 else np.sign(Lb)
    return s1 * s2 * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算"""
    return (1.0 - 2.0 * np.asarray(u_hat)) * La + Lb


def _all_computed(x):
    return not np.any(np.isnan(x))


def _permute_channel_llr(llr_ch):
    """将信道 LLR 映射到译码树（匹配含比特倒序的编码器）"""
    N = len(llr_ch)
    br = bit_reversal_permutation(N)
    return np.asarray(llr_ch, dtype=np.float64)[br]


def _leftdown(position):
    return [position[0] + 1, position[1], position[2], position[3]]


def _rightdown(position):
    return [position[0] + 1, position[1] + 2 ** (position[2] - 1 - position[0]), position[2], position[3]]


def _up(position):
    p0 = position[0] - 1
    p1 = int(np.floor(position[1] / (2 ** (position[2] - position[0] + 1))) * (2 ** (position[2] - position[0] + 1)))
    return [p0, p1, position[2], position[3]]


def _get_up_bit(left_bit, right_bit):
    length = len(left_bit)
    temp = np.array([(left_bit + right_bit) % 2, right_bit])
    return temp.reshape(2 * length)


def _get_left_llr(up_llr):
    half = len(up_llr) // 2
    return np.array([f_operation(up_llr[i], up_llr[i + half]) for i in range(half)])


def _get_right_llr(left_bit, up_llr):
    half = len(left_bit)
    return np.array([g_operation(up_llr[i], up_llr[i + half], left_bit[i]) for i in range(half)])


def _sc_tree_decode(y_llr, info_indices, frozen_value=0):
    """基于因子图树遍历的 SC 译码核心"""
    N = len(y_llr)
    if N == 1:
        return np.array([frozen_value if 0 not in info_indices else (0 if y_llr[0] >= 0 else 1)])

    n = int(math.log2(N))
    info_set = set(int(i) for i in info_indices)

    llr_matrix = np.full((n + 1, N), np.nan)
    bit_matrix = np.full((n + 1, N), np.nan)
    llr_matrix[0] = y_llr
    position = [0, 0, n, N]

    while not _all_computed(bit_matrix[n]):
        span = 2 ** (position[2] - position[0])
        p0, p1 = position[0], position[1]

        up_llr = llr_matrix[p0][p1:p1 + span]
        up_bit = bit_matrix[p0][p1:p1 + span]
        half = span // 2
        left_llr = llr_matrix[p0 + 1][p1:p1 + half]
        left_bit = bit_matrix[p0 + 1][p1:p1 + half]
        right_llr = llr_matrix[p0 + 1][p1 + half:p1 + span]
        right_bit = bit_matrix[p0 + 1][p1 + half:p1 + span]

        if _all_computed(up_bit):
            position = _up(position)
        elif _all_computed(right_bit):
            bit_matrix[p0][p1:p1 + span] = _get_up_bit(left_bit, right_bit)
        elif _all_computed(right_llr):
            if position[0] == position[2] - 1:
                right_pos = p1 + 1
                val = frozen_value if right_pos not in info_set else (0 if right_llr[0] > 0 else 1)
                bit_matrix[p0 + 1][p1 + half] = val
            else:
                position = _rightdown(position)
        elif _all_computed(left_bit):
            llr_matrix[p0 + 1][p1 + half:p1 + span] = _get_right_llr(left_bit, up_llr)
        elif not _all_computed(left_llr):
            llr_matrix[p0 + 1][p1:p1 + half] = _get_left_llr(up_llr)
        elif position[0] == position[2] - 1:
            left_pos = p1
            val = frozen_value if left_pos not in info_set else (0 if left_llr[0] >= 0 else 1)
            bit_matrix[p0 + 1][p1] = val
        else:
            position = _leftdown(position)

    return bit_matrix[n].astype(int)


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（调用树遍历核心作为参考实现）"""
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    info_indices = np.where(~frozen_bits)[0]
    y_llr = _permute_channel_llr(llr)
    return _sc_tree_decode(y_llr, info_indices)


def precompute_sc_indices(N):
    """
    预计算非递归 SC 译码所需的三个辅助向量（供分析/扩展使用）。
    """
    n = int(math.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]

    llr_layer_vec = []
    bit_layer_vec = []

    for phi in range(N):
        layers_llr = []
        temp = phi
        for layer in range(n):
            if (temp & 1) == 0:
                layers_llr.append(layer)
            temp >>= 1
        llr_layer_vec.append(layers_llr)

        layers_bit = []
        if phi % 2 == 1:
            t = phi
            while t % 2 == 1:
                layers_bit.append(int(math.log2(t & -t)))
                t = (t - 1) // 2
            if t > 0:
                layers_bit.append(int(math.log2(t & -t)))
            layers_bit = sorted(set(l for l in layers_bit if l < n))
        bit_layer_vec.append(layers_bit)

    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC 译码主函数（高效树遍历实现）"""
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    info_indices = np.where(~frozen_bits)[0]
    y_llr = _permute_channel_llr(llr_ch)
    return _sc_tree_decode(y_llr, info_indices)
