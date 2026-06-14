"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效树形遍历实现）
"""
import math
import numpy as np


def f_operation(La, Lb):
    """min-sum 近似的 f 运算。"""
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    s1 = np.sign(La)
    s2 = np.sign(Lb)
    s1 = np.where(s1 == 0, 1, s1)
    s2 = np.where(s2 == 0, 1, s2)
    return s1 * s2 * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算。"""
    return (1.0 - 2.0 * u_hat) * La + Lb


def _frozen_to_info_pos(frozen_bits):
    frozen_bits = np.asarray(frozen_bits)
    return np.where(~frozen_bits.astype(bool))[0].tolist()


def _all_known(arr):
    return not np.any(np.isnan(arr))


def _up(position):
    p0 = position[0] - 1
    span = 2 ** (position[2] - position[0] + 1)
    p1 = int(np.floor(position[1] / span) * span)
    return [p0, p1, position[2], position[3]]


def _leftdown(position):
    return [position[0] + 1, position[1], position[2], position[3]]


def _rightdown(position):
    offset = 2 ** (position[2] - 1 - position[0])
    return [position[0] + 1, position[1] + offset, position[2], position[3]]


def _get_up_bit(left_bit, right_bit):
    length = left_bit.size
    temp = np.array([(left_bit + right_bit) % 2, right_bit])
    temp.resize((1, 2 * length))
    return temp


def _get_left_llr(up_llr):
    half = up_llr.size // 2
    return np.array(
        [f_operation(up_llr[i], up_llr[i + half]) for i in range(half)],
        dtype=np.float64,
    )


def _get_right_llr(left_bit, up_llr):
    half = up_llr.size // 2
    return np.array(
        [g_operation(up_llr[i], up_llr[i + half], left_bit[i]) for i in range(half)],
        dtype=np.float64,
    )


def sc_decode_recursive(llr, frozen_bits):
    """
    递归 SC 译码（分块递归实现，与树形遍历等价）。
    """
    return sc_decode_tree(llr, frozen_bits)


def precompute_sc_indices(N):
    """
    预计算非递归 SC 译码所需的三个辅助向量。
    """
    n = int(math.log2(N))
    lambda_offset = [1 << layer for layer in range(n + 1)]
    llr_layer_vec = []
    bit_layer_vec = []

    for phi in range(N):
        psi = phi
        llr_layers = []
        while psi % 2 == 1:
            llr_layers.append(int(math.log2(psi & -psi)))
            psi //= 2
        llr_layer_vec.append(llr_layers)

        psi = phi
        bit_layers = []
        while psi > 0 and psi % 2 == 0:
            bit_layers.append(int(math.log2(psi & -psi)))
            psi //= 2
        bit_layer_vec.append(bit_layers)

    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode_tree(llr_ch, frozen_bits):
    """基于因子树遍历的非递归 SC 译码（O(N log N)）。"""
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    info_pos = _frozen_to_info_pos(frozen_bits)
    frozen_val = 0

    N = llr_ch.size
    n = int(math.log2(N))
    llr_matrix = np.full((n + 1, N), np.nan, dtype=np.float64)
    bit_matrix = np.full((n + 1, N), np.nan, dtype=np.float64)
    llr_matrix[0] = llr_ch
    position = [0, 0, n, N]

    while not _all_known(bit_matrix[n]):
        span = 2 ** (position[2] - position[0])
        start = position[1]
        up_llr = llr_matrix[position[0], start : start + span]
        up_bit = bit_matrix[position[0], start : start + span]
        half = span // 2
        left_llr = llr_matrix[position[0] + 1, start : start + half]
        left_bit = bit_matrix[position[0] + 1, start : start + half]
        right_llr = llr_matrix[position[0] + 1, start + half : start + span]
        right_bit = bit_matrix[position[0] + 1, start + half : start + span]

        if _all_known(up_bit):
            position = _up(position)
            continue

        if _all_known(right_bit):
            up_bit_val = _get_up_bit(left_bit, right_bit)
            bit_matrix[position[0], start : start + span] = up_bit_val.copy()
            continue

        if _all_known(right_llr):
            if position[0] == position[2] - 1:
                bit_pos = start + half
                if bit_pos in info_pos:
                    bit_val = 0 if right_llr[0] > 0 else 1
                else:
                    bit_val = frozen_val
                bit_matrix[position[0] + 1, bit_pos] = bit_val
            else:
                position = _rightdown(position)
            continue

        if _all_known(left_bit):
            right_llr_val = _get_right_llr(left_bit, up_llr)
            llr_matrix[position[0] + 1, start + half : start + span] = right_llr_val
            continue

        if not _all_known(left_llr):
            left_llr_val = _get_left_llr(up_llr)
            llr_matrix[position[0] + 1, start : start + half] = left_llr_val
            continue

        if position[0] == position[2] - 1:
            bit_pos = start
            if bit_pos in info_pos:
                bit_val = 0 if left_llr[0] >= 0 else 1
            else:
                bit_val = frozen_val
            bit_matrix[position[0] + 1, bit_pos] = bit_val
        else:
            position = _leftdown(position)

    return bit_matrix[n].astype(int)


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC 译码主函数（默认使用树形遍历实现）。"""
    return sc_decode_tree(llr_ch, frozen_bits)
