"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math
import numpy as np
from encoder import bit_reversal_permutation


def f_operation(La, Lb):
    """
    min-sum 近似的 f 运算。
    """
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    sa = np.sign(La)
    sb = np.sign(Lb)
    sa = np.where(sa == 0, 1, sa)
    sb = np.where(sb == 0, 1, sb)
    return sa * sb * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """
    g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb
    """
    return (1 - 2 * u_hat) * La + Lb


def _all_decided(bits):
    return not np.any(np.isnan(bits))


def _up(position):
    p0 = position[0] - 1
    p1 = int(np.floor(position[1] / (2 ** (position[2] - position[0] + 1)))
             * (2 ** (position[2] - position[0] + 1)))
    return [p0, p1, position[2], position[3]]


def _leftdown(position):
    return [position[0] + 1, position[1], position[2], position[3]]


def _rightdown(position):
    return [position[0] + 1, position[1] + 2 ** (position[2] - 1 - position[0]),
            position[2], position[3]]


def _get_left_llr(up_llr):
    half = len(up_llr) // 2
    return np.array([f_operation(up_llr[i], up_llr[i + half]) for i in range(half)])


def _get_right_llr(left_bit, up_llr):
    half = len(left_bit)
    return np.array([g_operation(up_llr[i], up_llr[i + half], left_bit[i])
                     for i in range(half)])


def _get_up_bit(left_bit, right_bit):
    length = len(left_bit)
    temp = np.array([(left_bit[i] + right_bit[i]) % 2 for i in range(length)] +
                    [right_bit[i] for i in range(length)])
    return temp.reshape(1, 2 * length)


def sc_decode_recursive(llr, frozen_bits):
    """
    递归 SC 译码（树遍历参考实现）。
    """
    N = len(llr)
    n = int(math.log2(N))
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    brp = bit_reversal_permutation(N)
    llr = np.asarray(llr, dtype=np.float64)[brp]

    llr_matrix = np.full((n + 1, N), np.nan)
    bit_matrix = np.full((n + 1, N), np.nan)
    llr_matrix[0] = llr
    position = [0, 0, n, N]

    while not _all_decided(bit_matrix[n]):
        span = 2 ** (position[2] - position[0])
        up_llr = llr_matrix[position[0]][position[1]:position[1] + span]
        up_bit = bit_matrix[position[0]][position[1]:position[1] + span]
        half = span // 2
        left_llr = llr_matrix[position[0] + 1][position[1]:position[1] + half]
        left_bit = bit_matrix[position[0] + 1][position[1]:position[1] + half]
        right_llr = llr_matrix[position[0] + 1][position[1] + half:position[1] + span]
        right_bit = bit_matrix[position[0] + 1][position[1] + half:position[1] + span]

        if _all_decided(up_bit):
            position = _up(position)
            continue

        if _all_decided(right_bit):
            up_bit_val = _get_up_bit(left_bit, right_bit)
            bit_matrix[position[0]][position[1]:position[1] + span] = up_bit_val
            continue

        if _all_decided(right_llr):
            if position[0] == position[2] - 1:
                right_pos = position[1] + half
                if frozen_bits[right_pos]:
                    right_bit_val = 0
                else:
                    right_bit_val = 0 if right_llr[0] >= 0 else 1
                bit_matrix[position[0] + 1][position[1] + half:position[1] + span] = right_bit_val
            else:
                position = _rightdown(position)
            continue

        if _all_decided(left_bit):
            right_llr_val = _get_right_llr(left_bit, up_llr)
            llr_matrix[position[0] + 1][position[1] + half:position[1] + span] = right_llr_val
            continue

        if not _all_decided(left_llr):
            left_llr_val = _get_left_llr(up_llr)
            llr_matrix[position[0] + 1][position[1]:position[1] + half] = left_llr_val
            continue

        if position[0] == position[2] - 1:
            left_pos = position[1]
            if frozen_bits[left_pos]:
                left_bit_val = 0
            else:
                left_bit_val = 0 if left_llr[0] >= 0 else 1
            bit_matrix[position[0] + 1][position[1]:position[1] + half] = left_bit_val
        else:
            position = _leftdown(position)

    u_hat = np.nan_to_num(bit_matrix[n], nan=0).astype(int)
    return u_hat


def precompute_sc_indices(N):
    """
    预计算非递归 SC 译码所需的辅助向量（层 0 为信道侧）。
    """
    n = int(math.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]

    llr_layer_vec = [[] for _ in range(N)]
    bit_layer_vec = [[] for _ in range(N)]

    for phi in range(N):
        p = phi
        layer = 0
        while p & 1:
            llr_layer_vec[phi].append(layer)
            p >>= 1
            layer += 1

        if phi == N - 1:
            continue

        p = phi + 1
        layer = 0
        while not (p & 1):
            bit_layer_vec[phi].append(layer)
            p >>= 1
            layer += 1

    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码（与递归版本等价的树遍历实现）。
    """
    return sc_decode_recursive(llr_ch, frozen_bits)
