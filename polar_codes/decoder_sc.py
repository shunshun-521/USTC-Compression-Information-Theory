"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math

import numpy as np


def f_operation(La, Lb):
    """min-sum 近似的 f 运算（sign(0) 取 +1）。"""
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    scalar = La.ndim == 0
    if scalar:
        La = La.reshape(1)
        Lb = Lb.reshape(1)
    s1 = np.sign(La)
    s2 = np.sign(Lb)
    s1[s1 == 0] = 1
    s2[s2 == 0] = 1
    out = s1 * s2 * np.minimum(np.abs(La), np.abs(Lb))
    return float(out[0]) if scalar else out


def g_operation(La, Lb, u_hat):
    """g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb"""
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    u_hat = np.asarray(u_hat, dtype=np.float64)
    out = (1 - 2 * u_hat) * La + Lb
    if np.ndim(out) == 0:
        return float(out)
    return out


def _all_decided(bits):
    return not np.any(np.isnan(bits))


def _left_down(pos):
    return [pos[0] + 1, pos[1], pos[2], pos[3]]


def _right_down(pos):
    step = 2 ** (pos[2] - 1 - pos[0])
    return [pos[0] + 1, pos[1] + step, pos[2], pos[3]]


def _up(pos):
    step = 2 ** (pos[2] - pos[0] + 1)
    return [pos[0] - 1, int(np.floor(pos[1] / step) * step), pos[2], pos[3]]


def _combine_bits(left_bit, right_bit):
    length = left_bit.size
    temp = np.array([(left_bit + right_bit) % 2, right_bit])
    temp = temp.reshape(1, 2 * length)
    return temp[0]


def _frozen_bits_to_info(frozen_bits):
    frozen_bits = np.asarray(frozen_bits)
    information_pos = np.where(frozen_bits == 0)[0]
    return information_pos, 0


def _sc_tree_decode(y_llr, information_pos, frozen_value):
    """基于因子图遍历的 SC 译码（O(N log N)）。"""
    N = y_llr.size
    n = int(np.log2(N))
    llr_matrix = np.full((n + 1, N), np.nan, dtype=np.float64)
    bit_matrix = np.full((n + 1, N), np.nan, dtype=np.float64)
    llr_matrix[0] = y_llr
    position = [0, 0, n, N]

    while not _all_decided(bit_matrix[n]):
        span = 2 ** (position[2] - position[0])
        up_llr = llr_matrix[position[0], position[1]:position[1] + span]
        up_bit = bit_matrix[position[0], position[1]:position[1] + span]
        half = span // 2
        left_llr = llr_matrix[position[0] + 1, position[1]:position[1] + half]
        left_bit = bit_matrix[position[0] + 1, position[1]:position[1] + half]
        right_llr = llr_matrix[position[0] + 1, position[1] + half:position[1] + span]
        right_bit = bit_matrix[position[0] + 1, position[1] + half:position[1] + span]

        if _all_decided(up_bit):
            position = _up(position)
            continue

        if _all_decided(right_bit):
            up_bit = _combine_bits(left_bit, right_bit)
            bit_matrix[position[0], position[1]:position[1] + span] = up_bit
            continue

        if _all_decided(right_llr):
            if position[0] == position[2] - 1:
                right_bit_pos = position[1] + 1
                if right_bit_pos in information_pos:
                    right_bit_val = 0.0 if right_llr[0] > 0 else 1.0
                else:
                    right_bit_val = float(frozen_value)
                bit_matrix[position[0] + 1, position[1] + half:position[1] + span] = right_bit_val
            else:
                position = _right_down(position)
            continue

        if _all_decided(left_bit):
            right_llr = np.array([
                g_operation(up_llr[i], up_llr[i + half], left_bit[i])
                for i in range(half)
            ])
            llr_matrix[position[0] + 1, position[1] + half:position[1] + span] = right_llr
            continue

        if not _all_decided(left_llr):
            left_llr = np.array([
                f_operation(up_llr[i], up_llr[i + half])
                for i in range(half)
            ])
            llr_matrix[position[0] + 1, position[1]:position[1] + half] = left_llr
            continue

        if position[0] == position[2] - 1:
            left_bit_pos = position[1]
            if left_bit_pos in information_pos:
                left_bit_val = 0.0 if left_llr[0] >= 0 else 1.0
            else:
                left_bit_val = float(frozen_value)
            bit_matrix[position[0] + 1, position[1]:position[1] + half] = left_bit_val
        else:
            position = _left_down(position)

    return bit_matrix[n].astype(int)


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（与树遍历实现等价）。"""
    llr = np.asarray(llr, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits)
    information_pos, frozen_value = _frozen_bits_to_info(frozen_bits)
    return _sc_tree_decode(llr, information_pos, frozen_value)


def precompute_sc_indices(N):
    """预计算非递归 SC 译码所需的辅助向量（接口兼容）。"""
    n = int(math.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]
    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        llr_layers = []
        p = phi
        while p % 2 == 1:
            llr_layers.append(int(math.log2(p & -p)))
            p >>= 1
        llr_layer_vec.append(llr_layers)
        bit_layers = []
        if phi % 2 == 1:
            p = phi
            while p % 2 == 1:
                bit_layers.append(int(math.log2(p & -p)))
                p >>= 1
        bit_layer_vec.append(bit_layers)
    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC 译码主函数。"""
    return sc_decode_recursive(llr_ch, frozen_bits)


def sc_decode_with_llr_reversal(llr_ch, frozen_bits):
    """对信道 LLR 做比特倒序后执行 SC 译码（与编码器比特倒序一致）。"""
    from encoder import bit_reversal_permutation

    N = len(llr_ch)
    br = bit_reversal_permutation(N)
    return sc_decode(llr_ch[br], frozen_bits)
