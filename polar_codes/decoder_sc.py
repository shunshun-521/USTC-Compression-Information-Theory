"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import numpy as np

from _polar_ref_function import (
    all_num,
    up,
    leftdown,
    rightdown,
    get_up_bit,
    get_right_bit,
    get_left_bit,
    get_right_llr,
    get_left_llr,
)


def f_operation(La, Lb):
    """min-sum 近似的 f 运算"""
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    s1 = np.sign(La)
    s2 = np.sign(Lb)
    s1[s1 == 0] = 1
    s2[s2 == 0] = 1
    return s1 * s2 * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算"""
    u_hat = np.asarray(u_hat)
    return (1 - 2 * u_hat) * La + Lb


def _frozen_bits_to_info_pos(frozen_bits):
    fb = np.asarray(frozen_bits)
    if fb.dtype != bool:
        fb = fb.astype(bool)
    return [int(i) for i in np.where(~fb)[0]]


def _sc_decode_core(y_llr, information_pos, frozen_bit=0):
    """树形 SC 译码核心（信道 LLR 在第 0 层）。"""
    N = y_llr.size
    n = int(np.log2(N))
    llr_matrix = np.ones((n + 1, N), dtype=np.float64)
    llr_matrix[llr_matrix == 1] = np.nan
    bit_matrix = llr_matrix.copy()
    llr_matrix[0] = y_llr
    position = [0, 0, n, N]

    while all_num(bit_matrix[n]) == 0:
        up_llr = llr_matrix[position[0]][position[1]:position[1] + 2 ** (position[2] - position[0])]
        up_bit = bit_matrix[position[0]][position[1]:position[1] + 2 ** (position[2] - position[0])]
        left_llr = llr_matrix[position[0] + 1][position[1]:position[1] + 2 ** (position[2] - position[0] - 1)]
        left_bit = bit_matrix[position[0] + 1][position[1]:position[1] + 2 ** (position[2] - position[0] - 1)]
        right_llr = llr_matrix[position[0] + 1][position[1] + 2 ** (position[2] - position[0] - 1):position[1] + 2 ** (position[2] - position[0])]
        right_bit = bit_matrix[position[0] + 1][position[1] + 2 ** (position[2] - position[0] - 1):position[1] + 2 ** (position[2] - position[0])]

        if all_num(up_bit) == 1:
            position = up(position)
        else:
            if all_num(right_bit) == 1:
                up_bit_new = get_up_bit(left_bit, right_bit)
                bit_matrix[position[0]][position[1]:position[1] + 2 ** (position[2] - position[0])] = up_bit_new.copy()
            else:
                if all_num(right_llr) == 1:
                    if position[0] == position[2] - 1:
                        right_bit_pos = position[1] + 1
                        rb = get_right_bit(right_llr, information_pos, frozen_bit, right_bit_pos)
                        bit_matrix[position[0] + 1][position[1] + 2 ** (position[2] - position[0] - 1):position[1] + 2 ** (position[2] - position[0])] = rb
                    else:
                        position = rightdown(position)
                else:
                    if all_num(left_bit) == 1:
                        right_llr_new = get_right_llr(left_bit, up_llr)
                        llr_matrix[position[0] + 1][position[1] + 2 ** (position[2] - position[0] - 1):position[1] + 2 ** (position[2] - position[0])] = right_llr_new
                    else:
                        if all_num(left_llr) == 0:
                            left_llr_new = get_left_llr(up_llr)
                            llr_matrix[position[0] + 1][position[1]:position[1] + 2 ** (position[2] - position[0] - 1)] = left_llr_new
                        else:
                            if position[0] == position[2] - 1:
                                left_bit_pos = position[1]
                                lb = get_left_bit(left_llr, information_pos, frozen_bit, left_bit_pos)
                                bit_matrix[position[0] + 1][position[1]:position[1] + 2 ** (position[2] - position[0] - 1)] = lb
                            else:
                                position = leftdown(position)

    return bit_matrix[n]


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码。"""
    y_llr = np.asarray(llr, dtype=np.float64)
    information_pos = _frozen_bits_to_info_pos(frozen_bits)
    return _sc_decode_core(y_llr, information_pos, 0).astype(int)


def precompute_sc_indices(N):
    """预计算非递归 SC 译码辅助向量。"""
    n = int(np.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]
    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        llr_layers = []
        bit_layers = []
        psi = phi
        layer = 0
        while psi % 2 == 1:
            bit_layers.append(layer)
            psi //= 2
            layer += 1
        llr_layers.append(layer)
        while layer < n:
            llr_layers.append(layer)
            layer += 1
        llr_layer_vec.append(llr_layers)
        bit_layer_vec.append(bit_layers)
    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC 译码（调用树形实现）。"""
    return sc_decode_recursive(llr_ch, frozen_bits)
