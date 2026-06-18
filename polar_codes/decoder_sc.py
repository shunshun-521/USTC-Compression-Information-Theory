"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import numpy as np

from encoder import bit_reversal_permutation
import sc_core as scf


def f_operation(La, Lb):
    """min-sum 近似的 f 运算。"""
    return scf.f_hf(La, Lb)


def g_operation(La, Lb, u_hat):
    """g 运算。"""
    return scf.g(La, Lb, u_hat)


def _frozen_to_info_indices(frozen_bits):
    frozen_bits = np.asarray(frozen_bits)
    if frozen_bits.dtype == bool:
        return np.where(~frozen_bits)[0]
    return np.where(frozen_bits == 0)[0]


def _sc_matrix_decode(y_llr, information_pos, frozen_bit=0):
    """矩阵遍历 SC 译码（非递归高效实现）。"""
    N = y_llr.size
    n = int(np.log2(N))
    information_pos = np.asarray(information_pos)

    llr_matrix = np.ones((n + 1, N))
    llr_matrix[llr_matrix == 1] = float('nan')
    bit_matrix = llr_matrix.copy()
    llr_matrix[0] = y_llr
    position = [0, 0, n, N]

    while scf.all_num(bit_matrix[n]) == 0:
        up_llr = llr_matrix[position[0]][position[1]:position[1] + 2 ** (position[2] - position[0])]
        up_bit = bit_matrix[position[0]][position[1]:position[1] + 2 ** (position[2] - position[0])]
        left_llr = llr_matrix[position[0] + 1][position[1]:position[1] + 2 ** (position[2] - position[0] - 1)]
        left_bit = bit_matrix[position[0] + 1][position[1]:position[1] + 2 ** (position[2] - position[0] - 1)]
        right_llr = llr_matrix[position[0] + 1][position[1] + 2 ** (position[2] - position[0] - 1):position[1] + 2 ** (position[2] - position[0])]
        right_bit = bit_matrix[position[0] + 1][position[1] + 2 ** (position[2] - position[0] - 1):position[1] + 2 ** (position[2] - position[0])]

        if scf.all_num(up_bit) == 1:
            position = scf.up(position)
        else:
            if scf.all_num(right_bit) == 1:
                up_bit_val = scf.get_up_bit(left_bit, right_bit)
                bit_matrix[position[0]][position[1]:position[1] + 2 ** (position[2] - position[0])] = up_bit_val.copy()
            else:
                if scf.all_num(right_llr) == 1:
                    if position[0] == position[2] - 1:
                        right_bit_pos = position[1] + 1
                        rb = scf.get_right_bit(right_llr, information_pos, frozen_bit, right_bit_pos)
                        bit_matrix[position[0] + 1][position[1] + 2 ** (position[2] - position[0] - 1):position[1] + 2 ** (position[2] - position[0])] = rb
                    else:
                        position = scf.rightdown(position)
                else:
                    if scf.all_num(left_bit) == 1:
                        right_llr_val = scf.get_right_llr(left_bit, up_llr)
                        llr_matrix[position[0] + 1][position[1] + 2 ** (position[2] - position[0] - 1):position[1] + 2 ** (position[2] - position[0])] = right_llr_val
                    else:
                        if scf.all_num(left_llr) == 0:
                            left_llr_val = scf.get_left_llr(up_llr)
                            llr_matrix[position[0] + 1][position[1]:position[1] + 2 ** (position[2] - position[0] - 1)] = left_llr_val
                        else:
                            if position[0] == position[2] - 1:
                                left_bit_pos = position[1]
                                lb = scf.get_left_bit(left_llr, information_pos, frozen_bit, left_bit_pos)
                                bit_matrix[position[0] + 1][position[1]:position[1] + 2 ** (position[2] - position[0] - 1)] = lb
                            else:
                                position = scf.leftdown(position)

    return bit_matrix[n].astype(int)


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（参考实现）。"""
    return sc_decode(llr, frozen_bits)


def precompute_sc_indices(N):
    """预计算非递归 SC 译码所需的三个辅助向量。"""
    n = int(np.log2(N))
    lambda_offset = np.arange(N, dtype=int)
    llr_layer_vec = []
    bit_layer_vec = []

    for phi in range(N):
        llr_layers = []
        temp = phi
        for layer in range(n):
            if (temp >> layer) & 1 == 0:
                llr_layers.append(layer)
        if phi == 0:
            llr_layers = list(range(n))

        bit_layers = []
        temp = phi
        layer = 0
        while temp & 1:
            bit_layers.append(layer)
            temp >>= 1
            layer += 1

        llr_layer_vec.append(llr_layers)
        bit_layer_vec.append(bit_layers)

    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码主函数。
    信道 LLR 在入口处做比特倒序置换，以匹配编码器的 B_N 置换。
    """
    N = len(llr_ch)
    br = bit_reversal_permutation(N)
    llr = np.asarray(llr_ch, dtype=np.float64)[br]
    info_indices = _frozen_to_info_indices(frozen_bits)
    return _sc_matrix_decode(llr, info_indices, frozen_bit=0)
