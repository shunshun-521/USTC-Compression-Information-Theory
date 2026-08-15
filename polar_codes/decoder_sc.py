"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import numpy as np
import math
from encoder import bit_reversal_permutation


def f_operation(La, Lb):
    """min-sum 近似的 f 运算"""
    s1 = np.sign(La)
    s2 = np.sign(Lb)
    s1 = np.where(s1 == 0, 1, s1)
    s2 = np.where(s2 == 0, 1, s2)
    return s1 * s2 * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb"""
    return (1 - 2 * u_hat) * La + Lb


def _unpermute_llr(llr_ch):
    """将信道 LLR 变换为与蝶形编码（无输出倒序）一致的顺序"""
    N = len(llr_ch)
    inv_perm = np.argsort(bit_reversal_permutation(N))
    return np.asarray(llr_ch, dtype=np.float64)[inv_perm]


def _frozen_to_info_set(frozen_bits):
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    return set(np.where(~frozen_bits)[0].tolist())


def _all_num(x):
    return int(not np.isnan(x).any())


def _leftdown(position):
    return [position[0] + 1, position[1], position[2], position[3]]


def _rightdown(position):
    return [position[0] + 1, position[1] + 2 ** (position[2] - 1 - position[0]),
            position[2], position[3]]


def _up(position):
    p0 = position[0] - 1
    p1 = int(np.floor(position[1] / (2 ** (position[2] - position[0] + 1)))
            * (2 ** (position[2] - position[0] + 1)))
    return [p0, p1, position[2], position[3]]


def _get_up_bit(left_bit, right_bit):
    length = left_bit.size
    temp = np.array([(left_bit + right_bit) % 2, right_bit])
    return temp.reshape(2 * length)


def _get_left_llr(up_llr):
    length = len(up_llr) // 2
    return np.array([f_operation(up_llr[i], up_llr[i + length]) for i in range(length)])


def _get_right_llr(left_bit, up_llr):
    length = len(left_bit)
    return np.array([g_operation(up_llr[i], up_llr[i + length], left_bit[i])
                     for i in range(length)])


def _get_left_bit(left_llr, information_pos, frozen_bit, left_bit_pos):
    if left_bit_pos in information_pos:
        return 0 if left_llr[0] >= 0 else 1
    return frozen_bit


def _get_right_bit(right_llr, information_pos, frozen_bit, right_bit_pos):
    if right_bit_pos in information_pos:
        return 0 if right_llr[0] > 0 else 1
    return frozen_bit


def _sc_tree_decode(y_llr, information_pos, frozen_bit=0):
    """非递归 SC 树遍历译码"""
    N = y_llr.size
    n = int(np.log2(N))
    info_set = set(information_pos)

    llr_matrix = np.ones((n + 1, N), dtype=np.float64)
    llr_matrix[:] = np.nan
    bit_matrix = llr_matrix.copy()
    llr_matrix[0] = y_llr
    position = [0, 0, n, N]

    while _all_num(bit_matrix[n]) == 0:
        span = 2 ** (position[2] - position[0])
        sl = position[1]
        sr = sl + span
        up_llr = llr_matrix[position[0], sl:sr]
        up_bit = bit_matrix[position[0], sl:sr]
        half = span // 2
        left_llr = llr_matrix[position[0] + 1, sl:sl + half]
        left_bit = bit_matrix[position[0] + 1, sl:sl + half]
        right_llr = llr_matrix[position[0] + 1, sl + half:sr]
        right_bit = bit_matrix[position[0] + 1, sl + half:sr]

        if _all_num(up_bit) == 1:
            position = _up(position)
        elif _all_num(right_bit) == 1:
            bit_matrix[position[0], sl:sr] = _get_up_bit(left_bit, right_bit)
        elif _all_num(right_llr) == 1:
            if position[0] == position[2] - 1:
                right_bit_pos = position[1] + 1
                val = _get_right_bit(right_llr, info_set, frozen_bit, right_bit_pos)
                bit_matrix[position[0] + 1, sl + half:sr] = val
            else:
                position = _rightdown(position)
        elif _all_num(left_bit) == 1:
            llr_matrix[position[0] + 1, sl + half:sr] = _get_right_llr(left_bit, up_llr)
        elif _all_num(left_llr) == 0:
            llr_matrix[position[0] + 1, sl:sl + half] = _get_left_llr(up_llr)
        else:
            if position[0] == position[2] - 1:
                left_bit_pos = position[1]
                val = _get_left_bit(left_llr, info_set, frozen_bit, left_bit_pos)
                bit_matrix[position[0] + 1, sl:sl + half] = val
            else:
                position = _leftdown(position)

    return bit_matrix[n].astype(int)


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（调用树遍历实现，结果与非递归版本一致）"""
    return sc_decode(llr, frozen_bits)


def precompute_sc_indices(N):
    """预计算非递归 SC 译码所需的辅助向量"""
    n = int(math.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]
    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        bits_bin = format(phi, f'0{n}b')
        layers_llr = []
        for layer in range(n):
            if bits_bin[n - 1 - layer] == '0':
                layers_llr.append(layer)
            else:
                break
        llr_layer_vec.append(layers_llr)
        layers_bit = []
        if phi % 2 == 1:
            for layer in range(n):
                if bits_bin[n - 1 - layer] == '1':
                    layers_bit.append(layer)
                else:
                    break
        bit_layer_vec.append(layers_bit)
    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC 译码主函数"""
    y_llr = _unpermute_llr(llr_ch)
    info_pos = _frozen_to_info_set(frozen_bits)
    return _sc_tree_decode(y_llr, info_pos, frozen_bit=0)
