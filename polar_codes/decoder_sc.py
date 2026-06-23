"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import numpy as np
from encoder import bit_reversal_permutation


def f_operation(La, Lb):
    """
    min-sum 近似的 f 运算：
    f(La, Lb) ≈ sign(La) * sign(Lb) * min(|La|, |Lb|)
    """
    s1 = np.sign(La)
    s2 = np.sign(Lb)
    s1 = np.where(s1 == 0, 1, s1)
    s2 = np.where(s2 == 0, 1, s2)
    return s1 * s2 * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb"""
    return (1 - 2 * u_hat) * La + Lb


def _all_computed(row):
    return not np.any(np.isnan(row))


def _leftdown(pos):
    return [pos[0] + 1, pos[1], pos[2], pos[3]]


def _rightdown(pos):
    return [pos[0] + 1, pos[1] + 2 ** (pos[2] - 1 - pos[0]), pos[2], pos[3]]


def _up(pos):
    p1 = int(np.floor(pos[1] / (2 ** (pos[2] - pos[0] + 1))) * (2 ** (pos[2] - pos[0] + 1)))
    return [pos[0] - 1, p1, pos[2], pos[3]]


def _get_up_bit(left_bit, right_bit):
    length = len(left_bit)
    temp = np.array([(left_bit[i] + right_bit[i]) % 2 for i in range(length)] + list(right_bit))
    return temp.reshape(1, 2 * length)


def _get_left_llr(up_llr):
    half = len(up_llr) // 2
    return np.array([f_operation(up_llr[i], up_llr[i + half]) for i in range(half)])


def _get_right_llr(left_bit, up_llr):
    half = len(left_bit)
    return np.array([g_operation(up_llr[i], up_llr[i + half], left_bit[i]) for i in range(half)])


def _sc_tree_decode(y_llr, frozen_bits):
    """基于因子图树遍历的 SC 译码（与含比特倒序的编码器配套）"""
    N = len(y_llr)
    n = int(np.log2(N))
    frozen_bits = np.asarray(frozen_bits, dtype=bool)

    llr_matrix = np.full((n + 1, N), np.nan, dtype=np.float64)
    bit_matrix = np.full((n + 1, N), np.nan, dtype=np.float64)
    llr_matrix[0] = y_llr
    position = [0, 0, n, N]

    while not _all_computed(bit_matrix[n]):
        span = 2 ** (position[2] - position[0])
        up_llr = llr_matrix[position[0], position[1] : position[1] + span]
        up_bit = bit_matrix[position[0], position[1] : position[1] + span]
        half = span // 2
        left_llr = llr_matrix[position[0] + 1, position[1] : position[1] + half]
        left_bit = bit_matrix[position[0] + 1, position[1] : position[1] + half]
        right_llr = llr_matrix[position[0] + 1, position[1] + half : position[1] + span]
        right_bit = bit_matrix[position[0] + 1, position[1] + half : position[1] + span]

        if _all_computed(up_bit):
            position = _up(position)
        else:
            if _all_computed(right_bit):
                up_bit_val = _get_up_bit(left_bit.astype(int), right_bit.astype(int))
                bit_matrix[position[0], position[1] : position[1] + span] = up_bit_val
            else:
                if _all_computed(right_llr):
                    if position[0] == position[2] - 1:
                        right_bit_pos = position[1] + 1
                        if frozen_bits[right_bit_pos]:
                            right_bit_val = 0
                        else:
                            right_bit_val = 0 if right_llr[0] >= 0 else 1
                        bit_matrix[position[0] + 1, position[1] + half : position[1] + span] = right_bit_val
                    else:
                        position = _rightdown(position)
                else:
                    if _all_computed(left_bit):
                        right_llr_val = _get_right_llr(left_bit.astype(int), up_llr)
                        llr_matrix[position[0] + 1, position[1] + half : position[1] + span] = right_llr_val
                    else:
                        if not _all_computed(left_llr):
                            left_llr_val = _get_left_llr(up_llr)
                            llr_matrix[position[0] + 1, position[1] : position[1] + half] = left_llr_val
                        else:
                            if position[0] == position[2] - 1:
                                left_bit_pos = position[1]
                                if frozen_bits[left_bit_pos]:
                                    left_bit_val = 0
                                else:
                                    left_bit_val = 0 if left_llr[0] >= 0 else 1
                                bit_matrix[position[0] + 1, position[1] : position[1] + half] = left_bit_val
                            else:
                                position = _leftdown(position)

    return bit_matrix[n].astype(int)


def precompute_sc_indices(N):
    """
    预计算非递归 SC 译码辅助向量（层更新列表）
    """
    n = int(np.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]

    llr_layer_vec = []
    bit_layer_vec = []

    for phi in range(N):
        llr_layers = []
        bit_layers = []
        temp = phi
        for layer in range(n):
            if temp % 2 == 0:
                llr_layers.append(layer)
            else:
                bit_layers.append(layer)
            temp //= 2
        llr_layer_vec.append(llr_layers)
        bit_layer_vec.append(bit_layers)

    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode_recursive(llr_ch, frozen_bits):
    """递归 SC 译码（包装树遍历实现）"""
    return sc_decode(llr_ch, frozen_bits)


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码主函数。
    编码器含比特倒序时，先将信道 LLR 做比特倒序置换。
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr_ch)
    br = bit_reversal_permutation(N)
    llr = llr_ch[br].copy()
    return _sc_tree_decode(llr, frozen_bits)
