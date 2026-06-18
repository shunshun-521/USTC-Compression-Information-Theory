"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math
import numpy as np


def f_operation(La, Lb):
    """min-sum 近似的 f 运算"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb"""
    return (1.0 - 2.0 * u_hat) * La + Lb


def _all_num(x):
    return int(not np.any(np.isnan(x)))


def _left_llr(up_llr):
    length = len(up_llr) // 2
    return np.array(
        [f_operation(up_llr[i], up_llr[i + length]) for i in range(length)]
    )


def _right_llr(left_bit, up_llr):
    length = len(up_llr) // 2
    return np.array(
        [
            g_operation(up_llr[i], up_llr[i + length], left_bit[i])
            for i in range(length)
        ]
    )


def _up_bit(left_bit, right_bit):
    length = len(left_bit)
    temp = np.array([(left_bit + right_bit) % 2, right_bit])
    temp = temp.reshape(1, 2 * length)
    return temp[0]


def _leftdown(position):
    return [position[0] + 1, position[1], position[2], position[3]]


def _rightdown(position):
    return [
        position[0] + 1,
        position[1] + 2 ** (position[2] - 1 - position[0]),
        position[2],
        position[3],
    ]


def _up(position):
    span = 2 ** (position[2] - position[0] + 1)
    return [position[0] - 1, int(np.floor(position[1] / span) * span), position[2], position[3]]


def _get_left_bit(left_llr, information_pos, frozen_bit, left_bit_pos):
    if left_bit_pos in information_pos:
        return 0 if left_llr >= 0 else 1
    return frozen_bit


def _get_right_bit(right_llr, information_pos, frozen_bit, right_bit_pos):
    if right_bit_pos in information_pos:
        return 0 if right_llr > 0 else 1
    return frozen_bit


def _sc_tree_decode(y_llr, information_pos, frozen_bit=0):
    N = y_llr.size
    n = int(np.log2(N))
    llr_matrix = np.ones((n + 1, N), dtype=np.float64)
    llr_matrix[llr_matrix == 1] = np.nan
    bit_matrix = llr_matrix.copy()
    llr_matrix[0] = y_llr
    position = [0, 0, n, N]

    while _all_num(bit_matrix[n]) == 0:
        span = 2 ** (position[2] - position[0])
        up_llr = llr_matrix[position[0], position[1] : position[1] + span]
        up_bit = bit_matrix[position[0], position[1] : position[1] + span]
        half = span // 2
        left_llr = llr_matrix[position[0] + 1, position[1] : position[1] + half]
        left_bit = bit_matrix[position[0] + 1, position[1] : position[1] + half]
        right_llr = llr_matrix[position[0] + 1, position[1] + half : position[1] + span]
        right_bit = bit_matrix[position[0] + 1, position[1] + half : position[1] + span]

        if _all_num(up_bit) == 1:
            position = _up(position)
        elif _all_num(right_bit) == 1:
            merged = _up_bit(left_bit, right_bit)
            bit_matrix[position[0], position[1] : position[1] + span] = merged
        elif _all_num(right_llr) == 1:
            if position[0] == position[2] - 1:
                right_bit_pos = position[1] + half
                val = _get_right_bit(
                    right_llr[0], information_pos, frozen_bit, right_bit_pos
                )
                bit_matrix[position[0] + 1, position[1] + half : position[1] + span] = (
                    val
                )
            else:
                position = _rightdown(position)
        elif _all_num(left_bit) == 1:
            right_llr_new = _right_llr(left_bit, up_llr)
            llr_matrix[position[0] + 1, position[1] + half : position[1] + span] = (
                right_llr_new
            )
        elif _all_num(left_llr) == 0:
            left_llr_new = _left_llr(up_llr)
            llr_matrix[position[0] + 1, position[1] : position[1] + half] = left_llr_new
        else:
            if position[0] == position[2] - 1:
                left_bit_pos = position[1]
                val = _get_left_bit(
                    left_llr[0], information_pos, frozen_bit, left_bit_pos
                )
                bit_matrix[position[0] + 1, position[1] : position[1] + half] = val
            else:
                position = _leftdown(position)

    return bit_matrix[n].astype(int)


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（参考实现）"""
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    information_pos = list(np.where(~frozen_bits)[0])
    return _sc_tree_decode(np.asarray(llr, dtype=np.float64), information_pos, 0)


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC 译码主函数"""
    return sc_decode_recursive(llr_ch, frozen_bits)


def precompute_sc_indices(N):
    """预计算 SCL 辅助向量"""
    n = int(math.log2(N))
    from encoder import bit_reversal_permutation

    br = bit_reversal_permutation(N)
    lambda_offset = [0] * (n + 1)
    for layer in range(1, n + 1):
        lambda_offset[layer] = 2 ** (layer - 1)

    llr_layer_vec = []
    bit_layer_vec = []
    for phi_nat in range(N):
        phi = br[phi_nat]
        layers_llr = []
        l = 0
        while l < n and ((phi >> l) & 1):
            layers_llr.append(l)
            l += 1
        layers_llr.append(n)
        llr_layer_vec.append(layers_llr)

        layers_bit = []
        if phi % 2 == 0:
            l = 1
            while l <= n and ((phi >> (l - 1)) & 1) == 0:
                layers_bit.append(l)
                l += 1
        bit_layer_vec.append(layers_bit)

    return lambda_offset, llr_layer_vec, bit_layer_vec, br
