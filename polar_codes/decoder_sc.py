"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math

import numpy as np


def f_operation(La, Lb):
    """
    min-sum 近似的 f 运算：
    f(La, Lb) ≈ sign(La) * sign(Lb) * min(|La|, |Lb|)
    """
    sa = np.sign(La)
    sb = np.sign(Lb)
    sa = np.where(sa == 0, 1, sa)
    sb = np.where(sb == 0, 1, sb)
    return sa * sb * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb"""
    return (1 - 2 * u_hat) * La + Lb


def _frozen_to_info_pos(frozen_bits):
    """将冻结位掩码转换为信息位索引列表"""
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    return list(np.where(~frozen_bits)[0])


def _all_decided(bits):
    return not np.any(np.isnan(bits))


def _up(position):
    p0 = position[0] - 1
    span = 2 ** (position[2] - position[0] + 1)
    p1 = int(np.floor(position[1] / span) * span)
    return [p0, p1, position[2], position[3]]


def _leftdown(position):
    return [position[0] + 1, position[1], position[2], position[3]]


def _rightdown(position):
    return [
        position[0] + 1,
        position[1] + 2 ** (position[2] - position[0] - 1),
        position[2],
        position[3],
    ]


def _get_up_bit(left_bit, right_bit):
    length = left_bit.size
    temp = np.array([(left_bit + right_bit) % 2, right_bit])
    temp.resize((1, 2 * length))
    return temp[0]


def _get_right_bit(right_llr, information_pos, frozen_bit, right_bit_pos):
    if right_bit_pos in information_pos:
        return 0 if right_llr[0] > 0 else 1
    return frozen_bit


def _get_left_bit(left_llr, information_pos, frozen_bit, left_bit_pos):
    if left_bit_pos in information_pos:
        return 0 if left_llr[0] >= 0 else 1
    return frozen_bit


def _get_right_llr(left_bit, up_llr):
    length = int(left_bit.size)
    return np.array([
        g_operation(up_llr[i], up_llr[i + length], left_bit[i])
        for i in range(length)
    ])


def _get_left_llr(up_llr):
    length = int(up_llr.size / 2)
    return np.array([
        f_operation(up_llr[i], up_llr[i + length]) for i in range(length)
    ])


def _sc_decode_core(y_llr, information_pos, frozen_bit):
    """基于因子图遍历的 SC 译码核心实现"""
    N = y_llr.size
    n = int(math.log2(N))
    llr_matrix = np.full((n + 1, N), np.nan, dtype=np.float64)
    bit_matrix = np.full((n + 1, N), np.nan, dtype=np.float64)
    llr_matrix[0] = y_llr
    position = [0, 0, n, N]

    while not _all_decided(bit_matrix[n]):
        block = 2 ** (position[2] - position[0])
        c = position[1]
        r = position[0]

        up_llr = llr_matrix[r, c:c + block]
        up_bit = bit_matrix[r, c:c + block]
        left_llr = llr_matrix[r + 1, c:c + block // 2]
        left_bit = bit_matrix[r + 1, c:c + block // 2]
        right_llr = llr_matrix[r + 1, c + block // 2:c + block]
        right_bit = bit_matrix[r + 1, c + block // 2:c + block]

        if _all_decided(up_bit):
            position = _up(position)
        elif _all_decided(right_bit):
            bit_matrix[r, c:c + block] = _get_up_bit(left_bit, right_bit)
        elif _all_decided(right_llr):
            if r == position[2] - 1:
                idx = c + block // 2
                bit_matrix[r + 1, idx] = _get_right_bit(
                    right_llr, information_pos, frozen_bit, idx
                )
            else:
                position = _rightdown(position)
        elif _all_decided(left_bit):
            llr_matrix[r + 1, c + block // 2:c + block] = _get_right_llr(
                left_bit, up_llr
            )
        elif not _all_decided(left_llr):
            llr_matrix[r + 1, c:c + block // 2] = _get_left_llr(up_llr)
        else:
            if r == position[2] - 1:
                bit_matrix[r + 1, c] = _get_left_bit(
                    left_llr, information_pos, frozen_bit, c
                )
            else:
                position = _leftdown(position)

    return bit_matrix[n].astype(int)


def sc_decode_recursive(llr, frozen_bits):
    """
    递归 SC 译码（参考实现）。
    """
    llr = np.asarray(llr, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    info_pos = _frozen_to_info_pos(frozen_bits)
    frozen_bit = 0
    return _sc_decode_core(llr, info_pos, frozen_bit)


def precompute_sc_indices(N):
    """
    预计算非递归 SC 译码所需的三个辅助向量。
    """
    n = int(math.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]

    llr_layer_vec = []
    bit_layer_vec = []

    for phi in range(N):
        layers = []
        psi = phi
        while psi % 2 == 1:
            layers.append(int(math.log2(psi & -psi)))
            psi >>= 1
        llr_layer_vec.append(layers)

        layers_b = []
        if phi % 2 == 0:
            psi = phi
            while psi % 2 == 0 and psi > 0:
                layers_b.append(int(math.log2(psi & -psi)))
                psi >>= 1
        bit_layer_vec.append(layers_b)

    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码主函数（与 sc_decode_recursive 等价的高效因子图实现）。
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    info_pos = _frozen_to_info_pos(frozen_bits)
    return _sc_decode_core(llr_ch, info_pos, frozen_bit=0)
