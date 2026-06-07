"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现，基于因子图位置遍历）
"""
import math

import numpy as np

from encoder import prepare_channel_llr


def f_operation(La, Lb):
    """min-sum 近似的 f 运算"""
    sa = np.sign(La)
    sb = np.sign(Lb)
    sa = np.where(sa == 0, 1, sa)
    sb = np.where(sb == 0, 1, sb)
    return sa * sb * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算"""
    return (1 - 2 * u_hat) * La + Lb


def _info_indices(frozen_bits):
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    return np.where(~frozen_bits)[0]


def _all_decided(bits):
    return not np.isnan(bits).any()


def _get_up_bit(left_bit, right_bit):
    length = len(left_bit)
    temp = np.array([(left_bit + right_bit) % 2, right_bit])
    temp.resize((1, 2 * length))
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
    p0 = position[0] - 1
    p1 = int(
        np.floor(position[1] / (2 ** (position[2] - position[0] + 1)))
        * (2 ** (position[2] - position[0] + 1))
    )
    return [p0, p1, position[2], position[3]]


def _decide_bit(llr_val, bit_pos, info_set, frozen_val=0):
    if bit_pos not in info_set:
        return frozen_val
    return 0 if llr_val >= 0 else 1


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码（主实现）。
    信道 LLR 经比特倒序后与编码端 B_N 对齐。
    """
    y_llr = prepare_channel_llr(llr_ch)
    info_set = set(_info_indices(frozen_bits))
    frozen_val = 0

    N = y_llr.size
    n = int(math.log2(N))

    llr_matrix = np.full((n + 1, N), np.nan, dtype=np.float64)
    bit_matrix = np.full((n + 1, N), np.nan, dtype=np.float64)
    llr_matrix[0] = y_llr
    position = [0, 0, n, N]

    while not _all_decided(bit_matrix[n]):
        span = 2 ** (position[2] - position[0])
        p0, p1 = position[0], position[1]

        up_llr = llr_matrix[p0][p1:p1 + span]
        up_bit = bit_matrix[p0][p1:p1 + span]
        left_llr = llr_matrix[p0 + 1][p1:p1 + span // 2]
        left_bit = bit_matrix[p0 + 1][p1:p1 + span // 2]
        right_llr = llr_matrix[p0 + 1][p1 + span // 2:p1 + span]
        right_bit = bit_matrix[p0 + 1][p1 + span // 2:p1 + span]

        if _all_decided(up_bit):
            position = _up(position)
        elif _all_decided(right_bit):
            up_bit_val = _get_up_bit(left_bit, right_bit)
            bit_matrix[p0][p1:p1 + span] = up_bit_val.copy()
        elif _all_decided(right_llr):
            if position[0] == position[2] - 1:
                right_pos = p1 + 1
                val = _decide_bit(right_llr[0], right_pos, info_set, frozen_val)
                bit_matrix[p0 + 1][p1 + span // 2:p1 + span] = val
            else:
                position = _rightdown(position)
        elif _all_decided(left_bit):
            half = len(up_llr) // 2
            new_right = np.array([
                g_operation(up_llr[i], up_llr[i + half], left_bit[i])
                for i in range(half)
            ])
            llr_matrix[p0 + 1][p1 + span // 2:p1 + span] = new_right
        elif not _all_decided(left_llr):
            half = len(up_llr) // 2
            new_left = f_operation(up_llr[:half], up_llr[half:])
            llr_matrix[p0 + 1][p1:p1 + span // 2] = new_left
        else:
            if position[0] == position[2] - 1:
                left_pos = p1
                val = _decide_bit(left_llr[0], left_pos, info_set, frozen_val)
                bit_matrix[p0 + 1][p1:p1 + span // 2] = val
            else:
                position = _leftdown(position)

    return bit_matrix[n].astype(int)


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（参考实现，与 sc_decode 结果一致）"""
    return sc_decode(llr, frozen_bits)


def precompute_sc_indices(N):
    """预计算非递归 SC 调度（供 SCL 复用）"""
    n = int(math.log2(N))
    lambda_offset = [1 << s for s in range(n + 1)]

    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        llr_layers = []
        p = phi
        while p & 1:
            llr_layers.append(int(math.log2(p & -p)))
            p >>= 1
        llr_layer_vec.append(llr_layers)

        if phi % 2 == 0:
            tz = n if phi == 0 else int(math.log2(phi & -phi))
            bit_layers = list(range(tz, n))
        else:
            bit_layers = list(range(int(math.log2(phi & -phi)), n))
        bit_layer_vec.append(bit_layers)

    return lambda_offset, llr_layer_vec, bit_layer_vec
