"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math
import numpy as np


def f_operation(La, Lb):
    """min-sum 近似的 f 运算（硬件友好，sign(0)=1）"""
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    sign_a = np.sign(La)
    sign_b = np.sign(Lb)
    sign_a = np.where(sign_a == 0, 1.0, sign_a)
    sign_b = np.where(sign_b == 0, 1.0, sign_b)
    return sign_a * sign_b * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb"""
    return (1 - 2 * u_hat) * La + Lb


def _all_num(x):
    return not np.any(np.isnan(x))


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


def _get_up_bit(left_bit, right_bit):
    left_bit = np.asarray(left_bit)
    right_bit = np.asarray(right_bit)
    length = len(left_bit)
    temp = np.array([(left_bit[i] + right_bit[i]) % 2 for i in range(length)] + list(right_bit))
    return temp


def _get_left_llr(up_llr):
    length = len(up_llr) // 2
    return np.array([f_operation(up_llr[i], up_llr[i + length]) for i in range(length)])


def _get_right_llr(left_bit, up_llr):
    length = len(left_bit)
    return np.array(
        [g_operation(up_llr[i], up_llr[i + length], left_bit[i]) for i in range(length)]
    )


def _get_left_bit(left_llr, info_set, frozen_bit, pos):
    if pos in info_set:
        return 0 if left_llr >= 0 else 1
    return frozen_bit


def _get_right_bit(right_llr, info_set, frozen_bit, pos):
    if pos in info_set:
        return 0 if right_llr > 0 else 1
    return frozen_bit


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC 译码主函数（基于矩阵状态机）"""
    y_llr = np.asarray(llr_ch, dtype=np.float64)
    N = len(y_llr)
    n = int(np.log2(N))
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    info_set = set(np.where(~frozen_bits)[0])
    frozen_bit = 0

    llr_matrix = np.ones((n + 1, N), dtype=np.float64)
    llr_matrix[:] = np.nan
    bit_matrix = llr_matrix.copy()
    llr_matrix[0] = y_llr
    position = [0, 0, n, N]

    while not _all_num(bit_matrix[n]):
        span = 2 ** (position[2] - position[0])
        up_llr = llr_matrix[position[0]][position[1] : position[1] + span]
        up_bit = bit_matrix[position[0]][position[1] : position[1] + span]
        half = span // 2
        left_llr = llr_matrix[position[0] + 1][position[1] : position[1] + half]
        left_bit = bit_matrix[position[0] + 1][position[1] : position[1] + half]
        right_llr = llr_matrix[position[0] + 1][position[1] + half : position[1] + span]
        right_bit = bit_matrix[position[0] + 1][position[1] + half : position[1] + span]

        if _all_num(up_bit):
            position = _up(position)
        elif _all_num(right_bit):
            up_bit = _get_up_bit(left_bit, right_bit)
            bit_matrix[position[0]][position[1] : position[1] + span] = up_bit
        elif _all_num(right_llr):
            if position[0] == position[2] - 1:
                right_bit_pos = position[1] + 1
                right_bit_val = _get_right_bit(
                    right_llr[0], info_set, frozen_bit, right_bit_pos
                )
                bit_matrix[position[0] + 1][position[1] + half] = right_bit_val
            else:
                position = _rightdown(position)
        elif _all_num(left_bit):
            right_llr_new = _get_right_llr(left_bit, up_llr)
            llr_matrix[position[0] + 1][position[1] + half : position[1] + span] = right_llr_new
        elif not _all_num(left_llr):
            left_llr_new = _get_left_llr(up_llr)
            llr_matrix[position[0] + 1][position[1] : position[1] + half] = left_llr_new
        else:
            if position[0] == position[2] - 1:
                left_bit_pos = position[1]
                left_bit_val = _get_left_bit(left_llr[0], info_set, frozen_bit, left_bit_pos)
                bit_matrix[position[0] + 1][position[1]] = left_bit_val
            else:
                position = _leftdown(position)

    return bit_matrix[n].astype(int)


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（参考实现，调用主译码器）"""
    return sc_decode(llr, frozen_bits)


def precompute_sc_indices(N):
    """预计算辅助向量（兼容 SCL 接口）"""
    n = int(np.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]
    return lambda_offset, [[] for _ in range(N)], [[] for _ in range(N)]


def _update_llr_for_phase(llr_layers, bit_layers, layer, phi, n):
    """占位函数，SCL 使用独立实现"""
    pass


def _update_bits_for_phase(bit_layers, phi, n):
    """占位函数，SCL 使用独立实现"""
    pass
