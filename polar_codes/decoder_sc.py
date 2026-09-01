"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效状态机实现）
"""
import math
import numpy as np


def f_operation(La, Lb):
    """min-sum 近似的 f 运算。"""
    sa = np.sign(La)
    sb = np.sign(Lb)
    sa = np.where(sa == 0, 1, sa)
    sb = np.where(sb == 0, 1, sb)
    return sa * sb * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb"""
    u_hat = np.asarray(u_hat)
    return (1.0 - 2.0 * u_hat) * La + Lb


def _all_ready(arr):
    return not np.any(np.isnan(arr))


def _leftdown(position):
    return [position[0] + 1, position[1], position[2], position[3]]


def _rightdown(position):
    return [
        position[0] + 1,
        position[1] + (1 << (position[2] - 1 - position[0])),
        position[2],
        position[3],
    ]


def _up(position):
    p0 = position[0] - 1
    p1 = int(np.floor(position[1] / (1 << (position[2] - position[0] + 1)))
             * (1 << (position[2] - position[0] + 1)))
    return [p0, p1, position[2], position[3]]


def _get_up_bit(left_bit, right_bit):
    length = len(left_bit)
    temp = np.vstack([(left_bit + right_bit) % 2, right_bit]).reshape(1, 2 * length)
    return temp.flatten()


def _get_left_llr(up_llr):
    half = len(up_llr) // 2
    return np.array([f_operation(up_llr[i], up_llr[i + half]) for i in range(half)])


def _get_right_llr(left_bit, up_llr):
    half = len(left_bit)
    return np.array([
        g_operation(up_llr[i], up_llr[i + half], left_bit[i]) for i in range(half)
    ])


def _decide_bit(llr, idx, info_indices, frozen_bits):
    info_set = set(info_indices) if info_indices is not None else None
    if info_set is None:
        is_info = not frozen_bits[idx]
    else:
        is_info = idx in info_set
    if not is_info:
        return 0
    return 0 if llr >= 0 else 1


def sc_decode(llr_ch, frozen_bits, info_indices=None):
    """
    非递归 SC 译码（状态机实现，与 x = u @ F^⊗n 编码配套）。
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    N = len(llr_ch)
    n = int(math.log2(N))
    frozen_bits = np.asarray(frozen_bits, dtype=bool)

    if info_indices is None:
        info_indices = np.where(~frozen_bits)[0]
    info_indices = np.asarray(info_indices, dtype=int)

    llr_matrix = np.full((n + 1, N), np.nan, dtype=np.float64)
    bit_matrix = np.full((n + 1, N), np.nan, dtype=np.float64)
    llr_matrix[0] = llr_ch
    position = [0, 0, n, N]

    while not _all_ready(bit_matrix[n]):
        span = 1 << (position[2] - position[0])
        up_llr = llr_matrix[position[0], position[1]:position[1] + span]
        up_bit = bit_matrix[position[0], position[1]:position[1] + span]
        half = span // 2
        left_llr = llr_matrix[position[0] + 1, position[1]:position[1] + half]
        left_bit = bit_matrix[position[0] + 1, position[1]:position[1] + half]
        right_llr = llr_matrix[position[0] + 1, position[1] + half:position[1] + span]
        right_bit = bit_matrix[position[0] + 1, position[1] + half:position[1] + span]

        if _all_ready(up_bit):
            position = _up(position)
        elif _all_ready(right_bit):
            up_bit_val = _get_up_bit(left_bit.astype(int), right_bit.astype(int))
            bit_matrix[position[0], position[1]:position[1] + span] = up_bit_val
        elif _all_ready(right_llr):
            if position[0] == position[2] - 1:
                right_pos = position[1] + 1
                bit_val = _decide_bit(right_llr[0], right_pos, info_indices, frozen_bits)
                bit_matrix[position[0] + 1, position[1] + half:position[1] + span] = bit_val
            else:
                position = _rightdown(position)
        elif _all_ready(left_bit):
            right_llr_val = _get_right_llr(left_bit.astype(int), up_llr)
            llr_matrix[position[0] + 1, position[1] + half:position[1] + span] = right_llr_val
        elif not _all_ready(left_llr):
            left_llr_val = _get_left_llr(up_llr)
            llr_matrix[position[0] + 1, position[1]:position[1] + half] = left_llr_val
        else:
            if position[0] == position[2] - 1:
                left_pos = position[1]
                bit_val = _decide_bit(left_llr[0], left_pos, info_indices, frozen_bits)
                bit_matrix[position[0] + 1, position[1]:position[1] + half] = bit_val
            else:
                position = _leftdown(position)

    return bit_matrix[n].astype(int)


def sc_decode_recursive(llr, frozen_bits, info_indices=None):
    """递归 SC 译码（调用非递归实现）。"""
    return sc_decode(llr, frozen_bits, info_indices)


def precompute_sc_indices(N):
    """兼容 SCL 的辅助向量接口。"""
    n = int(math.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]
    llr_layer_vec = [list(range(n)) for _ in range(N)]
    bit_layer_vec = [list(range(n)) for _ in range(N)]
    return lambda_offset, llr_layer_vec, bit_layer_vec
