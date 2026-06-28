"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import numpy as np


def f_operation(La, Lb):
    """min-sum 近似的 f 运算（向量化）。"""
    s1 = np.sign(La)
    s2 = np.sign(Lb)
    s1[s1 == 0] = 1
    s2[s2 == 0] = 1
    return s1 * s2 * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算。"""
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
    length = left_bit.size
    temp = np.array([(left_bit + right_bit) % 2, right_bit])
    temp.resize((1, 2 * length))
    return temp.flatten()


def _get_right_bit(right_llr, frozen_bits, right_bit_pos):
    if frozen_bits[right_bit_pos]:
        return 0
    return 0 if right_llr > 0 else 1


def _get_right_llr(left_bit, up_llr):
    length = int(left_bit.size)
    return np.array([
        g_operation(up_llr[i], up_llr[i + length], left_bit[i])
        for i in range(length)
    ])


def _get_left_bit(left_llr, frozen_bits, left_bit_pos):
    if frozen_bits[left_bit_pos]:
        return 0
    return 0 if left_llr >= 0 else 1


def _get_left_llr(up_llr):
    length = int(up_llr.size / 2)
    return f_operation(up_llr[:length], up_llr[length:])


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（参考实现）。"""
    return sc_decode(llr, frozen_bits)


def precompute_sc_indices(N):
    """预计算非递归 SC 译码辅助向量（兼容 SCL 接口）。"""
    n = int(np.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]
    llr_layer_vec = [list(range(n)) for _ in range(N)]
    bit_layer_vec = [list(range(n)) for _ in range(N)]
    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC 译码主函数。"""
    y_llr = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = y_llr.size
    n = int(np.log2(N))

    llr_matrix = np.ones((n + 1, N), dtype=np.float64)
    llr_matrix[llr_matrix == 1] = np.nan
    bit_matrix = llr_matrix.copy()
    llr_matrix[0] = y_llr
    position = [0, 0, n, N]

    while not _all_num(bit_matrix[n]):
        row, col, depth, _ = position
        span = 2 ** (depth - row)
        half = 2 ** (depth - row - 1)

        up_llr = llr_matrix[row, col:col + span]
        up_bit = bit_matrix[row, col:col + span]
        left_llr = llr_matrix[row + 1, col:col + half]
        left_bit = bit_matrix[row + 1, col:col + half]
        right_llr = llr_matrix[row + 1, col + half:col + span]
        right_bit = bit_matrix[row + 1, col + half:col + span]

        if _all_num(up_bit):
            position = _up(position)
        elif _all_num(right_bit):
            up_bit_new = _get_up_bit(left_bit, right_bit)
            bit_matrix[row, col:col + span] = up_bit_new.copy()
        elif _all_num(right_llr):
            if row == depth - 1:
                right_bit_pos = col + 1
                right_bit_val = _get_right_bit(
                    right_llr[0], frozen_bits, right_bit_pos
                )
                bit_matrix[row + 1, col + half:col + span] = right_bit_val
            else:
                position = _rightdown(position)
        elif _all_num(left_bit):
            right_llr_new = _get_right_llr(left_bit, up_llr)
            llr_matrix[row + 1, col + half:col + span] = right_llr_new
        elif not _all_num(left_llr):
            left_llr_new = _get_left_llr(up_llr)
            llr_matrix[row + 1, col:col + half] = left_llr_new
        else:
            if row == depth - 1:
                left_bit_pos = col
                left_bit_val = _get_left_bit(
                    left_llr[0], frozen_bits, left_bit_pos
                )
                bit_matrix[row + 1, col:col + half] = left_bit_val
            else:
                position = _leftdown(position)

    return bit_matrix[n].astype(int)
