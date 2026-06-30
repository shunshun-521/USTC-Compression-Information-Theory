"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import numpy as np
from encoder import bit_reversal_permutation


def f_operation(La, Lb):
    """min-sum 近似的 f 运算。"""
    sa = np.sign(La)
    sb = np.sign(Lb)
    sa = np.where(sa == 0, 1, sa)
    sb = np.where(sb == 0, 1, sb)
    return sa * sb * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb"""
    return (1 - 2 * u_hat) * La + Lb


def _all_computed(arr):
    return not np.any(np.isnan(arr))


def _leftdown(position):
    return [position[0] + 1, position[1], position[2], position[3]]


def _rightdown(position):
    p0 = position[0] + 1
    p1 = position[1] + 2 ** (position[2] - 1 - position[0])
    return [p0, p1, position[2], position[3]]


def _up(position):
    p0 = position[0] - 1
    p1 = int(np.floor(position[1] / (2 ** (position[2] - position[0] + 1)))
             * (2 ** (position[2] - position[0] + 1)))
    return [p0, p1, position[2], position[3]]


def _get_up_bit(left_bit, right_bit):
    length = left_bit.size
    temp = np.array([(left_bit + right_bit) % 2, right_bit])
    temp.resize((1, 2 * length))
    return temp


def _get_right_llr(left_bit, up_llr):
    length = int(left_bit.size)
    return np.array([g_operation(up_llr[i], up_llr[i + length], left_bit[i])
                     for i in range(length)])


def _get_left_llr(up_llr):
    length = int(up_llr.size / 2)
    return np.array([f_operation(up_llr[i], up_llr[i + length]) for i in range(length)])


def _get_left_bit(left_llr, info_set, frozen_val, pos):
    if pos in info_set:
        return 0 if left_llr >= 0 else 1
    return frozen_val


def _get_right_bit(right_llr, info_set, frozen_val, pos):
    if pos in info_set:
        return 0 if right_llr > 0 else 1
    return frozen_val


def _sc_decode_core(y_llr, info_indices, frozen_val=0):
    """基于因子图状态机的 SC 译码核心实现。"""
    N = y_llr.size
    n = int(np.log2(N))
    info_set = set(int(i) for i in info_indices)

    llr_matrix = np.ones((n + 1, N), dtype=np.float64)
    llr_matrix[:] = np.nan
    bit_matrix = llr_matrix.copy()
    llr_matrix[0] = y_llr
    position = [0, 0, n, N]

    while not _all_computed(bit_matrix[n]):
        block = 2 ** (position[2] - position[0])
        start = position[1]
        up_llr = llr_matrix[position[0]][start:start + block]
        up_bit = bit_matrix[position[0]][start:start + block]
        half = block // 2
        left_llr = llr_matrix[position[0] + 1][start:start + half]
        left_bit = bit_matrix[position[0] + 1][start:start + half]
        right_llr = llr_matrix[position[0] + 1][start + half:start + block]
        right_bit = bit_matrix[position[0] + 1][start + half:start + block]

        if _all_computed(up_bit):
            position = _up(position)
        elif _all_computed(right_bit):
            up_bit_new = _get_up_bit(left_bit, right_bit)
            bit_matrix[position[0]][start:start + block] = up_bit_new.copy()
        elif _all_computed(right_llr):
            if position[0] == position[2] - 1:
                right_bit_pos = position[1] + 1
                right_bit_val = _get_right_bit(
                    right_llr, info_set, frozen_val, right_bit_pos
                )
                bit_matrix[position[0] + 1][start + half:start + block] = right_bit_val
            else:
                position = _rightdown(position)
        elif _all_computed(left_bit):
            right_llr_new = _get_right_llr(left_bit, up_llr)
            llr_matrix[position[0] + 1][start + half:start + block] = right_llr_new
        elif not _all_computed(left_llr):
            left_llr_new = _get_left_llr(up_llr)
            llr_matrix[position[0] + 1][start:start + half] = left_llr_new
        else:
            if position[0] == position[2] - 1:
                left_bit_pos = position[1]
                left_bit_val = _get_left_bit(
                    left_llr, info_set, frozen_val, left_bit_pos
                )
                bit_matrix[position[0] + 1][start:start + half] = left_bit_val
            else:
                position = _leftdown(position)

    return bit_matrix[n].astype(int)


def _get_up_loc(bit_matrix):
    N = int(bit_matrix[0].size)
    n = int(np.log2(N))
    detect_array = bit_matrix[n]
    detect = -1
    for i in range(N):
        if not (detect_array[i] == 0 or detect_array[i] == 1):
            detect = i - 1
            break
    if detect == -1:
        return [0, 0]
    if detect % 2 == 0:
        return [n - 1, detect]
    return [n - 1, detect - 1]


def sc_stepping_decoder(llr_matrix, bit_matrix, info_indices, frozen_val, split_pos):
    """SC 译码至 split_pos 位（含）。"""
    N = int(bit_matrix.shape[1])
    n = int(np.log2(N))
    info_set = set(int(i) for i in info_indices)
    loc = _get_up_loc(bit_matrix)
    position = [loc[0], loc[1], n, N]

    while not (bit_matrix[n][split_pos] == 0 or bit_matrix[n][split_pos] == 1):
        block = 2 ** (position[2] - position[0])
        start = position[1]
        up_llr = llr_matrix[position[0]][start:start + block]
        up_bit = bit_matrix[position[0]][start:start + block]
        half = block // 2
        left_llr = llr_matrix[position[0] + 1][start:start + half]
        left_bit = bit_matrix[position[0] + 1][start:start + half]
        right_llr = llr_matrix[position[0] + 1][start + half:start + block]
        right_bit = bit_matrix[position[0] + 1][start + half:start + block]

        if _all_computed(up_bit):
            position = _up(position)
        elif _all_computed(right_bit):
            up_bit_new = _get_up_bit(left_bit, right_bit)
            bit_matrix[position[0]][start:start + block] = up_bit_new.copy()
        elif _all_computed(right_llr):
            if position[0] == position[2] - 1:
                right_bit_pos = position[1] + 1
                bit_matrix[position[0] + 1][start + half:start + block] = _get_right_bit(
                    right_llr, info_set, frozen_val, right_bit_pos
                )
            else:
                position = _rightdown(position)
        elif _all_computed(left_bit):
            llr_matrix[position[0] + 1][start + half:start + block] = _get_right_llr(left_bit, up_llr)
        elif not _all_computed(left_llr):
            llr_matrix[position[0] + 1][start:start + half] = _get_left_llr(up_llr)
        else:
            if position[0] == position[2] - 1:
                left_bit_pos = position[1]
                bit_matrix[position[0] + 1][start:start + half] = _get_left_bit(
                    left_llr, info_set, frozen_val, left_bit_pos
                )
            else:
                position = _leftdown(position)

    return llr_matrix, bit_matrix


def _pm_update(llr_array, bit_array):
    pm = 0.0
    for i in range(llr_array.size):
        expected = 0 if llr_array[i] >= 0 else 1
        if int(bit_array[i]) != expected:
            pm += abs(llr_array[i])
    return pm


def _init_matrices(y_llr):
    N = y_llr.size
    n = int(np.log2(N))
    llr_matrix = np.ones((n + 1, N), dtype=np.float64)
    llr_matrix[:] = np.nan
    bit_matrix = llr_matrix.copy()
    llr_matrix[0] = y_llr
    return llr_matrix, bit_matrix


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（参考实现，调用非递归核心）。"""
    return sc_decode(llr, frozen_bits)


def precompute_sc_indices(N):
    """预计算辅助信息（接口兼容）。"""
    n = int(np.log2(N))
    return np.arange(N), [[] for _ in range(N)], [[] for _ in range(N)]


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码主函数。
    与 B_N F^{⊗n} 编码配套：内部对信道 LLR 做比特倒序置换。
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    N = len(llr_ch)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    info_indices = np.where(~frozen_bits)[0]
    brp = bit_reversal_permutation(N)
    y_llr = llr_ch[brp]
    return _sc_decode_core(y_llr, info_indices, frozen_val=0)
