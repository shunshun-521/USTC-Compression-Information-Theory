"""
极化码 SC 译码核心辅助函数（树形遍历实现）
"""
import numpy as np


def f_operation(La, Lb):
    """min-sum 近似的 f 运算（向量化）。"""
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    scalar = La.ndim == 0
    if scalar:
        La = La.reshape(1)
        Lb = Lb.reshape(1)
    s1 = np.sign(La)
    s2 = np.sign(Lb)
    s1[s1 == 0] = 1
    s2[s2 == 0] = 1
    out = s1 * s2 * np.minimum(np.abs(La), np.abs(Lb))
    return out.item() if scalar else out


def g_operation(La, Lb, u_hat):
    """g 运算（支持标量/数组 u_hat）。"""
    return (1.0 - 2.0 * np.asarray(u_hat)) * La + Lb


def f_min_sum_alpha(La, Lb, alpha=0.9375):
    """带缩放因子的 min-sum f 运算（BP 用）。"""
    return alpha * f_operation(La, Lb)


def _all_filled(arr):
    return not np.any(np.isnan(arr))


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
    p1 = int(np.floor(position[1] / (2 ** (position[2] - position[0] + 1)))
            * (2 ** (position[2] - position[0] + 1)))
    return [p0, p1, position[2], position[3]]


def _get_up_bit(left_bit, right_bit):
    length = len(left_bit)
    temp = np.array([(left_bit + right_bit) % 2, right_bit])
    temp.resize((1, 2 * length))
    return temp


def _get_right_llr(left_bit, up_llr):
    length = len(left_bit)
    return np.array([g_operation(up_llr[i], up_llr[i + length], left_bit[i])
                     for i in range(length)])


def _get_left_llr(up_llr):
    length = len(up_llr) // 2
    return np.array([f_operation(up_llr[i], up_llr[i + length]) for i in range(length)])


def _is_info(bit_pos, info_set):
    return bit_pos in info_set


def _get_left_bit(left_llr, info_set, left_bit_pos):
    if _is_info(left_bit_pos, info_set):
        return 0 if left_llr >= 0 else 1
    return 0


def _get_right_bit(right_llr, info_set, right_bit_pos):
    if _is_info(right_bit_pos, info_set):
        return 0 if right_llr > 0 else 1
    return 0


def _get_up_loc(bit_matrix, n):
    detect_array = bit_matrix[n]
    detect = -1
    for i in range(len(detect_array)):
        if detect_array[i] != 0 and detect_array[i] != 1:
            detect = i - 1
            break
    if detect == -1:
        return [0, 0]
    if detect % 2 == 0:
        return [n - 1, detect]
    return [n - 1, detect - 1]


def _frozen_to_info_set(frozen_bits):
    fb = np.asarray(frozen_bits)
    if fb.dtype == bool:
        frozen_idx = np.where(fb)[0]
    else:
        frozen_idx = np.where(fb.astype(int) != 0)[0]
    N = len(fb)
    frozen_set = set(frozen_idx.tolist())
    return [i for i in range(N) if i not in frozen_set]


def _init_matrices(N, y_llr):
    n = int(np.log2(N))
    llr_matrix = np.ones((n + 1, N), dtype=np.float64)
    llr_matrix[:] = np.nan
    bit_matrix = llr_matrix.copy()
    llr_matrix[0] = y_llr
    return llr_matrix, bit_matrix, n


def _sc_decode_core(y_llr, info_set):
    """SC 译码核心（树形遍历）。"""
    N = y_llr.size
    llr_matrix, bit_matrix, n = _init_matrices(N, y_llr)
    position = [0, 0, n, N]

    while not _all_filled(bit_matrix[n]):
        span = 2 ** (position[2] - position[0])
        p0, p1 = position[0], position[1]
        up_llr = llr_matrix[p0][p1:p1 + span]
        up_bit = bit_matrix[p0][p1:p1 + span]
        half = span // 2
        left_llr = llr_matrix[p0 + 1][p1:p1 + half]
        left_bit = bit_matrix[p0 + 1][p1:p1 + half]
        right_llr = llr_matrix[p0 + 1][p1 + half:p1 + span]
        right_bit = bit_matrix[p0 + 1][p1 + half:p1 + span]

        if _all_filled(up_bit):
            position = _up(position)
        elif _all_filled(right_bit):
            up_bit = _get_up_bit(left_bit, right_bit)
            bit_matrix[p0][p1:p1 + span] = up_bit.copy()
        elif _all_filled(right_llr):
            if position[0] == position[2] - 1:
                right_bit_pos = position[1] + 1
                rb = _get_right_bit(right_llr[0], info_set, right_bit_pos)
                bit_matrix[p0 + 1][p1 + half:p1 + span] = rb
            else:
                position = _rightdown(position)
        elif _all_filled(left_bit):
            right_llr_new = _get_right_llr(left_bit, up_llr)
            llr_matrix[p0 + 1][p1 + half:p1 + span] = right_llr_new
        elif not _all_filled(left_llr):
            left_llr_new = _get_left_llr(up_llr)
            llr_matrix[p0 + 1][p1:p1 + half] = left_llr_new
        else:
            if position[0] == position[2] - 1:
                left_bit_pos = position[1]
                lb = _get_left_bit(left_llr[0], info_set, left_bit_pos)
                bit_matrix[p0 + 1][p1:p1 + half] = lb
            else:
                position = _leftdown(position)

    u_hat = bit_matrix[n].astype(int)
    return u_hat, llr_matrix, bit_matrix


def _sc_step_to(llr_matrix, bit_matrix, info_set, split_pos):
    """SC 译码到指定比特位置（SCL 用）。"""
    N = int(bit_matrix[0].size)
    n = int(np.log2(N))
    loc = _get_up_loc(bit_matrix, n)
    position = [loc[0], loc[1], n, N]

    while bit_matrix[n][split_pos] != 0 and bit_matrix[n][split_pos] != 1:
        span = 2 ** (position[2] - position[0])
        p0, p1 = position[0], position[1]
        up_llr = llr_matrix[p0][p1:p1 + span]
        up_bit = bit_matrix[p0][p1:p1 + span]
        half = span // 2
        left_llr = llr_matrix[p0 + 1][p1:p1 + half]
        left_bit = bit_matrix[p0 + 1][p1:p1 + half]
        right_llr = llr_matrix[p0 + 1][p1 + half:p1 + span]
        right_bit = bit_matrix[p0 + 1][p1 + half:p1 + span]

        if _all_filled(up_bit):
            position = _up(position)
        elif _all_filled(right_bit):
            up_bit = _get_up_bit(left_bit, right_bit)
            bit_matrix[p0][p1:p1 + span] = up_bit.copy()
        elif _all_filled(right_llr):
            if position[0] == position[2] - 1:
                right_bit_pos = position[1] + 1
                rb = _get_right_bit(right_llr[0], info_set, right_bit_pos)
                bit_matrix[p0 + 1][p1 + half:p1 + span] = rb
            else:
                position = _rightdown(position)
        elif _all_filled(left_bit):
            right_llr_new = _get_right_llr(left_bit, up_llr)
            llr_matrix[p0 + 1][p1 + half:p1 + span] = right_llr_new
        elif not _all_filled(left_llr):
            left_llr_new = _get_left_llr(up_llr)
            llr_matrix[p0 + 1][p1:p1 + half] = left_llr_new
        else:
            if position[0] == position[2] - 1:
                left_bit_pos = position[1]
                lb = _get_left_bit(left_llr[0], info_set, left_bit_pos)
                bit_matrix[p0 + 1][p1:p1 + half] = lb
            else:
                position = _leftdown(position)

    return llr_matrix, bit_matrix


def _path_metric_update(llr_array, bit_array):
    """路径度量：与 LLR 符号不一致时加 |LLR|。"""
    pm = 0.0
    for llr, bit in zip(llr_array, bit_array):
        hard = 0 if llr >= 0 else 1
        if hard != bit:
            pm += abs(llr)
    return pm
