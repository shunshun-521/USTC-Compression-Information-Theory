"""
极化码 SC/SCL 核心：树遍历译码（参考 PolarCodesPython）
"""
import numpy as np
from encoder import bit_reversal_permutation


def f_hf(L1, L2):
    s1 = 1 if L1 == 0 else np.sign(L1)
    s2 = 1 if L2 == 0 else np.sign(L2)
    return s1 * s2 * min(abs(L1), abs(L2))


def g(L1, L2, U1):
    return (1 - 2 * U1) * L1 + L2


def _all_num(x):
    return not np.any(np.isnan(x))


def _leftdown(pos):
    return [pos[0] + 1, pos[1], pos[2], pos[3]]


def _rightdown(pos):
    return [pos[0] + 1, pos[1] + 2 ** (pos[2] - pos[0] - 1), pos[2], pos[3]]


def _up(pos):
    p1 = int(np.floor(pos[1] / (2 ** (pos[2] - pos[0] + 1))) * (2 ** (pos[2] - pos[0] + 1)))
    return [pos[0] - 1, p1, pos[2], pos[3]]


def _get_up_bit(left_bit, right_bit):
    length = len(left_bit)
    temp = np.concatenate([
        np.array([(left_bit[i] + right_bit[i]) % 2 for i in range(length)]),
        right_bit,
    ])
    return temp


def _get_left_llr(up_llr):
    half = len(up_llr) // 2
    return np.array([f_hf(up_llr[i], up_llr[i + half]) for i in range(half)])


def _get_right_llr(left_bit, up_llr):
    half = len(left_bit)
    return np.array([g(up_llr[i], up_llr[i + half], left_bit[i]) for i in range(half)])


def _get_left_bit(left_llr, info_set, frozen_val, pos):
    if pos in info_set:
        return 0 if left_llr >= 0 else 1
    return frozen_val


def _get_right_bit(right_llr, info_set, frozen_val, pos):
    if pos in info_set:
        return 0 if right_llr > 0 else 1
    return frozen_val


def _get_up_loc(bit_matrix_n):
    N = len(bit_matrix_n)
    n = int(np.log2(N))
    detect = -1
    for i in range(N):
        if not (bit_matrix_n[i] == 0 or bit_matrix_n[i] == 1):
            detect = i - 1
            break
    if detect == -1:
        return [0, 0]
    if detect % 2 == 0:
        return [n - 1, detect]
    return [n - 1, detect - 1]


def _sc_tree_step(llr_matrix, bit_matrix, info_set, frozen_val, stop_pos=None):
    """树遍历一步或多步，直到 bit_matrix[n][stop_pos] 已判决"""
    N = llr_matrix.shape[1]
    n = int(np.log2(N))

    loc = _get_up_loc(bit_matrix[n])
    position = [loc[0], loc[1], n, N]

    if stop_pos is None:
        def done():
            return _all_num(bit_matrix[n])
    else:
        def done():
            v = bit_matrix[n][stop_pos]
            return v == 0 or v == 1

    while not done():
        up_llr = llr_matrix[position[0]][position[1]:position[1] + 2 ** (position[2] - position[0])]
        up_bit = bit_matrix[position[0]][position[1]:position[1] + 2 ** (position[2] - position[0])]
        span = 2 ** (position[2] - position[0] - 1)
        left_llr = llr_matrix[position[0] + 1][position[1]:position[1] + span]
        left_bit = bit_matrix[position[0] + 1][position[1]:position[1] + span]
        right_llr = llr_matrix[position[0] + 1][position[1] + span:position[1] + 2 * span]
        right_bit = bit_matrix[position[0] + 1][position[1] + span:position[1] + 2 * span]

        if _all_num(up_bit):
            position = _up(position)
        elif _all_num(right_bit):
            up_bit = _get_up_bit(left_bit, right_bit)
            bit_matrix[position[0]][position[1]:position[1] + 2 ** (position[2] - position[0])] = up_bit
        elif _all_num(right_llr):
            if position[0] == position[2] - 1:
                right_bit_pos = position[1] + 1
                rb = _get_right_bit(right_llr[0], info_set, frozen_val, right_bit_pos)
                bit_matrix[position[0] + 1][position[1] + span:position[1] + 2 * span] = rb
            else:
                position = _rightdown(position)
        elif _all_num(left_bit):
            right_llr_new = _get_right_llr(left_bit, up_llr)
            llr_matrix[position[0] + 1][position[1] + span:position[1] + 2 * span] = right_llr_new
        elif not _all_num(left_llr):
            left_llr_new = _get_left_llr(up_llr)
            llr_matrix[position[0] + 1][position[1]:position[1] + span] = left_llr_new
        else:
            if position[0] == position[2] - 1:
                left_bit_pos = position[1]
                lb = _get_left_bit(left_llr[0], info_set, frozen_val, left_bit_pos)
                bit_matrix[position[0] + 1][position[1]:position[1] + span] = lb
            else:
                position = _leftdown(position)

    return llr_matrix, bit_matrix


def sc_tree_decode(y_llr, info_indices, frozen_val=0):
    """完整树遍历 SC 译码"""
    y_llr = np.asarray(y_llr, dtype=np.float64)
    N = len(y_llr)
    n = int(np.log2(N))
    info_set = set(int(i) for i in info_indices)

    llr_matrix = np.full((n + 1, N), np.nan)
    bit_matrix = np.full((n + 1, N), np.nan)
    llr_matrix[0] = y_llr

    llr_matrix, bit_matrix = _sc_tree_step(
        llr_matrix, bit_matrix, info_set, frozen_val, stop_pos=None
    )
    return bit_matrix[n].astype(int)


def sc_step_to_position(llr_matrix, bit_matrix, info_indices, frozen_val, split_pos):
    """译码至 split_pos 位完成判决"""
    info_set = set(int(i) for i in info_indices)
    return _sc_tree_step(llr_matrix, bit_matrix, info_set, frozen_val, stop_pos=split_pos)


def get_pm_update(llr_array, bit_array):
    """路径度量：与硬判决不一致时累加 |LLR|"""
    pm = 0.0
    for llr, bit in zip(llr_array, bit_array):
        hard = 0 if llr >= 0 else 1
        if int(bit) != hard:
            pm += abs(llr)
    return pm


def preprocess_llr_for_polar_encode(llr_ch):
    """
    将信道 LLR 重排为与 x = u @ F^{\\otimes n} 编码一致的顺序。
  polar_encode 使用蝶形 + 比特倒序，等价于 u @ G[br, :].
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    br = bit_reversal_permutation(len(llr_ch))
    inv_br = np.argsort(br)
    return llr_ch[inv_br]


def bp_f(x, y, alpha=0.9375):
    return alpha * np.sign(x) * np.sign(y) * np.minimum(np.abs(x), np.abs(y))


def bp_update_left(left_array, right_array, layer_n):
    N = len(left_array)
    interval = 2 ** (layer_n - 1)
    num = N // (interval * 2)
    value = np.zeros(N)
    for i in range(num):
        for j in range(interval):
            li = 2 * i * interval + j
            ri = li + interval
            l0, l1 = left_array[li], left_array[ri]
            r0, r1 = right_array[li], right_array[ri]
            value[li] = bp_f(r1 + l1, l0, 1.0)
            value[ri] = bp_f(l0, r0, 1.0) + l1
    return value


def bp_update_right(left_array, right_array, layer_n):
    N = len(left_array)
    interval = 2 ** (layer_n - 1)
    num = N // (interval * 2)
    value = np.zeros(N)
    for i in range(num):
        for j in range(interval):
            li = 2 * i * interval + j
            ri = li + interval
            l0, l1 = left_array[li], left_array[ri]
            r0, r1 = right_array[li], right_array[ri]
            value[li] = bp_f(r1 + l1, r0, 1.0)
            value[ri] = bp_f(l0, l1, 1.0) + r1
    return value
