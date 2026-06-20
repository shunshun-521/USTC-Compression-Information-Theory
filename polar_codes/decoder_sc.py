"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
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
    return (1 - 2 * u_hat) * La + Lb


def _all_decided(bits):
    return not np.any(np.isnan(bits))


def _position_up(position):
    p0, p1, p2, p3 = position
    p0 -= 1
    block = int(np.floor(p1 / (2 ** (p2 - p0))))
    p1 = block * (2 ** (p2 - p0))
    return [p0, p1, p2, p3]


def _position_leftdown(position):
    p0, p1, p2, p3 = position
    return [p0 + 1, p1, p2, p3]


def _position_rightdown(position):
    p0, p1, p2, p3 = position
    return [p0 + 1, p1 + 2 ** (p2 - 1 - p0), p2, p3]


def _get_up_bit(left_bit, right_bit):
    length = len(left_bit)
    temp = np.array([(left_bit[i] + right_bit[i]) % 2 for i in range(length)] + list(right_bit))
    return temp


def _get_left_llr(up_llr):
    half = len(up_llr) // 2
    return np.array([f_operation(up_llr[i], up_llr[i + half]) for i in range(half)])


def _get_right_llr(left_bit, up_llr):
    half = len(left_bit)
    return np.array(
        [g_operation(up_llr[i], up_llr[i + half], left_bit[i]) for i in range(half)]
    )


def sc_decode_recursive(llr, frozen_bits):
    """
    递归 SC 译码（参考实现，基于分治树遍历）。
    与 sc_decode 采用相同 LLR 符号约定，结果在数值上等价。
    """
    llr = np.asarray(llr, dtype=np.float64)
    N = len(llr)
    n = int(math.log2(N))
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    info_indices = set(np.where(~frozen_bits)[0])

    llr_matrix = np.full((n + 1, N), np.nan, dtype=np.float64)
    bit_matrix = np.full((n + 1, N), np.nan, dtype=np.float64)
    llr_matrix[0, :] = llr

    def recurse(position):
        p0, p1, p2, _ = position
        span = 2 ** (p2 - p0)
        up_llr = llr_matrix[p0, p1 : p1 + span]
        up_bit = bit_matrix[p0, p1 : p1 + span]
        left_llr = llr_matrix[p0 + 1, p1 : p1 + span // 2]
        left_bit = bit_matrix[p0 + 1, p1 : p1 + span // 2]
        right_llr = llr_matrix[p0 + 1, p1 + span // 2 : p1 + span]
        right_bit = bit_matrix[p0 + 1, p1 + span // 2 : p1 + span]

        if _all_decided(up_bit):
            return _position_up(position)

        if _all_decided(right_bit):
            bit_matrix[p0, p1 : p1 + span] = _get_up_bit(
                left_bit.astype(int), right_bit.astype(int)
            )
            return position

        if _all_decided(right_llr):
            if p0 == p2 - 1:
                right_bit_pos = p1 + 1
                val = (0 if right_llr[0] >= 0 else 1) if right_bit_pos in info_indices else 0
                bit_matrix[p0 + 1, p1 + span // 2 : p1 + span] = val
                return position
            return _position_rightdown(position)

        if _all_decided(left_bit):
            llr_matrix[p0 + 1, p1 + span // 2 : p1 + span] = _get_right_llr(
                left_bit.astype(int), up_llr
            )
            return position

        if not _all_decided(left_llr):
            llr_matrix[p0 + 1, p1 : p1 + span // 2] = _get_left_llr(up_llr)
            return position

        if p0 == p2 - 1:
            left_bit_pos = p1
            val = (0 if left_llr[0] >= 0 else 1) if left_bit_pos in info_indices else 0
            bit_matrix[p0 + 1, p1 : p1 + span // 2] = val
            return position
        return _position_leftdown(position)

    position = [0, 0, n, N]
    while not _all_decided(bit_matrix[n, :]):
        position = recurse(position)
    return bit_matrix[n, :].astype(int)


def precompute_sc_indices(N):
    """预计算非递归 SC 译码所需的三个辅助向量。"""
    n = int(math.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]

    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        temp = phi
        llr_layers = []
        layer = 0
        while temp & 1:
            llr_layers.append(layer)
            temp >>= 1
            layer += 1
        llr_layer_vec.append(llr_layers)

        bit_layers = []
        temp = phi
        while temp & 1:
            bit_layers.append(len(bit_layers))
            temp >>= 1
        bit_layer_vec.append(bit_layers)

    return lambda_offset, llr_layer_vec, bit_layer_vec


def _get_up_loc(bit_matrix, n, N):
    detect_array = bit_matrix[n, :]
    detect = -1
    for i in range(N):
        if np.isnan(detect_array[i]):
            detect = i - 1
            break
    if detect % 2 == 0:
        loc_row = n - 1
        loc_col = detect
    else:
        loc_row = n - 1
        loc_col = detect - 1
    if detect == -1:
        loc_row = 0
        loc_col = 0
    return [loc_row, loc_col]


def sc_stepping_decode(llr_matrix, bit_matrix, frozen_bits, split_pos):
    """运行 SC 直到判决完成 split_pos 位置（含）。"""
    N = bit_matrix.shape[1]
    n = int(math.log2(N))
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    info_indices = set(np.where(~frozen_bits)[0])

    llr_matrix = llr_matrix.copy()
    bit_matrix = bit_matrix.copy()
    loc = _get_up_loc(bit_matrix, n, N)
    position = [loc[0], loc[1], n, N]

    while np.isnan(bit_matrix[n, split_pos]):
        p0, p1, p2, _ = position
        span = 2 ** (p2 - p0)
        up_llr = llr_matrix[p0, p1 : p1 + span]
        up_bit = bit_matrix[p0, p1 : p1 + span]
        left_llr = llr_matrix[p0 + 1, p1 : p1 + span // 2]
        left_bit = bit_matrix[p0 + 1, p1 : p1 + span // 2]
        right_llr = llr_matrix[p0 + 1, p1 + span // 2 : p1 + span]
        right_bit = bit_matrix[p0 + 1, p1 + span // 2 : p1 + span]

        if _all_decided(up_bit):
            position = _position_up(position)
            continue

        if _all_decided(right_bit):
            merged = _get_up_bit(left_bit.astype(int), right_bit.astype(int))
            bit_matrix[p0, p1 : p1 + span] = merged
            continue

        if _all_decided(right_llr):
            if p0 == p2 - 1:
                right_bit_pos = p1 + 1
                if right_bit_pos in info_indices:
                    right_val = 0 if right_llr[0] >= 0 else 1
                else:
                    right_val = 0
                bit_matrix[p0 + 1, p1 + span // 2 : p1 + span] = right_val
            else:
                position = _position_rightdown(position)
            continue

        if _all_decided(left_bit):
            right_llr_new = _get_right_llr(left_bit.astype(int), up_llr)
            llr_matrix[p0 + 1, p1 + span // 2 : p1 + span] = right_llr_new
            continue

        if not _all_decided(left_llr):
            left_llr_new = _get_left_llr(up_llr)
            llr_matrix[p0 + 1, p1 : p1 + span // 2] = left_llr_new
            continue

        if p0 == p2 - 1:
            left_bit_pos = p1
            if left_bit_pos in info_indices:
                left_val = 0 if left_llr[0] >= 0 else 1
            else:
                left_val = 0
            bit_matrix[p0 + 1, p1 : p1 + span // 2] = left_val
        else:
            position = _position_leftdown(position)

    return llr_matrix, bit_matrix


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码（因子图遍历实现，信道 LLR 位于第 0 层）。
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    N = len(llr_ch)
    n = int(math.log2(N))
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    info_indices = set(np.where(~frozen_bits)[0])

    llr_matrix = np.full((n + 1, N), np.nan, dtype=np.float64)
    bit_matrix = np.full((n + 1, N), np.nan, dtype=np.float64)
    llr_matrix[0, :] = llr_ch
    position = [0, 0, n, N]

    while not _all_decided(bit_matrix[n, :]):
        p0, p1, p2, _ = position
        span = 2 ** (p2 - p0)
        up_llr = llr_matrix[p0, p1 : p1 + span]
        up_bit = bit_matrix[p0, p1 : p1 + span]
        left_llr = llr_matrix[p0 + 1, p1 : p1 + span // 2]
        left_bit = bit_matrix[p0 + 1, p1 : p1 + span // 2]
        right_llr = llr_matrix[p0 + 1, p1 + span // 2 : p1 + span]
        right_bit = bit_matrix[p0 + 1, p1 + span // 2 : p1 + span]

        if _all_decided(up_bit):
            position = _position_up(position)
            continue

        if _all_decided(right_bit):
            merged = _get_up_bit(left_bit.astype(int), right_bit.astype(int))
            bit_matrix[p0, p1 : p1 + span] = merged
            continue

        if _all_decided(right_llr):
            if p0 == p2 - 1:
                right_bit_pos = p1 + 1
                if right_bit_pos in info_indices:
                    right_val = 0 if right_llr[0] >= 0 else 1
                else:
                    right_val = 0
                bit_matrix[p0 + 1, p1 + span // 2 : p1 + span] = right_val
            else:
                position = _position_rightdown(position)
            continue

        if _all_decided(left_bit):
            right_llr_new = _get_right_llr(left_bit.astype(int), up_llr)
            llr_matrix[p0 + 1, p1 + span // 2 : p1 + span] = right_llr_new
            continue

        if not _all_decided(left_llr):
            left_llr_new = _get_left_llr(up_llr)
            llr_matrix[p0 + 1, p1 : p1 + span // 2] = left_llr_new
            continue

        if p0 == p2 - 1:
            left_bit_pos = p1
            if left_bit_pos in info_indices:
                left_val = 0 if left_llr[0] >= 0 else 1
            else:
                left_val = 0
            bit_matrix[p0 + 1, p1 : p1 + span // 2] = left_val
        else:
            position = _position_leftdown(position)

    return bit_matrix[n, :].astype(int)
