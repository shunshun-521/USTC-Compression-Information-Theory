"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import copy

import numpy as np

from decoder_sc import sc_decode_recursive


_CRC8_POLY_BITS = np.array([1, 0, 0, 0, 0, 0, 1, 1], dtype=int)


def _get_poly_bits(crc_length):
    if crc_length == 8:
        return _CRC8_POLY_BITS
    return np.array(
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], dtype=int
    )


def _crc_shift_register(data_bits, crc_length):
    poly = _get_poly_bits(crc_length)
    reg = np.zeros(crc_length, dtype=int)
    for bit in data_bits:
        fb = int(bit) ^ reg[0]
        reg[:-1] = reg[1:]
        reg[-1] = 0
        if fb:
            reg ^= poly
    return reg


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=int).ravel()
    crc_bits = _crc_shift_register(info_bits, crc_length)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC 是否正确。"""
    if crc_length <= 0:
        return True
    bits = np.asarray(bits, dtype=int).ravel()
    return np.all(_crc_shift_register(bits, crc_length) == 0)


def _all_filled(arr):
    return not np.any(np.isnan(arr))


def _sign_ms(x):
    s = np.sign(x)
    return 1.0 if s == 0 else s


def _f_hf(l1, l2):
    return _sign_ms(l1) * _sign_ms(l2) * min(abs(l1), abs(l2))


def _g(l1, l2, u):
    return (1 - 2 * u) * l1 + l2


def _leftdown(pos):
    return [pos[0] + 1, pos[1], pos[2], pos[3]]


def _rightdown(pos):
    return [pos[0] + 1, pos[1] + 2 ** (pos[2] - 1 - pos[0]), pos[2], pos[3]]


def _up(pos):
    p0 = pos[0] - 1
    p1 = int(np.floor(pos[1] / (2 ** (pos[2] - pos[0] + 1))) * (2 ** (pos[2] - pos[0] + 1)))
    return [p0, p1, pos[2], pos[3]]


def _get_up_bit(left_bit, right_bit):
    length = len(left_bit)
    temp = np.array([(left_bit + right_bit) % 2, right_bit])
    return temp.reshape(1, 2 * length)


def _get_left_llr(up_llr):
    length = len(up_llr) // 2
    return np.array([_f_hf(up_llr[i], up_llr[i + length]) for i in range(length)])


def _get_right_llr(left_bit, up_llr):
    length = len(left_bit)
    return np.array([_g(up_llr[i], up_llr[i + length], left_bit[i]) for i in range(length)])


def _get_bit(llr_val, idx, info_set, frozen_val):
    if idx in info_set:
        return 0 if llr_val >= 0 else 1
    return frozen_val


def _get_up_loc(bit_matrix):
    n = int(np.log2(bit_matrix.shape[1]))
    detect_array = bit_matrix[n]
    detect = -1
    for i in range(len(detect_array)):
        if not (detect_array[i] == 0 or detect_array[i] == 1):
            detect = i - 1
            break
    if detect == -1:
        return [0, 0]
    if detect % 2 == 0:
        return [n - 1, detect]
    return [n - 1, detect - 1]


def _pm_update(llr_array, bit_array):
    pm = 0.0
    for llr, bit in zip(llr_array, bit_array):
        hard = 0 if llr >= 0 else 1
        if bit != hard:
            pm += abs(llr)
    return pm


def _sc_step_to(llr_matrix, bit_matrix, info_set, split_pos, frozen_val=0):
    """SC 逐步译码直到完成 split_pos 处比特判决。"""
    N = bit_matrix.shape[1]
    n = int(np.log2(N))
    loc = _get_up_loc(bit_matrix)
    position = [loc[0], loc[1], n, N]

    while not (bit_matrix[n][split_pos] == 0 or bit_matrix[n][split_pos] == 1):
        span = 2 ** (position[2] - position[0])
        half = span // 2
        up_llr = llr_matrix[position[0]][position[1] : position[1] + span]
        up_bit = bit_matrix[position[0]][position[1] : position[1] + span]
        left_llr = llr_matrix[position[0] + 1][position[1] : position[1] + half]
        left_bit = bit_matrix[position[0] + 1][position[1] : position[1] + half]
        right_llr = llr_matrix[position[0] + 1][position[1] + half : position[1] + span]
        right_bit = bit_matrix[position[0] + 1][position[1] + half : position[1] + span]

        if _all_filled(up_bit):
            position = _up(position)
        elif _all_filled(right_bit):
            up_bit_new = _get_up_bit(left_bit, right_bit)
            bit_matrix[position[0]][position[1] : position[1] + span] = up_bit_new.copy()
        elif _all_filled(right_llr):
            if position[0] == position[2] - 1:
                right_bit_pos = position[1] + half
                bit_matrix[position[0] + 1][position[1] + half] = _get_bit(
                    right_llr[0], right_bit_pos, info_set, frozen_val
                )
            else:
                position = _rightdown(position)
        elif _all_filled(left_bit):
            right_llr_new = _get_right_llr(left_bit, up_llr)
            llr_matrix[position[0] + 1][position[1] + half : position[1] + span] = (
                right_llr_new
            )
        elif not _all_filled(left_llr):
            left_llr_new = _get_left_llr(up_llr)
            llr_matrix[position[0] + 1][position[1] : position[1] + half] = left_llr_new
        elif position[0] == position[2] - 1:
            left_bit_pos = position[1]
            bit_matrix[position[0] + 1][position[1]] = _get_bit(
                left_llr[0], left_bit_pos, info_set, frozen_val
            )
        else:
            position = _leftdown(position)

    return llr_matrix, bit_matrix


def _scl_decode(llr_ch, info_indices, list_size, crc_length=0):
    """SCL 译码核心。"""
    if list_size <= 1:
        frozen = np.ones(len(llr_ch), dtype=int)
        frozen[info_indices] = 0
        u_hat = sc_decode_recursive(llr_ch, frozen)
        return u_hat, 0.0

    y_llr = np.asarray(llr_ch, dtype=np.float64)
    N = len(y_llr)
    n = int(np.log2(N))
    info_set = set(info_indices)
    split_pos = list(info_indices)
    frozen_val = 0

    def _new_matrices():
        llr_m = np.full((n + 1, N), np.nan, dtype=np.float64)
        bit_m = np.full((n + 1, N), np.nan, dtype=np.float64)
        llr_m[0] = y_llr.copy()
        return llr_m, bit_m

    llr_list, bit_list, pm_list = [], [], []
    llr0, bit0 = _new_matrices()
    llr_list.append(llr0)
    bit_list.append(bit0)
    pm_list.append(0.0)

    for loc, sp in enumerate(split_pos):
        new_llr, new_bit, new_pm = [], [], []
        prev_start = split_pos[loc - 1] + 1 if loc > 0 else 0

        for idx in range(len(llr_list)):
            llr_m = copy.deepcopy(llr_list[idx])
            bit_m = copy.deepcopy(bit_list[idx])
            pm0 = pm_list[idx]

            llr_m, bit_m = _sc_step_to(llr_m, bit_m, info_set, sp, frozen_val)
            llr_slice = llr_m[n][prev_start : sp + 1]
            bit_slice = bit_m[n][prev_start : sp + 1]
            pm_base = pm0 + _pm_update(llr_slice, bit_slice)

            new_llr.append(llr_m)
            new_bit.append(bit_m)
            new_pm.append(pm_base)

            bit_wrong = copy.deepcopy(bit_m)
            bit_wrong[n][sp] = 1 - bit_wrong[n][sp]
            bit_slice_w = bit_wrong[n][prev_start : sp + 1]
            new_llr.append(copy.deepcopy(llr_m))
            new_bit.append(bit_wrong)
            new_pm.append(pm0 + _pm_update(llr_slice, bit_slice_w))

        order = np.argsort(new_pm)[:list_size]
        llr_list = [new_llr[i] for i in order]
        bit_list = [new_bit[i] for i in order]
        pm_list = [new_pm[i] for i in order]

    last_sp = split_pos[-1]
    if last_sp != N - 1:
        for idx in range(len(llr_list)):
            llr_m = copy.deepcopy(llr_list[idx])
            bit_m = copy.deepcopy(bit_list[idx])
            llr_m, bit_m = _sc_step_to(llr_m, bit_m, info_set, N - 1, frozen_val)
            prev_start = last_sp + 1
            pm_list[idx] += _pm_update(
                llr_m[n][prev_start:N], bit_m[n][prev_start:N]
            )
            llr_list[idx] = llr_m
            bit_list[idx] = bit_m

    order = np.argsort(pm_list)
    best_u = None
    best_pm = pm_list[order[0]]

    if crc_length > 0:
        for i in order:
            u_cand = bit_list[i][n].astype(int)
            info_bits = u_cand[info_indices]
            if crc_check(info_bits, crc_length):
                return u_cand, pm_list[i]

    best_u = bit_list[order[0]][n].astype(int)
    return best_u, best_pm


class SCLDecoder:
    """SCL 译码器。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.list_size = list_size
        self.crc_length = crc_length
        self.frozen_bits = np.asarray(frozen_bits).astype(int)
        self.info_indices = np.where(self.frozen_bits == 0)[0]

    def decode(self, llr_ch):
        return _scl_decode(
            llr_ch, self.info_indices, self.list_size, self.crc_length
        )
