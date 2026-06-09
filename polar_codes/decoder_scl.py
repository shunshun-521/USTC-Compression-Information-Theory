"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
import math
from decoder_sc import (
    _prepare_llr, _sc_decode_core, _all_num, _leftdown, _rightdown, _up,
    _get_up_bit, _get_right_bit, _get_left_bit, _get_right_llr, _get_left_llr,
    f_operation, g_operation,
)


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, crc_length):
    reg = 0
    for bit in bits:
        reg = (reg << 1) | int(bit)
        if reg & (1 << crc_length):
            reg ^= poly
    return reg & ((1 << crc_length) - 1)


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    remainder = _crc_remainder(info_bits, poly, crc_length)
    crc_bits = np.array(
        [(remainder >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=int,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC。"""
    bits = np.asarray(bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    return _crc_remainder(bits, poly, crc_length) == 0


def _get_up_loc(bit_matrix):
    N = bit_matrix[0].size
    n = int(np.log2(N))
    detect = -1
    for i in range(N):
        if bit_matrix[n][i] != 0 and bit_matrix[n][i] != 1:
            detect = i - 1
            break
    if detect % 2 == 0:
        loc_row, loc_col = n - 1, detect
    else:
        loc_row, loc_col = n - 1, detect - 1
    if detect == -1:
        loc_row, loc_col = 0, 0
    return [loc_row, loc_col]


def _sc_stepping_decoder(llr_matrix, bit_matrix, information_pos, frozen_bit, split_pos):
    N = int(bit_matrix[0].size)
    n = int(np.log2(N))
    loc = _get_up_loc(bit_matrix)
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

        if _all_num(up_bit) == 1:
            position = _up(position)
        elif _all_num(right_bit) == 1:
            up_bit = _get_up_bit(left_bit, right_bit)
            bit_matrix[p0][p1:p1 + span] = up_bit.copy()
        elif _all_num(right_llr) == 1:
            if position[0] == position[2] - 1:
                right_bit_pos = p1 + 1
                rb = _get_right_bit(right_llr, information_pos, frozen_bit, right_bit_pos)
                bit_matrix[p0 + 1][p1 + half:p1 + span] = rb
            else:
                position = _rightdown(position)
        elif _all_num(left_bit) == 1:
            right_llr = _get_right_llr(left_bit, up_llr)
            llr_matrix[p0 + 1][p1 + half:p1 + span] = right_llr
        elif _all_num(left_llr) == 0:
            left_llr = _get_left_llr(up_llr)
            llr_matrix[p0 + 1][p1:p1 + half] = left_llr
        elif position[0] == position[2] - 1:
            left_bit_pos = p1
            lb = _get_left_bit(left_llr, information_pos, frozen_bit, left_bit_pos)
            bit_matrix[p0 + 1][p1:p1 + half] = lb
        else:
            position = _leftdown(position)

    return llr_matrix, bit_matrix


def _get_pm_update(llr_array, bit_array):
    pm = 0.0
    for i in range(llr_array.size):
        expected = 1 - 2 * bit_array[i]
        if np.sign(llr_array[i]) != np.sign(expected):
            pm += abs(llr_array[i])
    return pm


def _scl_decode_core(y_llr, information_pos, list_size, crc_length=0):
    N = y_llr.size
    n = int(np.log2(N))
    info_list = list(np.sort(information_pos))
    frozen_bit = 0

    llr_matrix = np.ones((n + 1, N), dtype=np.float64)
    llr_matrix[llr_matrix == 1] = np.nan
    bit_matrix = llr_matrix.copy()
    llr_matrix[0] = y_llr

    llr_list = [llr_matrix.copy()]
    bit_list = [bit_matrix.copy()]
    pm_list = [0.0]
    split_pos = info_list
    split_loc = 0
    l_now = 1

    while split_loc < len(split_pos):
        new_llr, new_bit, new_pm = [], [], []
        for i in range(l_now):
            lm, bm = llr_list[i].copy(), bit_list[i].copy()
            pm_temp = pm_list[i]
            lm, bm = _sc_stepping_decoder(lm, bm, set(info_list), frozen_bit, split_pos[split_loc])

            prev = split_pos[split_loc - 1] if split_loc > 0 else -1
            llr_slice = lm[n][prev + 1:split_pos[split_loc] + 1]
            bit_slice = bm[n][prev + 1:split_pos[split_loc] + 1]
            pm0 = pm_temp + _get_pm_update(llr_slice, bit_slice)

            new_llr.append(lm)
            new_bit.append(bm)
            new_pm.append(pm0)

            bm1 = bm.copy()
            bm1[n][split_pos[split_loc]] = 1 - bm1[n][split_pos[split_loc]]
            bit_slice1 = bm1[n][prev + 1:split_pos[split_loc] + 1]
            pm1 = pm_temp + _get_pm_update(llr_slice, bit_slice1)
            new_llr.append(lm.copy())
            new_bit.append(bm1)
            new_pm.append(pm1)

        order = np.argsort(new_pm)[:list_size]
        llr_list = [new_llr[i] for i in order]
        bit_list = [new_bit[i] for i in order]
        pm_list = [new_pm[i] for i in order]
        l_now = len(pm_list)
        split_loc += 1

    if split_pos[-1] != N - 1:
        for i in range(l_now):
            lm, bm = llr_list[i].copy(), bit_list[i].copy()
            pm_temp = pm_list[i]
            lm, bm = _sc_stepping_decoder(lm, bm, set(info_list), frozen_bit, N - 1)
            prev = split_pos[-1]
            llr_slice = lm[n][prev + 1:N]
            bit_slice = bm[n][prev + 1:N]
            pm_list[i] = pm_temp + _get_pm_update(llr_slice, bit_slice)
            llr_list[i] = lm
            bit_list[i] = bm

    order = np.argsort(pm_list)
    best_u = None
    best_pm = None
    if crc_length > 0:
        for idx in order:
            u_cand = bit_list[idx][n].astype(int)
            info_part = u_cand[info_list]
            if crc_check(info_part, crc_length):
                return u_cand, pm_list[idx]

    idx = order[0]
    return bit_list[idx][n].astype(int), pm_list[idx]


class SCLDecoder:
    """SCL 译码器。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]

    def decode(self, llr_ch):
        llr = _prepare_llr(llr_ch)
        if self.list_size == 1:
            u_hat = _sc_decode_core(llr, self.info_indices)
            return u_hat, 0.0
        return _scl_decode_core(llr, self.info_indices, self.list_size, self.crc_length)
