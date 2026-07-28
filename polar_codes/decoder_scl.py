"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np
from decoder_sc import (
    sc_decode, f_operation, g_operation, _all_num, _leftdown, _rightdown, _up,
    _get_up_bit, _get_right_bit, _get_left_bit, _get_right_llr, _get_left_llr,
)


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_process(bits, poly, crc_length):
    reg = 0
    mask = (1 << crc_length) - 1
    for bit in bits:
        feedback = ((reg >> (crc_length - 1)) & 1) ^ int(bit)
        reg = (reg << 1) & mask
        if feedback:
            reg ^= poly
    return reg


def crc_encode(info_bits, crc_length=8):
    info_bits = np.asarray(info_bits, dtype=np.int8)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    remainder = _crc_process(info_bits, poly, crc_length)
    crc_bits = np.array(
        [(remainder >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=np.int8,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=np.int8)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    return _crc_process(bits, poly, crc_length) == 0


def _get_up_loc(bit_matrix):
    N = int(bit_matrix[0].size)
    n = int(np.log2(N))
    detect_array = bit_matrix[n]
    detect = N - 1
    for i in range(N):
        if detect_array[i] != 0 and detect_array[i] != 1:
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


def _get_pm_update(llr_array, bit_array):
    pm = 0.0
    for i in range(len(llr_array)):
        if np.sign(llr_array[i]) != np.sign(1 - 2 * bit_array[i]):
            pm += abs(llr_array[i])
    return pm


def _sc_stepping_decoder(llr_matrix, bit_matrix, information_pos, frozen_bit, split_pos):
    N = int(bit_matrix[0].size)
    n = int(np.log2(N))
    loc = _get_up_loc(bit_matrix)
    position = [loc[0], loc[1], n, N]

    while bit_matrix[n][split_pos] != 0 and bit_matrix[n][split_pos] != 1:
        up_llr = llr_matrix[position[0]][position[1]:position[1] + 2 ** (position[2] - position[0])]
        up_bit = bit_matrix[position[0]][position[1]:position[1] + 2 ** (position[2] - position[0])]
        left_llr = llr_matrix[position[0] + 1][position[1]:position[1] + 2 ** (position[2] - position[0] - 1)]
        left_bit = bit_matrix[position[0] + 1][position[1]:position[1] + 2 ** (position[2] - position[0] - 1)]
        right_llr = llr_matrix[position[0] + 1][position[1] + 2 ** (position[2] - position[0] - 1):position[1] + 2 ** (position[2] - position[0])]
        right_bit = bit_matrix[position[0] + 1][position[1] + 2 ** (position[2] - position[0] - 1):position[1] + 2 ** (position[2] - position[0])]

        if _all_num(up_bit):
            position = _up(position)
        elif _all_num(right_bit):
            up_bit_val = _get_up_bit(left_bit, right_bit)
            bit_matrix[position[0]][position[1]:position[1] + 2 ** (position[2] - position[0])] = up_bit_val.copy()
        elif _all_num(right_llr):
            if position[0] == position[2] - 1:
                right_bit_pos = position[1] + 1
                right_bit_val = _get_right_bit(right_llr, information_pos, frozen_bit, right_bit_pos)
                bit_matrix[position[0] + 1][position[1] + 2 ** (position[2] - position[0] - 1):position[1] + 2 ** (position[2] - position[0])] = right_bit_val
            else:
                position = _rightdown(position)
        elif _all_num(left_bit):
            right_llr_val = _get_right_llr(left_bit, up_llr)
            llr_matrix[position[0] + 1][position[1] + 2 ** (position[2] - position[0] - 1):position[1] + 2 ** (position[2] - position[0])] = right_llr_val
        elif not _all_num(left_llr):
            left_llr_val = _get_left_llr(up_llr)
            llr_matrix[position[0] + 1][position[1]:position[1] + 2 ** (position[2] - position[0] - 1)] = left_llr_val
        else:
            if position[0] == position[2] - 1:
                left_bit_pos = position[1]
                left_bit_val = _get_left_bit(left_llr, information_pos, frozen_bit, left_bit_pos)
                bit_matrix[position[0] + 1][position[1]:position[1] + 2 ** (position[2] - position[0] - 1)] = left_bit_val
            else:
                position = _leftdown(position)

    return llr_matrix, bit_matrix


def _scl_decode(y_llr, information_pos, frozen_bit, list_size, crc_length):
    N = y_llr.size
    n = int(np.log2(N))
    llr_matrix = np.ones((n + 1, N))
    llr_matrix[llr_matrix == 1] = float('nan')
    bit_matrix = llr_matrix.copy()
    llr_matrix[0] = y_llr

    split_pos = list(information_pos)
    llr_list = [llr_matrix]
    bit_list = [bit_matrix]
    pm_list = [0.0]
    split_loc = 0
    split_len = len(split_pos)
    l_now = 1

    while split_len - 1 >= split_loc:
        for i in range(l_now):
            llr_temp = llr_list[i]
            bit_temp = bit_list[i]
            pm_temp = pm_list[i]
            llr_out, bit_out = _sc_stepping_decoder(
                llr_temp.copy(), bit_temp.copy(), information_pos, frozen_bit, split_pos[split_loc]
            )
            llr_list[i] = llr_out
            bit_list[i] = bit_out

            if split_loc > 0:
                sl = slice(split_pos[split_loc - 1] + 1, split_pos[split_loc] + 1)
            else:
                sl = slice(0, split_pos[split_loc] + 1)
            pm_list[i] = pm_temp + _get_pm_update(llr_out[n][sl], bit_out[n][sl])

            llr_list.append(llr_out.copy())
            bit_wrong = bit_out.copy()
            bit_wrong[n][split_pos[split_loc]] = 1 - bit_wrong[n][split_pos[split_loc]]
            bit_list.append(bit_wrong)
            pm_list.append(pm_temp + _get_pm_update(llr_out[n][sl], bit_wrong[n][sl]))

        if l_now > list_size / 2:
            keep = np.argsort(pm_list)[:list_size]
            pm_list = [pm_list[i] for i in keep]
            llr_list = [llr_list[i] for i in keep]
            bit_list = [bit_list[i] for i in keep]
            l_now = len(pm_list)
        else:
            l_now = len(pm_list)
        split_loc += 1

    if split_pos and split_pos[-1] != N - 1:
        for i in range(l_now):
            llr_temp = llr_list[i]
            bit_temp = bit_list[i]
            pm_temp = pm_list[i]
            llr_out, bit_out = _sc_stepping_decoder(
                llr_temp.copy(), bit_temp.copy(), information_pos, frozen_bit, N - 1
            )
            llr_list[i] = llr_out
            bit_list[i] = bit_out
            sl = slice(split_pos[split_loc - 1] + 1, N)
            pm_list[i] = pm_temp + _get_pm_update(llr_out[n][sl], bit_out[n][sl])

    pm_argsort = np.argsort(pm_list)
    if crc_length > 0:
        for idx in pm_argsort:
            u_candidate = bit_list[idx][n].astype(np.int8)
            info_bits = u_candidate[list(information_pos)]
            if crc_check(info_bits, crc_length):
                return u_candidate, pm_list[idx]
        best = pm_argsort[0]
        return bit_list[best][n].astype(np.int8), pm_list[best]

    best = pm_argsort[0]
    return bit_list[best][n].astype(np.int8), pm_list[best]


class SCLDecoder:
    """SCL 译码器"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.information_pos = list(np.where(~self.frozen_bits)[0])
        self.list_size = list_size
        self.crc_length = crc_length

    def decode(self, llr_ch):
        if self.list_size == 1 and self.crc_length == 0:
            u_hat = sc_decode(llr_ch, self.frozen_bits)
            return u_hat, 0.0
        u_hat, pm = _scl_decode(
            np.asarray(llr_ch, dtype=np.float64),
            self.information_pos,
            0,
            self.list_size,
            self.crc_length,
        )
        return u_hat, pm
