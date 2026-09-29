"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np

from decoder_sc import (
    sc_decode,
    _sc_tree_decode,
    f_operation,
    g_operation,
    _all_filled,
    _leftdown,
    _rightdown,
    _up,
    _get_up_bit,
    _get_left_llr,
    _get_right_llr,
)

CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_update(reg, bit, poly, crc_length):
    reg ^= int(bit) << (crc_length - 1)
    if reg & (1 << (crc_length - 1)):
        reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
    else:
        reg = (reg << 1) & ((1 << crc_length) - 1)
    return reg


def crc_encode(info_bits, crc_length=8):
    info_bits = np.asarray(info_bits, dtype=int).ravel()
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    reg = 0
    for b in info_bits:
        reg = _crc_update(reg, b, poly, crc_length)
    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=int).ravel()
    if len(bits) < crc_length:
        return False
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    reg = 0
    for b in bits:
        reg = _crc_update(reg, b, poly, crc_length)
    return reg == 0


def _get_up_loc(bit_matrix):
    n = int(np.log2(bit_matrix.shape[1]))
    detect_array = bit_matrix[n]
    detect = -1
    for i in range(detect_array.size):
        if detect_array[i] not in (0, 1):
            detect = i - 1
            break
    if detect == -1:
        return [0, 0]
    if detect % 2 == 0:
        return [n - 1, detect]
    return [n - 1, detect - 1]


def _get_pm_update(llr_array, bit_array):
    pm = 0.0
    for llr, bit in zip(llr_array, bit_array):
        hard = 0 if llr >= 0 else 1
        if hard != bit:
            pm += abs(llr)
    return pm


def _sc_stepping(llr_matrix, bit_matrix, information_pos, split_pos):
    n = int(np.log2(bit_matrix.shape[1]))
    n_layers = n + 1
    info_set = set(information_pos)
    loc = _get_up_loc(bit_matrix)
    position = [loc[0], loc[1], n, bit_matrix.shape[1]]

    while bit_matrix[n][split_pos] not in (0, 1):
        span = 2 ** (position[2] - position[0])
        up_llr = llr_matrix[position[0]][position[1]: position[1] + span]
        up_bit = bit_matrix[position[0]][position[1]: position[1] + span]
        half = span // 2
        left_llr = llr_matrix[position[0] + 1][position[1]: position[1] + half]
        left_bit = bit_matrix[position[0] + 1][position[1]: position[1] + half]
        right_llr = llr_matrix[position[0] + 1][position[1] + half: position[1] + span]
        right_bit = bit_matrix[position[0] + 1][position[1] + half: position[1] + span]

        if _all_filled(up_bit):
            position = _up(position)
        elif _all_filled(right_bit):
            up_bit_new = _get_up_bit(left_bit, right_bit)
            bit_matrix[position[0]][position[1]: position[1] + span] = up_bit_new
        elif _all_filled(right_llr):
            if position[0] == position[2] - 1:
                pos = position[1] + 1
                val = (0 if right_llr[0] >= 0 else 1) if pos in info_set else 0
                bit_matrix[position[0] + 1][position[1] + half] = val
            else:
                position = _rightdown(position)
        elif _all_filled(left_bit):
            right_llr_new = _get_right_llr(left_bit, up_llr)
            llr_matrix[position[0] + 1][position[1] + half: position[1] + span] = right_llr_new
        elif not _all_filled(left_llr):
            left_llr_new = _get_left_llr(up_llr)
            llr_matrix[position[0] + 1][position[1]: position[1] + half] = left_llr_new
        elif position[0] == position[2] - 1:
            pos = position[1]
            val = (0 if left_llr[0] >= 0 else 1) if pos in info_set else 0
            bit_matrix[position[0] + 1][position[1]] = val
        else:
            position = _leftdown(position)

    return llr_matrix, bit_matrix


def _scl_decode(llr_ch, information_pos, list_size, crc_length):
    n = int(np.log2(len(llr_ch)))
    n_layers = n + 1
    llr0 = np.ones((n_layers, len(llr_ch))) * np.nan
    bit0 = np.ones((n_layers, len(llr_ch))) * np.nan
    llr0[0] = llr_ch
    llr_list = [llr0]
    bit_list = [bit0]
    pm_list = [0.0]

    split_pos = list(information_pos)
    split_loc = 0
    l_now = 1

    while split_loc < len(split_pos):
        new_llr, new_bit, new_pm = [], [], []
        for i in range(l_now):
            lm, bm = _sc_stepping(
                llr_list[i].copy(), bit_list[i].copy(), information_pos, split_pos[split_loc]
            )
            prev_start = split_pos[split_loc - 1] + 1 if split_loc > 0 else 0
            seg_llr = lm[n][prev_start: split_pos[split_loc] + 1]
            seg_bit = bm[n][prev_start: split_pos[split_loc] + 1]
            pm0 = pm_list[i] + _get_pm_update(seg_llr, seg_bit)
            new_llr.append(lm)
            new_bit.append(bm)
            new_pm.append(pm0)
            bm1 = bm.copy()
            bm1[n][split_pos[split_loc]] = 1 - bm1[n][split_pos[split_loc]]
            seg_bit1 = bm1[n][prev_start: split_pos[split_loc] + 1]
            pm1 = pm_list[i] + _get_pm_update(seg_llr, seg_bit1)
            new_llr.append(lm.copy())
            new_bit.append(bm1)
            new_pm.append(pm1)

        order = np.argsort(new_pm)[:list_size]
        llr_list = [new_llr[i] for i in order]
        bit_list = [new_bit[i] for i in order]
        pm_list = [new_pm[i] for i in order]
        l_now = len(pm_list)
        split_loc += 1

    if split_pos and split_pos[-1] != len(llr_ch) - 1:
        for i in range(l_now):
            lm, bm = _sc_stepping(llr_list[i], bit_list[i], information_pos, len(llr_ch) - 1)
            llr_list[i], bit_list[i] = lm, bm

    order = np.argsort(pm_list)
    for idx in order:
        u = bit_list[idx][n].astype(int)
        if crc_length > 0:
            payload = u[information_pos]
            if crc_check(payload, crc_length):
                return u, pm_list[idx]
    best = order[0]
    return bit_list[best][n].astype(int), pm_list[best]


class SCLDecoder:
    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length

    def decode(self, llr_ch):
        info_pos = sorted(np.where(~self.frozen_bits)[0].tolist())
        if self.list_size == 1 and self.crc_length == 0:
            u = sc_decode(llr_ch, self.frozen_bits)
            return u, 0.0
        u, pm = _scl_decode(
            np.asarray(llr_ch, dtype=np.float64),
            info_pos,
            self.list_size,
            self.crc_length,
        )
        return u, pm
