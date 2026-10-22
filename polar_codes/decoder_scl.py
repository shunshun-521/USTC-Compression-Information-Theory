"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
import math

from decoder_sc import (
    _sc_decode_core,
    _all_filled,
    _leftdown,
    _rightdown,
    _up,
    _get_up_bit,
    _get_right_bit,
    _get_left_bit,
    _get_right_llr,
    _get_left_llr,
    sc_decode,
)

_CRC8_POLY = 0x07
_CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, crc_length):
    reg = 0
    for b in bits:
        reg <<= 1
        reg |= int(b) & 1
        if reg & (1 << crc_length):
            reg ^= poly
    return reg & ((1 << crc_length) - 1)


def crc_encode(info_bits, crc_length=8):
    info_bits = np.asarray(info_bits, dtype=np.int8).ravel()
    poly = _CRC8_POLY if crc_length == 8 else _CRC16_POLY
    if crc_length not in (8, 16):
        raise ValueError("crc_length must be 8 or 16")
    rem = _crc_remainder(info_bits, poly, crc_length)
    crc_bits = np.array(
        [(rem >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=np.int8
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=np.int8).ravel()
    poly = _CRC8_POLY if crc_length == 8 else _CRC16_POLY
    return _crc_remainder(bits, poly, crc_length) == 0


def _get_up_loc(bit_matrix):
    N = bit_matrix.shape[1]
    n = int(np.log2(N))
    detect = -1
    for i in range(N):
        v = bit_matrix[n, i]
        if v != 0 and v != 1:
            detect = i - 1
            break
    if detect == -1:
        return [0, 0]
    if detect % 2 == 0:
        return [n - 1, detect]
    return [n - 1, detect - 1]


def _pm_update(llr_array, bit_array):
    pm = 0.0
    for llr, b in zip(llr_array, bit_array):
        if np.sign(llr) != np.sign(1 - 2 * b):
            pm += abs(llr)
    return pm


def _sc_stepping(llr_matrix, bit_matrix, information_pos, frozen_bit, split_pos):
    N = bit_matrix.shape[1]
    n = int(math.log2(N))
    info_set = set(information_pos)
    loc = _get_up_loc(bit_matrix)
    position = [loc[0], loc[1], n, N]

    while bit_matrix[n, split_pos] != 0 and bit_matrix[n, split_pos] != 1:
        span = 2 ** (position[2] - position[0])
        half = span // 2
        up_llr = llr_matrix[position[0], position[1] : position[1] + span]
        up_bit = bit_matrix[position[0], position[1] : position[1] + span]
        left_llr = llr_matrix[position[0] + 1, position[1] : position[1] + half]
        left_bit = bit_matrix[position[0] + 1, position[1] : position[1] + half]
        right_llr = llr_matrix[position[0] + 1, position[1] + half : position[1] + span]
        right_bit = bit_matrix[position[0] + 1, position[1] + half : position[1] + span]

        if _all_filled(up_bit):
            position = _up(position)
        elif _all_filled(right_bit):
            up_bit_new = _get_up_bit(left_bit, right_bit)
            bit_matrix[position[0], position[1] : position[1] + span] = up_bit_new.copy()
        elif _all_filled(right_llr):
            if position[0] == position[2] - 1:
                right_bit_pos = position[1] + 1
                bit_matrix[position[0] + 1, position[1] + half] = _get_right_bit(
                    right_llr[0], info_set, frozen_bit, right_bit_pos
                )
            else:
                position = _rightdown(position)
        elif _all_filled(left_bit):
            llr_matrix[position[0] + 1, position[1] + half : position[1] + span] = _get_right_llr(
                left_bit, up_llr
            )
        elif not _all_filled(left_llr):
            llr_matrix[position[0] + 1, position[1] : position[1] + half] = _get_left_llr(up_llr)
        else:
            if position[0] == position[2] - 1:
                left_bit_pos = position[1]
                bit_matrix[position[0] + 1, position[1]] = _get_left_bit(
                    left_llr[0], info_set, frozen_bit, left_bit_pos
                )
            else:
                position = _leftdown(position)
    return llr_matrix, bit_matrix


def _scl_decode(llr_ch, information_pos, list_size, crc_length=0):
    N = len(llr_ch)
    n = int(math.log2(N))
    frozen_bit = 0

    llr_matrix = np.full((n + 1, N), np.nan, dtype=np.float64)
    bit_matrix = np.full((n + 1, N), np.nan, dtype=np.float64)
    llr_matrix[0] = llr_ch

    llr_list = [llr_matrix.copy()]
    bit_list = [bit_matrix.copy()]
    pm_list = [0.0]

    split_pos = list(information_pos)
    split_len = len(split_pos)

    for split_loc in range(split_len):
        new_llr, new_bit, new_pm = [], [], []
        for i in range(len(llr_list)):
            lm, bm = _sc_stepping(
                llr_list[i].copy(), bit_list[i].copy(), information_pos, frozen_bit, split_pos[split_loc]
            )
            prev_idx = split_pos[split_loc - 1] + 1 if split_loc > 0 else 0
            cur_idx = split_pos[split_loc] + 1
            pm_inc = _pm_update(lm[n, prev_idx:cur_idx], bm[n, prev_idx:cur_idx])

            new_llr.append(lm)
            new_bit.append(bm)
            new_pm.append(pm_list[i] + pm_inc)

            bm_wrong = bm.copy()
            bm_wrong[n, split_pos[split_loc]] = 1 - bm_wrong[n, split_pos[split_loc]]
            pm_wrong = pm_list[i] + _pm_update(
                lm[n, prev_idx:cur_idx], bm_wrong[n, prev_idx:cur_idx]
            )
            new_llr.append(lm.copy())
            new_bit.append(bm_wrong)
            new_pm.append(pm_wrong)

        order = np.argsort(new_pm)
        keep = order[:list_size]
        llr_list = [new_llr[i] for i in keep]
        bit_list = [new_bit[i] for i in keep]
        pm_list = [new_pm[i] for i in keep]

    if split_pos and split_pos[-1] != N - 1:
        for i in range(len(llr_list)):
            lm, bm = _sc_stepping(
                llr_list[i].copy(), bit_list[i].copy(), information_pos, frozen_bit, N - 1
            )
            prev_idx = split_pos[-1] + 1
            pm_list[i] += _pm_update(lm[n, prev_idx:N], bm[n, prev_idx:N])
            llr_list[i] = lm
            bit_list[i] = bm

    order = np.argsort(pm_list)
    for idx in order:
        u_hat = bit_list[idx][n].astype(int)
        if crc_length > 0:
            if crc_check(u_hat, crc_length):
                return u_hat, pm_list[idx]
        else:
            return u_hat, pm_list[idx]

    best = order[0]
    return bit_list[best][n].astype(int), pm_list[best]


class SCLDecoder:
    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_pos = np.where(~self.frozen_bits)[0].tolist()

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        if self.list_size <= 1:
            u = sc_decode(llr_ch, self.frozen_bits)
            return u, 0.0
        return _scl_decode(llr_ch, self.info_pos, self.list_size, self.crc_length)
