"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
import math
from decoder_sc import (
    sc_decode, bit_reversal_permutation,
    _all_filled, _up, _leftdown, _rightdown,
    _get_up_bit, _get_left_llr, _get_right_llr,
    f_operation, g_operation,
)

CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, crc_length):
    reg = 0
    for bit in bits:
        reg = ((reg << 1) | int(bit)) & ((1 << crc_length) - 1)
        top = (reg >> (crc_length - 1)) & 1
        reg = (reg << 1) & ((1 << crc_length) - 1)
        if top ^ int(bit):
            reg ^= poly
    for _ in range(crc_length):
        top = (reg >> (crc_length - 1)) & 1
        reg = (reg << 1) & ((1 << crc_length) - 1)
        if top:
            reg ^= poly
    return reg


def crc_encode(info_bits, crc_length=8):
    info_bits = np.asarray(info_bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    remainder = _crc_remainder(info_bits, poly, crc_length)
    crc_bits = np.array(
        [(remainder >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=int,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    padded = np.concatenate([bits, np.zeros(crc_length, dtype=int)])
    return _crc_remainder(padded, poly, crc_length) == 0


def _get_up_loc(bit_matrix, n):
    detect_array = bit_matrix[n]
    detect = -1
    for i in range(len(detect_array)):
        if detect_array[i] != 0 and detect_array[i] != 1:
            detect = i - 1
            break
    if detect % 2 == 0:
        return [n - 1, detect]
    return [n - 1, detect - 1] if detect >= 0 else [0, 0]


def _pm_update(llr_arr, bit_arr):
    pm = 0.0
    for llr, bit in zip(llr_arr, bit_arr):
        hard = 0 if llr >= 0 else 1
        if hard != bit:
            pm += abs(llr)
    return pm


def _sc_step(llr_matrix, bit_matrix, frozen_bits, split_pos):
    """Run SC until bit split_pos is decided."""
    N = bit_matrix.shape[1]
    n = int(math.log2(N))
    frozen_bit_val = 0
    loc = _get_up_loc(bit_matrix, n)
    position = [loc[0], loc[1], n, N]

    while bit_matrix[n][split_pos] != 0 and bit_matrix[n][split_pos] != 1:
        p0, p1, p2, p3 = position
        span = 2 ** (p2 - p0)
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
            bit_matrix[p0][p1:p1 + span] = _get_up_bit(left_bit, right_bit)
        elif _all_filled(right_llr):
            if p0 == p2 - 1:
                right_bit_pos = p1 + 1
                if frozen_bits[right_bit_pos]:
                    bit_matrix[p0 + 1][p1 + half] = frozen_bit_val
                else:
                    bit_matrix[p0 + 1][p1 + half] = 0 if right_llr[0] >= 0 else 1
            else:
                position = _rightdown(position)
        elif _all_filled(left_bit):
            llr_matrix[p0 + 1][p1 + half:p1 + span] = _get_right_llr(left_bit, up_llr)
        elif not _all_filled(left_llr):
            llr_matrix[p0 + 1][p1:p1 + half] = _get_left_llr(up_llr)
        else:
            if p0 == p2 - 1:
                left_bit_pos = p1
                if frozen_bits[left_bit_pos]:
                    bit_matrix[p0 + 1][p1] = frozen_bit_val
                else:
                    bit_matrix[p0 + 1][p1] = 0 if left_llr[0] >= 0 else 1
            else:
                position = _leftdown(position)

    return llr_matrix, bit_matrix


class SCLDecoder:
    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N = self.N
        n = self.n

        if self.list_size == 1:
            return sc_decode(llr_ch, self.frozen_bits), 0.0

        br = bit_reversal_permutation(N)
        y_llr = llr_ch[br]

        llr_init = np.full((n + 1, N), np.nan, dtype=np.float64)
        bit_init = np.full((n + 1, N), np.nan, dtype=np.float64)
        llr_init[0] = y_llr

        split_pos = list(self.info_indices)
        llr_list = [llr_init.copy()]
        bit_list = [bit_init.copy()]
        pm_list = [0.0]
        split_loc = 0
        l_now = 1

        while split_loc < len(split_pos):
            for i in range(l_now):
                llr_m, bit_m = _sc_step(
                    llr_list[i].copy(), bit_list[i].copy(),
                    self.frozen_bits, split_pos[split_loc],
                )
                llr_list[i] = llr_m
                bit_list[i] = bit_m

                prev = split_pos[split_loc - 1] + 1 if split_loc > 0 else 0
                curr = split_pos[split_loc] + 1
                pm_add = _pm_update(llr_m[n][prev:curr], bit_m[n][prev:curr])
                pm_list[i] += pm_add

                llr_list.append(llr_m.copy())
                bit_wrong = bit_m.copy()
                if not self.frozen_bits[split_pos[split_loc]]:
                    bit_wrong[n][split_pos[split_loc]] = 1 - bit_wrong[n][split_pos[split_loc]]
                bit_list.append(bit_wrong)
                wrong_pm = _pm_update(llr_m[n][prev:curr], bit_wrong[n][prev:curr])
                pm_list.append(pm_list[i] - pm_add + wrong_pm)

            if l_now > self.list_size // 2:
                keep = np.argsort(pm_list)[:self.list_size]
                pm_list = [pm_list[i] for i in keep]
                llr_list = [llr_list[i] for i in keep]
                bit_list = [bit_list[i] for i in keep]
            l_now = len(pm_list)
            split_loc += 1

        if split_pos[-1] != N - 1:
            for i in range(l_now):
                llr_m, bit_m = _sc_step(
                    llr_list[i].copy(), bit_list[i].copy(),
                    self.frozen_bits, N - 1,
                )
                llr_list[i] = llr_m
                bit_list[i] = bit_m
                prev = split_pos[-1] + 1
                pm_list[i] += _pm_update(llr_m[n][prev:N], bit_m[n][prev:N])

        order = np.argsort(pm_list)
        best_u = None
        best_pm = pm_list[order[0]]

        if self.crc_length > 0:
            for idx in order:
                u_cand = bit_list[idx][n].astype(int)
                if crc_check(u_cand[self.info_indices], self.crc_length):
                    return u_cand, pm_list[idx]
            best_u = bit_list[order[0]][n].astype(int)
        else:
            best_u = bit_list[order[0]][n].astype(int)

        return best_u, best_pm
