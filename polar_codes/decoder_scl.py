"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np

from decoder_sc import (
    _all_computed,
    _get_left_llr,
    _get_right_llr,
    g_operation,
)


def _get_up_loc(bit_matrix, n):
    detect_array = bit_matrix[n]
    detect = -1
    for i in range(len(detect_array)):
        if bit_matrix[n, i] != 0 and bit_matrix[n, i] != 1:
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


def sc_stepping_decoder(llr_matrix, bit_matrix, frozen_bits, split_pos):
    """SC 译码至 split_pos 位判决完成。"""
    N = bit_matrix.shape[1]
    n = int(math.log2(N))
    loc = _get_up_loc(bit_matrix, n)
    position = [loc[0], loc[1], n, N]

    def up(pos):
        p0 = pos[0] - 1
        p1 = int(np.floor(pos[1] / (2 ** (pos[2] - pos[0] + 1))) * (2 ** (pos[2] - pos[0] + 1)))
        return [p0, p1, pos[2], pos[3]]

    def leftdown(pos):
        return [pos[0] + 1, pos[1], pos[2], pos[3]]

    def rightdown(pos):
        return [pos[0] + 1, pos[1] + 2 ** (pos[2] - 1 - pos[0]), pos[2], pos[3]]

    while bit_matrix[n, split_pos] != 0 and bit_matrix[n, split_pos] != 1:
        span = 2 ** (position[2] - position[0])
        half = span // 2
        start = position[1]

        up_llr = llr_matrix[position[0], start:start + span]
        up_bit = bit_matrix[position[0], start:start + span]
        left_llr = llr_matrix[position[0] + 1, start:start + half]
        left_bit = bit_matrix[position[0] + 1, start:start + half]
        right_llr = llr_matrix[position[0] + 1, start + half:start + span]
        right_bit = bit_matrix[position[0] + 1, start + half:start + span]

        if _all_computed(up_bit):
            position = up(position)
            continue

        if _all_computed(right_bit):
            combined = np.zeros(span, dtype=int)
            combined[:half] = (left_bit.astype(int) + right_bit.astype(int)) % 2
            combined[half:] = right_bit.astype(int)
            bit_matrix[position[0], start:start + span] = combined
            continue

        if _all_computed(right_llr):
            if position[0] == position[2] - 1:
                right_pos = start + half
                if frozen_bits[right_pos]:
                    val = 0
                else:
                    val = 0 if right_llr[0] >= 0 else 1
                bit_matrix[position[0] + 1, right_pos] = val
            else:
                position = rightdown(position)
            continue

        if _all_computed(left_bit):
            new_right = _get_right_llr(left_bit, up_llr)
            llr_matrix[position[0] + 1, start + half:start + span] = new_right
            continue

        if not _all_computed(left_llr):
            new_left = _get_left_llr(up_llr)
            llr_matrix[position[0] + 1, start:start + half] = new_left
            continue

        if position[0] == position[2] - 1:
            left_pos = start
            if frozen_bits[left_pos]:
                val = 0
            else:
                val = 0 if left_llr[0] >= 0 else 1
            bit_matrix[position[0] + 1, left_pos] = val
        else:
            position = leftdown(position)

    return llr_matrix, bit_matrix


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, crc_length):
    reg = 0
    for bit in bits:
        reg ^= (int(bit) << (crc_length - 1))
        for _ in range(crc_length):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)
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
    info = bits[:-crc_length]
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    remainder = _crc_remainder(info, poly, crc_length)
    received = 0
    for i in range(crc_length):
        received = (received << 1) | int(bits[-crc_length + i])
    return remainder == received


def _pm_update(llr_slice, bit_slice):
    pm = 0.0
    for llr_val, bit_val in zip(llr_slice, bit_slice):
        hard = 0 if llr_val >= 0 else 1
        if hard != bit_val:
            pm += abs(llr_val)
    return pm


class SCLDecoder:
    """SCL 译码器。"""

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
        info_pos = list(self.info_indices)

        llr_matrix = np.full((n + 1, N), np.nan, dtype=np.float64)
        bit_matrix = np.full((n + 1, N), np.nan, dtype=np.float64)
        llr_matrix[0] = llr_ch

        llr_list = [llr_matrix.copy()]
        bit_list = [bit_matrix.copy()]
        pm_list = [0.0]
        l_now = 1

        split_pos = info_pos
        split_loc = 0

        while split_loc < len(split_pos):
            for i in range(l_now):
                lm, bm = sc_stepping_decoder(
                    llr_list[i].copy(),
                    bit_list[i].copy(),
                    self.frozen_bits,
                    split_pos[split_loc],
                )
                llr_list[i] = lm
                bit_list[i] = bm

                start = split_pos[split_loc - 1] + 1 if split_loc > 0 else 0
                end = split_pos[split_loc] + 1
                pm_add = _pm_update(lm[n, start:end], bm[n, start:end])
                pm_list[i] += pm_add

                llr_list.append(lm.copy())
                bm_wrong = bm.copy()
                bm_wrong[n, split_pos[split_loc]] = 1 - bm_wrong[n, split_pos[split_loc]]
                bit_list.append(bm_wrong)
                pm_wrong = _pm_update(lm[n, start:end], bm_wrong[n, start:end])
                pm_list.append(pm_list[i] - pm_add + pm_wrong)

            if l_now > self.list_size // 2:
                order = np.argsort(pm_list)
                keep = order[:self.list_size]
                pm_list = [pm_list[i] for i in keep]
                llr_list = [llr_list[i] for i in keep]
                bit_list = [bit_list[i] for i in keep]
                l_now = len(pm_list)
            else:
                l_now = len(pm_list)

            split_loc += 1

        if split_pos[-1] != N - 1:
            for i in range(l_now):
                lm, bm = sc_stepping_decoder(
                    llr_list[i].copy(),
                    bit_list[i].copy(),
                    self.frozen_bits,
                    N - 1,
                )
                llr_list[i] = lm
                bit_list[i] = bm
                start = split_pos[-1] + 1
                pm_list[i] += _pm_update(lm[n, start:N], bm[n, start:N])

        order = np.argsort(pm_list)
        best_pm = pm_list[order[0]]
        best_u = bit_list[order[0]][n].astype(int)

        if self.crc_length > 0:
            for idx in order:
                u_candidate = bit_list[idx][n].astype(int)
                info_bits = u_candidate[self.info_indices]
                if crc_check(info_bits, self.crc_length):
                    return u_candidate, pm_list[idx]

        return best_u, best_pm
