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
    _get_up_bit,
    _leftdown,
    _rightdown,
    _up,
    sc_decode,
)


CRC_POLYNOMIALS = {8: 0x07, 16: 0x8005}


def crc_encode(info_bits, crc_length=8):
    info_bits = np.asarray(info_bits, dtype=np.int32)
    poly = CRC_POLYNOMIALS[crc_length]
    reg = 0
    for b in info_bits:
        reg ^= int(b) << (crc_length - 1)
        for _ in range(1):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)
    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=np.int32,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=np.int32)
    poly = CRC_POLYNOMIALS[crc_length]
    reg = 0
    for b in bits:
        reg ^= int(b) << (crc_length - 1)
        for _ in range(1):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)
    return reg == 0


def _get_up_loc(bit_row):
    for i, v in enumerate(bit_row):
        if np.isnan(v):
            detect = i - 1
            break
    else:
        detect = len(bit_row) - 1
    if detect < 0:
        return 0, 0
    loc_row = int(math.log2(len(bit_row))) - 1
    loc_col = detect if detect % 2 == 0 else detect - 1
    return loc_row, loc_col


def _sc_step_to(llr_matrix, bit_matrix, info_positions, split_pos):
    """步进 SC 直到 split_pos 完成判决"""
    N = bit_matrix.shape[1]
    n = int(math.log2(N))
    loc_row, loc_col = _get_up_loc(bit_matrix[n])
    position = [loc_row, loc_col, n, N]

    while np.isnan(bit_matrix[n][split_pos]):
        span = 2 ** (position[2] - position[0])
        up_llr = llr_matrix[position[0]][position[1] : position[1] + span]
        up_bit = bit_matrix[position[0]][position[1] : position[1] + span]
        half = span // 2
        left_llr = llr_matrix[position[0] + 1][position[1] : position[1] + half]
        left_bit = bit_matrix[position[0] + 1][position[1] : position[1] + half]
        right_llr = llr_matrix[position[0] + 1][position[1] + half : position[1] + span]
        right_bit = bit_matrix[position[0] + 1][position[1] + half : position[1] + span]

        if _all_computed(up_bit):
            position = _up(position)
        elif _all_computed(right_bit):
            up_bit_new = _get_up_bit(left_bit, right_bit)
            bit_matrix[position[0]][position[1] : position[1] + span] = up_bit_new[0]
        elif _all_computed(right_llr):
            if position[0] == position[2] - 1:
                right_bit_pos = position[1] + 1
                if right_bit_pos in info_positions:
                    val = 0 if right_llr[0] > 0 else 1
                else:
                    val = 0
                bit_matrix[position[0] + 1][position[1] + half] = val
            else:
                position = _rightdown(position)
        elif _all_computed(left_bit):
            right_llr_new = _get_right_llr(left_bit, up_llr)
            llr_matrix[position[0] + 1][position[1] + half : position[1] + span] = right_llr_new
        elif not _all_computed(left_llr):
            left_llr_new = _get_left_llr(up_llr)
            llr_matrix[position[0] + 1][position[1] : position[1] + half] = left_llr_new
        else:
            if position[0] == position[2] - 1:
                left_bit_pos = position[1]
                if left_bit_pos in info_positions:
                    val = 0 if left_llr[0] >= 0 else 1
                else:
                    val = 0
                bit_matrix[position[0] + 1][position[1]] = val
            else:
                position = _leftdown(position)

    return llr_matrix, bit_matrix


def _hf_pm(llr, bit):
    if (llr >= 0 and bit == 1) or (llr < 0 and bit == 0):
        return abs(llr)
    return 0.0


class SCLDecoder:
    """SCL 译码器（参考 PolarCodesPython 步进分裂）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_positions = sorted(np.where(~self.frozen_bits)[0].tolist())

    def decode(self, llr_ch):
        if self.list_size == 1 and self.crc_length == 0:
            return sc_decode(llr_ch, self.frozen_bits), 0.0

        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N, n = self.N, self.n
        info_set = set(self.info_positions)

        def new_mats():
            lm = np.full((n + 1, N), np.nan)
            bm = np.full((n + 1, N), np.nan)
            lm[0] = llr_ch.copy()
            return lm, bm

        llr_list = [new_mats()[0]]
        bit_list = [new_mats()[1]]
        pm_list = [0.0]

        for split_pos in self.info_positions:
            new_llr, new_bit, new_pm = [], [], []
            for i in range(len(llr_list)):
                lm, bm = _sc_step_to(
                    llr_list[i].copy(), bit_list[i].copy(), info_set, split_pos
                )
                llr_at_pos = lm[n - 1][split_pos] if not np.isnan(lm[n - 1][split_pos]) else lm[0][split_pos]
                for bit in (0, 1):
                    lm2, bm2 = lm.copy(), bm.copy()
                    bm2[n][split_pos] = bit
                    pm = pm_list[i] + _hf_pm(llr_at_pos, bit)
                    new_llr.append(lm2)
                    new_bit.append(bm2)
                    new_pm.append(pm)

            order = np.argsort(new_pm)[: self.list_size]
            llr_list = [new_llr[i] for i in order]
            bit_list = [new_bit[i] for i in order]
            pm_list = [new_pm[i] for i in order]

        best = int(np.argmin(pm_list))
        u_hat = bit_list[best][n].astype(np.int32)
        u_hat[np.isnan(u_hat)] = 0

        if self.crc_length > 0:
            info_bits = u_hat[self.info_positions]
            if not crc_check(info_bits, self.crc_length):
                for i in np.argsort(pm_list):
                    ib = bit_list[i][n][self.info_positions].astype(int)
                    if crc_check(ib, self.crc_length):
                        u_hat = bit_list[i][n].astype(np.int32)
                        pm_list[best] = pm_list[i]
                        break

        return u_hat, pm_list[best]
