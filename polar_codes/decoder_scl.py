"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math

import numpy as np

from decoder_sc import (
    _pm_update_segment,
    _sc_decode_core,
    _sc_stepping_decoder,
    sc_decode,
)


def _crc_poly(crc_length):
    if crc_length == 8:
        loc = [8, 2, 1, 0]
    elif crc_length == 16:
        loc = [16, 15, 2, 0]
    else:
        raise ValueError("Unsupported CRC length")
    p = [0] * (crc_length + 1)
    for i in loc:
        p[i] = 1
    return p[::-1]


def _crc_division(info_bits, crc_length):
    p = _crc_poly(crc_length)
    work = list(map(int, info_bits)) + [0] * crc_length
    times = len(info_bits)
    for i in range(times):
        if work[i] == 1:
            for j in range(crc_length + 1):
                work[i + j] ^= p[j]
    return work[-crc_length:]


def crc_encode(info_bits, crc_length=8):
    info_bits = np.asarray(info_bits, dtype=int)
    check = _crc_division(info_bits.tolist(), crc_length)
    return np.concatenate([info_bits, np.array(check, dtype=int)])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=int)
    if len(bits) < crc_length:
        return False
    info = bits[:-crc_length]
    return np.array_equal(bits, crc_encode(info, crc_length))


class SCLDecoder:
    """SCL 译码器"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.info_indices = np.sort(np.where(self.frozen_bits == 0)[0])
        self.info_set = set(self.info_indices.tolist())
        self.frozen_val = 0
        self.list_size = list_size
        self.crc_length = crc_length

    def _init_matrices(self, llr_ch):
        llr_matrix = np.ones((self.n + 1, self.N), dtype=np.float64)
        llr_matrix[:] = np.nan
        bit_matrix = llr_matrix.copy()
        llr_matrix[0] = llr_ch.copy()
        return llr_matrix, bit_matrix

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        if self.list_size == 1:
            return sc_decode(llr_ch, self.frozen_bits), 0.0

        llr_list = [self._init_matrices(llr_ch)[0]]
        bit_list = [self._init_matrices(llr_ch)[1]]
        pm_list = [0.0]

        split_pos = self.info_indices.tolist()
        split_loc = 0
        l_now = 1

        while split_loc < len(split_pos):
            prev = split_pos[split_loc - 1] if split_loc > 0 else -1
            cur = split_pos[split_loc]
            new_llr, new_bit, new_pm = [], [], []
            for i in range(l_now):
                llr_m, bit_m = llr_list[i].copy(), bit_list[i].copy()
                pm0 = pm_list[i]
                llr_m, bit_m = _sc_stepping_decoder(
                    llr_m, bit_m, self.info_set, self.frozen_val, cur
                )
                seg_llr = llr_m[self.n][prev + 1:cur + 1]
                seg_bit = bit_m[self.n][prev + 1:cur + 1]
                pm_right = pm0 + _pm_update_segment(seg_llr, seg_bit)
                new_llr.append(llr_m)
                new_bit.append(bit_m)
                new_pm.append(pm_right)

                bit_wrong = bit_m.copy()
                bit_wrong[self.n][cur] = 1 - bit_wrong[self.n][cur]
                seg_bit_w = bit_wrong[self.n][prev + 1:cur + 1]
                pm_wrong = pm0 + _pm_update_segment(seg_llr, seg_bit_w)
                new_llr.append(llr_m.copy())
                new_bit.append(bit_wrong)
                new_pm.append(pm_wrong)

            order = np.argsort(new_pm)[: self.list_size]
            llr_list = [new_llr[i] for i in order]
            bit_list = [new_bit[i] for i in order]
            pm_list = [new_pm[i] for i in order]
            l_now = len(pm_list)
            split_loc += 1

        if split_pos and split_pos[-1] != self.N - 1:
            prev = split_pos[-1]
            for i in range(l_now):
                llr_m, bit_m = llr_list[i].copy(), bit_list[i].copy()
                llr_m, bit_m = _sc_stepping_decoder(
                    llr_m, bit_m, self.info_set, self.frozen_val, self.N - 1
                )
                seg_llr = llr_m[self.n][prev + 1:self.N]
                seg_bit = bit_m[self.n][prev + 1:self.N]
                pm_list[i] += _pm_update_segment(seg_llr, seg_bit)
                llr_list[i], bit_list[i] = llr_m, bit_m

        order = np.argsort(pm_list)
        if self.crc_length > 0:
            info_nat = self.info_indices
            for idx in order:
                u_hat = bit_list[idx][self.n].astype(int)
                if crc_check(u_hat[info_nat], self.crc_length):
                    return u_hat, pm_list[idx]

        best = order[0]
        return bit_list[best][self.n].astype(int), pm_list[best]
