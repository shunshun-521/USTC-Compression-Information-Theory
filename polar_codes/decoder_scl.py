"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from encoder import bit_reversal_permutation
from decoder_sc import (
    sc_stepping_decoder,
    _init_matrices,
    _pm_update,
    _sc_decode_core,
    sc_decode,
)


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, crc_length):
    reg = 0
    for bit in bits:
        reg ^= int(bit) << (crc_length - 1)
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
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    return _crc_remainder(bits, poly, crc_length) == 0


def _scl_decode(y_llr, info_indices, list_size, crc_length=0):
    N = y_llr.size
    n = int(np.log2(N))
    info_indices = np.asarray(info_indices, dtype=int)
    split_pos = list(info_indices)
    frozen_val = 0

    llr_matrix, bit_matrix = _init_matrices(y_llr)
    llr_list = [llr_matrix]
    bit_list = [bit_matrix]
    pm_list = [0.0]
    split_loc = 0
    l_now = 1

    while split_loc < len(split_pos):
        new_llr, new_bit, new_pm = [], [], []
        prev_idx = split_pos[split_loc - 1] if split_loc > 0 else -1
        cur_idx = split_pos[split_loc]

        for i in range(l_now):
            llr_m, bit_m = sc_stepping_decoder(
                llr_list[i].copy(), bit_list[i].copy(), info_indices, frozen_val, cur_idx
            )
            llr_slice = llr_m[n][prev_idx + 1:cur_idx + 1]
            bit_slice = bit_m[n][prev_idx + 1:cur_idx + 1]

            pm0 = pm_list[i] + _pm_update(llr_slice, bit_slice)
            new_llr.append(llr_m)
            new_bit.append(bit_m)
            new_pm.append(pm0)

            bit_wrong = bit_m.copy()
            bit_wrong[n][cur_idx] = 1 - int(bit_m[n][cur_idx])
            pm1 = pm_list[i] + _pm_update(llr_slice, bit_wrong[n][prev_idx + 1:cur_idx + 1])
            new_llr.append(llr_m.copy())
            new_bit.append(bit_wrong)
            new_pm.append(pm1)

        order = np.argsort(new_pm)[:list_size]
        llr_list = [new_llr[i] for i in order]
        bit_list = [new_bit[i] for i in order]
        pm_list = [new_pm[i] for i in order]
        l_now = len(pm_list)
        split_loc += 1

    if split_pos and split_pos[-1] != N - 1:
        for i in range(l_now):
            llr_m, bit_m = sc_stepping_decoder(
                llr_list[i], bit_list[i], info_indices, frozen_val, N - 1
            )
            llr_list[i] = llr_m
            bit_list[i] = bit_m
            prev_idx = split_pos[-1]
            pm_list[i] += _pm_update(llr_m[n][prev_idx + 1:N], bit_m[n][prev_idx + 1:N])

    order = np.argsort(pm_list)
    if crc_length > 0:
        for idx in order:
            u_hat = bit_list[idx][n].astype(int)
            payload = u_hat[info_indices]
            if crc_check(payload, crc_length):
                return u_hat, pm_list[idx]
    best = order[0]
    return bit_list[best][n].astype(int), pm_list[best]


class SCLDecoder:
    """SCL 译码器。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        brp = bit_reversal_permutation(self.N)
        y_llr = llr_ch[brp]

        if self.list_size == 1 and self.crc_length == 0:
            u_hat = _sc_decode_core(y_llr, self.info_indices)
            return u_hat, 0.0

        return _scl_decode(y_llr, self.info_indices, self.list_size, self.crc_length)
