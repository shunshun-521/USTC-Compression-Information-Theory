"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from encoder import bit_reversal_permutation
from decoder_sc import (
    _all_filled, _get_up_bit, _get_left_llr, _get_right_llr,
    _up_position, _leftdown, _rightdown, f_operation, g_operation,
)

# ==================== CRC 工具 ====================

_CRC8_POLY = 0x07
_CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, crc_length):
    reg = 0
    for bit in bits:
        reg ^= (int(bit) << (crc_length - 1))
        for _ in range(8 if crc_length <= 8 else 1):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)
    return reg


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=int)
    poly = _CRC8_POLY if crc_length == 8 else _CRC16_POLY
    remainder = _crc_remainder(info_bits, poly, crc_length)
    crc_bits = np.array([(remainder >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC 是否正确。"""
    bits = np.asarray(bits, dtype=int)
    poly = _CRC8_POLY if crc_length == 8 else _CRC16_POLY
    return _crc_remainder(bits, poly, crc_length) == 0


# ==================== SCL 辅助 ====================


def _get_up_loc(bit_matrix):
    N = bit_matrix.shape[1]
    n = int(np.log2(N))
    detect_array = bit_matrix[n]
    detect = -1
    for i in range(N):
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
    for i in range(len(llr_array)):
        hard = 0 if llr_array[i] >= 0 else 1
        if int(bit_array[i]) != hard:
            pm += abs(llr_array[i])
    return pm


def _sc_step(llr_matrix, bit_matrix, info_pos, frozen_bit, split_pos):
    """译码至信息位 split_pos 判决完成。"""
    N = bit_matrix.shape[1]
    n = int(np.log2(N))
    info_set = set(info_pos)
    loc = _get_up_loc(bit_matrix)
    position = [loc[0], loc[1], n, N]

    while not (bit_matrix[n][split_pos] == 0 or bit_matrix[n][split_pos] == 1):
        up_llr = llr_matrix[position[0]][position[1]:position[1] + 2 ** (position[2] - position[0])]
        up_bit = bit_matrix[position[0]][position[1]:position[1] + 2 ** (position[2] - position[0])]
        left_llr = llr_matrix[position[0] + 1][position[1]:position[1] + 2 ** (position[2] - position[0] - 1)]
        left_bit = bit_matrix[position[0] + 1][position[1]:position[1] + 2 ** (position[2] - position[0] - 1)]
        right_llr = llr_matrix[position[0] + 1][position[1] + 2 ** (position[2] - position[0] - 1):position[1] + 2 ** (position[2] - position[0])]
        right_bit = bit_matrix[position[0] + 1][position[1] + 2 ** (position[2] - position[0] - 1):position[1] + 2 ** (position[2] - position[0])]

        if _all_filled(up_bit):
            position = _up_position(position)
        elif _all_filled(right_bit):
            up_bit_new = _get_up_bit(left_bit.astype(int), right_bit.astype(int))
            bit_matrix[position[0]][position[1]:position[1] + 2 ** (position[2] - position[0])] = up_bit_new
        elif _all_filled(right_llr):
            if position[0] == position[2] - 1:
                right_bit_pos = position[1] + 1
                if right_bit_pos in info_set:
                    right_bit_val = 0 if right_llr[0] > 0 else 1
                else:
                    right_bit_val = frozen_bit
                bit_matrix[position[0] + 1][position[1] + 2 ** (position[2] - position[0] - 1):position[1] + 2 ** (position[2] - position[0])] = right_bit_val
            else:
                position = _rightdown(position)
        elif _all_filled(left_bit):
            right_llr_new = _get_right_llr(left_bit.astype(int), up_llr)
            llr_matrix[position[0] + 1][position[1] + 2 ** (position[2] - position[0] - 1):position[1] + 2 ** (position[2] - position[0])] = right_llr_new
        elif not _all_filled(left_llr):
            left_llr_new = _get_left_llr(up_llr)
            llr_matrix[position[0] + 1][position[1]:position[1] + 2 ** (position[2] - position[0] - 1)] = left_llr_new
        else:
            if position[0] == position[2] - 1:
                left_bit_pos = position[1]
                if left_bit_pos in info_set:
                    left_bit_val = 0 if left_llr[0] >= 0 else 1
                else:
                    left_bit_val = frozen_bit
                bit_matrix[position[0] + 1][position[1]:position[1] + 2 ** (position[2] - position[0] - 1)] = left_bit_val
            else:
                position = _leftdown(position)

    return llr_matrix, bit_matrix


def _scl_decode_core(y_llr, info_pos, frozen_bit, list_size):
    N = len(y_llr)
    n = int(np.log2(N))
    split_pos = sorted(info_pos)
    llr_matrix = np.full((n + 1, N), np.nan, dtype=np.float64)
    bit_matrix = np.full((n + 1, N), np.nan, dtype=np.float64)
    llr_matrix[0] = y_llr

    llr_list = [llr_matrix.copy()]
    bit_list = [bit_matrix.copy()]
    pm_list = [0.0]
    split_loc = 0

    while split_loc < len(split_pos):
        new_llr, new_bit, new_pm = [], [], []
        for idx in range(len(llr_list)):
            lm, bm, pm = llr_list[idx].copy(), bit_list[idx].copy(), pm_list[idx]
            lm, bm = _sc_step(lm, bm, split_pos, frozen_bit, split_pos[split_loc])
            seg_start = split_pos[split_loc - 1] + 1 if split_loc > 0 else 0
            seg_end = split_pos[split_loc] + 1
            pm_add = _pm_update(lm[n][seg_start:seg_end], bm[n][seg_start:seg_end])

            bm_flip = bm.copy()
            bm_flip[n][split_pos[split_loc]] = 1 - bm[n][split_pos[split_loc]]
            pm_flip_add = _pm_update(lm[n][seg_start:seg_end], bm_flip[n][seg_start:seg_end])

            new_llr.extend([lm, lm.copy()])
            new_bit.extend([bm, bm_flip])
            new_pm.extend([pm + pm_add, pm + pm_flip_add])

        order = np.argsort(new_pm)[:list_size]
        llr_list = [new_llr[i] for i in order]
        bit_list = [new_bit[i] for i in order]
        pm_list = [new_pm[i] for i in order]
        split_loc += 1

    if split_pos[-1] != N - 1:
        for idx in range(len(llr_list)):
            lm, bm, pm = llr_list[idx].copy(), bit_list[idx].copy(), pm_list[idx]
            lm, bm = _sc_step(lm, bm, split_pos, frozen_bit, N - 1)
            seg_start = split_pos[-1] + 1
            pm_list[idx] = pm + _pm_update(lm[n][seg_start:N], bm[n][seg_start:N])
            llr_list[idx], bit_list[idx] = lm, bm

    best = int(np.argmin(pm_list))
    return bit_list[best][n].astype(int), pm_list[best]


# ==================== SCL 译码器 ====================


class SCLDecoder:
    """SCL 译码器。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_pos = np.where(~self.frozen_bits)[0].tolist()

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        br = bit_reversal_permutation(self.N)
        y_llr = llr_ch[br]

        if self.list_size == 1:
            from decoder_sc import _sc_decode_core
            u_hat = _sc_decode_core(y_llr, self.frozen_bits)
            return u_hat, 0.0

        u_hat, pm = _scl_decode_core(y_llr, self.info_pos, 0, self.list_size)

        if self.crc_length > 0:
            info_bits = u_hat[self.info_pos]
            if crc_check(info_bits, self.crc_length):
                return u_hat, pm
            # CRC 未通过时仍返回最小 PM 路径
        return u_hat, pm
