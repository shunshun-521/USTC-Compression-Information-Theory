"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np
from encoder import bit_reversal_permutation
from decoder_sc import (
    _prepare_llr,
    _frozen_mask_to_info_pos,
    _sc_decode_core,
    _all_filled,
    _get_up_bit,
    _get_left_bit,
    _get_right_bit,
    _get_left_llr,
    _get_right_llr,
    _leftdown,
    _rightdown,
    _up,
)

_CRC_POLY_LOC = {
    8: [8, 2, 1, 0],
    16: [16, 15, 2, 0],
}


def _crc_poly(crc_length):
    loc = _CRC_POLY_LOC[crc_length]
    p = [0] * (crc_length + 1)
    for i in loc:
        p[i] = 1
    return p[::-1]


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=int)
    info = info_bits.tolist()
    p = _crc_poly(crc_length)
    times = len(info)
    work = info + [0] * crc_length
    for i in range(times):
        if work[i] == 1:
            for j in range(crc_length + 1):
                work[j + i] ^= p[j]
    check_code = work[-crc_length:]
    return np.array(info + check_code, dtype=int)


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。"""
    bits = np.asarray(bits, dtype=int)
    if len(bits) < crc_length:
        return False
    encoded = crc_encode(bits[:-crc_length], crc_length)
    return np.array_equal(encoded, bits)


def _get_up_loc(bit_matrix):
    N = bit_matrix[0].size
    n = int(math.log2(N))
    detect_array = bit_matrix[n]
    detect = N - 1
    for i in range(N):
        if np.isnan(detect_array[i]):
            detect = i - 1
            break
    if detect % 2 == 0:
        loc_row, loc_col = n - 1, detect
    else:
        loc_row, loc_col = n - 1, detect - 1
    if detect == -1:
        loc_row, loc_col = 0, 0
    return [loc_row, loc_col]


def _sc_step_to(llr_matrix, bit_matrix, information_pos, frozen_bit, split_pos):
    """推进 SC 状态机直到 bit_matrix[n, split_pos] 被判决。"""
    N = int(bit_matrix[0].size)
    n = int(math.log2(N))
    loc = _get_up_loc(bit_matrix)
    position = [loc[0], loc[1], n, N]

    while np.isnan(bit_matrix[n, split_pos]):
        span = 2 ** (position[2] - position[0])
        up_llr = llr_matrix[position[0]][position[1] : position[1] + span]
        up_bit = bit_matrix[position[0]][position[1] : position[1] + span]
        half = span // 2
        left_llr = llr_matrix[position[0] + 1][position[1] : position[1] + half]
        left_bit = bit_matrix[position[0] + 1][position[1] : position[1] + half]
        right_llr = llr_matrix[position[0] + 1][position[1] + half : position[1] + span]
        right_bit = bit_matrix[position[0] + 1][position[1] + half : position[1] + span]

        if _all_filled(up_bit):
            position = _up(position)
        elif _all_filled(right_bit):
            up_bit_val = _get_up_bit(left_bit, right_bit)
            bit_matrix[position[0]][position[1] : position[1] + span] = up_bit_val
        elif _all_filled(right_llr):
            if position[0] == position[2] - 1:
                right_bit_pos = position[1] + 1
                right_bit_val = _get_right_bit(
                    right_llr, information_pos, frozen_bit, right_bit_pos
                )
                bit_matrix[position[0] + 1][position[1] + half : position[1] + span] = right_bit_val
            else:
                position = _rightdown(position)
        elif _all_filled(left_bit):
            right_llr = _get_right_llr(left_bit, up_llr)
            llr_matrix[position[0] + 1][position[1] + half : position[1] + span] = right_llr
        elif not _all_filled(left_llr):
            left_llr = _get_left_llr(up_llr)
            llr_matrix[position[0] + 1][position[1] : position[1] + half] = left_llr
        else:
            if position[0] == position[2] - 1:
                left_bit_pos = position[1]
                left_bit_val = _get_left_bit(
                    left_llr, information_pos, frozen_bit, left_bit_pos
                )
                bit_matrix[position[0] + 1][position[1] : position[1] + half] = left_bit_val
            else:
                position = _leftdown(position)

    return llr_matrix, bit_matrix


def _pm_update(llr_array, bit_array):
    """硬件友好路径度量。"""
    pm = 0.0
    for llr, bit in zip(llr_array, bit_array):
        hard = 0 if llr >= 0 else 1
        if hard != bit:
            pm += abs(llr)
    return pm


def _scl_decode_core(y_llr, information_pos, frozen_bit, list_size, crc_length):
    N = y_llr.size
    n = int(math.log2(N))
    split_pos = list(information_pos)

    llr_list, bit_list, pm_list = [], [], []
    llr0 = np.full((n + 1, N), np.nan, dtype=np.float64)
    bit0 = np.full((n + 1, N), np.nan, dtype=np.float64)
    llr0[0] = y_llr
    llr_list.append(llr0)
    bit_list.append(bit0)
    pm_list.append(0.0)

    split_loc = 0
    prev_pos = -1

    while split_loc < len(split_pos):
        pos = split_pos[split_loc]
        new_llr, new_bit, new_pm = [], [], []

        for idx in range(len(llr_list)):
            lm, bm = llr_list[idx].copy(), bit_list[idx].copy()
            pm = pm_list[idx]
            lm, bm = _sc_step_to(lm, bm, information_pos, frozen_bit, pos)

            seg_llr = lm[n][prev_pos + 1 : pos + 1]
            seg_bit = bm[n][prev_pos + 1 : pos + 1]

            pm0 = pm + _pm_update(seg_llr, seg_bit)
            new_llr.append(lm)
            new_bit.append(bm)
            new_pm.append(pm0)

            bm1 = bm.copy()
            bm1[n, pos] = 1 - bm1[n, pos]
            seg_bit1 = bm1[n][prev_pos + 1 : pos + 1]
            pm1 = pm + _pm_update(seg_llr, seg_bit1)
            new_llr.append(lm.copy())
            new_bit.append(bm1)
            new_pm.append(pm1)

        order = np.argsort(new_pm)
        keep = order[:list_size]
        llr_list = [new_llr[i] for i in keep]
        bit_list = [new_bit[i] for i in keep]
        pm_list = [new_pm[i] for i in keep]

        split_loc += 1
        prev_pos = pos

    if split_pos and split_pos[-1] != N - 1:
        for idx in range(len(llr_list)):
            lm, bm = llr_list[idx], bit_list[idx]
            lm, bm = _sc_step_to(lm, bm, information_pos, frozen_bit, N - 1)
            llr_list[idx], bit_list[idx] = lm, bm

    order = np.argsort(pm_list)
    if crc_length > 0:
        for idx in order:
            u_hat = bit_list[idx][n].astype(int)
            info_bits = u_hat[information_pos]
            if crc_check(info_bits, crc_length):
                return u_hat, pm_list[idx]

    best = order[0]
    u_hat = bit_list[best][n].astype(int)
    return u_hat, pm_list[best]


class SCLDecoder:
    """SCL 译码器。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.frozen_bits = frozen_bits
        self.information_pos = _frozen_mask_to_info_pos(frozen_bits)
        self.list_size = list_size
        self.crc_length = crc_length

    def decode(self, llr_ch):
        y_llr = _prepare_llr(llr_ch)
        u_hat, pm = _scl_decode_core(
            y_llr,
            self.information_pos,
            frozen_bit=0,
            list_size=self.list_size,
            crc_length=self.crc_length,
        )
        return u_hat, pm
