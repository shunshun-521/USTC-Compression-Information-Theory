"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math

import numpy as np

from decoder_sc import (
    _all_num,
    _get_left_bit,
    _get_left_llr,
    _get_right_bit,
    _get_right_llr,
    _get_up_bit,
    _leftdown,
    _rightdown,
    _up,
    sc_decode,
)
from encoder import bit_reversal_permutation


CRC_POLYNOMIALS = {
    8: 0x07,
    16: 0x8005,
}


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=int)
    poly = CRC_POLYNOMIALS[crc_length]
    reg = 0
    for bit in info_bits:
        reg <<= 1
        reg |= int(bit)
        if reg & (1 << crc_length):
            reg ^= poly
    for _ in range(crc_length):
        reg <<= 1
        if reg & (1 << crc_length):
            reg ^= poly
    crc_bits = np.array([(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。"""
    if crc_length <= 0:
        return True
    expected = crc_encode(bits[:-crc_length], crc_length)
    return np.array_equal(bits[-crc_length:], expected)


def _get_up_loc(bit_matrix):
    N = bit_matrix.shape[1]
    n = int(math.log2(N))
    detect_array = bit_matrix[n]
    detect = -1
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


def _sc_step_to_split(llr_matrix, bit_matrix, frozen_bits, split_pos):
    """SC 译码至 split_pos 位判决完成。"""
    N = bit_matrix.shape[1]
    n = int(math.log2(N))
    loc = _get_up_loc(bit_matrix)
    position = [loc[0], loc[1], n, N]

    while bit_matrix[n][split_pos] != 0 and bit_matrix[n][split_pos] != 1:
        span = 2 ** (position[2] - position[0])
        half = span // 2
        up_llr = llr_matrix[position[0]][position[1]: position[1] + span]
        up_bit = bit_matrix[position[0]][position[1]: position[1] + span]
        left_llr = llr_matrix[position[0] + 1][position[1]: position[1] + half]
        left_bit = bit_matrix[position[0] + 1][position[1]: position[1] + half]
        right_llr = llr_matrix[position[0] + 1][position[1] + half: position[1] + span]
        right_bit = bit_matrix[position[0] + 1][position[1] + half: position[1] + span]

        if _all_num(up_bit):
            position = _up(position)
        elif _all_num(right_bit):
            up_bit = _get_up_bit(left_bit, right_bit)
            bit_matrix[position[0]][position[1]: position[1] + span] = up_bit.copy()
        elif _all_num(right_llr):
            if position[0] == position[2] - 1:
                right_bit_pos = position[1] + 1
                right_bit_val = _get_right_bit(right_llr[0], frozen_bits, right_bit_pos)
                bit_matrix[position[0] + 1][position[1] + half: position[1] + span] = right_bit_val
            else:
                position = _rightdown(position)
        elif _all_num(left_bit):
            right_llr_new = _get_right_llr(left_bit, up_llr)
            llr_matrix[position[0] + 1][position[1] + half: position[1] + span] = right_llr_new
        elif not _all_num(left_llr):
            left_llr_new = _get_left_llr(up_llr)
            llr_matrix[position[0] + 1][position[1]: position[1] + half] = left_llr_new
        else:
            if position[0] == position[2] - 1:
                left_bit_pos = position[1]
                left_bit_val = _get_left_bit(left_llr[0], frozen_bits, left_bit_pos)
                bit_matrix[position[0] + 1][position[1]] = left_bit_val
            else:
                position = _leftdown(position)

    return llr_matrix, bit_matrix


def _path_metric_update(llr_slice, bit_slice):
    pm = 0.0
    for llr, bit in zip(llr_slice, bit_slice):
        hard = 0 if llr >= 0 else 1
        if hard != bit:
            pm += abs(llr)
    return pm


class SCLDecoder:
    """SCL 译码器。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.brp = bit_reversal_permutation(N)
        self.info_indices = np.where(~self.frozen_bits)[0]

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        y_llr = llr_ch[self.brp]

        if self.list_size == 1:
            u_hat = sc_decode(llr_ch, self.frozen_bits)
            return u_hat, 0.0

        split_pos = list(self.info_indices)
        if len(split_pos) == 0:
            u_hat = np.zeros(self.N, dtype=int)
            return u_hat, 0.0

        llr_list = []
        bit_list = []
        pm_list = []

        llr_matrix = np.full((self.n + 1, self.N), np.nan)
        bit_matrix = np.full((self.n + 1, self.N), np.nan)
        llr_matrix[0] = y_llr
        llr_list.append(llr_matrix.copy())
        bit_list.append(bit_matrix.copy())
        pm_list.append(0.0)

        split_loc = 0
        l_now = 1

        while split_loc < len(split_pos):
            pos = split_pos[split_loc]
            prev = split_pos[split_loc - 1] if split_loc > 0 else -1

            new_llr_list = []
            new_bit_list = []
            new_pm_list = []

            for idx in range(l_now):
                llr_m = llr_list[idx].copy()
                bit_m = bit_list[idx].copy()
                pm_base = pm_list[idx]

                llr_ok, bit_ok = _sc_step_to_split(llr_m, bit_m, self.frozen_bits, pos)
                llr_slice = llr_ok[self.n][prev + 1: pos + 1]
                bit_slice = bit_ok[self.n][prev + 1: pos + 1].astype(int)
                pm_ok = pm_base + _path_metric_update(llr_slice, bit_slice)

                new_llr_list.append(llr_ok)
                new_bit_list.append(bit_ok)
                new_pm_list.append(pm_ok)

                if not self.frozen_bits[pos]:
                    bit_wrong = bit_ok.copy()
                    bit_wrong[self.n][pos] = 1 - int(bit_ok[self.n][pos])
                    wrong_slice = bit_wrong[self.n][prev + 1: pos + 1].astype(int)
                    pm_wrong = pm_base + _path_metric_update(llr_slice, wrong_slice)
                    new_llr_list.append(llr_ok.copy())
                    new_bit_list.append(bit_wrong)
                    new_pm_list.append(pm_wrong)

            order = np.argsort(new_pm_list)[: self.list_size]
            llr_list = [new_llr_list[i] for i in order]
            bit_list = [new_bit_list[i] for i in order]
            pm_list = [new_pm_list[i] for i in order]
            l_now = len(pm_list)
            split_loc += 1

        if split_pos[-1] != self.N - 1:
            for idx in range(l_now):
                llr_m = llr_list[idx].copy()
                bit_m = bit_list[idx].copy()
                pm_base = pm_list[idx]
                prev = split_pos[-1]
                llr_ok, bit_ok = _sc_step_to_split(llr_m, bit_m, self.frozen_bits, self.N - 1)
                llr_slice = llr_ok[self.n][prev + 1: self.N]
                bit_slice = bit_ok[self.n][prev + 1: self.N].astype(int)
                pm_list[idx] = pm_base + _path_metric_update(llr_slice, bit_slice)
                llr_list[idx] = llr_ok
                bit_list[idx] = bit_ok

        order = np.argsort(pm_list)
        best_u = None
        best_pm = pm_list[order[0]]

        if self.crc_length > 0:
            for idx in order:
                u_hat = bit_list[idx][self.n].astype(int)
                info_bits = u_hat[self.info_indices]
                if crc_check(info_bits, self.crc_length):
                    return u_hat, pm_list[idx]
            best_u = bit_list[order[0]][self.n].astype(int)
        else:
            best_u = bit_list[order[0]][self.n].astype(int)

        return best_u, best_pm
