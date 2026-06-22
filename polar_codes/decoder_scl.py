"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import copy
import math

import numpy as np

from decoder_sc import (
    _all_num,
    _decide_bit,
    _get_left_llr,
    _get_right_llr,
    _get_up_bit,
    _leftdown,
    _rightdown,
    _up,
    sc_decode,
)


CRC_POLYNOMIALS = {
    8: 0x07,
    16: 0x8005,
}


def _crc_division(bits, poly, crc_length):
    reg = np.zeros(crc_length, dtype=int)
    for bit in bits:
        feedback = bit ^ reg[0]
        reg[:-1] = reg[1:]
        reg[-1] = 0
        if feedback:
            poly_bits = np.array(
                [(poly >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
                dtype=int,
            )
            reg ^= poly_bits
    return reg


def crc_encode(info_bits, crc_length=8):
    info_bits = np.asarray(info_bits, dtype=int)
    poly = CRC_POLYNOMIALS[crc_length]
    remainder = _crc_division(info_bits, poly, crc_length)
    return np.concatenate([info_bits, remainder])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=int)
    poly = CRC_POLYNOMIALS[crc_length]
    remainder = _crc_division(bits, poly, crc_length)
    return np.all(remainder == 0)


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


def _pm_update(llr_array, bit_array):
    pm = 0.0
    for llr, bit in zip(llr_array, bit_array):
        hard = 0 if llr >= 0 else 1
        if int(bit) != hard:
            pm += abs(llr)
    return pm


def _sc_stepping_decoder(llr_matrix, bit_matrix, frozen_bits, split_pos):
    N = bit_matrix.shape[1]
    n = int(math.log2(N))
    loc = _get_up_loc(bit_matrix)
    position = [loc[0], loc[1], n, N]

    while bit_matrix[n, split_pos] != 0 and bit_matrix[n, split_pos] != 1:
        up_llr = llr_matrix[position[0]][position[1]:position[1] + 2 ** (position[2] - position[0])]
        up_bit = bit_matrix[position[0]][position[1]:position[1] + 2 ** (position[2] - position[0])]
        left_llr = llr_matrix[position[0] + 1][position[1]:position[1] + 2 ** (position[2] - position[0] - 1)]
        left_bit = bit_matrix[position[0] + 1][position[1]:position[1] + 2 ** (position[2] - position[0] - 1)]
        right_llr = llr_matrix[position[0] + 1][
            position[1] + 2 ** (position[2] - position[0] - 1):position[1] + 2 ** (position[2] - position[0])
        ]
        right_bit = bit_matrix[position[0] + 1][
            position[1] + 2 ** (position[2] - position[0] - 1):position[1] + 2 ** (position[2] - position[0])
        ]

        if _all_num(up_bit) == 1:
            position = _up(position)
        elif _all_num(right_bit) == 1:
            bit_matrix[position[0]][position[1]:position[1] + 2 ** (position[2] - position[0])] = _get_up_bit(
                left_bit, right_bit
            )
        elif _all_num(right_llr) == 1:
            if position[0] == position[2] - 1:
                right_bit_pos = position[1] + 1
                bit_matrix[position[0] + 1][
                    position[1] + 2 ** (position[2] - position[0] - 1):position[1] + 2 ** (position[2] - position[0])
                ] = _decide_bit(right_llr[0], frozen_bits[right_bit_pos])
            else:
                position = _rightdown(position)
        elif _all_num(left_bit) == 1:
            llr_matrix[position[0] + 1][
                position[1] + 2 ** (position[2] - position[0] - 1):position[1] + 2 ** (position[2] - position[0])
            ] = _get_right_llr(left_bit, up_llr)
        elif _all_num(left_llr) == 0:
            llr_matrix[position[0] + 1][position[1]:position[1] + 2 ** (position[2] - position[0] - 1)] = _get_left_llr(up_llr)
        else:
            if position[0] == position[2] - 1:
                left_bit_pos = position[1]
                bit_matrix[position[0] + 1][position[1]:position[1] + 2 ** (position[2] - position[0] - 1)] = _decide_bit(
                    left_llr[0], frozen_bits[left_bit_pos]
                )
            else:
                position = _leftdown(position)

    return llr_matrix, bit_matrix


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
        if self.list_size == 1 and self.crc_length == 0:
            return sc_decode(llr_ch, self.frozen_bits), 0.0

        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n = self.n
        N = self.N

        llr_matrix = np.ones((n + 1, N), dtype=np.float64)
        llr_matrix[llr_matrix == 1] = np.nan
        bit_matrix = llr_matrix.copy()
        llr_matrix[0] = llr_ch

        split_pos = list(self.info_indices)
        llr_list = [llr_matrix]
        bit_list = [bit_matrix]
        pm_list = [0.0]
        split_loc = 0
        l_now = 1

        while split_loc < len(split_pos):
            for i in range(l_now):
                llr_temp = llr_list[i]
                bit_temp = bit_list[i]
                pm_temp = pm_list[i]

                llr_new, bit_new = _sc_stepping_decoder(
                    copy.deepcopy(llr_temp), copy.deepcopy(bit_temp), self.frozen_bits, split_pos[split_loc]
                )

                prev = 0 if split_loc == 0 else split_pos[split_loc - 1] + 1
                cur = split_pos[split_loc] + 1
                pm_add = _pm_update(llr_new[n, prev:cur], bit_new[n, prev:cur])

                llr_list[i] = llr_new
                bit_list[i] = bit_new
                pm_list[i] = pm_temp + pm_add

                if not self.frozen_bits[split_pos[split_loc]]:
                    llr_list.append(llr_new.copy())
                    bit_wrong = bit_new.copy()
                    bit_wrong[n, split_pos[split_loc]] = 1 - bit_wrong[n, split_pos[split_loc]]
                    bit_list.append(bit_wrong)
                    pm_wrong = _pm_update(llr_new[n, prev:cur], bit_wrong[n, prev:cur])
                    pm_list.append(pm_temp + pm_wrong)

            if l_now > self.list_size // 2:
                keep = np.argsort(pm_list)[: self.list_size]
                pm_list = [pm_list[i] for i in keep]
                llr_list = [llr_list[i] for i in keep]
                bit_list = [bit_list[i] for i in keep]

            l_now = len(pm_list)
            split_loc += 1

        if split_pos and split_pos[-1] != N - 1:
            for i in range(l_now):
                llr_temp = llr_list[i]
                bit_temp = bit_list[i]
                pm_temp = pm_list[i]
                llr_new, bit_new = _sc_stepping_decoder(
                    copy.deepcopy(llr_temp), copy.deepcopy(bit_temp), self.frozen_bits, N - 1
                )
                prev = split_pos[-1] + 1
                pm_add = _pm_update(llr_new[n, prev:N], bit_new[n, prev:N])
                llr_list[i] = llr_new
                bit_list[i] = bit_new
                pm_list[i] = pm_temp + pm_add

        order = np.argsort(pm_list)
        if self.crc_length > 0:
            for idx in order:
                u_hat = bit_list[idx][n].astype(int)
                info_bits = u_hat[self.info_indices]
                if crc_check(info_bits, self.crc_length):
                    return u_hat, pm_list[idx]

        best = order[0]
        return bit_list[best][n].astype(int), pm_list[best]
