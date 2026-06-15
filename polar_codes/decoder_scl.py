"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import copy

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


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, crc_length):
    reg = 0
    for bit in bits:
        reg = ((reg << 1) | int(bit)) & ((1 << crc_length) - 1)
        if reg & (1 << (crc_length - 1)):
            reg ^= poly
    for _ in range(crc_length):
        reg = (reg << 1) & ((1 << crc_length) - 1)
        if reg & (1 << (crc_length - 1)):
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
    return _crc_remainder(bits, poly, crc_length) == 0


def _get_up_loc(bit_matrix):
    N = bit_matrix[0].size
    n = int(np.log2(N))
    detect_array = bit_matrix[n]
    detect = -1
    for i in range(N):
        if detect_array[i] not in (0, 1):
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


def _get_pm_update(llr_array, bit_array):
    pm = 0.0
    for i in range(len(llr_array)):
        if np.sign(llr_array[i]) != np.sign(1 - 2 * bit_array[i]):
            pm += abs(llr_array[i])
    return pm


def _sc_stepping_decoder(llr_matrix, bit_matrix, frozen_bits, split_pos):
    N = int(bit_matrix[0].size)
    n = int(np.log2(N))
    loc = _get_up_loc(bit_matrix)
    position = [loc[0], loc[1], n, N]

    while bit_matrix[n][split_pos] not in (0, 1):
        up_llr = llr_matrix[position[0]][position[1] : position[1] + 2 ** (position[2] - position[0])]
        up_bit = bit_matrix[position[0]][position[1] : position[1] + 2 ** (position[2] - position[0])]
        left_llr = llr_matrix[position[0] + 1][position[1] : position[1] + 2 ** (position[2] - position[0] - 1)]
        left_bit = bit_matrix[position[0] + 1][position[1] : position[1] + 2 ** (position[2] - position[0] - 1)]
        right_llr = llr_matrix[position[0] + 1][
            position[1] + 2 ** (position[2] - position[0] - 1) : position[1] + 2 ** (position[2] - position[0])
        ]
        right_bit = bit_matrix[position[0] + 1][
            position[1] + 2 ** (position[2] - position[0] - 1) : position[1] + 2 ** (position[2] - position[0])
        ]

        if _all_num(up_bit):
            position = _up(position)
        else:
            if _all_num(right_bit):
                up_bit = _get_up_bit(left_bit, right_bit)
                bit_matrix[position[0]][position[1] : position[1] + 2 ** (position[2] - position[0])] = up_bit
            else:
                if _all_num(right_llr):
                    if position[0] == position[2] - 1:
                        right_bit_pos = position[1] + 1
                        right_bit = _get_right_bit(right_llr, frozen_bits, right_bit_pos)
                        bit_matrix[position[0] + 1][
                            position[1] + 2 ** (position[2] - position[0] - 1) : position[1] + 2 ** (position[2] - position[0])
                        ] = right_bit
                    else:
                        position = _rightdown(position)
                else:
                    if _all_num(left_bit):
                        right_llr = _get_right_llr(left_bit, up_llr)
                        llr_matrix[position[0] + 1][
                            position[1] + 2 ** (position[2] - position[0] - 1) : position[1] + 2 ** (position[2] - position[0])
                        ] = right_llr
                    else:
                        if not _all_num(left_llr):
                            left_llr = _get_left_llr(up_llr)
                            llr_matrix[position[0] + 1][
                                position[1] : position[1] + 2 ** (position[2] - position[0] - 1)
                            ] = left_llr
                        else:
                            if position[0] == position[2] - 1:
                                left_bit_pos = position[1]
                                left_bit = _get_left_bit(left_llr, frozen_bits, left_bit_pos)
                                bit_matrix[position[0] + 1][
                                    position[1] : position[1] + 2 ** (position[2] - position[0] - 1)
                                ] = left_bit
                            else:
                                position = _leftdown(position)

    return llr_matrix, bit_matrix


def _scl_decode(y_llr, info_indices, frozen_bits, list_size, crc_length):
    N = y_llr.size
    n = int(np.log2(N))
    info_indices = list(info_indices)

    llr_matrix = np.ones((n + 1, N), dtype=np.float64)
    llr_matrix[:] = np.nan
    bit_matrix = llr_matrix.copy()
    llr_matrix[0] = y_llr

    llr_list = [llr_matrix]
    bit_list = [bit_matrix]
    pm_list = [0.0]
    split_loc = 0
    l_now = 1

    while split_loc < len(info_indices):
        split_pos = info_indices[split_loc]
        new_llr_list = []
        new_bit_list = []
        new_pm_list = []

        for i in range(l_now):
            llr_temp = copy.deepcopy(llr_list[i])
            bit_temp = copy.deepcopy(bit_list[i])
            pm_temp = pm_list[i]

            llr_temp, bit_temp = _sc_stepping_decoder(llr_temp, bit_temp, frozen_bits, split_pos)

            start = info_indices[split_loc - 1] + 1 if split_loc > 0 else 0
            pm_slice_llr = llr_temp[n][start : split_pos + 1]
            pm_slice_bit = bit_temp[n][start : split_pos + 1]
            pm_right = pm_temp + _get_pm_update(pm_slice_llr, pm_slice_bit)

            new_llr_list.append(llr_temp)
            new_bit_list.append(bit_temp)
            new_pm_list.append(pm_right)

            llr_wrong = copy.deepcopy(llr_temp)
            bit_wrong = copy.deepcopy(bit_temp)
            bit_wrong[n][split_pos] = 1 - bit_wrong[n][split_pos]
            pm_slice_bit_wrong = bit_wrong[n][start : split_pos + 1]
            pm_wrong = pm_temp + _get_pm_update(pm_slice_llr, pm_slice_bit_wrong)

            new_llr_list.append(llr_wrong)
            new_bit_list.append(bit_wrong)
            new_pm_list.append(pm_wrong)

        order = np.argsort(new_pm_list)[:list_size]
        llr_list = [new_llr_list[i] for i in order]
        bit_list = [new_bit_list[i] for i in order]
        pm_list = [new_pm_list[i] for i in order]
        l_now = len(order)
        split_loc += 1

    if info_indices and info_indices[-1] != N - 1:
        for i in range(l_now):
            llr_list[i], bit_list[i] = _sc_stepping_decoder(
                llr_list[i], bit_list[i], frozen_bits, N - 1
            )

    order = np.argsort(pm_list)
    u_d = None
    if crc_length > 0:
        for idx in order:
            u_cand = bit_list[idx][n].astype(int)
            if crc_check(u_cand[info_indices], crc_length):
                u_d = u_cand
                break
    if u_d is None:
        u_d = bit_list[order[0]][n].astype(int)

    return u_d, pm_list[order[0]]


class SCLDecoder:
    """SCL 译码器。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.info_indices = np.where(~self.frozen_bits)[0]
        self.list_size = list_size
        self.crc_length = crc_length

    def decode(self, llr_ch):
        if self.list_size == 1 and self.crc_length == 0:
            return sc_decode(llr_ch, self.frozen_bits), 0.0
        return _scl_decode(
            np.asarray(llr_ch, dtype=np.float64),
            self.info_indices,
            self.frozen_bits,
            self.list_size,
            self.crc_length,
        )
