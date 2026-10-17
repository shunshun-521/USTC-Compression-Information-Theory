"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np

from decoder_sc import _sc_decode_tree, f_operation, g_operation

CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, crc_length):
    reg = 0
    mask = (1 << crc_length) - 1
    top = 1 << (crc_length - 1)
    for b in bits:
        reg ^= int(b) << (crc_length - 1)
        if reg & top:
            reg = ((reg << 1) ^ poly) & mask
        else:
            reg = (reg << 1) & mask
    return reg


def crc_encode(info_bits, crc_length=8):
    info_bits = np.asarray(info_bits, dtype=int).ravel()
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    rem = _crc_remainder(info_bits, poly, crc_length)
    crc_bits = np.array([(rem >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=int).ravel()
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    return _crc_remainder(bits, poly, crc_length) == 0


def _pm_update(llr_slice, bit_slice):
    pm = 0.0
    for llr, b in zip(llr_slice, bit_slice):
        hard = 0 if llr >= 0 else 1
        if hard != b:
            pm += abs(llr)
    return pm


def _init_matrices(y_llr):
    N = y_llr.size
    n = int(math.log2(N))
    llr_matrix = np.ones((n + 1, N), dtype=np.float64)
    llr_matrix[:] = np.nan
    bit_matrix = np.ones((n + 1, N), dtype=np.float64)
    bit_matrix[:] = np.nan
    llr_matrix[0] = y_llr
    return llr_matrix, bit_matrix, n


def _get_up_loc(bit_matrix):
    N = int(bit_matrix[0].size)
    n = int(math.log2(N))
    detect_array = bit_matrix[n]
    detect = -1
    for i in range(N):
        if detect_array[i] not in (0, 1):
            detect = i - 1
            break
    if detect == -1:
        return [0, 0]
    if detect % 2 == 0:
        return [n - 1, detect]
    return [n - 1, detect - 1]


def _sc_step_to_split(llr_matrix, bit_matrix, info_set, frozen_val, split_pos):
    """SC 译码至 split_pos 判决完成"""
    from decoder_sc import (
        _all_filled,
        _get_left_bit,
        _get_left_llr,
        _get_right_bit,
        _get_right_llr,
        _get_up_bit,
        _leftdown,
        _rightdown,
        _up,
    )

    N = int(bit_matrix[0].size)
    n = int(math.log2(N))
    loc = _get_up_loc(bit_matrix)
    position = [loc[0], loc[1], n, N]

    while bit_matrix[n][split_pos] not in (0, 1):
        up_llr = llr_matrix[position[0]][
            position[1] : position[1] + 2 ** (position[2] - position[0])
        ]
        up_bit = bit_matrix[position[0]][
            position[1] : position[1] + 2 ** (position[2] - position[0])
        ]
        left_llr = llr_matrix[position[0] + 1][
            position[1] : position[1] + 2 ** (position[2] - position[0] - 1)
        ]
        left_bit = bit_matrix[position[0] + 1][
            position[1] : position[1] + 2 ** (position[2] - position[0] - 1)
        ]
        right_llr = llr_matrix[position[0] + 1][
            position[1]
            + 2 ** (position[2] - position[0] - 1) : position[1]
            + 2 ** (position[2] - position[0])
        ]
        right_bit = bit_matrix[position[0] + 1][
            position[1]
            + 2 ** (position[2] - position[0] - 1) : position[1]
            + 2 ** (position[2] - position[0])
        ]

        if _all_filled(up_bit):
            position = _up(position)
        elif _all_filled(right_bit):
            up_bit = _get_up_bit(left_bit, right_bit)
            bit_matrix[position[0]][
                position[1] : position[1] + 2 ** (position[2] - position[0])
            ] = up_bit.copy()
        elif _all_filled(right_llr):
            if position[0] == position[2] - 1:
                right_bit_pos = position[1] + 1
                right_bit = _get_right_bit(
                    right_llr, info_set, frozen_val, right_bit_pos
                )
                bit_matrix[position[0] + 1][
                    position[1]
                    + 2 ** (position[2] - position[0] - 1) : position[1]
                    + 2 ** (position[2] - position[0])
                ] = right_bit
            else:
                position = _rightdown(position)
        elif _all_filled(left_bit):
            right_llr = _get_right_llr(left_bit, up_llr)
            llr_matrix[position[0] + 1][
                position[1]
                + 2 ** (position[2] - position[0] - 1) : position[1]
                + 2 ** (position[2] - position[0])
            ] = right_llr
        elif not _all_filled(left_llr):
            left_llr = _get_left_llr(up_llr)
            llr_matrix[position[0] + 1][
                position[1] : position[1] + 2 ** (position[2] - position[0] - 1)
            ] = left_llr
        else:
            if position[0] == position[2] - 1:
                left_bit_pos = position[1]
                left_bit = _get_left_bit(
                    left_llr, info_set, frozen_val, left_bit_pos
                )
                bit_matrix[position[0] + 1][
                    position[1] : position[1] + 2 ** (position[2] - position[0] - 1)
                ] = left_bit
            else:
                position = _leftdown(position)

    return llr_matrix, bit_matrix


class SCLDecoder:
    """SCL 译码器（基于 SC 步进 + 路径剪枝）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_pos = np.where(~self.frozen_bits)[0]
        self.info_set = set(int(i) for i in self.info_pos)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        if self.list_size == 1:
            u_hat = _sc_decode_tree(llr_ch, self.info_pos, frozen_val=0)
            return u_hat, 0.0

        split_pos = list(self.info_pos)
        if len(split_pos) == 0:
            return np.zeros(self.N, dtype=int), 0.0

        llr0, bit0, _ = _init_matrices(llr_ch)
        llr_list = [llr0]
        bit_list = [bit0]
        pm_list = [0.0]
        split_loc = 0

        while split_loc < len(split_pos):
            new_llr = []
            new_bit = []
            new_pm = []
            for idx in range(len(llr_list)):
                lm, bm = _sc_step_to_split(
                    llr_list[idx].copy(),
                    bit_list[idx].copy(),
                    self.info_set,
                    0,
                    split_pos[split_loc],
                )
                prev_pm = pm_list[idx]
                start = split_pos[split_loc - 1] + 1 if split_loc > 0 else 0
                end = split_pos[split_loc] + 1
                llr_slice = lm[self.n][start:end]
                bit_slice = bm[self.n][start:end].astype(int)

                new_llr.append(lm)
                new_bit.append(bm)
                new_pm.append(prev_pm + _pm_update(llr_slice, bit_slice))

                bm_wrong = bm.copy()
                sp = split_pos[split_loc]
                bm_wrong[self.n][sp] = 1 - bm_wrong[self.n][sp]
                wrong_slice = bm_wrong[self.n][start:end].astype(int)
                new_llr.append(lm.copy())
                new_bit.append(bm_wrong)
                new_pm.append(prev_pm + _pm_update(llr_slice, wrong_slice))

            order = np.argsort(new_pm)
            keep = order[: self.list_size]
            llr_list = [new_llr[i] for i in keep]
            bit_list = [new_bit[i] for i in keep]
            pm_list = [new_pm[i] for i in keep]
            split_loc += 1

        if split_pos[-1] != self.N - 1:
            for idx in range(len(llr_list)):
                lm, bm = _sc_step_to_split(
                    llr_list[idx], bit_list[idx], self.info_set, 0, self.N - 1
                )
                llr_list[idx] = lm
                bit_list[idx] = bm
                start = split_pos[-1] + 1
                pm_list[idx] += _pm_update(
                    lm[self.n][start:], bm[self.n][start:].astype(int)
                )

        order = np.argsort(pm_list)
        for idx in order:
            u_hat = bit_list[idx][self.n].astype(int)
            if self.crc_length > 0:
                payload = u_hat[self.info_pos]
                if crc_check(payload, self.crc_length):
                    return u_hat, pm_list[idx]
        best = order[0]
        return bit_list[best][self.n].astype(int), pm_list[best]
