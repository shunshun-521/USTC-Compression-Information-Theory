"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np

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
    sc_decode,
)


CRC_POLYS = {
    8: 0x07,
    16: 0x8005,
}


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=np.int8)
    poly = CRC_POLYS[crc_length]
    reg = 0
    for bit in info_bits:
        reg ^= int(bit) << (crc_length - 1)
        for _ in range(crc_length):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)
    crc_bits = np.array(
        [(reg >> i) & 1 for i in range(crc_length - 1, -1, -1)], dtype=np.int8
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC"""
    if crc_length == 0:
        return True
    expected = crc_encode(bits[:-crc_length], crc_length)
    return np.array_equal(bits[-crc_length:], expected[-crc_length:])


def _get_up_loc(bit_matrix):
    n = int(math.log2(bit_matrix.shape[1]))
    detect_array = bit_matrix[n]
    detect = -1
    for i in range(bit_matrix.shape[1]):
        if detect_array[i] not in (0, 1):
            detect = i - 1
            break
    if detect == -1:
        return [0, 0]
    if detect % 2 == 0:
        return [n - 1, detect]
    return [n - 1, detect - 1]


def _pm_update(llr_array, bit_array):
    pm = 0.0
    for llr, bit in zip(llr_array, bit_array):
        hard = 0 if llr >= 0 else 1
        if hard != bit:
            pm += abs(llr)
    return pm


def _sc_stepping(llr_matrix, bit_matrix, info_set, frozen_val, split_pos):
    N = bit_matrix.shape[1]
    n = int(math.log2(N))
    loc = _get_up_loc(bit_matrix)
    position = [loc[0], loc[1], n, N]

    while bit_matrix[n][split_pos] not in (0, 1):
        span = 2 ** (position[2] - position[0])
        half = span // 2
        up_llr = llr_matrix[position[0]][position[1] : position[1] + span]
        up_bit = bit_matrix[position[0]][position[1] : position[1] + span]
        left_llr = llr_matrix[position[0] + 1][position[1] : position[1] + half]
        left_bit = bit_matrix[position[0] + 1][position[1] : position[1] + half]
        right_llr = llr_matrix[position[0] + 1][position[1] + half : position[1] + span]
        right_bit = bit_matrix[position[0] + 1][position[1] + half : position[1] + span]

        if _all_filled(up_bit):
            position = _up(position)
        elif _all_filled(right_bit):
            up_bit = _get_up_bit(left_bit, right_bit)
            bit_matrix[position[0]][position[1] : position[1] + span] = up_bit
        elif _all_filled(right_llr):
            if position[0] == position[2] - 1:
                right_bit_pos = position[1] + 1
                bit_matrix[position[0] + 1][position[1] + half] = _get_right_bit(
                    right_llr[0], info_set, frozen_val, right_bit_pos
                )
            else:
                position = _rightdown(position)
        elif _all_filled(left_bit):
            right_llr = _get_right_llr(left_bit, up_llr)
            llr_matrix[position[0] + 1][position[1] + half : position[1] + span] = right_llr
        elif not _all_filled(left_llr):
            left_llr = _get_left_llr(up_llr)
            llr_matrix[position[0] + 1][position[1] : position[1] + half] = left_llr
        elif position[0] == position[2] - 1:
            left_bit_pos = position[1]
            bit_matrix[position[0] + 1][position[1]] = _get_left_bit(
                left_llr[0], info_set, frozen_val, left_bit_pos
            )
        else:
            position = _leftdown(position)

    return llr_matrix, bit_matrix


class SCLDecoder:
    """SCL 译码器"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.info_positions = sorted(np.where(self.frozen_bits == 0)[0].tolist())
        self.frozen_val = 0
        self.list_size = list_size
        self.crc_length = crc_length

    def decode(self, llr_ch):
        if self.list_size == 1:
            u_hat = sc_decode(llr_ch, self.frozen_bits)
            return u_hat, 0.0

        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n = self.n
        N = self.N
        info_set = set(self.info_positions)

        llr_matrix = np.full((n + 1, N), np.nan)
        bit_matrix = np.full((n + 1, N), np.nan)
        llr_matrix[0] = llr_ch

        llr_list = [llr_matrix.copy()]
        bit_list = [bit_matrix.copy()]
        pm_list = [0.0]

        split_positions = [-1] + self.info_positions
        for split_loc in range(1, len(split_positions)):
            split_pos = split_positions[split_loc]
            prev_pos = split_positions[split_loc - 1]
            new_llr_list = []
            new_bit_list = []
            new_pm_list = []

            for llr_m, bit_m, pm in zip(llr_list, bit_list, pm_list):
                llr_m, bit_m = _sc_stepping(
                    llr_m.copy(), bit_m.copy(), info_set, self.frozen_val, split_pos
                )
                llr_slice = llr_m[n][prev_pos + 1 : split_pos + 1]
                bit_slice = bit_m[n][prev_pos + 1 : split_pos + 1]

                new_llr_list.append(llr_m)
                new_bit_list.append(bit_m)
                new_pm_list.append(pm + _pm_update(llr_slice, bit_slice))

                bit_wrong = bit_m.copy()
                bit_wrong[n][split_pos] = 1 - bit_wrong[n][split_pos]
                wrong_slice = bit_wrong[n][prev_pos + 1 : split_pos + 1]
                new_llr_list.append(llr_m.copy())
                new_bit_list.append(bit_wrong)
                new_pm_list.append(pm + _pm_update(llr_slice, wrong_slice))

            order = np.argsort(new_pm_list)[: self.list_size]
            llr_list = [new_llr_list[i] for i in order]
            bit_list = [new_bit_list[i] for i in order]
            pm_list = [new_pm_list[i] for i in order]

        if split_positions[-1] != N - 1:
            for i in range(len(llr_list)):
                llr_list[i], bit_list[i] = _sc_stepping(
                    llr_list[i], bit_list[i], info_set, self.frozen_val, N - 1
                )
                prev_pos = split_positions[-1]
                pm_list[i] += _pm_update(
                    llr_list[i][n][prev_pos + 1 : N], bit_list[i][n][prev_pos + 1 : N]
                )

        order = np.argsort(pm_list)
        if self.crc_length > 0:
            info_positions = self.info_positions
            for idx in order:
                u_hat = bit_list[idx][n].astype(int)
                info_bits = u_hat[info_positions]
                if crc_check(info_bits, self.crc_length):
                    return u_hat, pm_list[idx]
            return bit_list[order[0]][n].astype(int), pm_list[order[0]]

        best = order[0]
        return bit_list[best][n].astype(int), pm_list[best]
