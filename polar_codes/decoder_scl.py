"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np

from decoder_sc import (
    sc_decode,
    _all_ready,
    _decide_bit,
    _get_left_llr,
    _get_right_llr,
    _get_up_bit,
    _leftdown,
    _rightdown,
    _up,
)


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_remainder(bits, crc_length):
    if crc_length == 8:
        poly = CRC8_POLY
    elif crc_length == 16:
        poly = CRC16_POLY
    else:
        raise ValueError(f'Unsupported CRC length: {crc_length}')
    reg = 0
    for bit in bits:
        reg ^= int(bit) << (crc_length - 1)
        for _ in range(8):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)
    return reg


def crc_encode(info_bits, crc_length=8):
    info_bits = np.asarray(info_bits, dtype=int)
    remainder = _crc_remainder(info_bits, crc_length)
    crc_bits = np.array(
        [(remainder >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=int,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=int)
    return np.array_equal(bits, crc_encode(bits[:-crc_length], crc_length))


def _pm_update(llr_array, bit_array):
    pm = 0.0
    for llr, bit in zip(llr_array, bit_array):
        hard = 0 if llr >= 0 else 1
        if bit != hard:
            pm += abs(llr)
    return pm


def _sc_step_to(llr_matrix, bit_matrix, info_indices, frozen_bits, stop_pos):
    N = llr_matrix.shape[1]
    n = int(math.log2(N))

    detect = -1
    for i in range(N):
        if not np.isnan(bit_matrix[n, i]):
            continue
        detect = i - 1
        break

    if detect < 0:
        position = [0, 0, n, N]
    elif detect % 2 == 0:
        position = [n - 1, detect, n, N]
    else:
        position = [n - 1, detect - 1, n, N]

    while np.isnan(bit_matrix[n, stop_pos]):
        span = 1 << (position[2] - position[0])
        up_llr = llr_matrix[position[0], position[1]:position[1] + span]
        up_bit = bit_matrix[position[0], position[1]:position[1] + span]
        half = span // 2
        left_llr = llr_matrix[position[0] + 1, position[1]:position[1] + half]
        left_bit = bit_matrix[position[0] + 1, position[1]:position[1] + half]
        right_llr = llr_matrix[position[0] + 1, position[1] + half:position[1] + span]
        right_bit = bit_matrix[position[0] + 1, position[1] + half:position[1] + span]

        if _all_ready(up_bit):
            position = _up(position)
        elif _all_ready(right_bit):
            up_bit_val = _get_up_bit(left_bit.astype(int), right_bit.astype(int))
            bit_matrix[position[0], position[1]:position[1] + span] = up_bit_val
        elif _all_ready(right_llr):
            if position[0] == position[2] - 1:
                right_pos = position[1] + 1
                bit_val = _decide_bit(right_llr[0], right_pos, info_indices, frozen_bits)
                bit_matrix[position[0] + 1, position[1] + half:position[1] + span] = bit_val
            else:
                position = _rightdown(position)
        elif _all_ready(left_bit):
            right_llr_val = _get_right_llr(left_bit.astype(int), up_llr)
            llr_matrix[position[0] + 1, position[1] + half:position[1] + span] = right_llr_val
        elif not _all_ready(left_llr):
            left_llr_val = _get_left_llr(up_llr)
            llr_matrix[position[0] + 1, position[1]:position[1] + half] = left_llr_val
        else:
            if position[0] == position[2] - 1:
                left_pos = position[1]
                bit_val = _decide_bit(left_llr[0], left_pos, info_indices, frozen_bits)
                bit_matrix[position[0] + 1, position[1]:position[1] + half] = bit_val
            else:
                position = _leftdown(position)

    return llr_matrix, bit_matrix


class SCLDecoder:
    """SCL 译码器。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0, info_indices=None):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        if info_indices is None:
            info_indices = np.where(~self.frozen_bits)[0]
        self.info_indices = np.asarray(info_indices, dtype=int)

    def decode(self, llr_ch):
        if self.list_size == 1:
            u_hat = sc_decode(llr_ch, self.frozen_bits, self.info_indices)
            return u_hat, 0.0

        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N = self.N
        n = self.n

        llr0 = np.full((n + 1, N), np.nan, dtype=np.float64)
        bit0 = np.full((n + 1, N), np.nan, dtype=np.float64)
        llr0[0] = llr_ch

        paths = [(llr0.copy(), bit0.copy(), 0.0)]
        split_positions = [i for i in self.info_indices]

        for split_pos in split_positions:
            new_paths = []
            for llr_m, bit_m, pm in paths:
                llr_m, bit_m = _sc_step_to(
                    llr_m, bit_m, self.info_indices, self.frozen_bits, split_pos,
                )
                llr_bit = llr_m[n, split_pos]
                bit_val = int(bit_m[n, split_pos])

                pm_add = _pm_update(np.array([llr_bit]), np.array([bit_val]))
                new_paths.append((llr_m.copy(), bit_m.copy(), pm + pm_add))

                alt = bit_m.copy()
                alt[n, split_pos] = 1 - bit_val
                pm_alt = _pm_update(np.array([llr_bit]), np.array([alt[n, split_pos]]))
                new_paths.append((llr_m.copy(), alt, pm + pm_alt))

            new_paths.sort(key=lambda x: x[2])
            paths = new_paths[: self.list_size]

        last_pos = N - 1
        final_paths = []
        for llr_m, bit_m, pm in paths:
            llr_m, bit_m = _sc_step_to(
                llr_m, bit_m, self.info_indices, self.frozen_bits, last_pos,
            )
            final_paths.append((bit_m[n].astype(int), pm))

        final_paths.sort(key=lambda x: x[1])

        if self.crc_length > 0:
            for u_hat, pm in final_paths:
                info_bits = u_hat[self.info_indices]
                if crc_check(info_bits, self.crc_length):
                    return u_hat.copy(), pm

        return final_paths[0][0].copy(), final_paths[0][1]
