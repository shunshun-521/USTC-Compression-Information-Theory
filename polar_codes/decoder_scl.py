"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from decoder_sc import (
    sc_decode,
    f_operation,
    g_operation,
    _all_num,
    _leftdown,
    _rightdown,
    _up,
    _get_up_bit,
    _get_left_llr,
    _get_right_llr,
    _get_left_bit,
    _get_right_bit,
)


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, crc_length):
    reg = 0
    for bit in bits:
        reg ^= int(bit) << (crc_length - 1)
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


def _pm_penalty(llr_val, bit_val):
    hard = 0 if llr_val >= 0 else 1
    return 0.0 if bit_val == hard else abs(llr_val)


def _sc_step_to_split(llr_matrix, bit_matrix, info_set, frozen_bit, split_pos, n, N):
    """SC 译码到 split_pos 位置"""
    loc = _get_up_loc(bit_matrix, n, N)
    position = [loc[0], loc[1], n, N]

    while bit_matrix[n][split_pos] != 0 and bit_matrix[n][split_pos] != 1:
        span = 2 ** (position[2] - position[0])
        up_llr = llr_matrix[position[0]][position[1] : position[1] + span]
        half = span // 2
        left_llr = llr_matrix[position[0] + 1][position[1] : position[1] + half]
        left_bit = bit_matrix[position[0] + 1][position[1] : position[1] + half]
        right_llr = llr_matrix[position[0] + 1][position[1] + half : position[1] + span]
        right_bit = bit_matrix[position[0] + 1][position[1] + half : position[1] + span]

        if _all_num(bit_matrix[position[0]][position[1] : position[1] + span]):
            position = _up(position)
        elif _all_num(right_bit):
            up_bit = _get_up_bit(left_bit, right_bit)
            bit_matrix[position[0]][position[1] : position[1] + span] = up_bit
        elif _all_num(right_llr):
            if position[0] == position[2] - 1:
                right_bit_pos = position[1] + 1
                val = _get_right_bit(right_llr[0], info_set, frozen_bit, right_bit_pos)
                bit_matrix[position[0] + 1][position[1] + half] = val
            else:
                position = _rightdown(position)
        elif _all_num(left_bit):
            right_llr_new = _get_right_llr(left_bit, up_llr)
            llr_matrix[position[0] + 1][position[1] + half : position[1] + span] = right_llr_new
        elif not _all_num(left_llr):
            left_llr_new = _get_left_llr(up_llr)
            llr_matrix[position[0] + 1][position[1] : position[1] + half] = left_llr_new
        else:
            if position[0] == position[2] - 1:
                left_bit_pos = position[1]
                val = _get_left_bit(left_llr[0], info_set, frozen_bit, left_bit_pos)
                bit_matrix[position[0] + 1][position[1]] = val
            else:
                position = _leftdown(position)

    return llr_matrix, bit_matrix


def _get_up_loc(bit_matrix, n, N):
    detect_array = bit_matrix[n]
    detect = -1
    for i in range(N):
        if detect_array[i] == 1 or detect_array[i] == 0:
            pass
        else:
            detect = i - 1
            break
    if detect % 2 == 0:
        return [n - 1, detect]
    return [n - 1, detect - 1] if detect >= 0 else [0, 0]


def _init_matrices(y_llr, n, N):
    llr_matrix = np.full((n + 1, N), np.nan)
    bit_matrix = np.full((n + 1, N), np.nan)
    llr_matrix[0] = y_llr
    return llr_matrix, bit_matrix


class SCLDecoder:
    """SCL 译码器"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]
        self.info_set = set(self.info_indices.tolist())
        self.frozen_bit = 0

    def decode(self, llr_ch):
        y_llr = np.asarray(llr_ch, dtype=np.float64)
        N, n = self.N, self.n

        if self.list_size == 1:
            return sc_decode(y_llr, self.frozen_bits), 0.0

        llr_list = [_init_matrices(y_llr, n, N)[0]]
        bit_list = [_init_matrices(y_llr, n, N)[1]]
        pm_list = [0.0]

        split_positions = [i for i in self.info_indices]

        for split_pos in split_positions:
            new_llr, new_bit, new_pm = [], [], []
            for idx in range(len(llr_list)):
                llr_m = llr_list[idx].copy()
                bit_m = bit_list[idx].copy()
                pm = pm_list[idx]

                llr_m, bit_m = _sc_step_to_split(
                    llr_m, bit_m, self.info_set, self.frozen_bit, split_pos, n, N
                )

                loc = _get_up_loc(bit_m, n, N)
                span = 2 ** (n - loc[0])
                left_llr = llr_m[loc[0] + 1][loc[1] : loc[1] + span // 2]

                for bit in (0, 1):
                    lm = llr_m.copy()
                    bm = bit_m.copy()
                    bm[n][split_pos] = bit
                    penalty = _pm_penalty(left_llr[0] if len(left_llr) > 0 else 0, bit)
                    new_llr.append(lm)
                    new_bit.append(bm)
                    new_pm.append(pm + penalty)

            order = np.argsort(new_pm)[: self.list_size]
            llr_list = [new_llr[i] for i in order]
            bit_list = [new_bit[i] for i in order]
            pm_list = [new_pm[i] for i in order]

        for idx in range(len(llr_list)):
            u_hat = sc_decode(y_llr, self.frozen_bits)
            bit_list[idx][n] = u_hat.astype(float)

        best_idx = int(np.argmin(pm_list))
        u_hat = bit_list[best_idx][n].astype(int)

        if self.crc_length > 0:
            for idx in np.argsort(pm_list):
                candidate = bit_list[idx][n].astype(int)
                info_bits = candidate[self.info_indices]
                if crc_check(info_bits, self.crc_length):
                    return candidate, pm_list[idx]

        return u_hat, pm_list[best_idx]
