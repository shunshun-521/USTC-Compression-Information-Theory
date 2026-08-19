"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from decoder_sc import (
    _all_computed, _get_up_bit, _leftdown, _rightdown, _up,
    f_operation, g_operation,
)

CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, crc_length):
    reg = 0
    mask = (1 << crc_length) - 1
    top_bit = 1 << (crc_length - 1)
    for bit in bits:
        reg ^= int(bit) << (crc_length - 1)
        if reg & top_bit:
            reg = ((reg << 1) ^ poly) & mask
        else:
            reg = (reg << 1) & mask
    return reg


def crc_encode(info_bits, crc_length=8):
    info_bits = np.asarray(info_bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    remainder = _crc_remainder(info_bits, poly, crc_length)
    crc_bits = np.array([(remainder >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    return _crc_remainder(bits, poly, crc_length) == 0


def _get_up_loc(bit_matrix):
    n = int(np.log2(bit_matrix.shape[1]))
    detect_array = bit_matrix[n]
    detect = -1
    for i in range(len(detect_array)):
        if np.isnan(detect_array[i]):
            detect = i - 1
            break
    if detect == -1:
        return [0, 0]
    if detect % 2 == 0:
        return [n - 1, detect]
    return [n - 1, detect - 1]


def _sc_step_to(llr_matrix, bit_matrix, info_pos, frozen_val, target_pos):
    """SC 译码推进到 target_pos 判决完成"""
    N = bit_matrix.shape[1]
    n = int(np.log2(N))
    loc = _get_up_loc(bit_matrix)
    position = [loc[0], loc[1], n, N]

    while np.isnan(bit_matrix[n, target_pos]):
        span = 2 ** (position[2] - position[0])
        up_llr = llr_matrix[position[0], position[1]:position[1] + span]
        up_bit = bit_matrix[position[0], position[1]:position[1] + span]
        half = span // 2
        left_llr = llr_matrix[position[0] + 1, position[1]:position[1] + half]
        left_bit = bit_matrix[position[0] + 1, position[1]:position[1] + half]
        right_llr = llr_matrix[position[0] + 1, position[1] + half:position[1] + span]
        right_bit = bit_matrix[position[0] + 1, position[1] + half:position[1] + span]

        if _all_computed(up_bit):
            position = _up(position)
            continue
        if _all_computed(right_bit):
            bit_matrix[position[0], position[1]:position[1] + span] = _get_up_bit(left_bit, right_bit).flatten()
            continue
        if _all_computed(right_llr):
            if position[0] == position[2] - 1:
                rp = position[1] + half
                val = (0 if right_llr[0] >= 0 else 1) if rp in info_pos else frozen_val
                bit_matrix[position[0] + 1, position[1] + half:position[1] + span] = val
            else:
                position = _rightdown(position)
            continue
        if _all_computed(left_bit):
            llr_matrix[position[0] + 1, position[1] + half:position[1] + span] = g_operation(
                up_llr[:half], up_llr[half:], left_bit
            )
            continue
        if not _all_computed(left_llr):
            llr_matrix[position[0] + 1, position[1]:position[1] + half] = f_operation(up_llr[:half], up_llr[half:])
            continue
        if position[0] == position[2] - 1:
            lp = position[1]
            val = (0 if left_llr[0] >= 0 else 1) if lp in info_pos else frozen_val
            bit_matrix[position[0] + 1, position[1]:position[1] + half] = val
        else:
            position = _leftdown(position)

    return llr_matrix, bit_matrix


def _path_metric(llr_vec, bit_vec):
    pm = 0.0
    for llr, bit in zip(llr_vec, bit_vec):
        hard = 0 if llr >= 0 else 1
        if bit != hard:
            pm += abs(llr)
    return pm


class SCLDecoder:
    """SCL 译码器"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.info_pos = set(np.where(self.frozen_bits == 0)[0])
        self.info_indices = np.array(sorted(self.info_pos), dtype=int)
        self.list_size = list_size
        self.crc_length = crc_length
        self.frozen_val = 0

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N, n = self.N, self.n

        def new_state():
            llr_m = np.full((n + 1, N), np.nan, dtype=np.float64)
            bit_m = np.full((n + 1, N), np.nan, dtype=np.float64)
            llr_m[0] = llr_ch
            return llr_m, bit_m

        llr0, bit0 = new_state()
        llr_list, bit_list, pm_list = [llr0], [bit0], [0.0]
        split_positions = sorted(self.info_pos)

        prev = -1
        for sp in split_positions:
            new_llr, new_bit, new_pm = [], [], []
            for llr_m, bit_m, pm in zip(llr_list, bit_list, pm_list):
                llr_c, bit_c = llr_m.copy(), bit_m.copy()
                llr_c, bit_c = _sc_step_to(llr_c, bit_c, self.info_pos, self.frozen_val, sp)

                llr_seg = llr_c[n, prev + 1:sp + 1]
                bit0 = bit_c.copy()
                bit1 = bit_c.copy()
                bit1[n, sp] = 1 - bit1[n, sp]

                pm0 = pm + _path_metric(llr_seg, bit0[n, prev + 1:sp + 1])
                pm1 = pm + _path_metric(llr_seg, bit1[n, prev + 1:sp + 1])

                new_llr.extend([llr_c, llr_c])
                new_bit.extend([bit0, bit1])
                new_pm.extend([pm0, pm1])

            order = np.argsort(new_pm)[: self.list_size]
            llr_list = [new_llr[i] for i in order]
            bit_list = [new_bit[i] for i in order]
            pm_list = [new_pm[i] for i in order]
            prev = sp

        final_llr, final_bit, final_pm = [], [], []
        for llr_m, bit_m, pm in zip(llr_list, bit_list, pm_list):
            llr_c, bit_c = llr_m.copy(), bit_m.copy()
            llr_c, bit_c = _sc_step_to(llr_c, bit_c, self.info_pos, self.frozen_val, N - 1)
            llr_seg = llr_c[n, prev + 1:N]
            pm_new = pm + _path_metric(llr_seg, bit_c[n, prev + 1:N])
            final_llr.append(llr_c)
            final_bit.append(bit_c)
            final_pm.append(pm_new)

        order = np.argsort(final_pm)
        for idx in order:
            u_hat = final_bit[idx][n].astype(int)
            if self.crc_length > 0:
                info_bits = u_hat[self.info_indices]
                if crc_check(info_bits, self.crc_length):
                    return u_hat, final_pm[idx]
        best = order[0]
        return final_bit[best][n].astype(int), final_pm[best]
