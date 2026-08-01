"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np
from decoder_sc import (
    _all_filled,
    _get_bit,
    _get_left_llr,
    _get_right_llr,
    _get_up_bit,
    _leftdown,
    _rightdown,
    _up,
    _f_min_sum,
    g_operation,
)


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, crc_length):
    reg = 0
    poly_shifted = poly << (crc_length - 8) if crc_length == 8 else poly
    mask = (1 << crc_length) - 1
    for bit in bits:
        reg ^= int(bit) << (crc_length - 1)
        for _ in range(8 if crc_length == 8 else 1):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly_shifted) & mask
            else:
                reg = (reg << 1) & mask
    return reg


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    remainder = _crc_remainder(info_bits, poly, crc_length)
    crc_bits = np.array(
        [(remainder >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=int,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC 是否正确"""
    bits = np.asarray(bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    return _crc_remainder(bits, poly, crc_length) == 0


def _get_up_loc(bit_matrix):
    n = int(math.log2(bit_matrix.shape[1]))
    for i in range(n + 1):
        for j in range(bit_matrix.shape[1]):
            if np.isnan(bit_matrix[i, j]):
                return [i, j]
    return [n, 0]


def _sc_step_to(llr_matrix, bit_matrix, info_positions, target_pos):
    """SC 译码至 target_pos 并完成该位判决"""
    N = bit_matrix.shape[1]
    n = int(math.log2(N))
    info_set = set(info_positions)
    loc = _get_up_loc(bit_matrix)
    position = [loc[0], loc[1], n, N]

    while np.isnan(bit_matrix[n, target_pos]):
        span_pow = position[2] - position[0]
        up_llr = llr_matrix[position[0]][position[1]:position[1] + 2 ** span_pow]
        up_bit = bit_matrix[position[0]][position[1]:position[1] + 2 ** span_pow]
        half = 2 ** (span_pow - 1)
        left_llr = llr_matrix[position[0] + 1][position[1]:position[1] + half]
        left_bit = bit_matrix[position[0] + 1][position[1]:position[1] + half]
        right_llr = llr_matrix[position[0] + 1][position[1] + half:position[1] + 2 * half]
        right_bit = bit_matrix[position[0] + 1][position[1] + half:position[1] + 2 * half]

        if _all_filled(up_bit):
            position = _up(position)
        elif _all_filled(right_bit):
            up_bit_new = _get_up_bit(left_bit, right_bit)
            bit_matrix[position[0]][position[1]:position[1] + 2 ** span_pow] = up_bit_new[0]
        elif _all_filled(right_llr):
            if position[0] == position[2] - 1:
                right_bit_pos = position[1] + 1
                val = _get_bit(right_llr[0], right_bit_pos in info_set)
                bit_matrix[position[0] + 1][position[1] + half] = val
            else:
                position = _rightdown(position)
        elif _all_filled(left_bit):
            right_llr_new = _get_right_llr(left_bit, up_llr)
            llr_matrix[position[0] + 1][position[1] + half:position[1] + 2 * half] = right_llr_new
        elif not _all_filled(left_llr):
            left_llr_new = _get_left_llr(up_llr)
            llr_matrix[position[0] + 1][position[1]:position[1] + half] = left_llr_new
        elif position[0] == position[2] - 1:
            left_bit_pos = position[1]
            val = _get_bit(left_llr[0], left_bit_pos in info_set)
            bit_matrix[position[0] + 1][position[1]] = val
        else:
            position = _leftdown(position)

    return llr_matrix, bit_matrix


def _path_metric_update(llr_val, bit):
    hard = 0 if llr_val >= 0 else 1
    return 0.0 if bit == hard else abs(llr_val)


class SCLDecoder:
    """SCL 译码器"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_positions = np.where(self.frozen_bits == 0)[0]

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N = self.N
        n = self.n

        llr_matrix = np.full((n + 1, N), np.nan)
        bit_matrix = np.full((n + 1, N), np.nan)
        llr_matrix[0] = llr_ch.copy()

        llr_list = [llr_matrix.copy()]
        bit_list = [bit_matrix.copy()]
        pm_list = [0.0]

        for phi in range(N):
            is_info = phi in set(self.info_positions)
            new_llr, new_bit, new_pm = [], [], []

            for llr_m, bit_m, pm in zip(llr_list, bit_list, pm_list):
                if not is_info:
                    llr_out, bit_out = _sc_step_to(
                        llr_m.copy(), bit_m.copy(), self.info_positions, phi
                    )
                    llr_at_bit = llr_out[n, phi] if not np.isnan(llr_out[n, phi]) else 0.0
                    new_pm.append(pm + _path_metric_update(llr_at_bit, 0))
                    new_llr.append(llr_out)
                    new_bit.append(bit_out)
                else:
                    for bit in (0, 1):
                        llr_c = llr_m.copy()
                        bit_c = bit_m.copy()
                        llr_out, bit_out = _sc_step_to(
                            llr_c, bit_c, self.info_positions, phi
                        )
                        if bit_out[n, phi] != bit:
                            bit_out = bit_out.copy()
                            bit_out[n, phi] = bit
                        llr_at_bit = llr_out[n, phi] if not np.isnan(llr_out[n, phi]) else 0.0
                        new_pm.append(pm + _path_metric_update(llr_at_bit, bit))
                        new_llr.append(llr_out)
                        new_bit.append(bit_out)

            order = np.argsort(new_pm)
            keep = order[: self.list_size]
            llr_list = [new_llr[i] for i in keep]
            bit_list = [new_bit[i] for i in keep]
            pm_list = [new_pm[i] for i in keep]

        best_idx = 0
        if self.crc_length > 0:
            valid = []
            for i, bit_m in enumerate(bit_list):
                info_bits = bit_m[n, self.info_positions].astype(int)
                if crc_check(info_bits, self.crc_length):
                    valid.append(i)
            if valid:
                best_idx = min(valid, key=lambda i: pm_list[i])
            else:
                best_idx = int(np.argmin(pm_list))
        else:
            best_idx = int(np.argmin(pm_list))

        u_hat = bit_list[best_idx][n].astype(int)
        return u_hat, pm_list[best_idx]
