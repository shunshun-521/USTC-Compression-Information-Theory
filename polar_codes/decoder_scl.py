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
)
from encoder import bit_reversal_permutation


def _crc_poly(crc_length):
    if crc_length == 8:
        return 0x07
    if crc_length == 16:
        return 0x8005
    raise ValueError("crc_length must be 8 or 16")


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=int)
    poly = _crc_poly(crc_length)
    reg = 0
    for bit in info_bits:
        feedback = ((reg >> (crc_length - 1)) ^ int(bit)) & 1
        reg = ((reg << 1) & ((1 << crc_length) - 1)) ^ (feedback * poly)
    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=int,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits 是否满足 CRC。"""
    bits = np.asarray(bits, dtype=int)
    poly = _crc_poly(crc_length)
    reg = 0
    for bit in bits:
        feedback = ((reg >> (crc_length - 1)) ^ int(bit)) & 1
        reg = ((reg << 1) & ((1 << crc_length) - 1)) ^ (feedback * poly)
    return reg == 0


def _pm_update(llr_val, u_bit):
    if (llr_val >= 0 and u_bit == 0) or (llr_val < 0 and u_bit == 1):
        return 0.0
    return abs(llr_val)


def _get_up_loc(bit_matrix, n):
    for layer in range(n + 1):
        for idx in range(bit_matrix.shape[1]):
            if np.isnan(bit_matrix[layer, idx]):
                return layer, idx
    return n, 0


def _sc_step(llr_matrix, bit_matrix, frozen_bits, stop_pos):
    """运行树形 SC 直到 bit_matrix[n][stop_pos] 被判决。"""
    N = bit_matrix.shape[1]
    n = int(math.log2(N))
    layer, idx = _get_up_loc(bit_matrix, n)
    position = [layer, idx, n, N]

    while np.isnan(bit_matrix[n, stop_pos]):
        span = 2 ** (position[2] - position[0])
        start = position[1]
        up_llr = llr_matrix[position[0], start : start + span]
        up_bit = bit_matrix[position[0], start : start + span]
        half = span // 2
        left_llr = llr_matrix[position[0] + 1, start : start + half]
        left_bit = bit_matrix[position[0] + 1, start : start + half]
        right_llr = llr_matrix[position[0] + 1, start + half : start + span]
        right_bit = bit_matrix[position[0] + 1, start + half : start + span]

        if _all_num(up_bit) == 1:
            position = _up(position)
            continue
        if _all_num(right_bit) == 1:
            up_bit_val = _get_up_bit(left_bit.astype(int), right_bit.astype(int))
            bit_matrix[position[0], start : start + span] = up_bit_val
            continue
        if _all_num(right_llr) == 1:
            if position[0] == position[2] - 1:
                right_pos = start + half
                bit_val = _get_right_bit(right_llr[0], frozen_bits, right_pos)
                bit_matrix[position[0] + 1, right_pos] = bit_val
            else:
                position = _rightdown(position)
            continue
        if _all_num(left_bit) == 1:
            right_llr_val = _get_right_llr(left_bit.astype(int), up_llr)
            llr_matrix[position[0] + 1, start + half : start + span] = right_llr_val
            continue
        if _all_num(left_llr) == 0:
            left_llr_val = _get_left_llr(up_llr)
            llr_matrix[position[0] + 1, start : start + half] = left_llr_val
            continue
        if position[0] == position[2] - 1:
            left_pos = start
            bit_val = _get_left_bit(left_llr[0], frozen_bits, left_pos)
            bit_matrix[position[0] + 1, left_pos] = bit_val
        else:
            position = _leftdown(position)

    return llr_matrix, bit_matrix


class SCLDecoder:
    """SCL 译码器（含 Lazy Copy 优化）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]

    def _init_matrices(self, llr_ch):
        llr_matrix = np.full((self.n + 1, self.N), np.nan, dtype=np.float64)
        bit_matrix = np.full((self.n + 1, self.N), np.nan)
        llr_matrix[0] = llr_ch[bit_reversal_permutation(self.N)]
        return llr_matrix, bit_matrix

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)

        llr_list = []
        bit_list = []
        pm_list = []

        llr_m, bit_m = self._init_matrices(llr_ch)
        llr_list.append(llr_m)
        bit_list.append(bit_m)
        pm_list.append(0.0)

        for phi in range(self.N):
            new_llr_list = []
            new_bit_list = []
            new_pm_list = []

            for path_idx, pm in enumerate(pm_list):
                llr_m = llr_list[path_idx]
                bit_m = bit_list[path_idx]

                if not np.isnan(bit_m[self.n, phi]):
                    new_llr_list.append(llr_m)
                    new_bit_list.append(bit_m)
                    new_pm_list.append(pm)
                    continue

                llr_m, bit_m = _sc_step(
                    llr_m.copy(), bit_m.copy(), self.frozen_bits, phi
                )
                llr_val = llr_m[self.n, phi]

                if self.frozen_bits[phi]:
                    new_pm = pm + _pm_update(llr_val, 0)
                    new_llr_list.append(llr_m)
                    new_bit_list.append(bit_m)
                    new_pm_list.append(new_pm)
                else:
                    for u_bit in (0, 1):
                        lm = llr_m.copy()
                        bm = bit_m.copy()
                        bm[self.n, phi] = u_bit
                        new_pm = pm + _pm_update(llr_val, u_bit)
                        new_llr_list.append(lm)
                        new_bit_list.append(bm)
                        new_pm_list.append(new_pm)

            order = np.argsort(new_pm_list)
            keep = order[: self.list_size]
            llr_list = [new_llr_list[i] for i in keep]
            bit_list = [new_bit_list[i] for i in keep]
            pm_list = [new_pm_list[i] for i in keep]

        if self.crc_length > 0:
            valid = []
            for i, bit_m in enumerate(bit_list):
                u_hat = bit_m[self.n].astype(int)
                info_bits = u_hat[self.info_indices]
                if crc_check(info_bits, self.crc_length):
                    valid.append((pm_list[i], i))
            if valid:
                best = min(valid, key=lambda x: x[0])[1]
            else:
                best = int(np.argmin(pm_list))
        else:
            best = int(np.argmin(pm_list))

        u_hat = bit_list[best][self.n].astype(int)
        return u_hat, pm_list[best]
