"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math

import numpy as np

from decoder_sc import (
    _all_filled,
    _get_left_llr,
    _get_right_llr,
    _get_up_bit,
    _leftdown,
    _rightdown,
    _up,
    f_operation,
    g_operation,
)


CRC_POLYNOMIALS = {8: 0x07, 16: 0x8005}


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    poly = CRC_POLYNOMIALS[crc_length]
    reg = 0
    bits = np.asarray(info_bits, dtype=int)
    for bit in bits:
        reg ^= bit << (crc_length - 1)
        for _ in range(crc_length):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)
    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int
    )
    return np.concatenate([bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC。"""
    return np.array_equal(bits[-crc_length:], crc_encode(bits[:-crc_length], crc_length))


def _pm_update(pm, llr, bit):
    hard = 0 if llr >= 0 else 1
    return pm + (0.0 if bit == hard else abs(llr))


def _sc_step_to_bit(llr_matrix, bit_matrix, frozen_bits, target_bit):
    """将 SC 状态推进到 target_bit 并完成该位判决。"""
    N = llr_matrix.shape[1]
    n = int(math.log2(N))
    position = [0, 0, n, N]

    while not _all_filled(bit_matrix[n][: target_bit + 1]):
        span = 2 ** (position[2] - position[0])
        start = position[1]
        up_llr = llr_matrix[position[0]][start : start + span]
        up_bit = bit_matrix[position[0]][start : start + span]
        half = span // 2
        left_llr = llr_matrix[position[0] + 1][start : start + half]
        left_bit = bit_matrix[position[0] + 1][start : start + half]
        right_llr = llr_matrix[position[0] + 1][start + half : start + span]
        right_bit = bit_matrix[position[0] + 1][start + half : start + span]

        if _all_filled(up_bit):
            position = _up(position)
        elif _all_filled(right_bit):
            bit_matrix[position[0]][start : start + span] = _get_up_bit(
                left_bit, right_bit
            )
        elif _all_filled(right_llr):
            if position[0] == position[2] - 1:
                pos = start + half
                if frozen_bits[pos]:
                    bit_matrix[position[0] + 1][start + half : start + span] = 0
                else:
                    bit_matrix[position[0] + 1][start + half : start + span] = (
                        0 if right_llr[0] >= 0 else 1
                    )
            else:
                position = _rightdown(position)
        elif _all_filled(left_bit):
            llr_matrix[position[0] + 1][start + half : start + span] = _get_right_llr(
                left_bit, up_llr
            )
        elif not _all_filled(left_llr):
            llr_matrix[position[0] + 1][start : start + half] = _get_left_llr(up_llr)
        elif position[0] == position[2] - 1:
            pos = start
            if frozen_bits[pos]:
                bit_matrix[position[0] + 1][start : start + half] = 0
            else:
                bit_matrix[position[0] + 1][start : start + half] = (
                    0 if left_llr[0] >= 0 else 1
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
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n = self.n
        N = self.N

        llr_matrix = np.full((n + 1, N), np.nan)
        bit_matrix = np.full((n + 1, N), np.nan)
        llr_matrix[0] = llr_ch
        paths = [(0.0, llr_matrix.copy(), bit_matrix.copy())]

        for phi in range(N):
            new_paths = []
            for pm, llr_m, bit_m in paths:
                llr_m, bit_m = _sc_step_to_bit(
                    llr_m.copy(), bit_m.copy(), self.frozen_bits, phi
                )
                llr_leaf = llr_m[n][phi]
                if self.frozen_bits[phi]:
                    pm_new = _pm_update(pm, llr_leaf, 0)
                    bit_m[n][phi] = 0
                    new_paths.append((pm_new, llr_m, bit_m))
                else:
                    for bit in (0, 1):
                        bit_copy = bit_m.copy()
                        bit_copy[n][phi] = bit
                        pm_new = _pm_update(pm, llr_leaf, bit)
                        new_paths.append((pm_new, llr_m.copy(), bit_copy))

            new_paths.sort(key=lambda x: x[0])
            paths = new_paths[: self.list_size]

        if self.crc_length > 0:
            valid = []
            for pm, _, bit_m in paths:
                bits = bit_m[n].astype(int)
                bits[self.frozen_bits] = 0
                payload = bits[self.info_indices]
                if crc_check(payload, self.crc_length):
                    valid.append((pm, bits))
            if valid:
                best = min(valid, key=lambda x: x[0])
                return best[1], best[0]

        best = min(paths, key=lambda x: x[0])
        u_hat = best[2][n].astype(int)
        u_hat[self.frozen_bits] = 0
        return u_hat, best[0]
