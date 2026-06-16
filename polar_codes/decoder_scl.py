"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
import math

from decoder_sc import (
    f_operation, g_operation, _permute_channel_llr,
    _all_computed, _leftdown, _rightdown, _up,
    _get_up_bit, _get_left_llr, _get_right_llr,
)


CRC_POLYNOMIALS = {8: 0x07, 16: 0x8005}


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后"""
    poly = CRC_POLYNOMIALS[crc_length]
    info_bits = np.asarray(info_bits, dtype=np.int8)
    reg = 0
    for bit in info_bits:
        reg = ((reg << 1) | int(bit)) & ((1 << crc_length) - 1)
        if reg & (1 << (crc_length - 1)):
            reg ^= poly
    for _ in range(crc_length):
        reg = (reg << 1) & ((1 << crc_length) - 1)
        if reg & (1 << (crc_length - 1)):
            reg ^= poly
    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=np.int8
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC"""
    poly = CRC_POLYNOMIALS[crc_length]
    bits = np.asarray(bits, dtype=np.int8)
    reg = 0
    for bit in bits:
        reg = ((reg << 1) | int(bit)) & ((1 << crc_length) - 1)
        if reg & (1 << (crc_length - 1)):
            reg ^= poly
    return reg == 0


def _pm_penalty(llr, u):
    u_hard = 0 if llr >= 0 else 1
    return 0.0 if u == u_hard else abs(llr)


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
        return [n - 1, detect]
    return [n - 1, detect - 1] if detect >= 0 else [0, 0]


def _sc_step_to_phi(llr_matrix, bit_matrix, info_set, frozen_value, target_phi):
    """SC 逐步译码至比特 target_phi"""
    N = bit_matrix.shape[1]
    n = int(math.log2(N))
    loc = _get_up_loc(bit_matrix)
    position = [loc[0], loc[1], n, N]

    while bit_matrix[n][target_phi] != 0 and bit_matrix[n][target_phi] != 1:
        span = 2 ** (position[2] - position[0])
        p0, p1 = position[0], position[1]
        half = span // 2

        up_llr = llr_matrix[p0][p1:p1 + span]
        up_bit = bit_matrix[p0][p1:p1 + span]
        left_llr = llr_matrix[p0 + 1][p1:p1 + half]
        left_bit = bit_matrix[p0 + 1][p1:p1 + half]
        right_llr = llr_matrix[p0 + 1][p1 + half:p1 + span]
        right_bit = bit_matrix[p0 + 1][p1 + half:p1 + span]

        if _all_computed(up_bit):
            position = _up(position)
        elif _all_computed(right_bit):
            bit_matrix[p0][p1:p1 + span] = _get_up_bit(left_bit, right_bit)
        elif _all_computed(right_llr):
            if position[0] == position[2] - 1:
                right_pos = p1 + 1
                val = frozen_value if right_pos not in info_set else (0 if right_llr[0] > 0 else 1)
                bit_matrix[p0 + 1][p1 + half] = val
            else:
                position = _rightdown(position)
        elif _all_computed(left_bit):
            llr_matrix[p0 + 1][p1 + half:p1 + span] = _get_right_llr(left_bit, up_llr)
        elif not _all_computed(left_llr):
            llr_matrix[p0 + 1][p1:p1 + half] = _get_left_llr(up_llr)
        elif position[0] == position[2] - 1:
            left_pos = p1
            val = frozen_value if left_pos not in info_set else (0 if left_llr[0] >= 0 else 1)
            bit_matrix[p0 + 1][p1] = val
        else:
            position = _leftdown(position)

    return llr_matrix, bit_matrix


class SCLDecoder:
    """SCL 译码器（含 Lazy Copy 优化）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.info_indices = np.where(~self.frozen_bits)[0]
        self.info_set = set(int(i) for i in self.info_indices)
        self.list_size = list_size
        self.crc_length = crc_length

    def decode(self, llr_ch):
        y_llr = _permute_channel_llr(llr_ch)
        n, N = self.n, self.N

        llr_matrix = np.full((n + 1, N), np.nan)
        bit_matrix = np.full((n + 1, N), np.nan)
        llr_matrix[0] = y_llr

        paths = [(llr_matrix.copy(), bit_matrix.copy(), 0.0)]

        for phi in range(N):
            new_paths = []
            for llr_m, bit_m, pm in paths:
                llr_m, bit_m = _sc_step_to_phi(llr_m, bit_m, self.info_set, 0, phi)
                llr_val = llr_m[n][phi] if not np.isnan(llr_m[n][phi]) else 0.0

                if self.frozen_bits[phi]:
                    bit_m[n][phi] = 0
                    new_pm = pm + _pm_penalty(llr_val, 0)
                    new_paths.append((llr_m, bit_m, new_pm))
                else:
                    for u in (0, 1):
                        lm = llr_m.copy()
                        bm = bit_m.copy()
                        bm[n][phi] = u
                        new_pm = pm + _pm_penalty(llr_val, u)
                        new_paths.append((lm, bm, new_pm))

            new_paths.sort(key=lambda x: x[2])
            paths = new_paths[: self.list_size]

        best_crc = None
        best_all = min(paths, key=lambda x: x[2])

        if self.crc_length > 0:
            for _, bit_m, pm in paths:
                u_hat = bit_m[n].astype(int)
                payload = u_hat[self.info_indices[: len(self.info_indices)]]
                if crc_check(payload, self.crc_length):
                    if best_crc is None or pm < best_crc[2]:
                        best_crc = (_, bit_m, pm)

        chosen = best_crc if best_crc is not None else best_all
        u_hat = chosen[1][n].astype(int)
        return u_hat, chosen[2]
