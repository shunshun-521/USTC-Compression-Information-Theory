"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import copy
import math
import numpy as np

from decoder_sc import sc_decode, _sc_step_to
from encoder import bit_reversal_permutation

CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    reg = 0
    for bit in info_bits:
        reg = ((reg << 1) | int(bit)) & ((1 << crc_length) - 1)
        if reg & (1 << (crc_length - 1)):
            reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
        else:
            reg = (reg << 1) & ((1 << crc_length) - 1)
    for _ in range(crc_length):
        if reg & (1 << (crc_length - 1)):
            reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
        else:
            reg = (reg << 1) & ((1 << crc_length) - 1)
    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。"""
    bits = np.asarray(bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    reg = 0
    for bit in bits:
        reg = ((reg << 1) | int(bit)) & ((1 << crc_length) - 1)
        if reg & (1 << (crc_length - 1)):
            reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
        else:
            reg = (reg << 1) & ((1 << crc_length) - 1)
    return reg == 0


class SCLDecoder:
    """SCL 译码器（路径状态增量更新）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]
        self.information_pos = list(self.info_indices)
        self.br = bit_reversal_permutation(N)

    def _init_matrices(self, y_llr):
        llr_matrix = np.ones((self.n + 1, self.N))
        llr_matrix[llr_matrix == 1] = float('nan')
        bit_matrix = llr_matrix.copy()
        llr_matrix[0] = y_llr
        return llr_matrix, bit_matrix

    def decode(self, llr_ch):
        if self.list_size == 1:
            return sc_decode(llr_ch, self.frozen_bits), 0.0

        y_llr = llr_ch.astype(np.float64)[self.br]
        llr0, bit0 = self._init_matrices(y_llr)
        paths = [(0.0, llr0, bit0)]

        for phi in range(self.N):
            new_paths = []
            for pm, llr_m, bit_m in paths:
                llr_m = copy.deepcopy(llr_m)
                bit_m = copy.deepcopy(bit_m)
                llr_m, bit_m = _sc_step_to(
                    llr_m, bit_m, self.information_pos, 0, phi
                )
                llr_phi = llr_m[self.n, phi]
                bit_val = int(bit_m[self.n, phi])

                if self.frozen_bits[phi]:
                    new_paths.append((pm + (0.0 if llr_phi >= 0 else abs(llr_phi)), llr_m, bit_m))
                else:
                    for alt in (0, 1):
                        bm = copy.deepcopy(bit_m)
                        lm = copy.deepcopy(llr_m)
                        bm[self.n, phi] = alt
                        penalty = (
                            0.0
                            if (alt == 0 and llr_phi >= 0) or (alt == 1 and llr_phi < 0)
                            else abs(llr_phi)
                        )
                        new_paths.append((pm + penalty, lm, bm))

            new_paths.sort(key=lambda x: x[0])
            paths = new_paths[: self.list_size]

        if self.crc_length > 0:
            valid = []
            for pm, _, bm in paths:
                u = bm[self.n].astype(int)
                if crc_check(u[self.info_indices], self.crc_length):
                    valid.append((pm, u))
            pm, u_hat = min(valid, key=lambda x: x[0]) if valid else min(
                ((p, b[self.n].astype(int)) for p, _, b in paths), key=lambda x: x[0]
            )
        else:
            pm, u_hat = min(((p, b[self.n].astype(int)) for p, _, b in paths), key=lambda x: x[0])

        u_hat = u_hat.copy()
        u_hat[self.frozen_bits] = 0
        return u_hat, pm
