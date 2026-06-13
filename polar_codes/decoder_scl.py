"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import copy
import math

import numpy as np

from decoder_sc import _b_check, _li, _s_updater, f_operation, g_operation
from encoder import bit_reversal_permutation


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, crc_length):
    reg = 0
    for bit in bits:
        reg <<= 1
        reg |= int(bit)
        if reg & (1 << crc_length):
            reg ^= poly
    return reg & ((1 << crc_length) - 1)


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    remainder = _crc_remainder(info_bits, poly, crc_length)
    crc_bits = np.array([(remainder >> (crc_length - 1 - i)) & 1 for i in range(crc_length)])
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。"""
    bits = np.asarray(bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    remainder = _crc_remainder(bits, poly, crc_length)
    return remainder == 0


class SCLDecoder:
    """SCL 译码器（含 Lazy Copy 优化）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.br = bit_reversal_permutation(N)

    def _new_path(self, llr_ch):
        llrs = np.full((self.n + 1, self.N), -np.inf, dtype=np.float64)
        llrs[self.n, :] = llr_ch
        s = np.full((self.n + 1, self.N), -1, dtype=np.int8)
        return {"pm": 0.0, "llrs": llrs, "s": s, "u_hat": np.zeros(self.N, dtype=int)}

    def _path_copy(self, path):
        return {
            "pm": path["pm"],
            "llrs": path["llrs"].copy(),
            "s": path["s"].copy(),
            "u_hat": path["u_hat"].copy(),
        }

    def _pm_penalty(self, llr, u):
        u_hard = 0 if llr >= 0 else 1
        return 0.0 if u == u_hard else abs(llr)

    def decode(self, llr_ch):
        """主译码函数。"""
        llr_ch = np.asarray(llr_ch, dtype=np.float64)[self.br]
        paths = [self._new_path(llr_ch)]

        for phi in range(self.N):
            candidates = []
            for path in paths:
                if self.frozen_bits[phi]:
                    llr_phi = _li(0, phi, path["llrs"], path["s"], self.n)
                    new_path = self._path_copy(path)
                    new_path["pm"] += self._pm_penalty(llr_phi, 0)
                    new_path["u_hat"][phi] = 0
                    new_path["s"][0, phi] = 0
                    new_path["llrs"][0, phi] = np.inf
                    candidates.append(new_path)
                else:
                    llr_phi = _li(0, phi, path["llrs"], path["s"], self.n)
                    for u in (0, 1):
                        new_path = self._path_copy(path)
                        new_path["pm"] += self._pm_penalty(llr_phi, u)
                        new_path["u_hat"][phi] = u
                        new_path["s"][0, phi] = u
                        new_path["llrs"][0, phi] = llr_phi
                        candidates.append(new_path)

            candidates.sort(key=lambda p: p["pm"])
            paths = candidates[: self.list_size]

        paths.sort(key=lambda p: p["pm"])

        if self.crc_length > 0:
            info_positions = np.where(~self.frozen_bits)[0]
            for path in paths:
                info_bits = path["u_hat"][info_positions]
                if crc_check(info_bits, self.crc_length):
                    return path["u_hat"].copy(), path["pm"]

        best = paths[0]
        return best["u_hat"].copy(), best["pm"]
