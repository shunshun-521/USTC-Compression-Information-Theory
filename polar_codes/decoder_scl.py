"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np

from decoder_sc import (
    _active_bit_level,
    _active_llr_level,
    _update_bits,
    _update_llrs,
    f_operation,
    g_operation,
)
from encoder import bit_reversed


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, crc_length):
    reg = 0
    for bit in bits:
        reg ^= int(bit) << (crc_length - 1)
        for _ in range(crc_length):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)
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
    """检验 CRC"""
    bits = np.asarray(bits, dtype=int)
    expected = crc_encode(bits[:-crc_length], crc_length)
    return np.array_equal(bits, expected)


def _pm_update(pm, llr, u):
    u_from_llr = 0 if llr >= 0 else 1
    if u != u_from_llr:
        pm += abs(llr)
    return pm


class SCLDecoder:
    """SCL 译码器"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]

    def _new_path(self, llr_ch):
        L = np.full((self.N, self.n + 1), np.nan, dtype=np.float64)
        B = np.zeros((self.N, self.n + 1), dtype=int)
        L[:, 0] = llr_ch.astype(np.float64)
        return {"L": L, "B": B, "pm": 0.0, "u_hat": np.zeros(self.N, dtype=int)}

    def _copy_path(self, path):
        return {
            "L": path["L"].copy(),
            "B": path["B"].copy(),
            "pm": path["pm"],
            "u_hat": path["u_hat"].copy(),
        }

    def decode(self, llr_ch):
        """主译码函数，返回 u_hat, pm"""
        paths = [self._new_path(llr_ch)]

        for i in range(self.N):
            l = bit_reversed(i, self.n)
            new_paths = []

            for path in paths:
                _update_llrs(path["L"], path["B"], l, self.n)
                llr = path["L"][l, self.n]

                if l in self.frozen_set:
                    p = self._copy_path(path)
                    p["pm"] = _pm_update(p["pm"], llr, 0)
                    p["B"][l, self.n] = 0
                    p["u_hat"][l] = 0
                    _update_bits(p["B"], l, self.n, self.N)
                    new_paths.append(p)
                else:
                    for u in (0, 1):
                        p = self._copy_path(path)
                        p["pm"] = _pm_update(p["pm"], llr, u)
                        p["B"][l, self.n] = u
                        p["u_hat"][l] = u
                        _update_bits(p["B"], l, self.n, self.N)
                        new_paths.append(p)

            new_paths.sort(key=lambda p: p["pm"])
            paths = new_paths[: self.list_size]

        crc_valid = []
        for p in paths:
            if self.crc_length > 0:
                info_bits = p["u_hat"][self.info_indices]
                if crc_check(info_bits, self.crc_length):
                    crc_valid.append(p)
            else:
                crc_valid.append(p)

        best = min(crc_valid or paths, key=lambda p: p["pm"])
        return best["u_hat"], best["pm"]
