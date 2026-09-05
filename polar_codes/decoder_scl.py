"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math

import numpy as np

from decoder_sc import (
    _boxplus,
    _update_bits,
    _update_llrs,
    active_bit_level,
    active_llr_level,
    g_operation,
)
from encoder import bit_reversed


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_poly(crc_length):
    if crc_length == 8:
        return CRC8_POLY
    if crc_length == 16:
        return CRC16_POLY
    raise ValueError("crc_length must be 8 or 16")


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=np.int8)
    poly = _crc_poly(crc_length)
    reg = 0
    for bit in info_bits:
        reg ^= int(bit) << (crc_length - 1)
        for _ in range(8 if crc_length == 8 else 1):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)
    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=np.int8
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC"""
    if crc_length == 0:
        return True
    expected = crc_encode(bits[:-crc_length], crc_length)
    return np.array_equal(expected[-crc_length:], bits[-crc_length:])


class SCLDecoder:
    """SCL 译码器（Lazy Copy + mcba1n 相位顺序）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]

    @staticmethod
    def _pm_update(pm, llr, u):
        u_hard = 0 if llr >= 0 else 1
        if u != u_hard:
            pm += abs(llr)
        return pm

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n = self.n
        N = self.N

        paths = [{
            "L": np.full((N, n + 1), np.nan, dtype=np.float64),
            "B": np.zeros((N, n + 1), dtype=int),
            "pm": 0.0,
        }]
        paths[0]["L"][:, 0] = llr_ch

        for phase in range(N):
            l = bit_reversed(phase, n)
            candidates = []

            for path in paths:
                _update_llrs(path["L"], path["B"], l, n)
                llr0 = path["L"][l, n]

                if self.frozen_bits[l]:
                    pm = self._pm_update(path["pm"], llr0, 0)
                    new_path = {
                        "L": path["L"].copy(),
                        "B": path["B"].copy(),
                        "pm": pm,
                    }
                    new_path["B"][l, n] = 0
                    _update_bits(new_path["B"], l, n)
                    candidates.append(new_path)
                else:
                    for u in (0, 1):
                        pm = self._pm_update(path["pm"], llr0, u)
                        new_path = {
                            "L": path["L"].copy(),
                            "B": path["B"].copy(),
                            "pm": pm,
                        }
                        new_path["B"][l, n] = u
                        _update_bits(new_path["B"], l, n)
                        candidates.append(new_path)

            candidates.sort(key=lambda p: p["pm"])
            paths = candidates[: self.list_size]

        if self.crc_length > 0:
            valid = []
            for p in paths:
                bits = p["B"][:, n][self.info_indices]
                if crc_check(bits, self.crc_length):
                    valid.append(p)
            best = min(valid if valid else paths, key=lambda p: p["pm"])
        else:
            best = min(paths, key=lambda p: p["pm"])

        return best["B"][:, n].astype(int), best["pm"]
