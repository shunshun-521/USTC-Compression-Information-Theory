"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np

from decoder_sc import (
    f_operation,
    g_operation,
    sc_decode,
    _active_llr_level,
    _active_bit_level,
    _update_llrs,
    _update_bits,
)
from encoder import bit_reversed


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, crc_length):
    mask = (1 << crc_length) - 1
    msb = 1 << (crc_length - 1)
    reg = 0
    for bit in bits:
        reg ^= int(bit) << (crc_length - 1)
        if reg & msb:
            reg = ((reg << 1) ^ poly) & mask
        else:
            reg = (reg << 1) & mask
    return reg


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    remainder = _crc_remainder(info_bits, poly, crc_length)
    crc_bits = np.array(
        [(remainder >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=int,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC 是否正确。"""
    bits = np.asarray(bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    payload = bits[:-crc_length]
    expected = bits[-crc_length:]
    remainder = _crc_remainder(payload, poly, crc_length)
    actual = np.array(
        [(remainder >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=int,
    )
    return np.array_equal(expected, actual)


def _pm_penalty(llr, u):
    hard = 0 if llr >= 0 else 1
    return 0.0 if u == hard else abs(llr)


class SCLDecoder:
    """SCL 译码器（Lazy Copy 优化）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]

    def decode(self, llr_ch):
        if self.list_size == 1 and self.crc_length == 0:
            return sc_decode(llr_ch, self.frozen_bits), 0.0

        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [self._new_path(llr_ch)]

        for i in range(self.N):
            l = bit_reversed(i, self.n)
            candidates = []
            for path in paths:
                _update_llrs(path["L"], path["B"], l, self.n, self.N)
                cur_llr = path["L"][l, self.n]
                if self.frozen_bits[l]:
                    child = self._fork(path)
                    child["pm"] += _pm_penalty(cur_llr, 0)
                    child["u_hat"][l] = 0
                    child["B"][l, self.n] = 0
                    _update_bits(child["B"], l, self.n, self.N)
                    candidates.append(child)
                else:
                    for u in (0, 1):
                        child = self._fork(path)
                        child["pm"] += _pm_penalty(cur_llr, u)
                        child["u_hat"][l] = u
                        child["B"][l, self.n] = u
                        _update_bits(child["B"], l, self.n, self.N)
                        candidates.append(child)

            candidates.sort(key=lambda p: p["pm"])
            paths = candidates[: self.list_size]

        if self.crc_length > 0:
            valid = [
                p
                for p in paths
                if crc_check(p["u_hat"][self.info_indices], self.crc_length)
            ]
            if valid:
                paths = valid

        best = min(paths, key=lambda p: p["pm"])
        return best["u_hat"].copy(), best["pm"]

    def _new_path(self, llr_ch):
        L = np.zeros((self.N, self.n + 1), dtype=np.float64)
        B = np.zeros((self.N, self.n + 1), dtype=np.int8)
        L[:, 0] = llr_ch
        return {"pm": 0.0, "u_hat": np.zeros(self.N, dtype=int), "L": L, "B": B}

    def _fork(self, path):
        return {
            "pm": path["pm"],
            "u_hat": path["u_hat"].copy(),
            "L": path["L"].copy(),
            "B": path["B"].copy(),
        }
