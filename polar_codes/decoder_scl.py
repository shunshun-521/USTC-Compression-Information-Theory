"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np

from decoder_sc import (
    _PermutedSCD,
    bit_reversed,
    _hard_decision,
)

_CRC8_POLY = 0x07
_CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, crc_length):
    reg = 0
    for b in bits:
        reg ^= int(b) << (crc_length - 1)
        for _ in range(8):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)
    return reg


def crc_encode(info_bits, crc_length=8):
    info_bits = np.asarray(info_bits, dtype=int).ravel()
    if crc_length == 8:
        poly = _CRC8_POLY
    elif crc_length == 16:
        poly = _CRC16_POLY
    else:
        raise ValueError("crc_length must be 8 or 16")
    rem = _crc_remainder(info_bits, poly, crc_length)
    crc_bits = np.array([(rem >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=int).ravel()
    if crc_length == 8:
        poly = _CRC8_POLY
    elif crc_length == 16:
        poly = _CRC16_POLY
    else:
        raise ValueError("crc_length must be 8 or 16")
    return _crc_remainder(bits, poly, crc_length) == 0


class SCLDecoder:
    """SCL 译码器（基于 Permuted SC 的路径扩展）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.list_size = list_size
        self.crc_length = crc_length

    def _pm_penalty(self, llr_val, u):
        return 0.0 if u == _hard_decision(llr_val) else abs(llr_val)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [{"scd": _PermutedSCD(self.N, llr_ch, self.frozen_set), "pm": 0.0}]

        for i in range(self.N):
            l = bit_reversed(i, self.n)
            new_paths = []
            for path in paths:
                scd = path["scd"]
                scd._update_llrs(l)
                llr_dec = scd.L[l, scd.n]

                if l in self.frozen_set:
                    pen = self._pm_penalty(llr_dec, 0)
                    scd.B[l, scd.n] = 0
                    scd._update_bits(l)
                    path["pm"] += pen
                    new_paths.append(path)
                else:
                    for u in (0, 1):
                        child = {
                            "scd": _PermutedSCD(self.N, llr_ch, self.frozen_set),
                            "pm": path["pm"] + self._pm_penalty(llr_dec, u),
                        }
                        child["scd"].L = scd.L.copy()
                        child["scd"].B = scd.B.copy()
                        child["scd"].B[l, scd.n] = u
                        child["scd"]._update_bits(l)
                        new_paths.append(child)

            new_paths.sort(key=lambda p: p["pm"])
            paths = new_paths[: self.list_size]

        best = paths[0]
        u_hat = best["scd"].B[:, self.n].astype(int)

        if self.crc_length > 0:
            info_positions = np.where(self.frozen_bits == 0)[0]
            for path in sorted(paths, key=lambda p: p["pm"]):
                cand = path["scd"].B[:, self.n].astype(int)
                if crc_check(cand[info_positions], self.crc_length):
                    best = path
                    u_hat = cand
                    break

        return u_hat, best["pm"]
