"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np

from decoder_sc import (
    sc_decode,
    _bit_reversed,
    _update_llrs,
    _update_bits,
)


def _crc_poly(crc_length):
    if crc_length == 8:
        return 0x07
    if crc_length == 16:
        return 0x8005
    raise ValueError("crc_length must be 8 or 16")


def crc_encode(info_bits, crc_length=8):
    info_bits = np.asarray(info_bits, dtype=np.int8)
    poly = _crc_poly(crc_length)
    reg = 0
    for b in info_bits:
        reg ^= int(b) << (crc_length - 1)
        for _ in range(8):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)
    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=np.int8
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=np.int8)
    poly = _crc_poly(crc_length)
    reg = 0
    for b in bits:
        reg ^= int(b) << (crc_length - 1)
        for _ in range(8):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)
    return reg == 0


class SCLDecoder:
    """SCL 译码器（基于 Permuted SC 的列表扩展）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.L = list_size
        self.crc_length = crc_length
        self.frozen_set = set(np.where(self.frozen_bits)[0])

    def _penalty(self, llr, u):
        u_hat = 0 if llr >= 0 else 1
        return 0.0 if u == u_hat else abs(llr)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        if self.L == 1 and self.crc_length == 0:
            u = sc_decode(llr_ch, self.frozen_bits)
            return u, 0.0

        N, n = self.N, self.n
        paths = [
            {
                "pm": 0.0,
                "L": np.full((N, n + 1), np.nan, dtype=np.float64),
                "B": np.zeros((N, n + 1), dtype=np.float64),
            }
        ]
        paths[0]["L"][:, 0] = llr_ch

        for l in [_bit_reversed(i, n) for i in range(N)]:
            new_paths = []
            for path in paths:
                _update_llrs(path["L"], path["B"], l, n)
                llr_bit = path["L"][l, n]
                if l in self.frozen_set:
                    pm = path["pm"] + self._penalty(llr_bit, 0)
                    B = path["B"].copy()
                    L = path["L"].copy()
                    B[l, n] = 0
                    _update_bits(B, l, n)
                    new_paths.append({"pm": pm, "L": L, "B": B})
                else:
                    for u in (0, 1):
                        pm = path["pm"] + self._penalty(llr_bit, u)
                        B = path["B"].copy()
                        L = path["L"].copy()
                        B[l, n] = u
                        _update_bits(B, l, n)
                        new_paths.append({"pm": pm, "L": L, "B": B})
            new_paths.sort(key=lambda p: p["pm"])
            paths = new_paths[: self.L]

        if self.crc_length > 0:
            info_idx = np.where(~self.frozen_bits)[0]
            passed = []
            for p in paths:
                bits = p["B"][:, n].astype(int)[info_idx]
                if crc_check(bits, self.crc_length):
                    passed.append(p)
            if passed:
                paths = passed

        best = min(paths, key=lambda p: p["pm"])
        return best["B"][:, n].astype(int), best["pm"]
