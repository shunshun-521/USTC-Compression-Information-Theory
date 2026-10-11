"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np
from encoder import bit_reversal_permutation
from decoder_sc import (
    _bit_reversed,
    _update_llrs,
    _update_bits,
)


_CRC8_POLY = 0x07
_CRC16_POLY = 0x8005


def crc_encode(info_bits, crc_length=8):
    info_bits = np.asarray(info_bits, dtype=np.int8).ravel()
    poly = _CRC8_POLY if crc_length == 8 else _CRC16_POLY
    reg = 0
    mask = (1 << crc_length) - 1
    for bit in info_bits:
        reg ^= int(bit) << (crc_length - 1)
        if reg & (1 << (crc_length - 1)):
            reg = ((reg << 1) ^ poly) & mask
        else:
            reg = (reg << 1) & mask
    for _ in range(crc_length):
        if reg & (1 << (crc_length - 1)):
            reg = ((reg << 1) ^ poly) & mask
        else:
            reg = (reg << 1) & mask
    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=np.int8
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=np.int8).ravel()
    expected = crc_encode(bits[:-crc_length], crc_length)
    return np.array_equal(expected[-crc_length:], bits[-crc_length:])


class SCLDecoder:
    """SCL 译码器"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]

    def _new_state(self, llr_ch):
        L = np.zeros((self.N, self.n + 1), dtype=np.float64)
        B = np.zeros((self.N, self.n + 1), dtype=np.int8)
        L[:, 0] = llr_ch
        return L, B

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        llr_ch = llr_ch[bit_reversal_permutation(self.N)]
        L, B = self._new_state(llr_ch)
        paths = [{"pm": 0.0, "L": L, "B": B}]

        for phi in range(self.N):
            l = _bit_reversed(phi, self.n)
            new_paths = []
            for path in paths:
                _update_llrs(path["L"], path["B"], l, self.n)
                llr0 = path["L"][l, self.n]
                if self.frozen_bits[l]:
                    Lc, Bc = path["L"].copy(), path["B"].copy()
                    pm = path["pm"] + (abs(llr0) if llr0 < 0 else 0.0)
                    Bc[l, self.n] = 0
                    _update_bits(Bc, l, self.n, self.N)
                    new_paths.append({"pm": pm, "L": Lc, "B": Bc})
                else:
                    hard = 0 if llr0 >= 0 else 1
                    for u in (0, 1):
                        Lc, Bc = path["L"].copy(), path["B"].copy()
                        pm = path["pm"] + (abs(llr0) if u != hard else 0.0)
                        Bc[l, self.n] = u
                        _update_bits(Bc, l, self.n, self.N)
                        new_paths.append({"pm": pm, "L": Lc, "B": Bc})

            new_paths.sort(key=lambda p: p["pm"])
            paths = new_paths[: self.list_size]

        crc_ok = []
        for p in paths:
            if self.crc_length > 0:
                crc_ok.append(crc_check(p["B"][:, self.n][self.info_indices], self.crc_length))
            else:
                crc_ok.append(True)

        if any(crc_ok):
            best = min([p for p, ok in zip(paths, crc_ok) if ok], key=lambda p: p["pm"])
        else:
            best = paths[0]

        return best["B"][:, self.n].astype(int), best["pm"]
