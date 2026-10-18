"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL），Permuted SC 顺序
"""
import numpy as np
from encoder import bit_reversal_permutation

from decoder_sc import (
    _bit_reversed,
    _frozen_set_from_mask,
    _update_bits,
    _update_llrs,
)

CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, crc_length):
    reg = 0
    for b in bits:
        reg ^= int(b) << (crc_length - 1)
        if reg & (1 << (crc_length - 1)):
            reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
        else:
            reg = (reg << 1) & ((1 << crc_length) - 1)
    return reg


def crc_encode(info_bits, crc_length=8):
    info_bits = np.asarray(info_bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    rem = _crc_remainder(info_bits, poly, crc_length)
    crc_bits = np.array([(rem >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    return _crc_remainder(bits, poly, crc_length) == 0


class SCLDecoder:
    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits)
        self.frozen_set = _frozen_set_from_mask(frozen_bits)
        self.list_size = int(list_size)
        self.crc_length = int(crc_length)
        self.rev = bit_reversal_permutation(N)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N, n = self.N, self.n

        L0 = np.zeros((N, n + 1), dtype=np.float64)
        B0 = np.zeros((N, n + 1), dtype=np.int8)
        L0[:, 0] = llr_ch[self.rev]
        paths = [{"pm": 0.0, "L": L0, "B": B0}]

        for l in [_bit_reversed(i, n) for i in range(N)]:
            new_paths = []
            for path in paths:
                L = path["L"]
                B = path["B"]
                _update_llrs(L, B, l, n, N)
                llr_bit = float(L[l, n])

                if l in self.frozen_set:
                    pen = abs(llr_bit) if llr_bit < 0 else 0.0
                    Bc = B.copy()
                    Bc[l, n] = 0
                    _update_bits(Bc, l, n, N)
                    new_paths.append({"pm": path["pm"] + pen, "L": L, "B": Bc})
                else:
                    for u_bit in (0, 1):
                        pen = (
                            0.0
                            if (u_bit == 0 and llr_bit >= 0) or (u_bit == 1 and llr_bit < 0)
                            else abs(llr_bit)
                        )
                        Lc = L.copy()
                        Bc = B.copy()
                        Bc[l, n] = u_bit
                        _update_bits(Bc, l, n, N)
                        new_paths.append({"pm": path["pm"] + pen, "L": Lc, "B": Bc})

            new_paths.sort(key=lambda x: x["pm"])
            paths = new_paths[: self.list_size]

        best_crc = None
        best_pm = min(paths, key=lambda x: x["pm"])
        if self.crc_length > 0:
            fb = np.asarray(self.frozen_bits, dtype=bool)
            if fb.dtype != bool:
                fb = fb.astype(bool)
            for p in paths:
                u_hat = p["B"][:, n].astype(int)
                info_bits = u_hat[~fb]
                if crc_check(info_bits, self.crc_length):
                    if best_crc is None or p["pm"] < best_crc["pm"]:
                        best_crc = p
        chosen = best_crc if best_crc is not None else best_pm
        u_hat = chosen["B"][:, n].astype(int)
        return u_hat, chosen["pm"]
