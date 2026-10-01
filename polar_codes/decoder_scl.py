"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import copy
import numpy as np
from decoder_sc import sc_decode, _PermutedSCD, _bit_reversed, _hard_decision


def crc_encode(info_bits, crc_length=8):
    info_bits = np.asarray(info_bits, dtype=np.int8)
    poly = {8: 0x07, 16: 0x8005}[crc_length]
    reg = 0
    for b in info_bits:
        reg ^= int(b) << (crc_length - 1)
        for _ in range(8):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)
    crc_bits = np.array([(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=np.int8)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=np.int8)
    return np.array_equal(crc_encode(bits[:-crc_length], crc_length), bits)


class SCLDecoder:
    def __init__(self, N, frozen_bits, list_size=4, crc_length=0, info_indices=None):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.L_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.asarray(info_indices, dtype=int) if info_indices is not None else None

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        if self.L_size == 1 and self.crc_length == 0:
            return sc_decode(llr_ch, self.frozen_bits), 0.0

        paths = []
        base = _PermutedSCD(self.N, self.frozen_set)
        base.L[:, 0] = llr_ch
        paths.append({"pm": 0.0, "dec": base})

        for phi in [_bit_reversed(i, self.n) for i in range(self.N)]:
            l = phi
            expanded = []
            for path in paths:
                dec = path["dec"]
                dec._update_llrs(l)
                llr = dec.L[l, self.n]
                bit_opts = [0] if l in self.frozen_set else [0, 1]
                for bit in bit_opts:
                    if l in self.frozen_set:
                        bit = 0
                    child = copy.deepcopy(dec)
                    penalty = 0.0 if bit == _hard_decision(llr) else abs(llr)
                    child.B[l, self.n] = bit
                    child._update_bits(l)
                    expanded.append({"pm": path["pm"] + penalty, "dec": child})

            expanded.sort(key=lambda x: x["pm"])
            paths = expanded[: self.L_size]

        best = min(paths, key=lambda p: p["pm"])
        u_hat = best["dec"].B[:, self.n].astype(np.int8)

        if self.crc_length > 0:
            valid = []
            for p in paths:
                bits = p["dec"].B[:, self.n][self.info_indices] if self.info_indices is not None else p["dec"].B[:, self.n]
                if crc_check(bits, self.crc_length):
                    valid.append(p)
            if valid:
                best = min(valid, key=lambda p: p["pm"])
                u_hat = best["dec"].B[:, self.n].astype(np.int8)

        return u_hat.astype(int), float(best["pm"])
