"""
极化码 SCL（串行抵消列表）译码器，支持 CRC 辅助 CA-SCL
"""
import numpy as np
from decoder_sc import (
    _PermutedSCD,
    _permute_llr_to_decoder_order,
    _bit_reversed_index,
)


def _crc_remainder(bits, poly, crc_len):
    reg = 0
    mask = (1 << crc_len) - 1
    poly &= mask
    msb = 1 << (crc_len - 1)
    for b in bits:
        reg ^= int(b) << (crc_len - 1)
        for _ in range(crc_len):
            if reg & msb:
                reg = ((reg << 1) ^ poly) & mask
            else:
                reg = (reg << 1) & mask
    return reg


def crc_encode(info_bits, crc_length=8):
    info_bits = np.asarray(info_bits, dtype=np.int8)
    if crc_length == 8:
        poly = 0x07
    elif crc_length == 16:
        poly = 0x8005
    else:
        raise ValueError("crc_length must be 8 or 16")
    rem = _crc_remainder(info_bits, poly, crc_length)
    crc_bits = np.array([(rem >> i) & 1 for i in range(crc_length - 1, -1, -1)], dtype=np.int8)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=np.int8)
    if crc_length == 8:
        poly = 0x07
    elif crc_length == 16:
        poly = 0x8005
    else:
        raise ValueError("crc_length must be 8 or 16")
    return _crc_remainder(bits, poly, crc_length) == 0


class SCLDecoder:
    """SCL 译码器"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]

    @staticmethod
    def _pm_add(pm, llr, u):
        v = 0 if llr >= 0 else 1
        return pm + (0.0 if u == v else abs(llr))

    def _clone_path(self, path):
        scd = _PermutedSCD.__new__(_PermutedSCD)
        scd.N, scd.n, scd.frozen = path["scd"].N, path["scd"].n, path["scd"].frozen
        scd.L = path["scd"].L.copy()
        scd.B = path["scd"].B.copy()
        return {"scd": scd, "pm": path["pm"], "u": path["u"].copy()}

    def decode(self, llr_ch):
        llr_internal = _permute_llr_to_decoder_order(llr_ch)
        scd0 = _PermutedSCD(self.N, llr_internal, self.frozen_set)
        paths = [{"scd": scd0, "pm": 0.0, "u": np.zeros(self.N, dtype=np.int8)}]

        for i in range(self.N):
            l = _bit_reversed_index(i, self.n)
            candidates = []
            for path in paths:
                scd = path["scd"]
                scd._update_llrs(l)
                llr_l = scd.L[l, scd.n]
                if l in self.frozen_set:
                    scd.B[l, scd.n] = 0
                    path["u"][l] = 0
                    path["pm"] = self._pm_add(path["pm"], llr_l, 0)
                    scd._update_bits(l)
                    candidates.append(path)
                else:
                    for u in (0, 1):
                        p = self._clone_path(path)
                        p["pm"] = self._pm_add(path["pm"], llr_l, u)
                        p["u"][l] = u
                        p["scd"].B[l, p["scd"].n] = u
                        p["scd"]._update_bits(l)
                        candidates.append(p)
            candidates.sort(key=lambda p: p["pm"])
            paths = candidates[: self.list_size]

        if self.crc_length > 0:
            valid = [p for p in paths if self._crc_valid(p["u"])]
            if valid:
                paths = valid
        best = min(paths, key=lambda p: p["pm"])
        return best["u"].astype(int), float(best["pm"])

    def _crc_valid(self, u_hat):
        info = u_hat[self.info_indices]
        if len(info) <= self.crc_length:
            return False
        k_info = len(info) - self.crc_length
        payload = info[:k_info]
        full = np.concatenate([payload, info[k_info:]])
        return crc_check(full, self.crc_length)
