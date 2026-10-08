"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np
from decoder_sc import bit_reversed, sc_decode, _load_reference_scd

_CRC8_POLY = 0x07
_CRC16_POLY = 0x8005


def crc_encode(info_bits, crc_length=8):
    info_bits = np.asarray(info_bits, dtype=int).ravel()
    if crc_length == 8:
        poly = _CRC8_POLY
        rem = 0
        for b in info_bits:
            rem ^= int(b) << 7
            for _ in range(8):
                rem = ((rem << 1) ^ poly) & 0xFF if rem & 0x80 else (rem << 1) & 0xFF
        crc_bits = np.array([(rem >> (7 - i)) & 1 for i in range(8)], dtype=int)
    elif crc_length == 16:
        poly = _CRC16_POLY
        rem = 0
        for b in info_bits:
            rem ^= int(b) << 15
            for _ in range(16):
                rem = ((rem << 1) ^ poly) & 0xFFFF if rem & 0x8000 else (rem << 1) & 0xFFFF
        crc_bits = np.array([(rem >> (15 - i)) & 1 for i in range(16)], dtype=int)
    else:
        raise ValueError("crc_length must be 8 or 16")
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=int).ravel()
    return np.array_equal(bits, crc_encode(bits[:-crc_length], crc_length))


class _Path:
    __slots__ = ("scd", "pm", "u")

    def __init__(self, scd, pm=0.0, N=None):
        self.scd = scd
        self.pm = pm
        self.u = np.zeros(N if N is not None else scd.myPC.N, dtype=int)

    def clone(self):
        SCD = _load_reference_scd()
        scd = SCD(self.scd.myPC)
        scd.L = self.scd.L.copy()
        scd.B = self.scd.B.copy()
        child = _Path(scd, self.pm, self.scd.myPC.N)
        child.u = self.u.copy()
        return child


class SCLDecoder:
    """SCL 译码器（基于 vendor SCD 的路径分裂）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.L = list_size
        self.crc_length = crc_length
        self.info_idx = np.where(~self.frozen_bits)[0]
        self._SCD = _load_reference_scd()

    @staticmethod
    def _pm_add(pm, llr0, u_bit):
        pred = 0 if llr0 >= 0 else 1
        if u_bit != pred:
            pm += abs(llr0)
        return pm

    def _new_scd(self, llr_ch):
        class _PC:
            pass

        pc = _PC()
        pc.N = self.N
        pc.n = self.n
        pc.frozen = np.where(self.frozen_bits)[0]
        pc.likelihoods = llr_ch
        return self._SCD(pc)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        if self.L == 1 and self.crc_length == 0:
            return sc_decode(llr_ch, self.frozen_bits), 0.0

        paths = [_Path(self._new_scd(llr_ch))]

        for l in [bit_reversed(i, self.n) for i in range(self.N)]:
            new_paths = []
            for path in paths:
                path.scd.update_llrs(l)
                llr0 = path.scd.L[l, self.n]
                if l in self.frozen_set:
                    path.pm = self._pm_add(path.pm, llr0, 0)
                    path.u[l] = 0
                    path.scd.B[l, self.n] = 0
                    path.scd.update_bits(l)
                    new_paths.append(path)
                else:
                    for u_bit in (0, 1):
                        child = path.clone()
                        child.pm = self._pm_add(child.pm, llr0, u_bit)
                        child.u[l] = u_bit
                        child.scd.B[l, self.n] = u_bit
                        child.scd.update_bits(l)
                        new_paths.append(child)
            new_paths.sort(key=lambda p: p.pm)
            paths = new_paths[: self.L]

        if self.crc_length > 0:
            valid = [p for p in paths if crc_check(p.u[self.info_idx], self.crc_length)]
            if valid:
                paths = valid

        best = min(paths, key=lambda p: p.pm)
        return best.u.copy(), best.pm
