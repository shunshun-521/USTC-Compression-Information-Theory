"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from encoder import bit_reversal_permutation
from scd_core import SCD, bit_reversed


def _crc_remainder(bits, poly, crc_length):
    reg = 0
    mask = (1 << crc_length) - 1
    top = 1 << (crc_length - 1)
    for bit in bits:
        reg ^= int(bit) << (crc_length - 1)
        if reg & top:
            reg = ((reg << 1) ^ poly) & mask
        else:
            reg = (reg << 1) & mask
    return reg


def crc_encode(info_bits, crc_length=8):
    poly = 0x07 if crc_length == 8 else 0x8005
    info_bits = np.asarray(info_bits, dtype=np.int8)
    rem = _crc_remainder(info_bits, poly, crc_length)
    crc = np.array(
        [(rem >> i) & 1 for i in range(crc_length - 1, -1, -1)], dtype=np.int8
    )
    return np.concatenate([info_bits, crc])


def crc_check(bits, crc_length=8):
    poly = 0x07 if crc_length == 8 else 0x8005
    return _crc_remainder(np.asarray(bits, dtype=np.int8), poly, crc_length) == 0


class _Path:
    __slots__ = ("scd", "pm", "step")

    def __init__(self, scd, pm=0.0, step=0):
        self.scd = scd
        self.pm = pm
        self.step = step


class SCLDecoder:
    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        from decoder_sc import _frozen_for_scd

        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_scd = _frozen_for_scd(self.frozen_bits, N)
        self.frozen_idx = set(np.where(self.frozen_scd)[0])
        self.L = list_size
        self.crc_length = crc_length
        self.rev = bit_reversal_permutation(N)

    @staticmethod
    def _penalty(llr_val, u):
        if (llr_val >= 0 and u == 0) or (llr_val < 0 and u == 1):
            return 0.0
        return abs(llr_val)

    def _new_scd(self, llr_ch):
        return SCD(self.N, self.n, llr_ch, self.frozen_idx)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [_Path(self._new_scd(llr_ch), 0.0, 0)]

        for i in range(self.N):
            l = bit_reversed(i, self.n)
            expanded = []
            for path in paths:
                scd = path.scd
                scd.update_llrs(l)
                llr_l = scd.L[l, self.n]
                if l in self.frozen_idx:
                    cand = [(0, path.pm + self._penalty(llr_l, 0))]
                else:
                    cand = [
                        (0, path.pm + self._penalty(llr_l, 0)),
                        (1, path.pm + self._penalty(llr_l, 1)),
                    ]
                for u, pm in cand:
                    scd2 = self._new_scd(llr_ch)
                    scd2.L = path.scd.L.copy()
                    scd2.B = path.scd.B.copy()
                    scd2.update_llrs(l)
                    scd2.B[l, self.n] = u
                    scd2.update_bits(l)
                    expanded.append(_Path(scd2, pm, i + 1))

            expanded.sort(key=lambda p: p.pm)
            paths = expanded[: self.L]

        best = min(paths, key=lambda p: p.pm)
        u_hat = best.scd.B[:, self.n].astype(np.int8)[self.rev]

        if self.crc_length > 0:
            info_idx = np.where(~self.frozen_bits)[0]
            valid = []
            for p in paths:
                u_cand = p.scd.B[:, self.n].astype(np.int8)[self.rev]
                if crc_check(u_cand[info_idx], self.crc_length):
                    valid.append((p.pm, u_cand))
            if valid:
                valid.sort(key=lambda x: x[0])
                return valid[0][1], valid[0][0]

        return u_hat, best.pm
