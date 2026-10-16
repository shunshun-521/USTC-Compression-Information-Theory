"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import copy
import numpy as np

from decoder_sc import _SCD, _bit_reversed
from encoder import bit_reversal_permutation

_CRC8_POLY = 0x07
_CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, crc_length):
    mask = (1 << crc_length) - 1
    reg = 0
    for b in bits:
        reg ^= int(b) << (crc_length - 1)
        if reg & (1 << (crc_length - 1)):
            reg = ((reg << 1) ^ poly) & mask
        else:
            reg = (reg << 1) & mask
    return reg


def crc_encode(info_bits, crc_length=8):
    info_bits = np.asarray(info_bits, dtype=int)
    poly = _CRC8_POLY if crc_length == 8 else _CRC16_POLY
    rem = _crc_remainder(info_bits, poly, crc_length)
    crc_bits = np.array([(rem >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=int)
    poly = _CRC8_POLY if crc_length == 8 else _CRC16_POLY
    return _crc_remainder(bits, poly, crc_length) == 0


def _path_metric_update(pm, llr, u_bit):
    if u_bit == 0:
        return pm + (abs(llr) if llr < 0 else 0.0)
    return pm + (abs(llr) if llr >= 0 else 0.0)


class SCLDecoder:
    """SCL 译码器（路径复制）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_set = set(np.where(self.frozen_bits)[0].tolist())
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_positions = np.where(~self.frozen_bits)[0]
    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        br = bit_reversal_permutation(self.N)
        inv = np.zeros(self.N, dtype=int)
        inv[br] = np.arange(self.N)
        llr_internal = llr_ch[inv]

        paths = [{"scd": _SCD(llr_internal, self.frozen_set, self.n), "pm": 0.0}]

        for i in range(self.N):
            l = _bit_reversed(i, self.n)
            new_paths = []
            for path in paths:
                scd = path["scd"]
                scd._update_llrs(l)
                llr_bit = scd.L[l, scd.n]
                if l in self.frozen_set:
                    scd2 = copy.deepcopy(scd)
                    scd2.B[l, scd2.n] = 0
                    scd2._update_bits(l)
                    pm = _path_metric_update(path["pm"], llr_bit, 0)
                    new_paths.append({"scd": scd2, "pm": pm})
                else:
                    for bit in (0, 1):
                        scd2 = copy.deepcopy(scd)
                        scd2.B[l, scd2.n] = bit
                        scd2._update_bits(l)
                        pm = _path_metric_update(path["pm"], llr_bit, bit)
                        new_paths.append({"scd": scd2, "pm": pm})
            new_paths.sort(key=lambda p: p["pm"])
            paths = new_paths[: self.list_size]

        candidates = []
        for path in paths:
            u_hat = path["scd"].B[:, self.n].astype(int)
            u_hat = np.nan_to_num(u_hat, nan=0.0).astype(int)
            u_hat[self.frozen_bits] = 0
            candidates.append((path["pm"], u_hat))

        if self.crc_length > 0:
            valid = [(pm, u) for pm, u in candidates if crc_check(u[self.info_positions], self.crc_length)]
            if valid:
                pm, u_hat = min(valid, key=lambda t: t[0])
                return u_hat, pm

        pm, u_hat = min(candidates, key=lambda t: t[0])
        return u_hat, pm
