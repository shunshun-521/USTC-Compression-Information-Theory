"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from decoder_sc import (
    _active_bit_level,
    _active_llr_level,
    _bit_reversed,
    _lower_llr,
    _update_bits,
    _update_llrs,
    _upper_llr,
    sc_decode,
)

_CRC8_POLY = 0x07
_CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, reg_bits):
    reg = 0
    mask = (1 << reg_bits) - 1
    for b in bits:
        reg ^= int(b) << (reg_bits - 1)
        if reg & (1 << (reg_bits - 1)):
            reg = ((reg << 1) ^ poly) & mask
        else:
            reg = (reg << 1) & mask
    return reg


def crc_encode(info_bits, crc_length=8):
    """CRC 编码。"""
    info_bits = np.asarray(info_bits, dtype=int)
    poly = _CRC8_POLY if crc_length == 8 else _CRC16_POLY
    rem = _crc_remainder(info_bits, poly, crc_length)
    crc_bits = np.array([(rem >> i) & 1 for i in range(crc_length - 1, -1, -1)], dtype=int)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """CRC 校验。"""
    bits = np.asarray(bits, dtype=int)
    poly = _CRC8_POLY if crc_length == 8 else _CRC16_POLY
    return _crc_remainder(bits, poly, crc_length) == 0


def _pm_update(pm, llr, u_bit):
    hard = 0 if llr >= 0 else 1
    if u_bit != hard:
        return pm + abs(llr)
    return pm


class SCLDecoder:
    """SCL 译码器（多路径 L/B 矩阵，Lazy Copy 通过路径复制实现）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.frozen_set = set(np.where(self.frozen_bits)[0])

    def _new_path(self, llr_ch):
        L = np.full((self.N, self.n + 1), np.nan, dtype=np.float64)
        B = np.full((self.N, self.n + 1), np.nan)
        L[:, 0] = llr_ch
        return L, B

    def decode(self, llr_ch):
        """主译码函数，返回 (u_hat, pm)。"""
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        if self.list_size == 1:
            return sc_decode(llr_ch, self.frozen_bits), 0.0

        N, n = self.N, self.n
        paths = [(0.0, *self._new_path(llr_ch))]

        for l in [_bit_reversed(i, n) for i in range(N)]:
            new_paths = []
            for pm, L, B in paths:
                _update_llrs(L, B, l, n, N)
                llr_val = L[l, n]
                if l in self.frozen_set:
                    B[l, n] = 0
                    _update_bits(B, l, n, N)
                    new_paths.append((_pm_update(pm, llr_val, 0), L, B))
                else:
                    for bit in (0, 1):
                        Lc = L.copy()
                        Bc = B.copy()
                        Bc[l, n] = bit
                        _update_bits(Bc, l, n, N)
                        new_paths.append((_pm_update(pm, llr_val, bit), Lc, Bc))

            new_paths.sort(key=lambda x: x[0])
            paths = new_paths[: self.list_size]

        best_pm, best_L, best_B = paths[0]
        u_hat = best_B[:, n].astype(int)

        if self.crc_length > 0:
            info_pos = np.where(~self.frozen_bits)[0]
            valid = []
            for pm, _, B in paths:
                u = B[:, n].astype(int)
                if crc_check(u[info_pos], self.crc_length):
                    valid.append((pm, u))
            if valid:
                best_pm, u_hat = min(valid, key=lambda x: x[0])

        return u_hat, best_pm
