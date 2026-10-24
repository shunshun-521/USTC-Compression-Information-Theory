"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np

from decoder_sc import (
    _bit_reversed,
    _new_state,
    _update_bits,
    _update_llrs,
    sc_decode,
)

CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def crc_encode(info_bits, crc_length=8):
    """信息比特 + CRC 校验位"""
    info_bits = np.asarray(info_bits, dtype=np.int8)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    reg = 0
    for b in info_bits:
        reg ^= int(b) << (crc_length - 1)
        for _ in range(crc_length):
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
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    reg = 0
    for b in bits:
        reg ^= int(b) << (crc_length - 1)
        for _ in range(crc_length):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)
    return reg == 0


class _Path:
    __slots__ = ("L", "B", "pm", "u_hat")

    def __init__(self, L, B, pm=0.0):
        self.L = L
        self.B = B
        self.pm = pm
        self.u_hat = np.zeros(L.shape[0], dtype=np.int8)

    def clone(self):
        return _Path(self.L.copy(), self.B.copy(), self.pm)


class SCLDecoder:
    """SCL 译码器（路径分裂时复制 L/B 数组）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]

    @staticmethod
    def _pm_penalty(llr, u_bit):
        hard = 0 if llr >= 0 else 1
        return 0.0 if u_bit == hard else abs(llr)

    def decode(self, llr_ch):
        llr = np.asarray(llr_ch, dtype=np.float64)
        if self.list_size == 1:
            u = sc_decode(llr, self.frozen_bits)
            return u, 0.0

        L0, B0 = _new_state(self.N, self.n, llr)
        paths = [_Path(L0, B0)]

        for i in range(self.N):
            l = _bit_reversed(i, self.n)
            new_paths = []
            for path in paths:
                _update_llrs(path.L, path.B, l, self.n, self.N)
                llr_l = path.L[l, self.n]
                if self.frozen_bits[l]:
                    p = path.clone()
                    p.pm += self._pm_penalty(llr_l, 0)
                    p.B[l, self.n] = 0
                    p.u_hat[l] = 0
                    _update_bits(p.B, l, self.n, self.N)
                    new_paths.append(p)
                else:
                    for u_bit in (0, 1):
                        p = path.clone()
                        p.pm += self._pm_penalty(llr_l, u_bit)
                        p.B[l, self.n] = u_bit
                        p.u_hat[l] = u_bit
                        _update_bits(p.B, l, self.n, self.N)
                        new_paths.append(p)
            new_paths.sort(key=lambda p: p.pm)
            paths = new_paths[: self.list_size]

        crc_ok = []
        for p in paths:
            u = p.B[:, self.n].astype(int)
            if self.crc_length > 0:
                if crc_check(u[self.info_indices], self.crc_length):
                    crc_ok.append((p.pm, u))
            else:
                crc_ok.append((p.pm, u))

        if not crc_ok:
            best = min(paths, key=lambda p: p.pm)
            return best.B[:, self.n].astype(int), best.pm
        pm, u = min(crc_ok, key=lambda t: t[0])
        return u, pm
