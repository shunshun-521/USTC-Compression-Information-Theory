"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np

from decoder_sc import (
    _active_bit_level,
    _active_llr_level,
    _bit_reversed,
    _lower_llr,
    _update_bits,
    _update_llrs,
    precompute_sc_indices,
)


def _crc_poly(crc_length):
    if crc_length == 8:
        return 0x07
    if crc_length == 16:
        return 0x8005
    raise ValueError("crc_length must be 8 or 16")


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 并附加到信息比特后"""
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
    crc_bits = np.array([(reg >> i) & 1 for i in range(crc_length - 1, -1, -1)], dtype=np.int8)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC"""
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


class _Path:
    __slots__ = ("pm", "B", "L")

    def __init__(self, N, n):
        self.pm = 0.0
        self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.B = np.zeros((N, n + 1), dtype=np.int8)


class SCLDecoder:
    """SCL 译码器（Lazy Copy via array copy on split）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.phase_order, _, _ = precompute_sc_indices(N)
        self.info_indices = np.where(~self.frozen_bits)[0]

    def _pm_penalty(self, llr, u_bit):
        u_from_llr = 0 if llr >= 0 else 1
        return 0.0 if u_bit == u_from_llr else abs(llr)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N, n = self.N, self.n

        paths = [_Path(N, n)]
        paths[0].L[:, 0] = llr_ch

        for l in self.phase_order:
            new_paths = []
            for path in paths:
                _update_llrs(path.L, path.B, l, n)
                llr_leaf = path.L[l, n]

                if self.frozen_bits[l]:
                    path.pm += self._pm_penalty(llr_leaf, 0)
                    path.B[l, n] = 0
                    _update_bits(path.B, l, n, N)
                    new_paths.append(path)
                else:
                    for u_bit in (0, 1):
                        child = _Path(N, n)
                        child.pm = path.pm + self._pm_penalty(llr_leaf, u_bit)
                        child.L = path.L.copy()
                        child.B = path.B.copy()
                        child.B[l, n] = u_bit
                        _update_bits(child.B, l, n, N)
                        new_paths.append(child)

            new_paths.sort(key=lambda p: p.pm)
            paths = new_paths[: self.list_size]

        best = paths[0]
        u_hat = best.B[:, n].astype(np.int8)

        if self.crc_length > 0:
            crc_ok = []
            for p in paths:
                bits = p.B[:, n].astype(np.int8)[self.info_indices]
                if crc_check(bits, self.crc_length):
                    crc_ok.append(p)
            if crc_ok:
                best = min(crc_ok, key=lambda p: p.pm)
                u_hat = best.B[:, n].astype(np.int8)

        return u_hat, best.pm
