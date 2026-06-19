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
    _update_bits,
    _update_llrs,
    f_operation,
    g_operation,
)


def _crc_poly(crc_length):
    if crc_length == 8:
        return 0x07
    if crc_length == 16:
        return 0x8005
    raise ValueError("crc_length must be 8 or 16")


def crc_encode(info_bits, crc_length=8):
    """
    计算 CRC 校验位并附加到信息比特后。
  CRC-8: 0x07; CRC-16: 0x8005（MSB 先行）
    """
    info_bits = np.asarray(info_bits, dtype=int)
    poly = _crc_poly(crc_length)
    reg = 0
    for bit in info_bits:
        reg ^= int(bit) << (crc_length - 1)
        for _ in range(8 if crc_length == 8 else 1):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)
    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits 末尾 CRC 是否正确"""
    bits = np.asarray(bits, dtype=int)
    expected = crc_encode(bits[:-crc_length], crc_length)
    return np.array_equal(bits[-crc_length:], expected)


class _Path:
  __slots__ = ("L", "B", "pm")

  def __init__(self, N, n, llr_ch):
    self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
    self.B = np.full((N, n + 1), np.nan)
    self.L[:, 0] = llr_ch
    self.pm = 0.0


class SCLDecoder:
    """SCL 译码器（Lazy Copy：路径分裂时复制 LLR/比特数组）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]

    def _path_metric_update(self, pm, llr, bit):
        penalty = 0.0 if (llr >= 0 and bit == 0) or (llr < 0 and bit == 1) else abs(llr)
        return pm + penalty

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [_Path(self.N, self.n, llr_ch)]

        for i in range(self.N):
            l = _bit_reversed(i, self.n)
            new_paths = []

            for path in paths:
                _update_llrs(path.L, path.B, l, self.n, self.N)
                llr = path.L[l, self.n]

                if self.frozen_bits[l]:
                    bit = 0
                    pm = self._path_metric_update(path.pm, llr, bit)
                    path.B[l, self.n] = bit
                    _update_bits(path.B, l, self.n, self.N)
                    new_paths.append(_PathState(path.L, path.B, pm))
                else:
                    for bit in (0, 1):
                        Lc = path.L.copy()
                        Bc = path.B.copy()
                        pm = self._path_metric_update(path.pm, llr, bit)
                        Bc[l, self.n] = bit
                        _update_bits(Bc, l, self.n, self.N)
                        new_paths.append(_PathState(Lc, Bc, pm))

            new_paths.sort(key=lambda p: p.pm)
            paths = [
                _PathState(p.L, p.B, p.pm) for p in new_paths[: self.list_size]
            ]

        candidates = []
        for p in paths:
            u_hat = p.B[:, self.n].astype(int)
            candidates.append((p.pm, u_hat))

        if self.crc_length > 0:
            valid = [
                (pm, u)
                for pm, u in candidates
                if crc_check(u[self.info_indices], self.crc_length)
            ]
            if valid:
                valid.sort(key=lambda x: x[0])
                return valid[0][1], valid[0][0]

        candidates.sort(key=lambda x: x[0])
        return candidates[0][1], candidates[0][0]


class _PathState:
    __slots__ = ("L", "B", "pm")

    def __init__(self, L, B, pm):
        self.L = L
        self.B = B
        self.pm = pm
