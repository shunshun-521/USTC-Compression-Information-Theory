"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np
from decoder_sc import (
    _prepare_llr,
    _update_bits,
    _update_llrs,
    active_bit_level,
    active_llr_level,
    f_operation,
)
from encoder import bit_reversed


def _crc_poly(crc_length):
    if crc_length == 8:
        return 0x07
    if crc_length == 16:
        return 0x8005
    raise ValueError("crc_length must be 8 or 16")


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=np.int8).ravel()
    poly = _crc_poly(crc_length)
    reg = 0
    for bit in info_bits:
        reg <<= 1
        reg |= int(bit)
        if reg & (1 << crc_length):
            reg ^= poly
    crc_bits = np.zeros(crc_length, dtype=np.int8)
    for i in range(crc_length - 1, -1, -1):
        crc_bits[crc_length - 1 - i] = (reg >> i) & 1
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC 是否正确"""
    bits = np.asarray(bits, dtype=np.int8).ravel()
    expected = crc_encode(bits[:-crc_length], crc_length)
    return np.array_equal(bits, expected)


class _Path:
    __slots__ = ("L", "B", "pm")

    def __init__(self, n, N, llr):
        self.L = np.zeros((N, n + 1), dtype=np.float64)
        self.B = np.zeros((N, n + 1), dtype=np.int8)
        self.L[:, 0] = llr
        self.pm = 0.0


class SCLDecoder:
    """SCL 译码器（Lazy Copy 优化）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool).ravel()
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]
        self.frozen_set = set(np.where(self.frozen_bits)[0])

    def _copy_path(self, path):
        new_path = _Path(self.n, self.N, path.L[:, 0])
        new_path.L = path.L.copy()
        new_path.B = path.B.copy()
        new_path.pm = path.pm
        return new_path

    def decode(self, llr_ch):
        llr = _prepare_llr(llr_ch)
        active = [_Path(self.n, self.N, llr)]

        for phi in range(self.N):
            l = bit_reversed(phi, self.n)
            candidates = []
            for path in active:
                _update_llrs(path.L, path.B, l, self.n, self.N)
                llr_bit = path.L[l, self.n]

                if l in self.frozen_set:
                    new_path = self._copy_path(path)
                    if llr_bit < 0:
                        new_path.pm += abs(llr_bit)
                    new_path.B[l, self.n] = 0
                    _update_bits(new_path.B, l, self.n, self.N)
                    candidates.append(new_path)
                else:
                    for bit in (0, 1):
                        new_path = self._copy_path(path)
                        hard = 0 if llr_bit >= 0 else 1
                        if bit != hard:
                            new_path.pm += abs(llr_bit)
                        new_path.B[l, self.n] = bit
                        _update_bits(new_path.B, l, self.n, self.N)
                        candidates.append(new_path)

            candidates.sort(key=lambda p: p.pm)
            active = candidates[: self.list_size]

        if self.crc_length > 0:
            valid = []
            for path in active:
                u_hat = path.B[:, self.n].astype(np.int8)
                info_bits = u_hat[self.info_indices]
                if crc_check(info_bits, self.crc_length):
                    valid.append(path)
            if valid:
                active = valid

        best = min(active, key=lambda p: p.pm)
        u_hat = best.B[:, self.n].astype(np.int8)
        return u_hat, best.pm
