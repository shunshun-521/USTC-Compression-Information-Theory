"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from encoder import bit_reversed
from decoder_sc import (
  upper_llr, lower_llr, active_llr_level, active_bit_level,
  _update_llrs, _update_bits,
)

CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, crc_length):
    reg = 0
    for bit in bits:
        reg <<= 1
        reg |= int(bit)
        if reg & (1 << crc_length):
            reg ^= poly
    return reg & ((1 << crc_length) - 1)


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    remainder = _crc_remainder(info_bits, poly, crc_length)
    crc_bits = np.array([(remainder >> (crc_length - 1 - i)) & 1
                         for i in range(crc_length)], dtype=int)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits 的 CRC 是否正确"""
    bits = np.asarray(bits, dtype=int)
    if crc_length == 0:
        return True
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    expected = _crc_remainder(bits[:-crc_length], poly, crc_length)
    received = 0
    for i in range(crc_length):
        received = (received << 1) | int(bits[-(crc_length - i)])
    return expected == received


class Path:
    """单条译码路径"""
    __slots__ = ('L', 'B', 'pm', 'frozen_set')

    def __init__(self, N, n, llr_ch, frozen_set):
        self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.B = np.full((N, n + 1), np.nan)
        self.L[:, 0] = llr_ch.copy()
        self.pm = 0.0
        self.frozen_set = frozen_set


class SCLDecoder:
    """SCL 译码器"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.info_indices = np.where(~self.frozen_bits)[0]

    def _copy_path(self, path):
        new = Path(self.N, self.n, np.zeros(self.N), self.frozen_set)
        new.L = path.L.copy()
        new.B = path.B.copy()
        new.pm = path.pm
        return new

    def _pm_penalty(self, llr_val, u_val):
        hard = 0 if llr_val >= 0 else 1
        return 0.0 if u_val == hard else abs(llr_val)

    def _check_crc(self, u_hat):
        if self.crc_length == 0:
            return True
        return crc_check(u_hat[self.info_indices], self.crc_length)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N, n = self.N, self.n
        paths = [Path(N, n, llr_ch, self.frozen_set)]

        for i in range(N):
            l = bit_reversed(i, n)
            candidates = []

            for path in paths:
                _update_llrs(path.L, path.B, l, n, N)
                llr_val = path.L[l, n]

                if l in self.frozen_set:
                    new_path = self._copy_path(path)
                    new_path.pm += self._pm_penalty(llr_val, 0)
                    new_path.B[l, n] = 0
                    _update_bits(new_path.B, l, n, N)
                    candidates.append(new_path)
                else:
                    for u_val in (0, 1):
                        new_path = self._copy_path(path)
                        new_path.pm += self._pm_penalty(llr_val, u_val)
                        new_path.B[l, n] = u_val
                        _update_bits(new_path.B, l, n, N)
                        candidates.append(new_path)

            candidates.sort(key=lambda p: p.pm)
            paths = candidates[:self.list_size]

        crc_passed = [p for p in paths if self._check_crc(p.B[:, n].astype(int))]
        best = min(crc_passed if crc_passed else paths, key=lambda p: p.pm)
        return best.B[:, n].astype(int), best.pm
