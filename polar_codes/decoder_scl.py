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
    _reorder_channel_llrs,
    _update_bits,
    _update_llrs,
    f_operation,
    g_operation,
)

CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, crc_length):
    reg = 0
    mask = (1 << crc_length) - 1
    for bit in bits:
        reg ^= int(bit) << (crc_length - 1)
        if reg & (1 << (crc_length - 1)):
            reg = ((reg << 1) ^ poly) & mask
        else:
            reg = (reg << 1) & mask
    return reg


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    remainder = _crc_remainder(info_bits, poly, crc_length)
    crc_bits = np.array(
        [(remainder >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=int,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC 是否正确。"""
    if crc_length == 0:
        return True
    bits = np.asarray(bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    return _crc_remainder(bits, poly, crc_length) == 0


class Path:
    __slots__ = ('pm', 'L', 'B')

    def __init__(self, N, n, llr_ch):
        self.pm = 0.0
        self.L = np.zeros((N, n + 1))
        self.B = np.zeros((N, n + 1), dtype=int)
        self.L[:, 0] = _reorder_channel_llrs(llr_ch, N)


class SCLDecoder:
    """SCL 译码器（Lazy Copy 优化）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length

    def _copy_path(self, src):
        dst = Path(self.N, self.n, np.zeros(self.N))
        dst.pm = src.pm
        dst.L = src.L.copy()
        dst.B = src.B.copy()
        return dst

    @staticmethod
    def _pm_penalty(llr, bit):
        hard = 0 if llr >= 0 else 1
        return 0.0 if bit == hard else abs(llr)

    def decode(self, llr_ch):
        """主译码函数。"""
        paths = [Path(self.N, self.n, llr_ch)]

        for i in range(self.N):
            l = _bit_reversed(i, self.n)
            candidates = []

            for path in paths:
                _update_llrs(path.L, path.B, l, self.n, self.N)
                llr = path.L[l, self.n]

                if self.frozen_bits[l]:
                    new_path = self._copy_path(path)
                    new_path.pm += self._pm_penalty(llr, 0)
                    new_path.B[l, self.n] = 0
                    _update_bits(new_path.B, l, self.n, self.N)
                    candidates.append(new_path)
                else:
                    for bit in (0, 1):
                        new_path = self._copy_path(path)
                        new_path.pm += self._pm_penalty(llr, bit)
                        new_path.B[l, self.n] = bit
                        _update_bits(new_path.B, l, self.n, self.N)
                        candidates.append(new_path)

            candidates.sort(key=lambda p: p.pm)
            paths = candidates[: self.list_size]

        if self.crc_length > 0:
            u_list = [p.B[:, self.n].astype(int) for p in paths]
            crc_pass = [
                p for p, u in zip(paths, u_list)
                if crc_check(u[~self.frozen_bits], self.crc_length)
            ]
            best = min(crc_pass, key=lambda p: p.pm) if crc_pass else paths[0]
        else:
            best = paths[0]

        return best.B[:, self.n].astype(int).copy(), best.pm
