"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math

import numpy as np

from decoder_sc import (
    _update_bits,
    _update_llrs,
    bit_reversed,
)


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_division(info_bits, poly, crc_length):
    """GF(2) 多项式除法求 CRC 余数"""
    reg = 0
    for bit in info_bits:
        reg = ((reg << 1) | int(bit)) & ((1 << crc_length) - 1)
        if reg & (1 << (crc_length - 1)):
            reg ^= poly
    return reg


def crc_encode(info_bits, crc_length=8):
    """
    计算 CRC 校验位并附加到信息比特后。
    """
    info_bits = np.asarray(info_bits, dtype=int)
    if crc_length == 8:
        poly = CRC8_POLY
    elif crc_length == 16:
        poly = CRC16_POLY
    else:
        raise ValueError("crc_length must be 8 or 16")

    padded = np.concatenate([info_bits, np.zeros(crc_length, dtype=int)])
    remainder = _crc_division(padded, poly, crc_length)
    crc_bits = np.array([(remainder >> i) & 1 for i in range(crc_length - 1, -1, -1)], dtype=int)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """
    检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。
    """
    if crc_length == 0:
        return True
    bits = np.asarray(bits, dtype=int)
    info_bits = bits[:-crc_length]
    expected = crc_encode(info_bits, crc_length)
    return np.array_equal(bits, expected)


class _Path:
    """单条译码路径（Lazy Copy）"""

    __slots__ = ("pm", "L", "B", "parent", "copied")

    def __init__(self, N, n, llr_ch=None, parent=None):
        self.pm = 0.0
        self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.B = np.full((N, n + 1), np.nan)
        if llr_ch is not None:
            self.L[:, 0] = llr_ch
        self.parent = parent
        self.copied = parent is None

    def materialize(self):
        if not self.copied:
            self.L = self.parent.L.copy()
            self.B = self.parent.B.copy()
            self.copied = True

    def fork(self):
        child = _Path(self.L.shape[0], self.L.shape[1] - 1, parent=self)
        child.pm = self.pm
        return child


class SCLDecoder:
    """
    SCL 译码器（含 Lazy Copy 优化，Permuted SCD 结构）。
    """

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.info_indices = np.where(~self.frozen_bits)[0]
        self.decode_order = [bit_reversed(i, self.n) for i in range(N)]

    @staticmethod
    def _path_metric_penalty(llr_val, u_bit):
        hard = 0 if llr_val >= 0 else 1
        return 0.0 if u_bit == hard else abs(llr_val)

    def decode(self, llr_ch):
        """
        主译码函数。
        """
        N, n = self.N, self.n
        llr_ch = llr_ch.astype(np.float64)

        paths = [_Path(N, n, llr_ch)]

        for l in self.decode_order:
            new_paths = []

            for path in paths:
                path.materialize()
                _update_llrs(path.L, path.B, l, n)
                llr_val = path.L[l, n]

                if l in self.frozen_set:
                    path.pm += self._path_metric_penalty(llr_val, 0)
                    path.B[l, n] = 0
                    _update_bits(path.B, l, n, N)
                    new_paths.append(path)
                else:
                    for u_bit in (0, 1):
                        child = _Path(N, n, parent=path)
                        child.materialize()
                        child.pm = path.pm + self._path_metric_penalty(llr_val, u_bit)
                        child.B[l, n] = u_bit
                        _update_bits(child.B, l, n, N)
                        new_paths.append(child)

            new_paths.sort(key=lambda p: p.pm)
            paths = new_paths[: self.list_size]

        if self.crc_length > 0:
            info_bits_all = paths[0].B[:, n][self.info_indices]
            crc_pass = [
                p for p in paths
                if crc_check(p.B[:, n][self.info_indices], self.crc_length)
            ]
            best = min(crc_pass, key=lambda p: p.pm) if crc_pass else paths[0]
        else:
            best = paths[0]

        return best.B[:, n].astype(int), best.pm
