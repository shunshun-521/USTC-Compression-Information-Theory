"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np
from encoder import bit_reversal_permutation
from decoder_sc import (
    _bit_reversed, _update_llrs, _update_bits,
)


CRC_POLYS = {8: 0x07, 16: 0x8005}


def _int_to_bits(val, length):
    return np.array([(val >> (length - 1 - i)) & 1 for i in range(length)], dtype=int)


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    poly = CRC_POLYS[crc_length]
    reg = 0
    for bit in info_bits:
        reg ^= int(bit) << (crc_length - 1)
        for _ in range(crc_length):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)
    crc_bits = _int_to_bits(reg, crc_length)
    return np.concatenate([np.asarray(info_bits, dtype=int), crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC。"""
    expected = crc_encode(bits[:-crc_length], crc_length)
    return np.array_equal(bits[-crc_length:], expected[-crc_length:])


class Path:
    __slots__ = ('L', 'B', 'pm', 'u_hat')

    def __init__(self, N, n):
        self.L = np.zeros((N, n + 1), dtype=np.float64)
        self.B = np.zeros((N, n + 1), dtype=np.int_)
        self.pm = 0.0
        self.u_hat = np.zeros(N, dtype=np.int_)

    def copy(self):
        new_path = Path(self.L.shape[0], self.L.shape[1] - 1)
        new_path.L = self.L.copy()
        new_path.B = self.B.copy()
        new_path.pm = self.pm
        new_path.u_hat = self.u_hat.copy()
        return new_path


class SCLDecoder:
    """SCL 译码器（含 Lazy Copy 优化）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.info_indices = np.where(~self.frozen_bits)[0]

    def _path_metric_penalty(self, llr, bit):
        hard = 0 if llr >= 0 else 1
        return 0.0 if bit == hard else abs(llr)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        llr_ch = llr_ch[bit_reversal_permutation(self.N)]
        paths = [Path(self.N, self.n)]
        paths[0].L[:, 0] = llr_ch.copy()

        for i in range(self.N):
            l = _bit_reversed(i, self.n)
            candidates = []

            for path in paths:
                _update_llrs(path.L, path.B, l, self.n)
                llr = path.L[l, self.n]

                if l in self.frozen_set:
                    new_path = path.copy()
                    penalty = self._path_metric_penalty(llr, 0)
                    new_path.pm += penalty
                    new_path.B[l, self.n] = 0
                    new_path.u_hat[l] = 0
                    _update_bits(new_path.B, l, self.n)
                    candidates.append(new_path)
                else:
                    for bit in (0, 1):
                        new_path = path.copy()
                        new_path.pm += self._path_metric_penalty(llr, bit)
                        new_path.B[l, self.n] = bit
                        new_path.u_hat[l] = bit
                        _update_bits(new_path.B, l, self.n)
                        candidates.append(new_path)

            candidates.sort(key=lambda p: p.pm)
            paths = candidates[: self.list_size]

        if self.crc_length > 0:
            valid = [
                p for p in paths
                if crc_check(p.u_hat[self.info_indices], self.crc_length)
            ]
            chosen = min(valid, key=lambda p: p.pm) if valid else min(paths, key=lambda p: p.pm)
        else:
            chosen = min(paths, key=lambda p: p.pm)

        return chosen.u_hat.copy(), chosen.pm
