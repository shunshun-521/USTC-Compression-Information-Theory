"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
import math

from decoder_sc import (
    _bit_reversed,
    _update_llrs,
    _update_bits,
)


def _crc_division(bits, poly_bits):
    data = list(map(int, bits))
    r = len(poly_bits) - 1
    for i in range(len(data) - r):
        if data[i]:
            for j in range(len(poly_bits)):
                data[i + j] ^= poly_bits[j]
    return data[-r:]


def crc_encode(info_bits, crc_length=8):
    """
    计算 CRC 校验位并附加到信息比特后。
    CRC-8: 0x07 (x^8+x^2+x+1); CRC-16: 0x8005
    """
    info_bits = np.asarray(info_bits, dtype=int)
    poly8 = [1, 0, 0, 0, 0, 0, 1, 1, 1]
    poly16 = [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1]
    poly = poly16 if crc_length == 16 else poly8
    rem = _crc_division(
        np.concatenate([info_bits, np.zeros(crc_length, dtype=int)]), poly
    )
    return np.concatenate([info_bits, np.array(rem, dtype=int)])


def crc_check(bits, crc_length=8):
    """检验 bits 末尾 CRC 是否正确。"""
    bits = np.asarray(bits, dtype=int)
    poly8 = [1, 0, 0, 0, 0, 0, 1, 1, 1]
    poly16 = [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1]
    poly = poly16 if crc_length == 16 else poly8
    rem = _crc_division(bits, poly)
    return all(x == 0 for x in rem)


class _Path:
    __slots__ = ("L", "B", "pm")

    def __init__(self, N, n):
        self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.B = np.zeros((N, n + 1), dtype=int)
        self.pm = 0.0

    def copy_state(self):
        child = _Path(self.L.shape[0], self.L.shape[1] - 1)
        child.L = self.L.copy()
        child.B = self.B.copy()
        child.pm = self.pm
        return child


class SCLDecoder:
    """SCL 译码器。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.list_size = list_size
        self.crc_length = crc_length

    def decode(self, llr_ch):
        """返回：(u_hat, pm)"""
        from encoder import bit_reversal_permutation

        rev = bit_reversal_permutation(self.N)
        llr_ch = np.asarray(llr_ch, dtype=np.float64)[rev]

        paths = [_Path(self.N, self.n)]
        paths[0].L[:, 0] = llr_ch

        for l in [_bit_reversed(i, self.n) for i in range(self.N)]:
            new_paths = []
            for path in paths:
                _update_llrs(l, path.L, path.B, self.n)
                llr = path.L[l, self.n]

                if l in self.frozen_set:
                    penalty = abs(llr) if llr < 0 else 0.0
                    child = path.copy_state()
                    child.pm += penalty
                    child.B[l, self.n] = 0
                    _update_bits(l, child.B, self.n)
                    new_paths.append(child)
                else:
                    for bit in (0, 1):
                        child = path.copy_state()
                        if bit == 0 and llr < 0:
                            child.pm += abs(llr)
                        elif bit == 1 and llr >= 0:
                            child.pm += abs(llr)
                        child.B[l, self.n] = bit
                        _update_bits(l, child.B, self.n)
                        new_paths.append(child)

            new_paths.sort(key=lambda p: p.pm)
            paths = new_paths[:self.list_size]

        best = self._select_path(paths)
        return best.B[:, self.n].copy(), best.pm

    def _select_path(self, paths):
        if self.crc_length > 0:
            crc_ok = []
            for p in paths:
                u = p.B[:, self.n]
                info_bits = u[~self.frozen_bits]
                if crc_check(info_bits, self.crc_length):
                    crc_ok.append(p)
            if crc_ok:
                return min(crc_ok, key=lambda p: p.pm)
        return min(paths, key=lambda p: p.pm)
