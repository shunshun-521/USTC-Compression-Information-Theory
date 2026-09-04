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
)
from encoder import bit_reversal_permutation

CRC8_POLY_BITS = [1, 0, 0, 0, 0, 0, 1, 1, 1]
CRC16_POLY_BITS = [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1]


def _crc_poly_encode(msg_bits, poly_bits):
    n = len(poly_bits) - 1
    data = list(np.asarray(msg_bits, dtype=int)) + [0] * n
    for i in range(len(msg_bits)):
        if data[i]:
            for j in range(len(poly_bits)):
                data[i + j] ^= poly_bits[j]
    return list(np.asarray(msg_bits, dtype=int)) + data[len(msg_bits) :]


def _crc_poly_check(bits, poly_bits):
    n = len(poly_bits) - 1
    data = list(np.asarray(bits, dtype=int))
    for i in range(len(bits) - n):
        if data[i]:
            for j in range(len(poly_bits)):
                data[i + j] ^= poly_bits[j]
    return all(b == 0 for b in data[-n:])


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=int)
    if crc_length == 8:
        return np.array(_crc_poly_encode(info_bits, CRC8_POLY_BITS), dtype=int)
    if crc_length == 16:
        return np.array(_crc_poly_encode(info_bits, CRC16_POLY_BITS), dtype=int)
    raise ValueError("crc_length must be 8 or 16")


def crc_check(bits, crc_length=8):
    """检验 CRC。"""
    if crc_length == 8:
        return _crc_poly_check(bits, CRC8_POLY_BITS)
    if crc_length == 16:
        return _crc_poly_check(bits, CRC16_POLY_BITS)
    raise ValueError("crc_length must be 8 or 16")


class Path:
    __slots__ = ("pm", "L", "B", "u_hat")

    def __init__(self, N, n, llr_perm):
        self.pm = 0.0
        self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.B = np.full((N, n + 1), np.nan)
        self.L[:, 0] = llr_perm
        self.u_hat = np.zeros(N, dtype=int)


class SCLDecoder:
    """SCL 译码器（PSC 结构 + Lazy Copy）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.list_size = list_size
        self.crc_length = crc_length
        self.perm = bit_reversal_permutation(N)
        self.info_indices = np.where(~self.frozen_bits)[0]

    def _copy_path(self, path):
        new_path = Path(self.N, self.n, path.L[:, 0].copy())
        new_path.pm = path.pm
        new_path.u_hat = path.u_hat.copy()
        new_path.L = path.L.copy()
        new_path.B = path.B.copy()
        return new_path

    def _pm_penalty(self, llr, bit):
        hard = 0 if llr >= 0 else 1
        return 0.0 if bit == hard else abs(llr)

    def decode(self, llr_ch):
        """主译码函数，返回 u_hat, pm。"""
        llr_perm = np.asarray(llr_ch, dtype=np.float64)[self.perm]
        paths = [Path(self.N, self.n, llr_perm)]

        for i in range(self.N):
            l = _bit_reversed(i, self.n)
            new_paths = []

            for path in paths:
                _update_llrs(path.L, path.B, l, self.n, self.N)
                llr0 = path.L[l, self.n]

                if l in self.frozen_set:
                    path.pm += self._pm_penalty(llr0, 0)
                    path.B[l, self.n] = 0
                    path.u_hat[l] = 0
                    _update_bits(path.B, l, self.n, self.N)
                    new_paths.append(path)
                else:
                    for bit in (0, 1):
                        p = self._copy_path(path)
                        p.pm += self._pm_penalty(llr0, bit)
                        p.B[l, self.n] = bit
                        p.u_hat[l] = bit
                        _update_bits(p.B, l, self.n, self.N)
                        new_paths.append(p)

            new_paths.sort(key=lambda p: p.pm)
            paths = new_paths[: self.list_size]

        candidates = []
        for path in paths:
            if self.crc_length > 0:
                info_bits = path.u_hat[self.info_indices]
                if crc_check(info_bits, self.crc_length):
                    candidates.append(path)
            else:
                candidates.append(path)

        best = min(candidates or paths, key=lambda p: p.pm)
        return best.u_hat.copy(), best.pm
