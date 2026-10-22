"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from decoder_sc import (
    _active_bit_level,
    _active_llr_level,
    _bit_reversed,
    _llr_to_decoder_order,
    _lower_llr,
    _upper_llr,
)
from encoder import bit_reversal_permutation


_CRC8_GEN = 0x107   # x^8 + x^2 + x + 1
_CRC16_GEN = 0x11021  # CRC-16-IBM


def _bits_to_poly(bits):
    p = 0
    for b in bits:
        p = (p << 1) | int(b)
    return p


def _poly_remainder(bits, crc_length, gen):
    p = _bits_to_poly(bits)
    deg = crc_length
    while p.bit_length() >= deg + 1:
        shift = p.bit_length() - (deg + 1)
        p ^= gen << shift
    return p


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=int).ravel()
    gen = _CRC8_GEN if crc_length == 8 else _CRC16_GEN
    rem = _poly_remainder(np.concatenate([info_bits, np.zeros(crc_length, dtype=int)]), crc_length, gen)
    crc_bits = np.array([(rem >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC。"""
    bits = np.asarray(bits, dtype=int).ravel()
    if len(bits) < crc_length:
        return False
    gen = _CRC8_GEN if crc_length == 8 else _CRC16_GEN
    return _poly_remainder(bits, crc_length, gen) == 0


class _Path:
    __slots__ = ("L", "B", "pm", "u")

    def __init__(self, N, n, llr_ch):
        self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.B = np.full((N, n + 1), np.nan)
        self.L[:, 0] = llr_ch
        self.pm = 0.0
        self.u = np.zeros(N, dtype=int)

    def copy(self):
        p = _Path.__new__(_Path)
        p.L = self.L.copy()
        p.B = self.B.copy()
        p.pm = self.pm
        p.u = self.u.copy()
        return p


class SCLDecoder:
    """SCL 译码器（路径复制实现，列表规模适中时足够高效）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]

    def _update_llrs(self, path, l):
        for s in range(self.n - _active_llr_level(l, self.n), self.n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    path.L[j, s + 1] = _upper_llr(path.L[j, s], path.L[j + branch_size, s])
                else:
                    path.L[j, s + 1] = _lower_llr(
                        path.L[j, s], path.L[j - branch_size, s], path.B[j - branch_size, s + 1]
                    )

    def _update_bits(self, path, l):
        if l < self.N / 2:
            return
        for s in range(self.n, self.n - _active_bit_level(l, self.n), -1):
            block_size = 2 ** s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    path.B[j - branch_size, s - 1] = int(path.B[j, s]) ^ int(
                        path.B[j - branch_size, s]
                    )
                    path.B[j, s - 1] = path.B[j, s]

    def _pm_penalty(self, llr, u_bit):
        """与 LLR 不一致时加 |LLR|。"""
        hard = 0 if llr >= 0 else 1
        return 0.0 if u_bit == hard else abs(llr)

    def decode(self, llr_ch):
        llr_dec = _llr_to_decoder_order(llr_ch)
        paths = [_Path(self.N, self.n, llr_dec)]

        for i in range(self.N):
            l = _bit_reversed(i, self.n)
            candidates = []

            for path in paths:
                self._update_llrs(path, l)
                llr_bit = path.L[l, self.n]

                if l in self.frozen_set:
                    u_bit = 0
                    new_path = path.copy()
                    new_path.pm += self._pm_penalty(llr_bit, u_bit)
                    new_path.B[l, self.n] = 0
                    new_path.u[l] = 0
                    self._update_bits(new_path, l)
                    candidates.append(new_path)
                else:
                    for u_bit in (0, 1):
                        new_path = path.copy()
                        new_path.pm += self._pm_penalty(llr_bit, u_bit)
                        new_path.B[l, self.n] = u_bit
                        new_path.u[l] = u_bit
                        self._update_bits(new_path, l)
                        candidates.append(new_path)

            candidates.sort(key=lambda p: p.pm)
            paths = candidates[: self.list_size]

        best_pm = paths[0].pm
        best_u = paths[0].u.copy()

        if self.crc_length > 0:
            passed = []
            for p in paths:
                payload = p.u[self.info_indices]
                if crc_check(payload, self.crc_length):
                    passed.append(p)
            if passed:
                passed.sort(key=lambda p: p.pm)
                best_u = passed[0].u
                best_pm = passed[0].pm

        return best_u, best_pm
