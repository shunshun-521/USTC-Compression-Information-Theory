"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np
from decoder_sc import (
    bit_reversed,
    active_bit_level,
    active_llr_level,
    upper_llr,
    lower_llr,
)


CRC8_POLY = 0x107   # x^8 + x^2 + x + 1
CRC16_POLY = 0x11021  # CRC-16


def _crc_remainder(bits, crc_length):
    if crc_length == 8:
        poly = CRC8_POLY
    elif crc_length == 16:
        poly = CRC16_POLY
    else:
        raise ValueError("crc_length must be 8 or 16")

    data = list(map(int, bits)) + [0] * crc_length
    for i in range(len(bits)):
        if data[i]:
            for j in range(crc_length + 1):
                if (poly >> (crc_length - j)) & 1:
                    data[i + j] ^= 1
    return np.array(data[len(bits): len(bits) + crc_length], dtype=int)


def crc_encode(info_bits, crc_length=8):
    info_bits = np.asarray(info_bits, dtype=int)
    return np.concatenate([info_bits, _crc_remainder(info_bits, crc_length)])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=int)
    if crc_length == 0:
        return True
    if crc_length == 8:
        poly = CRC8_POLY
    else:
        poly = CRC16_POLY
    data = list(map(int, bits))
    for i in range(len(data) - crc_length):
        if data[i]:
            for j in range(crc_length + 1):
                if (poly >> (crc_length - j)) & 1:
                    data[i + j] ^= 1
    return sum(data[-crc_length:]) == 0


class _Path:
    __slots__ = ("L", "B", "pm", "u_hat", "L_ref", "B_ref")

    def __init__(self, N, n, llr_ch):
        self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.B = np.full((N, n + 1), np.nan)
        self.L[:, 0] = llr_ch
        self.pm = 0.0
        self.u_hat = np.zeros(N, dtype=int)
        self.L_ref = None
        self.B_ref = None

    def lazy_copy(self):
        new_path = _Path.__new__(_Path)
        new_path.L_ref = self.L
        new_path.B_ref = self.B
        new_path.L = None
        new_path.B = None
        new_path.pm = self.pm
        new_path.u_hat = self.u_hat.copy()
        return new_path

    def ensure_owned(self):
        if self.L is None:
            self.L = self.L_ref.copy()
            self.L_ref = None
        if self.B is None:
            self.B = self.B_ref.copy()
            self.B_ref = None

    def get_L(self):
        return self.L if self.L is not None else self.L_ref

    def get_B(self):
        return self.B if self.B is not None else self.B_ref


class SCLDecoder:
    """SCL 译码器（含 Lazy Copy 优化）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.frozen_set = set(np.where(self.frozen_bits == 1)[0])
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(self.frozen_bits == 0)[0]

    def _update_llrs(self, path, l):
        L = path.get_L()
        B = path.get_B()
        n = self.n
        for s in range(n - active_llr_level(l, n), n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = upper_llr(L[j, s], L[j + branch_size, s])
                else:
                    L[j, s + 1] = lower_llr(
                        L[j, s], L[j - branch_size, s], int(B[j - branch_size, s + 1])
                    )

    def _update_bits(self, path, l):
        B = path.get_B()
        if l < self.N // 2:
            return
        n = self.n
        for s in range(n, n - active_bit_level(l, n), -1):
            block_size = 2 ** s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                    B[j, s - 1] = B[j, s]

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [_Path(self.N, self.n, llr_ch)]

        for phi in range(self.N):
            l = bit_reversed(phi, self.n)
            candidates = []

            for path in paths:
                path.ensure_owned()
                self._update_llrs(path, l)
                llr_val = path.get_L()[l, self.n]

                if l in self.frozen_set:
                    penalty = 0.0 if llr_val >= 0 else abs(llr_val)
                    path.pm += penalty
                    path.u_hat[l] = 0
                    path.get_B()[l, self.n] = 0
                    self._update_bits(path, l)
                    candidates.append(path)
                else:
                    for u in (0, 1):
                        new_path = path.lazy_copy()
                        new_path.ensure_owned()
                        consistent = (u == 0 and llr_val >= 0) or (u == 1 and llr_val < 0)
                        new_path.pm += 0.0 if consistent else abs(llr_val)
                        new_path.u_hat[l] = u
                        new_path.get_B()[l, self.n] = u
                        self._update_bits(new_path, l)
                        candidates.append(new_path)

            candidates.sort(key=lambda p: p.pm)
            paths = candidates[: self.list_size]

        if self.crc_length > 0:
            valid = [p for p in paths if crc_check(p.u_hat[self.info_indices], self.crc_length)]
            best_path = min(valid or paths, key=lambda p: p.pm)
        else:
            best_path = min(paths, key=lambda p: p.pm)

        return best_path.u_hat.copy(), best_path.pm
