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
    _lower_llr_exact,
    _upper_llr_exact,
    sc_decode,
)
from encoder import bit_reversal_permutation


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
    crc_bits = np.array(
        [(remainder >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=int,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC"""
    bits = np.asarray(bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    return _crc_remainder(bits, poly, crc_length) == 0


def _pm_update(pm, llr, u):
    hard = 0 if llr >= 0 else 1
    if u != hard:
        pm += abs(llr)
    return pm


class _Path:
    __slots__ = ("pm", "L", "B", "u")

    def __init__(self, N, n, llr_init):
        self.pm = 0.0
        self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.B = np.full((N, n + 1), np.nan)
        self.u = np.zeros(N, dtype=int)
        self.L[:, 0] = llr_init


class SCLDecoder:
    """SCL 译码器"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.list_size = list_size
        self.crc_length = crc_length
        self.br = bit_reversal_permutation(N)
        self.info_idx = np.where(~self.frozen_bits)[0]

    def _update_llrs(self, path, phase):
        L, B = path.L, path.B
        for s in range(self.n - _active_llr_level(phase, self.n), self.n):
            block_size = 1 << (s + 1)
            branch_size = block_size >> 1
            for j in range(phase, self.N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = _upper_llr_exact(L[j, s], L[j + branch_size, s])
                else:
                    L[j, s + 1] = _lower_llr_exact(
                        L[j, s],
                        L[j - branch_size, s],
                        int(B[j - branch_size, s + 1]),
                    )

    def _update_bits(self, path, phase):
        if phase < self.N // 2:
            return
        B = path.B
        for s in range(self.n, self.n - _active_bit_level(phase, self.n), -1):
            block_size = 1 << s
            branch_size = block_size >> 1
            for j in range(phase, -1, -block_size):
                if j % block_size >= branch_size:
                    B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                    B[j, s - 1] = B[j, s]

    def decode(self, llr_ch):
        """主译码函数，返回 (u_hat, pm)"""
        if self.list_size == 1 and self.crc_length == 0:
            u_hat = sc_decode(llr_ch, self.frozen_bits)
            return u_hat, 0.0

        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        llr_init = llr_ch[self.br]
        phases = [_bit_reversed(i, self.n) for i in range(self.N)]

        paths = [_Path(self.N, self.n, llr_init.copy())]

        for phase in phases:
            candidates = []
            for path in paths:
                self._update_llrs(path, phase)
                cur_llr = path.L[phase, self.n]

                if phase in self.frozen_set:
                    path.pm = _pm_update(path.pm, cur_llr, 0)
                    path.u[phase] = 0
                    path.B[phase, self.n] = 0
                    self._update_bits(path, phase)
                else:
                    for u_val in (0, 1):
                        new_path = _Path(self.N, self.n, path.L[:, 0].copy())
                        new_path.L[:, 1:] = path.L[:, 1:].copy()
                        new_path.B[:, 1:] = path.B[:, 1:].copy()
                        new_path.u = path.u.copy()
                        new_path.pm = _pm_update(path.pm, cur_llr, u_val)
                        new_path.u[phase] = u_val
                        new_path.B[phase, self.n] = u_val
                        self._update_bits(new_path, phase)
                        candidates.append(new_path)

            if phase not in self.frozen_set:
                paths = sorted(candidates, key=lambda p: p.pm)[: self.list_size]

        if self.crc_length > 0:
            crc_ok = [
                p for p in paths if crc_check(p.u[self.info_idx], self.crc_length)
            ]
            if crc_ok:
                best = min(crc_ok, key=lambda p: p.pm)
            else:
                best = min(paths, key=lambda p: p.pm)
        else:
            best = min(paths, key=lambda p: p.pm)

        return best.u.copy(), best.pm
