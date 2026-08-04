"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
import math

from encoder import bit_reversal_permutation
from decoder_sc import (
    f_operation, g_operation, _bit_reversed_index,
    _active_llr_level, _active_bit_level,
)

_CRC8_POLY = 0x07
_CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, reg_bits):
    reg = 0
    for b in bits:
        reg = ((reg << 1) | int(b)) & ((1 << (reg_bits + 1)) - 1)
        if reg & (1 << reg_bits):
            reg ^= poly
    for _ in range(reg_bits):
        reg = (reg << 1) & ((1 << (reg_bits + 1)) - 1)
        if reg & (1 << reg_bits):
            reg ^= poly
    return reg & ((1 << reg_bits) - 1)


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=int)
    if crc_length == 8:
        poly, reg_bits = _CRC8_POLY, 8
    elif crc_length == 16:
        poly, reg_bits = _CRC16_POLY, 16
    else:
        raise ValueError("crc_length must be 8 or 16")

    remainder = _crc_remainder(info_bits, poly, reg_bits)
    crc_bits = np.array(
        [(remainder >> (reg_bits - 1 - i)) & 1 for i in range(reg_bits)], dtype=int
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC 是否正确。"""
    bits = np.asarray(bits, dtype=int)
    if crc_length == 8:
        poly, reg_bits = _CRC8_POLY, 8
    elif crc_length == 16:
        poly, reg_bits = _CRC16_POLY, 16
    else:
        raise ValueError("crc_length must be 8 or 16")
    return _crc_remainder(bits, poly, reg_bits) == 0


class _Path:
    __slots__ = ('pm', 'u_hat', 'L', 'B')

    def __init__(self, N, n):
        self.pm = 0.0
        self.u_hat = np.zeros(N, dtype=int)
        self.L = np.zeros((N, n + 1), dtype=np.float64)
        self.B = np.zeros((N, n + 1), dtype=np.float64)


class SCLDecoder:
    """SCL 译码器（Lazy Copy 优化）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.info_indices = np.where(~self.frozen_bits)[0]

    def _pm_penalty(self, llr, u_bit):
        llr_sign_bit = 0 if llr >= 0 else 1
        return 0.0 if u_bit == llr_sign_bit else abs(llr)

    def _update_llrs(self, path, l):
        for s in range(self.n - _active_llr_level(l, self.n), self.n):
            block_size = 1 << (s + 1)
            branch_size = block_size >> 1
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    path.L[j, s + 1] = f_operation(path.L[j, s], path.L[j + branch_size, s])
                else:
                    path.L[j, s + 1] = g_operation(
                        path.L[j - branch_size, s], path.L[j, s],
                        int(path.B[j - branch_size, s + 1])
                    )

    def _propagate_bit(self, path, l):
        if l < self.N // 2:
            return
        for s in range(self.n, self.n - _active_bit_level(l, self.n), -1):
            block_size = 1 << s
            branch_size = block_size >> 1
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    path.B[j - branch_size, s - 1] = (
                        int(path.B[j, s]) ^ int(path.B[j - branch_size, s])
                    )
                    path.B[j, s - 1] = path.B[j, s]

    def decode(self, llr_ch):
        """主译码函数，返回 (u_hat, pm)。"""
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        rev = bit_reversal_permutation(self.N)
        llr_ch = llr_ch[rev]

        paths = [_Path(self.N, self.n)]
        paths[0].L[:, 0] = llr_ch

        for i in range(self.N):
            l = _bit_reversed_index(i, self.n)
            new_paths = []

            for path in paths:
                self._update_llrs(path, l)
                llr = path.L[l, self.n]

                if l in self.frozen_set:
                    new_p = _Path(self.N, self.n)
                    new_p.pm = path.pm + self._pm_penalty(llr, 0)
                    new_p.L = path.L.copy()
                    new_p.B = path.B.copy()
                    new_p.u_hat = path.u_hat.copy()
                    new_p.u_hat[l] = 0
                    new_p.B[l, self.n] = 0
                    self._propagate_bit(new_p, l)
                    new_paths.append(new_p)
                else:
                    for u_bit in (0, 1):
                        new_p = _Path(self.N, self.n)
                        new_p.pm = path.pm + self._pm_penalty(llr, u_bit)
                        new_p.L = path.L.copy()
                        new_p.B = path.B.copy()
                        new_p.u_hat = path.u_hat.copy()
                        new_p.u_hat[l] = u_bit
                        new_p.B[l, self.n] = u_bit
                        self._propagate_bit(new_p, l)
                        new_paths.append(new_p)

            new_paths.sort(key=lambda p: p.pm)
            paths = new_paths[:self.list_size]

        if self.crc_length > 0:
            valid = [p for p in paths if crc_check(p.u_hat[self.info_indices], self.crc_length)]
            if valid:
                best = min(valid, key=lambda p: p.pm)
            else:
                best = min(paths, key=lambda p: p.pm)
        else:
            best = min(paths, key=lambda p: p.pm)

        return best.u_hat.copy(), best.pm
