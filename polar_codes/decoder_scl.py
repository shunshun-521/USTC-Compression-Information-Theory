"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np

from decoder_sc import (
    f_operation,
    g_operation,
    _active_bit_level,
    _active_llr_level,
    _bit_reversed_index,
    _reorder_channel_llrs,
    _update_bits,
)
from encoder import bit_reversal_permutation

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
    info_bits = np.asarray(info_bits, dtype=int)
    if crc_length == 8:
        poly = CRC8_POLY
    elif crc_length == 16:
        poly = CRC16_POLY
    else:
        raise ValueError("crc_length must be 8 or 16")
    remainder = _crc_remainder(info_bits, poly, crc_length)
    crc_bits = np.array(
        [(remainder >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=int,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=int)
    if crc_length == 8:
        poly = CRC8_POLY
    elif crc_length == 16:
        poly = CRC16_POLY
    else:
        raise ValueError("crc_length must be 8 or 16")
    expected = _crc_remainder(bits[:-crc_length], poly, crc_length)
    received = 0
    for i, b in enumerate(bits[-crc_length:]):
        received |= int(b) << (crc_length - 1 - i)
    return expected == received


class _Path:
    __slots__ = ("L", "B", "pm", "u_hat", "copied")

    def __init__(self, n, N, template_L=None, template_B=None):
        self.L = template_L.copy() if template_L is not None else np.zeros((N, n + 1))
        self.B = template_B.copy() if template_B is not None else np.zeros((N, n + 1))
        self.pm = 0.0
        self.u_hat = np.zeros(N, dtype=int)
        self.copied = template_L is None


class SCLDecoder:
    """SCL 译码器（Lazy Copy）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]
        self.decode_order = [_bit_reversed_index(i, self.n) for i in range(N)]

    def _update_llrs_path(self, path, l):
        n, N = self.n, self.N
        for s in range(n - _active_llr_level(l, n), n):
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(l, N, block_size):
                if j % block_size < branch_size:
                    path.L[j, s + 1] = f_operation(path.L[j, s], path.L[j + branch_size, s])
                else:
                    top_bit = path.B[j - branch_size, s + 1]
                    path.L[j, s + 1] = g_operation(
                        path.L[j - branch_size, s],
                        path.L[j, s],
                        top_bit,
                    )

    def _pm_penalty(self, llr, bit):
        hard = 0 if llr >= 0 else 1
        return 0.0 if bit == hard else abs(llr)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n, N = self.n, self.N

        base_L = np.zeros((N, n + 1), dtype=np.float64)
        base_L[:, 0] = _reorder_channel_llrs(llr_ch, n)
        base_B = np.zeros((N, n + 1), dtype=np.int32)
        paths = [_Path(n, N, base_L, base_B)]

        for l in self.decode_order:
            new_paths = []
            for path in paths:
                self._update_llrs_path(path, l)
                llr0 = path.L[l, n]
                if self.frozen_bits[l]:
                    child = _Path(n, N, path.L, path.B)
                    child.pm = path.pm + self._pm_penalty(llr0, 0)
                    child.u_hat = path.u_hat.copy()
                    child.u_hat[l] = 0
                    child.B[l, n] = 0
                    _update_bits(child.B, l, n, N)
                    new_paths.append(child)
                else:
                    for bit in (0, 1):
                        child = _Path(n, N, path.L, path.B)
                        child.pm = path.pm + self._pm_penalty(llr0, bit)
                        child.u_hat = path.u_hat.copy()
                        child.u_hat[l] = bit
                        child.B[l, n] = bit
                        _update_bits(child.B, l, n, N)
                        new_paths.append(child)
            new_paths.sort(key=lambda p: p.pm)
            paths = new_paths[: self.list_size]

        if self.crc_length > 0:
            valid = [
                p for p in paths
                if crc_check(p.u_hat[self.info_indices], self.crc_length)
            ]
            if valid:
                paths = valid

        best = min(paths, key=lambda p: p.pm)
        return best.u_hat.copy(), best.pm
