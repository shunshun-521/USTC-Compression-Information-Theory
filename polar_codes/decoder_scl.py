"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np

from decoder_sc import (
    _active_bit_level,
    _active_llr_level,
    _bit_reversed_index,
    _frozen_set_for_decoder,
    _lower_llr,
    _upper_llr,
)
from encoder import bit_reversal_permutation

_CRC_POLY = {8: 0x07, 16: 0x8005}


def _crc_remainder(bits, crc_length):
    """CRC-8/16 余数（多项式 0x07 / 0x8005，MSB 先行）"""
    poly = _CRC_POLY[crc_length]
    mask = (1 << crc_length) - 1
    top = 1 << (crc_length - 1)
    reg = 0
    width = 8 if crc_length == 8 else 16
    for bit in bits:
        reg ^= int(bit) << (crc_length - 1)
        for _ in range(width):
            if reg & top:
                reg = ((reg << 1) ^ poly) & mask
            else:
                reg = (reg << 1) & mask
    return reg


def _crc_bits_from_remainder(remainder, crc_length):
    return np.array(
        [(remainder >> i) & 1 for i in range(crc_length - 1, -1, -1)], dtype=int
    )


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=np.uint8)
    remainder = _crc_remainder(info_bits, crc_length)
    return np.concatenate([info_bits, _crc_bits_from_remainder(remainder, crc_length)])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC"""
    bits = np.asarray(bits, dtype=np.uint8)
    info = bits[:-crc_length]
    expected = _crc_bits_from_remainder(_crc_remainder(info, crc_length), crc_length)
    return np.array_equal(bits[-crc_length:], expected)


class _Path:
    __slots__ = ("L", "B", "pm", "u_nat")

    def __init__(self, N, n):
        self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.B = np.full((N, n + 1), np.nan)
        self.pm = 0.0
        self.u_nat = np.zeros(N, dtype=int)


class SCLDecoder:
    """SCL 译码器"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(self.frozen_bits == 0)[0]
        self.frozen_set = _frozen_set_for_decoder(self.frozen_bits)
        self.br = bit_reversal_permutation(N)
        self.decode_order = [_bit_reversed_index(i, self.n) for i in range(N)]

    def _llr_penalty(self, llr, bit):
        hard = 0 if llr >= 0 else 1
        return 0.0 if bit == hard else abs(llr)

    def _update_llrs(self, path, l):
        for s in range(self.n - _active_llr_level(l, self.n), self.n):
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    path.L[j, s + 1] = _upper_llr(path.L[j, s], path.L[j + branch_size, s])
                else:
                    top_bit = path.B[j - branch_size, s + 1]
                    if np.isnan(top_bit):
                        top_bit = 0
                    path.L[j, s + 1] = _lower_llr(
                        path.L[j, s], path.L[j - branch_size, s], top_bit
                    )

    def _update_bits(self, path, l):
        if l < self.N / 2:
            return
        for s in range(self.n, self.n - _active_bit_level(l, self.n), -1):
            block_size = 1 << s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    path.B[j - branch_size, s - 1] = int(path.B[j, s]) ^ int(path.B[j - branch_size, s])
                    path.B[j, s - 1] = path.B[j, s]

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [_Path(self.N, self.n)]
        paths[0].L[:, 0] = llr_ch

        for l in self.decode_order:
            active = []
            for path in paths:
                self._update_llrs(path, l)
                llr = path.L[l, self.n]
                if np.isnan(llr):
                    llr = 0.0

                nat_idx = int(self.br[l])
                if l in self.frozen_set:
                    new_path = _Path(self.N, self.n)
                    new_path.L[:] = path.L
                    new_path.B[:] = path.B
                    new_path.u_nat[:] = path.u_nat
                    new_path.pm = path.pm + self._llr_penalty(llr, 0)
                    new_path.B[l, self.n] = 0
                    new_path.u_nat[nat_idx] = 0
                    self._update_bits(new_path, l)
                    active.append(new_path)
                else:
                    for bit in (0, 1):
                        new_path = _Path(self.N, self.n)
                        new_path.L[:] = path.L
                        new_path.B[:] = path.B
                        new_path.u_nat[:] = path.u_nat
                        new_path.pm = path.pm + self._llr_penalty(llr, bit)
                        new_path.B[l, self.n] = bit
                        new_path.u_nat[nat_idx] = bit
                        self._update_bits(new_path, l)
                        active.append(new_path)

            active.sort(key=lambda p: p.pm)
            paths = active[: self.list_size]

        if self.crc_length > 0:
            valid = [
                p for p in paths
                if crc_check(p.u_nat[self.info_indices], self.crc_length)
            ]
            chosen = min(valid, key=lambda p: p.pm) if valid else min(paths, key=lambda p: p.pm)
        else:
            chosen = min(paths, key=lambda p: p.pm)

        return chosen.u_nat.copy(), chosen.pm
