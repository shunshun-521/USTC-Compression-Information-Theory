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
    _frozen_indices,
    _lower_llr_exact,
    _upper_llr_exact,
    sc_decode,
)


def crc_encode(info_bits, crc_length=8):
    """
    计算 CRC 校验位并附加到信息比特后。

    使用标准多项式：
      r=8:  CRC-8  (0x07, 即 x^8 + x^2 + x + 1)
      r=16: CRC-16 (0x8005)
    """
    info_bits = np.asarray(info_bits, dtype=np.uint8)
    if crc_length == 8:
        poly = 0x07
    elif crc_length == 16:
        poly = 0x8005
    else:
        raise ValueError("crc_length must be 8 or 16")

    reg = 0
    for bit in info_bits:
        reg ^= int(bit) << (crc_length - 1)
        for _ in range(8):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)

    crc_bits = np.array([(reg >> i) & 1 for i in range(crc_length - 1, -1, -1)], dtype=int)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。"""
    if crc_length == 0:
        return True
    bits = np.asarray(bits, dtype=np.uint8)
    if len(bits) < crc_length:
        return False
    expected = crc_encode(bits[:-crc_length], crc_length)
    return np.array_equal(bits[-crc_length:], expected[-crc_length:])


class PathState:
    """单条 SCL 路径状态"""

    __slots__ = ("pm", "u_hat", "llr", "bits")

    def __init__(self, n_len, n):
        self.pm = 0.0
        self.u_hat = np.zeros(n_len, dtype=int)
        self.llr = np.full((n_len, n + 1), np.nan, dtype=np.float64)
        self.bits = np.full((n_len, n + 1), np.nan)


class SCLDecoder:
    """SCL 译码器（Lazy Copy：路径分裂时复制 LLR/比特数组）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_set = _frozen_indices(frozen_bits)
        self.list_size = list_size
        self.crc_length = crc_length
        self.decode_order = [_bit_reversed(i, self.n) for i in range(N)]

    def _update_llrs(self, path, l):
        for s in range(self.n - _active_llr_level(l, self.n), self.n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    path.llr[j, s + 1] = _upper_llr_exact(
                        path.llr[j, s], path.llr[j + branch_size, s]
                    )
                else:
                    path.llr[j, s + 1] = _lower_llr_exact(
                        path.llr[j, s],
                        path.llr[j - branch_size, s],
                        int(path.bits[j - branch_size, s + 1]),
                    )

    def _update_bits(self, path, l):
        if l < self.N / 2:
            return
        for s in range(self.n, self.n - _active_bit_level(l, self.n), -1):
            block_size = 2 ** s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    path.bits[j - branch_size, s - 1] = int(path.bits[j, s]) ^ int(
                        path.bits[j - branch_size, s]
                    )
                    path.bits[j, s - 1] = path.bits[j, s]

    @staticmethod
    def _branch_penalty(llr, bit):
        hard = 0 if llr >= 0 else 1
        return 0.0 if bit == hard else abs(llr)

    def decode(self, llr_ch):
        if self.list_size == 1 and self.crc_length == 0:
            u_hat = sc_decode(llr_ch, [i in self.frozen_set for i in range(self.N)])
            return u_hat, 0.0

        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [PathState(self.N, self.n)]
        paths[0].llr[:, 0] = llr_ch

        for l in self.decode_order:
            new_paths = []
            for path in paths:
                self._update_llrs(path, l)
                llr = path.llr[l, self.n]

                if l in self.frozen_set:
                    child = PathState(self.N, self.n)
                    child.pm = path.pm + self._branch_penalty(llr, 0)
                    child.u_hat = path.u_hat.copy()
                    child.u_hat[l] = 0
                    child.llr = path.llr.copy()
                    child.bits = path.bits.copy()
                    child.bits[l, self.n] = 0
                    self._update_bits(child, l)
                    new_paths.append(child)
                else:
                    for bit in (0, 1):
                        child = PathState(self.N, self.n)
                        child.pm = path.pm + self._branch_penalty(llr, bit)
                        child.u_hat = path.u_hat.copy()
                        child.u_hat[l] = bit
                        child.llr = path.llr.copy()
                        child.bits = path.bits.copy()
                        child.bits[l, self.n] = bit
                        self._update_bits(child, l)
                        new_paths.append(child)

            new_paths.sort(key=lambda p: p.pm)
            paths = new_paths[: self.list_size]

        if self.crc_length > 0:
            valid = [p for p in paths if crc_check(p.u_hat, self.crc_length)]
            chosen = valid[0] if valid else paths[0]
        else:
            chosen = paths[0]

        return chosen.u_hat.copy(), chosen.pm
