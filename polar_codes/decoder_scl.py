"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math

import numpy as np

from decoder_sc import (
    _active_bit_level,
    _active_llr_level,
    _update_bits,
    _update_llrs,
)
from encoder import bit_reversed_index


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_register(bits, crc_length):
    """对比特序列运行 CRC 寄存器，返回最终寄存器值。"""
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    reg = 0
    for bit in bits:
        reg ^= int(bit)
        for _ in range(crc_length):
            if reg & 1:
                reg = (reg >> 1) ^ poly
            else:
                reg >>= 1
    return reg


def crc_encode(info_bits, crc_length=8):
    """
    计算 CRC 校验位并附加到信息比特后。
    """
    info_bits = np.asarray(info_bits, dtype=int)
    remainder = _crc_register(info_bits, crc_length)
    crc_bits = np.array(
        [(remainder >> i) & 1 for i in range(crc_length)],
        dtype=int,
    )[::-1]
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """
    检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。
    """
    bits = np.asarray(bits, dtype=int)
    return _crc_register(bits, crc_length) == 0


class _Path:
    """单条 SCL 路径。"""

    __slots__ = ("pm", "B", "L")

    def __init__(self, N, n, llr_ch):
        self.pm = 0.0
        self.L = np.zeros((N, n + 1), dtype=np.float64)
        self.B = np.zeros((N, n + 1), dtype=np.int8)
        self.L[:, 0] = llr_ch

    def copy(self):
        child = _Path(self.L.shape[0], int(math.log2(self.L.shape[0])), self.L[:, 0])
        child.pm = self.pm
        child.L = self.L.copy()
        child.B = self.B.copy()
        return child


class SCLDecoder:
    """
    SCL 译码器（置换 SC + 路径度量）。
    """

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.decode_order = [bit_reversed_index(i, self.n) for i in range(N)]

    def _pm_penalty(self, llr, bit):
        hard = 0 if llr >= 0 else 1
        return 0.0 if bit == hard else abs(llr)

    def decode(self, llr_ch):
        """
        主译码函数。
        """
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [_Path(self.N, self.n, llr_ch)]

        for l in self.decode_order:
            new_paths = []
            for path in paths:
                _update_llrs(path.L, path.B, l, self.n)
                llr0 = path.L[l, self.n]

                if self.frozen_bits[l]:
                    path.pm += self._pm_penalty(llr0, 0)
                    path.B[l, self.n] = 0
                    _update_bits(path.B, l, self.n)
                    new_paths.append(path)
                else:
                    for bit in (0, 1):
                        child = path.copy()
                        child.pm += self._pm_penalty(llr0, bit)
                        child.B[l, self.n] = bit
                        _update_bits(child.B, l, self.n)
                        new_paths.append(child)

            new_paths.sort(key=lambda p: p.pm)
            paths = new_paths[: self.list_size]

        if self.crc_length > 0:
            valid = []
            for p in paths:
                info_bits = p.B[:, self.n][~self.frozen_bits]
                if crc_check(info_bits, self.crc_length):
                    valid.append(p)
            chosen = min(valid, key=lambda p: p.pm) if valid else min(
                paths, key=lambda p: p.pm
            )
        else:
            chosen = min(paths, key=lambda p: p.pm)

        return chosen.B[:, self.n].astype(int), chosen.pm


def scl_decode_channel(llr_ch, frozen_bits, list_size=4, crc_length=0):
    """信道 LLR 直接 SCL 译码。"""
    decoder = SCLDecoder(len(llr_ch), frozen_bits, list_size=list_size, crc_length=crc_length)
    return decoder.decode(llr_ch)
