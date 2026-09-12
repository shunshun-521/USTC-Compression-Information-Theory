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
    _update_bits,
    _update_llrs,
    f_operation,
)


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, crc_length):
    """计算 CRC 余数（MSB first，标准 LFSR）。"""
    mask = (1 << crc_length) - 1
    top_bit = 1 << (crc_length - 1)
    reg = 0
    for b in bits:
        reg ^= int(b) << (crc_length - 1)
        if reg & top_bit:
            reg = ((reg << 1) ^ poly) & mask
        else:
            reg = (reg << 1) & mask
    return reg


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    remainder = _crc_remainder(info_bits, poly, crc_length)
    crc_bits = np.array(
        [(remainder >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=int,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits 是否通过 CRC。"""
    bits = np.asarray(bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    return _crc_remainder(bits, poly, crc_length) == 0


class Path:
    """SCL 单条路径（Lazy Copy）。"""

    __slots__ = ("pm", "L", "B", "parent")

    def __init__(self, N, n, llr_ch):
        self.pm = 0.0
        self.L = np.zeros((N, n + 1), dtype=np.float64)
        self.B = np.zeros((N, n + 1), dtype=np.int8)
        self.L[:, 0] = llr_ch
        self.parent = None

    def clone(self):
        child = Path.__new__(Path)
        child.pm = self.pm
        child.L = self.L
        child.B = self.B
        child.parent = self
        return child

    def ensure_owned(self):
        if self.parent is not None:
            self.L = self.L.copy()
            self.B = self.B.copy()
            self.parent = None


class SCLDecoder:
    """SCL 译码器（含 Lazy Copy 优化，Permuted SC 结构）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length

    def decode(self, llr_ch):
        """主译码函数，返回 (u_hat, pm)。"""
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [Path(self.N, self.n, llr_ch)]

        for phi in range(self.N):
            leaf = _bit_reversed(phi, self.n)
            new_paths = []

            for path in paths:
                _update_llrs(path.L, path.B, leaf, self.n)
                llr0 = path.L[leaf, self.n]

                if self.frozen_bits[leaf]:
                    path.ensure_owned()
                    if llr0 < 0:
                        path.pm += abs(llr0)
                    path.B[leaf, self.n] = 0
                    _update_bits(path.B, leaf, self.n)
                    new_paths.append(path)
                else:
                    for bit in (0, 1):
                        child = path.clone()
                        child.ensure_owned()
                        child.B[leaf, self.n] = bit
                        if (bit == 0 and llr0 < 0) or (bit == 1 and llr0 >= 0):
                            child.pm += abs(llr0)
                        _update_bits(child.B, leaf, self.n)
                        new_paths.append(child)

            new_paths.sort(key=lambda p: p.pm)
            paths = new_paths[: self.list_size]

        if self.crc_length > 0:
            info_positions = np.where(~self.frozen_bits)[0]
            valid = []
            for path in paths:
                info_bits = path.B[:, self.n][info_positions]
                if crc_check(info_bits, self.crc_length):
                    valid.append(path)
            if valid:
                paths = valid

        best = min(paths, key=lambda p: p.pm)
        return best.B[:, self.n].astype(int), best.pm
