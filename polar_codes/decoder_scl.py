"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
import math
from decoder_sc import (
    f_operation,
    g_operation,
    bit_reversed,
    _active_llr_level,
    _active_bit_level,
)


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, crc_length):
    reg = 0
    for bit in bits:
        reg = (reg << 1) | int(bit)
        if reg & (1 << crc_length):
            reg ^= poly
    return reg & ((1 << crc_length) - 1)


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=np.int8)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    remainder = _crc_remainder(info_bits, poly, crc_length)
    crc_bits = np.array(
        [(remainder >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=np.int8,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC 是否正确"""
    bits = np.asarray(bits, dtype=np.int8)
    expected = crc_encode(bits[:-crc_length], crc_length)[-crc_length:]
    return np.array_equal(bits[-crc_length:], expected)


class Path:
    """SCL 单条路径"""

    __slots__ = ("pm", "L", "B", "u_hat")

    def __init__(self, N, n, llr_ch):
        self.pm = 0.0
        self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.B = np.zeros((N, n + 1), dtype=np.int8)
        self.L[:, 0] = llr_ch.copy()
        self.u_hat = np.zeros(N, dtype=np.int8)

    def copy(self):
        new = Path.__new__(Path)
        new.pm = self.pm
        new.L = self.L.copy()
        new.B = self.B.copy()
        new.u_hat = self.u_hat.copy()
        return new


class SCLDecoder:
    """SCL 译码器（Lazy Copy）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length

    def _update_llrs(self, paths, l):
        for path in paths:
            for s in range(self.n - _active_llr_level(l, self.n), self.n):
                block_size = 2 ** (s + 1)
                branch_size = block_size // 2
                for j in range(l, self.N, block_size):
                    if j % block_size < branch_size:
                        path.L[j, s + 1] = f_operation(
                            path.L[j, s], path.L[j + branch_size, s]
                        )
                    else:
                        path.L[j, s + 1] = g_operation(
                            path.L[j - branch_size, s],
                            path.L[j, s],
                            path.B[j - branch_size, s + 1],
                        )

    def _update_bits(self, paths, l):
        if l < self.N // 2:
            return
        for path in paths:
            for s in range(self.n, self.n - _active_bit_level(l, self.n), -1):
                block_size = 2 ** s
                branch_size = block_size // 2
                for j in range(l, -1, -block_size):
                    if j % block_size >= branch_size:
                        path.B[j - branch_size, s - 1] = int(
                            path.B[j, s] ^ path.B[j - branch_size, s]
                        )
                        path.B[j, s - 1] = path.B[j, s]

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [Path(self.N, self.n, llr_ch)]

        for i in range(self.N):
            l = bit_reversed(i, self.n)
            self._update_llrs(paths, l)

            if self.frozen_bits[l]:
                for p in paths:
                    llr = p.L[l, self.n]
                    if llr < 0:
                        p.pm += abs(llr)
                    p.B[l, self.n] = 0
                    p.u_hat[l] = 0
            else:
                candidates = []
                for p in paths:
                    llr = p.L[l, self.n]
                    for bit in (0, 1):
                        cp = p.copy()
                        cp.B[l, self.n] = bit
                        cp.u_hat[l] = bit
                        hard = 0 if llr >= 0 else 1
                        if bit != hard:
                            cp.pm += abs(llr)
                        candidates.append(cp)
                candidates.sort(key=lambda x: x.pm)
                paths = candidates[: self.list_size]

            self._update_bits(paths, l)

        if self.crc_length > 0:
            valid = []
            for p in paths:
                info_bits = p.u_hat[~self.frozen_bits]
                if crc_check(info_bits, self.crc_length):
                    valid.append(p)
            best = min(valid if valid else paths, key=lambda x: x.pm)
        else:
            best = min(paths, key=lambda x: x.pm)

        return best.u_hat.astype(int), best.pm
