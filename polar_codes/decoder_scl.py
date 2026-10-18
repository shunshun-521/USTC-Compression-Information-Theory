"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL），与置换 SC 使用相同的 LLR 更新
"""
import numpy as np
import math
from decoder_sc import (
    bit_reversed,
    active_llr_level,
    active_bit_level,
    upper_llr,
    lower_llr,
    hard_decision,
    f_operation,
    g_operation,
)


def _crc_poly(crc_length):
    if crc_length == 8:
        return 0x07
    if crc_length == 16:
        return 0x8005
    raise ValueError("crc_length must be 8 or 16")


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=np.int8)
    poly = _crc_poly(crc_length)
    mask = (1 << crc_length) - 1
    reg = 0
    for b in info_bits:
        reg ^= int(b) << (crc_length - 1)
        for _ in range(1):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & mask
            else:
                reg = (reg << 1) & mask
    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=np.int8
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """校验 CRC（余式为零）"""
    bits = np.asarray(bits, dtype=np.int8)
    poly = _crc_poly(crc_length)
    mask = (1 << crc_length) - 1
    reg = 0
    for b in bits:
        reg ^= int(b) << (crc_length - 1)
        for _ in range(1):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & mask
            else:
                reg = (reg << 1) & mask
    return reg == 0


class _Path:
    __slots__ = ("pm", "L", "B", "u")

    def __init__(self, N, n, llr_ch):
        self.pm = 0.0
        self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.B = np.zeros((N, n + 1), dtype=np.int8)
        self.L[:, 0] = llr_ch
        self.u = np.zeros(N, dtype=np.int8)

    def copy(self):
        p = _Path(self.L.shape[0], int(math.log2(self.L.shape[0])), self.L[:, 0])
        p.pm = self.pm
        p.L = self.L.copy()
        p.B = self.B.copy()
        p.u = self.u.copy()
        return p


class SCLDecoder:
    """SCL 译码器（Lazy Copy：分裂时复制路径对象）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen = np.asarray(frozen_bits, dtype=bool)
        self.L_size = list_size
        self.crc_length = crc_length

    def _update_llrs(self, path, l):
        for s in range(self.n - active_llr_level(l, self.n), self.n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    path.L[j, s + 1] = upper_llr(path.L[j, s], path.L[j + branch_size, s])
                else:
                    top_bit = path.B[j - branch_size, s + 1]
                    path.L[j, s + 1] = lower_llr(
                        path.L[j, s], path.L[j - branch_size, s], top_bit
                    )

    def _update_bits(self, path, l):
        if l < self.N // 2:
            return
        for s in range(self.n, self.n - active_bit_level(l, self.n), -1):
            block_size = 2 ** s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    path.B[j - branch_size, s - 1] = int(path.B[j, s]) ^ int(
                        path.B[j - branch_size, s]
                    )
                    path.B[j, s - 1] = path.B[j, s]

    def _pm_penalty(self, llr, bit):
        """与 LLR 符号不一致时加 |LLR|"""
        if bit == 0 and llr < 0:
            return abs(llr)
        if bit == 1 and llr >= 0:
            return abs(llr)
        return 0.0

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [_Path(self.N, self.n, llr_ch)]

        for i in range(self.N):
            l = bit_reversed(i, self.n)
            new_paths = []

            for path in paths:
                self._update_llrs(path, l)
                llr = path.L[l, self.n]

                if self.frozen[l]:
                    bit = 0
                    path.pm += self._pm_penalty(llr, bit)
                    path.B[l, self.n] = bit
                    path.u[l] = bit
                    self._update_bits(path, l)
                    new_paths.append(path)
                else:
                    for bit in (0, 1):
                        p = path.copy()
                        p.pm += self._pm_penalty(llr, bit)
                        p.B[l, self.n] = bit
                        p.u[l] = bit
                        self._update_bits(p, l)
                        new_paths.append(p)

            new_paths.sort(key=lambda p: p.pm)
            paths = new_paths[: self.L_size]

        if self.crc_length > 0:
            valid = [p for p in paths if crc_check(p.u, self.crc_length)]
            if valid:
                paths = valid

        best = min(paths, key=lambda p: p.pm)
        return best.u.astype(int), best.pm
