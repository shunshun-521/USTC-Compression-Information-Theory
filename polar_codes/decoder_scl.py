"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
import math

from encoder import bit_reversed
from decoder_sc import (
    f_operation,
    g_operation,
    _active_llr_level,
    _active_bit_level,
)


# ==================== CRC 工具 ====================

_CRC8_POLY = 0x07
_CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, crc_length):
    reg = 0
    for b in bits:
        reg ^= int(b) << (crc_length - 1)
        if reg & (1 << (crc_length - 1)):
            reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
        else:
            reg = (reg << 1) & ((1 << crc_length) - 1)
    return reg


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=np.int8)
    poly = _CRC8_POLY if crc_length == 8 else _CRC16_POLY
    rem = _crc_remainder(info_bits, poly, crc_length)
    crc_bits = np.array([(rem >> i) & 1 for i in range(crc_length - 1, -1, -1)], dtype=np.int8)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC"""
    bits = np.asarray(bits, dtype=np.int8)
    if len(bits) < crc_length:
        return False
    poly = _CRC8_POLY if crc_length == 8 else _CRC16_POLY
    rem = _crc_remainder(bits, poly, crc_length)
    return rem == 0


class _Path:
    __slots__ = ("L", "B", "pm", "u_hat")

    def __init__(self, N, n, llr_ch):
        self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.B = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.L[:, 0] = llr_ch
        self.pm = 0.0
        self.u_hat = np.zeros(N, dtype=np.int8)


class SCLDecoder:
    """SCL 译码器（Lazy Copy：路径分裂时复制数组）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_set = set(int(i) for i in np.where(self.frozen_bits)[0])
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]

    def _update_llrs(self, path, l):
        L, B = path.L, path.B
        n = self.n
        for s in range(n - _active_llr_level(l, n), n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = f_operation(L[j, s], L[j + branch_size, s])
                else:
                    top_bit = B[j - branch_size, s + 1]
                    L[j, s + 1] = g_operation(L[j, s], L[j - branch_size, s], top_bit)

    def _update_bits(self, path, l):
        B = path.B
        n = self.n
        if l < self.N // 2:
            return
        for s in range(n, n - _active_bit_level(l, n), -1):
            block_size = 2 ** s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                    B[j, s - 1] = B[j, s]

    def _pm_penalty(self, llr, u):
        u_hard = 0 if llr >= 0 else 1
        return 0.0 if u == u_hard else abs(llr)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [_Path(self.N, self.n, llr_ch)]

        for i in range(self.N):
            l = bit_reversed(i, self.n)
            new_paths = []

            for path in paths:
                self._update_llrs(path, l)
                llr = path.L[l, self.n]

                if l in self.frozen_set:
                    path.pm += self._pm_penalty(llr, 0)
                    path.B[l, self.n] = 0
                    path.u_hat[l] = 0
                    self._update_bits(path, l)
                    new_paths.append(path)
                else:
                    for u in (0, 1):
                        p0 = _Path(self.N, self.n, llr_ch)
                        p0.L = path.L.copy()
                        p0.B = path.B.copy()
                        p0.pm = path.pm + self._pm_penalty(llr, u)
                        p0.u_hat = path.u_hat.copy()
                        p0.B[l, self.n] = u
                        p0.u_hat[l] = u
                        self._update_bits(p0, l)
                        new_paths.append(p0)

            new_paths.sort(key=lambda p: p.pm)
            paths = new_paths[: self.list_size]

        # 选择输出路径
        best = paths[0]
        if self.crc_length > 0:
            valid = [p for p in paths if crc_check(p.u_hat[self.info_indices], self.crc_length)]
            if valid:
                best = min(valid, key=lambda p: p.pm)

        return best.u_hat.astype(int), best.pm
