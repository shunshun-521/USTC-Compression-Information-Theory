"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np
from decoder_sc import (
    f_operation,
    g_operation,
    _bit_reversed,
    _active_llr_level,
    _active_bit_level,
)


_CRC8_POLY = 0x07
_CRC16_POLY = 0x8005


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=int)
    poly = _CRC8_POLY if crc_length == 8 else _CRC16_POLY
    reg = 0
    mask = (1 << crc_length) - 1
    for b in info_bits:
        reg ^= int(b) << (crc_length - 1)
        for _ in range(8):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & mask
            else:
                reg = (reg << 1) & mask
    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC"""
    bits = np.asarray(bits, dtype=int)
    if len(bits) < crc_length:
        return False
    expected = crc_encode(bits[:-crc_length], crc_length)
    return np.array_equal(bits[-crc_length:], expected[-crc_length:])


class _Path:
    __slots__ = ("L", "B", "u_hat", "pm")

    def __init__(self, N, n, llr_ch):
        self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.B = np.full((N, n + 1), np.nan)
        self.L[:, 0] = llr_ch
        self.u_hat = np.zeros(N, dtype=int)
        self.pm = 0.0


class SCLDecoder:
    """SCL 译码器（每条路径维护独立 L/B 矩阵）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_idx = np.where(~self.frozen_bits)[0]
        self.decode_order = [_bit_reversed(i, self.n) for i in range(N)]

    def _update_llrs(self, path, l):
        n = self.n
        N = self.N
        for s in range(n - _active_llr_level(l, n), n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, N, block_size):
                if j % block_size < branch_size:
                    path.L[j, s + 1] = f_operation(path.L[j, s], path.L[j + branch_size, s])
                else:
                    top_bit = path.B[j - branch_size, s + 1]
                    path.L[j, s + 1] = g_operation(
                        path.L[j - branch_size, s], path.L[j, s], top_bit
                    )

    def _update_bits(self, path, l):
        n = self.n
        N = self.N
        if l < N / 2:
            return
        for s in range(n, n - _active_bit_level(l, n), -1):
            block_size = 2**s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    path.B[j - branch_size, s - 1] = int(path.B[j, s]) ^ int(
                        path.B[j - branch_size, s]
                    )
                    path.B[j, s - 1] = path.B[j, s]

    def _pm_penalty(self, llr_val, u_bit):
        hard = 0 if llr_val >= 0 else 1
        return 0.0 if u_bit == hard else abs(llr_val)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        L_size = self.list_size
        paths = [_Path(self.N, self.n, llr_ch.copy())]

        for l in self.decode_order:
            new_paths = []
            for path in paths:
                self._update_llrs(path, l)
                llr0 = path.L[l, self.n]

                if self.frozen_bits[l]:
                    path.pm += self._pm_penalty(llr0, 0)
                    path.B[l, self.n] = 0
                    path.u_hat[l] = 0
                    self._update_bits(path, l)
                    new_paths.append(path)
                else:
                    for u_cand in (0, 1):
                        p2 = _Path(self.N, self.n, path.L[:, 0].copy())
                        p2.L[:, 1:] = path.L[:, 1:].copy()
                        p2.B[:, 1:] = path.B[:, 1:].copy()
                        p2.u_hat = path.u_hat.copy()
                        p2.pm = path.pm + self._pm_penalty(llr0, u_cand)
                        p2.B[l, self.n] = u_cand
                        p2.u_hat[l] = u_cand
                        self._update_bits(p2, l)
                        new_paths.append(p2)

            new_paths.sort(key=lambda p: p.pm)
            paths = new_paths[:L_size]

        crc_paths = []
        if self.crc_length > 0:
            for p in paths:
                info_bits = p.u_hat[self.info_idx]
                if crc_check(info_bits, self.crc_length):
                    crc_paths.append(p)

        best = min(crc_paths if crc_paths else paths, key=lambda p: p.pm)
        return best.u_hat.copy(), best.pm
