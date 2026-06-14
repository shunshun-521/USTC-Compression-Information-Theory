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
    f_operation,
    g_operation,
)


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    reg = 0
    shift = crc_length - 1
    mask = (1 << crc_length) - 1
    top = 1 << shift

    for bit in info_bits:
        reg ^= int(bit) << shift
        for _ in range(crc_length):
            if reg & top:
                reg = ((reg << 1) & mask) ^ poly
            else:
                reg = (reg << 1) & mask

    crc_bits = np.array([(reg >> (shift - i)) & 1 for i in range(crc_length)], dtype=int)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits 是否满足 CRC。"""
    bits = np.asarray(bits, dtype=int)
    return np.array_equal(bits, crc_encode(bits[:-crc_length], crc_length))


def _pm_penalty(llr, u):
    hard = 0 if llr >= 0 else 1
    return 0.0 if u == hard else abs(llr)


class _Path:
    __slots__ = ("pm", "L", "B", "u_hat")

    def __init__(self, N, n, llr_ch):
        self.pm = 0.0
        self.L = np.zeros((N, n + 1), dtype=np.float64)
        self.B = np.zeros((N, n + 1), dtype=np.int8)
        self.L[:, 0] = llr_ch
        self.u_hat = np.zeros(N, dtype=int)


class SCLDecoder:
    """SCL 译码器。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_positions = np.where(~self.frozen_bits)[0]

    def _update_llrs(self, path, l):
        n = self.n
        start = n - _active_llr_level(l, n)
        for s in range(start, n):
            block = 1 << (s + 1)
            half = block // 2
            for j in range(l, self.N, block):
                if j % block < half:
                    path.L[j, s + 1] = f_operation(path.L[j, s], path.L[j + half, s])
                else:
                    top_bit = path.B[j - half, s + 1]
                    path.L[j, s + 1] = g_operation(
                        path.L[j - half, s], path.L[j, s], top_bit
                    )

    def _update_bits(self, path, l):
        if l < self.N // 2:
            return
        n = self.n
        for s in range(n, n - _active_bit_level(l, n), -1):
            block = 1 << s
            half = block // 2
            for j in range(l, -1, -block):
                if j % block >= half:
                    path.B[j - half, s - 1] = path.B[j, s] ^ path.B[j - half, s]
                    path.B[j, s - 1] = path.B[j, s]

    def decode(self, llr_ch):
        from encoder import bit_reversal_permutation

        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        llr_ch = llr_ch[bit_reversal_permutation(self.N)]
        n = self.n
        paths = [_Path(self.N, n, llr_ch)]

        for phi in range(self.N):
            l = _bit_reversed(phi, n)
            candidates = []

            for path in paths:
                self._update_llrs(path, l)
                llr0 = path.L[l, n]

                if self.frozen_bits[l]:
                    child = _Path(self.N, n, llr_ch)
                    child.pm = path.pm + _pm_penalty(llr0, 0)
                    child.L = path.L.copy()
                    child.B = path.B.copy()
                    child.u_hat = path.u_hat.copy()
                    child.B[l, n] = 0
                    child.u_hat[l] = 0
                    candidates.append(child)
                else:
                    for u in (0, 1):
                        child = _Path(self.N, n, llr_ch)
                        child.pm = path.pm + _pm_penalty(llr0, u)
                        child.L = path.L.copy()
                        child.B = path.B.copy()
                        child.u_hat = path.u_hat.copy()
                        child.B[l, n] = u
                        child.u_hat[l] = u
                        candidates.append(child)

            candidates.sort(key=lambda p: p.pm)
            paths = candidates[: self.list_size]

            for path in paths:
                self._update_bits(path, l)

        if self.crc_length > 0:
            valid = []
            for path in paths:
                info_bits = path.u_hat[self.info_positions]
                if crc_check(info_bits, self.crc_length):
                    valid.append((path.pm, path.u_hat.copy()))
            if valid:
                valid.sort(key=lambda x: x[0])
                return valid[0][1], valid[0][0]

        best = min(paths, key=lambda p: p.pm)
        return best.u_hat.copy(), best.pm
