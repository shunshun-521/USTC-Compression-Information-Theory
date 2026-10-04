"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np

from decoder_sc import (
    _active_bit_level,
    _active_llr_level,
    f_operation,
    g_operation,
)
from encoder import bit_reversed

CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, crc_length):
    mask = (1 << crc_length) - 1
    reg = 0
    for b in bits:
        reg ^= int(b) << (crc_length - 1)
        if reg & (1 << (crc_length - 1)):
            reg = ((reg << 1) ^ poly) & mask
        else:
            reg = (reg << 1) & mask
    return reg


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=np.int64)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    reg = _crc_remainder(info_bits, poly, crc_length)
    crc_bits = np.array([(reg >> i) & 1 for i in range(crc_length - 1, -1, -1)], dtype=np.int64)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC。"""
    bits = np.asarray(bits, dtype=np.int64)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    return _crc_remainder(bits, poly, crc_length) == 0


class _Path:
    __slots__ = ("pm", "L", "B", "u_hat")

    def __init__(self, N, n):
        self.pm = 0.0
        self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.B = np.full((N, n + 1), np.nan)
        self.u_hat = np.zeros(N, dtype=np.int64)


class SCLDecoder:
    """SCL 译码器（路径分裂时复制 L/B 状态）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_set = set(np.where(np.asarray(frozen_bits).astype(int) != 0)[0])
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.array(sorted(set(range(N)) - self.frozen_set), dtype=int)

    def _llr_at(self, path, l):
        L, B = path.L, path.B
        n = self.n
        for s in range(n - _active_llr_level(l, n), n):
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = f_operation(L[j, s], L[j + branch_size, s])
                else:
                    top_bit = B[j - branch_size, s + 1]
                    L[j, s + 1] = g_operation(L[j, s], L[j - branch_size, s], top_bit)
        return L[l, n]

    def _update_bits(self, path, l):
        B = path.B
        n = self.n
        if l < self.N / 2:
            return
        for s in range(n, n - _active_bit_level(l, n), -1):
            block_size = 1 << s
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
        paths = [_Path(self.N, self.n)]
        paths[0].L[:, 0] = llr_ch

        for phi in range(self.N):
            l = bit_reversed(phi, self.n)
            candidates = []

            for pidx, path in enumerate(paths):
                llr = self._llr_at(path, l)
                if l in self.frozen_set:
                    candidates.append((path.pm + self._pm_penalty(llr, 0), pidx, 0))
                else:
                    for u in (0, 1):
                        candidates.append((path.pm + self._pm_penalty(llr, u), pidx, u))

            candidates.sort(key=lambda x: x[0])
            survivors = candidates[: self.list_size]

            new_paths = []
            for pm, pidx, u in survivors:
                parent = paths[pidx]
                child = _Path(self.N, self.n)
                child.pm = pm
                child.L = parent.L.copy()
                child.B = parent.B.copy()
                child.u_hat = parent.u_hat.copy()
                child.B[l, self.n] = u
                child.u_hat[l] = u
                self._update_bits(child, l)
                new_paths.append(child)
            paths = new_paths

        best = None
        if self.crc_length > 0:
            for path in paths:
                info_bits = path.u_hat[self.info_indices]
                if crc_check(info_bits, self.crc_length):
                    if best is None or path.pm < best.pm:
                        best = path
        if best is None:
            best = min(paths, key=lambda p: p.pm)

        return best.u_hat.copy(), best.pm
