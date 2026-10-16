"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np
from decoder_sc import (
    _prepare_channel_llr,
    _bit_reversed_index,
    _active_llr_level,
    _active_bit_level,
    _upper_llr_log,
    _lower_llr_log,
    f_operation,
)


def _crc_poly(crc_length):
    if crc_length == 8:
        return 0x07
    if crc_length == 16:
        return 0x8005
    raise ValueError("crc_length must be 8 or 16")


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 并附加到信息比特末尾。"""
    info_bits = np.asarray(info_bits, dtype=np.int8)
    poly = _crc_poly(crc_length)
    reg = 0
    for b in info_bits:
        reg ^= int(b) << (crc_length - 1)
        for _ in range(8):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)
    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int
    )
    return np.concatenate([info_bits.astype(int), crc_bits])


def crc_check(bits, crc_length=8):
    """校验 CRC。"""
    bits = np.asarray(bits, dtype=int)
    if len(bits) < crc_length:
        return False
    poly = _crc_poly(crc_length)
    reg = 0
    for b in bits:
        reg ^= int(b) << (crc_length - 1)
        for _ in range(8):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)
    return reg == 0


class _Path:
    __slots__ = ("L", "B", "pm", "u")

    def __init__(self, N, n):
        self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.B = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.pm = 0.0
        self.u = np.zeros(N, dtype=int)


class SCLDecoder:
    """SCL 译码器（路径复制实现，列表规模适中时足够高效）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(self.frozen_bits == 0)[0]

    def _llr_penalty(self, llr, u):
        u_hard = 0 if llr >= 0 else 1
        return 0.0 if u == u_hard else abs(llr)

    def _update_llrs(self, path, l):
        for s in range(self.n - _active_llr_level(l, self.n), self.n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    path.L[j, s + 1] = _upper_llr_log(
                        path.L[j, s], path.L[j + branch_size, s]
                    )
                else:
                    top_bit = path.B[j - branch_size, s + 1]
                    top_bit = 0 if np.isnan(top_bit) else int(top_bit)
                    path.L[j, s + 1] = _lower_llr_log(
                        path.L[j, s], path.L[j - branch_size, s], top_bit
                    )

    def _update_bits(self, path, l):
        if l < self.N / 2:
            return
        for s in range(self.n, self.n - _active_bit_level(l, self.n), -1):
            block_size = 2 ** s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    b_low = int(path.B[j, s]) if not np.isnan(path.B[j, s]) else 0
                    b_high = (
                        int(path.B[j - branch_size, s])
                        if not np.isnan(path.B[j - branch_size, s])
                        else 0
                    )
                    path.B[j - branch_size, s - 1] = b_low ^ b_high
                    path.B[j, s - 1] = b_low

    def decode(self, llr_ch):
        llr = _prepare_channel_llr(llr_ch, self.N)
        paths = [_Path(self.N, self.n)]
        paths[0].L[:, 0] = llr

        for i in range(self.N):
            l = _bit_reversed_index(i, self.n)
            new_paths = []

            for path in paths:
                self._update_llrs(path, l)
                cur_llr = path.L[l, self.n]
                if np.isnan(cur_llr):
                    cur_llr = 0.0

                if l in self.frozen_set:
                    p = _Path(self.N, self.n)
                    p.L = path.L.copy()
                    p.B = path.B.copy()
                    p.pm = path.pm + self._llr_penalty(cur_llr, 0)
                    p.u = path.u.copy()
                    p.B[l, self.n] = 0
                    p.u[l] = 0
                    self._update_bits(p, l)
                    new_paths.append(p)
                else:
                    for bit in (0, 1):
                        p = _Path(self.N, self.n)
                        p.L = path.L.copy()
                        p.B = path.B.copy()
                        p.pm = path.pm + self._llr_penalty(cur_llr, bit)
                        p.u = path.u.copy()
                        p.B[l, self.n] = bit
                        p.u[l] = bit
                        self._update_bits(p, l)
                        new_paths.append(p)

            new_paths.sort(key=lambda p: p.pm)
            paths = new_paths[: self.list_size]

        candidates = paths
        if self.crc_length > 0:
            passed = []
            for p in candidates:
                info_bits = p.u[self.info_indices]
                if crc_check(info_bits, self.crc_length):
                    passed.append(p)
            if passed:
                candidates = passed

        best = min(candidates, key=lambda p: p.pm)
        return best.u.copy(), best.pm
