"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np
from encoder import bit_reversed
from decoder_sc import (
    f_operation,
    g_operation,
    _frozen_mask,
    active_llr_level,
    active_bit_level,
)


def _crc_process(bits, crc_length):
    if crc_length == 8:
        poly = 0x07
    elif crc_length == 16:
        poly = 0x8005
    else:
        raise ValueError("crc_length must be 8 or 16")
    mask = (1 << crc_length) - 1
    reg = 0
    for bit in bits:
        msb = (reg >> (crc_length - 1)) & 1
        reg = (reg << 1) & mask
        if int(bit) ^ msb:
            reg ^= poly
    return reg


def crc_encode(info_bits, crc_length=8):
    info_bits = np.asarray(info_bits, dtype=np.int8)
    rem = _crc_process(info_bits, crc_length)
    crc_bits = np.array(
        [(rem >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=np.int8
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=np.int8)
    return _crc_process(bits, crc_length) == 0


class _Path:
    __slots__ = ("pm", "L", "B")

    def __init__(self, N, n, llr):
        self.pm = 0.0
        self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.B = np.full((N, n + 1), np.nan)
        self.L[:, 0] = llr

    def copy(self):
        p = _Path(self.L.shape[0], self.L.shape[1] - 1, self.L[:, 0])
        p.pm = self.pm
        p.L = self.L.copy()
        p.B = self.B.copy()
        return p


class SCLDecoder:
    """SCL 译码器（置换 SC + 路径度量）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen = _frozen_mask(frozen_bits)
        self.list_size = list_size
        self.crc_length = crc_length
        self.decode_order = [bit_reversed(i, self.n) for i in range(N)]

    def _update_llrs(self, path, l):
        L, B = path.L, path.B
        for s in range(self.n - active_llr_level(l, self.n), self.n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = f_operation(L[j, s], L[j + branch_size, s])
                else:
                    top_bit = int(B[j - branch_size, s + 1]) if not np.isnan(B[j - branch_size, s + 1]) else 0
                    L[j, s + 1] = g_operation(L[j, s], L[j - branch_size, s], top_bit)

    def _update_bits(self, path, l):
        if l < self.N / 2:
            return
        B = path.B
        for s in range(self.n, self.n - active_bit_level(l, self.n), -1):
            block_size = 2 ** s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                    B[j, s - 1] = B[j, s]

    def decode(self, llr_ch):
        llr = np.asarray(llr_ch, dtype=np.float64)
        paths = [_Path(self.N, self.n, llr)]

        for l in self.decode_order:
            candidates = []
            for path in paths:
                self._update_llrs(path, l)
                cur_llr = path.L[l, self.n]
                if np.isnan(cur_llr):
                    cur_llr = 0.0

                if self.frozen[l]:
                    p = path.copy()
                    if cur_llr < 0:
                        p.pm += abs(cur_llr)
                    p.B[l, self.n] = 0
                    candidates.append(p)
                else:
                    for u in (0, 1):
                        p = path.copy()
                        u_ml = 0 if cur_llr >= 0 else 1
                        if u != u_ml:
                            p.pm += abs(cur_llr)
                        p.B[l, self.n] = u
                        candidates.append(p)

            candidates.sort(key=lambda p: p.pm)
            paths = candidates[: self.list_size]
            for p in paths:
                self._update_bits(p, l)

        paths.sort(key=lambda p: p.pm)
        info_idx = np.where(~self.frozen)[0]
        if self.crc_length > 0:
            for p in paths:
                payload = p.B[:, self.n].astype(int)[info_idx]
                if crc_check(payload, self.crc_length):
                    return p.B[:, self.n].astype(int), p.pm
        best = paths[0]
        return best.B[:, self.n].astype(int), best.pm
