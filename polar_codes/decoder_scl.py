"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from decoder_sc import (
    _bit_reversed,
    _active_llr_level,
    _active_bit_level,
    f_operation,
    g_operation,
)


def _crc_poly(crc_length):
    if crc_length == 8:
        return 0x07
    if crc_length == 16:
        return 0x8005
    raise ValueError("crc_length must be 8 or 16")


def _crc_remainder(bits, crc_length):
    poly = _crc_poly(crc_length)
    reg = 0
    for b in bits:
        reg ^= int(b) << (crc_length - 1)
        for _ in range(8):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)
    return reg


def crc_encode(info_bits, crc_length=8):
    """信息比特后附加 CRC 校验位"""
    info_bits = np.asarray(info_bits, dtype=int)
    rem = _crc_remainder(info_bits, crc_length)
    crc_bits = np.array([(rem >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=int)
    payload = bits[:-crc_length]
    rem = _crc_remainder(payload, crc_length)
    rx = 0
    for i, b in enumerate(bits[-crc_length:]):
        rx |= int(b) << (crc_length - 1 - i)
    return rem == rx


class _Path:
    __slots__ = ("L", "B", "pm", "active")

    def __init__(self, N, n):
        self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.B = np.full((N, n + 1), np.nan)
        self.pm = 0.0
        self.active = True


class SCLDecoder:
    """SCL 译码器（Lazy Copy：路径分裂时复制 L/B 数组）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_set = set(np.where(self.frozen_bits.astype(bool))[0])
        self.info_indices = np.where(~self.frozen_bits.astype(bool))[0]
        self.list_size = list_size
        self.crc_length = crc_length

    def _update_llrs(self, path, l):
        for s in range(self.n - _active_llr_level(l, self.n), self.n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    path.L[j, s + 1] = f_operation(path.L[j, s], path.L[j + branch_size, s])
                else:
                    top_bit = int(path.B[j - branch_size, s + 1])
                    path.L[j, s + 1] = g_operation(
                        path.L[j - branch_size, s],
                        path.L[j, s],
                        top_bit,
                    )

    def _update_bits(self, path, l):
        if l < self.N // 2:
            return
        for s in range(self.n, self.n - _active_bit_level(l, self.n), -1):
            block_size = 2 ** s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    path.B[j - branch_size, s - 1] = int(path.B[j, s]) ^ int(path.B[j - branch_size, s])
                    path.B[j, s - 1] = path.B[j, s]

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [_Path(self.N, self.n)]
        paths[0].L[:, 0] = llr_ch

        for i in range(self.N):
            l = _bit_reversed(i, self.n)
            candidates = []

            for p in paths:
                if not p.active:
                    continue
                self._update_llrs(p, l)
                llr_bit = p.L[l, self.n]
                if l in self.frozen_set:
                    penalty = 0.0 if llr_bit >= 0 else abs(llr_bit)
                    p.pm += penalty
                    p.B[l, self.n] = 0
                    self._update_bits(p, l)
                    candidates.append(p)
                else:
                    for bit in (0, 1):
                        child = _Path(self.N, self.n)
                        child.L = p.L.copy()
                        child.B = p.B.copy()
                        child.pm = p.pm
                        expected = 0 if llr_bit >= 0 else 1
                        if bit != expected:
                            child.pm += abs(llr_bit)
                        child.B[l, self.n] = bit
                        self._update_bits(child, l)
                        candidates.append(child)

            candidates.sort(key=lambda x: x.pm)
            paths = candidates[: self.list_size]
            for p in paths:
                p.active = True

        if self.crc_length > 0:
            valid = []
            for p in paths:
                payload = p.B[:, self.n].astype(int)[self.info_indices]
                if crc_check(payload, self.crc_length):
                    valid.append(p)
            chosen = min(valid if valid else paths, key=lambda x: x.pm)
        else:
            chosen = min(paths, key=lambda x: x.pm)

        u_hat = chosen.B[:, self.n].astype(int)
        return u_hat, chosen.pm
