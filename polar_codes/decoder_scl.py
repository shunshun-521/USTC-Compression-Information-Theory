"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from decoder_sc import (
    _active_bit_level,
    _active_llr_level,
    _bit_reversed_index,
    f_operation,
    g_operation,
)


def crc_encode(info_bits, crc_length=8):
    """CRC-8 (0x07) 或 CRC-16 (0x8005)，返回信息位 + CRC"""
    info_bits = np.asarray(info_bits, dtype=np.int8)
    if crc_length == 8:
        poly = 0x07
        reg = 0
        for bit in info_bits:
            reg ^= int(bit) << 7
            for _ in range(8):
                if reg & 0x80:
                    reg = ((reg << 1) ^ poly) & 0xFF
                else:
                    reg = (reg << 1) & 0xFF
        crc_bits = np.array([(reg >> (7 - i)) & 1 for i in range(8)], dtype=np.int8)
    elif crc_length == 16:
        poly = 0x8005
        reg = 0
        for bit in info_bits:
            reg ^= int(bit) << 15
            for _ in range(16):
                if reg & 0x8000:
                    reg = ((reg << 1) ^ poly) & 0xFFFF
                else:
                    reg = (reg << 1) & 0xFFFF
        crc_bits = np.array([(reg >> (15 - i)) & 1 for i in range(16)], dtype=np.int8)
    else:
        raise ValueError("crc_length must be 8 or 16")
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=np.int8)
    expected = crc_encode(bits[:-crc_length], crc_length)
    return np.array_equal(bits, expected)


class _Path:
    __slots__ = ("L", "B", "pm", "u")

    def __init__(self, N, n, llr_ch):
        self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.B = np.zeros((N, n + 1), dtype=np.int8)
        self.L[:, 0] = llr_ch
        self.pm = 0.0
        self.u = np.zeros(N, dtype=np.int8)


class SCLDecoder:
    """SCL 译码器"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.list_size = list_size
        self.crc_length = crc_length

    def _update_llrs(self, path, l):
        N, n = self.N, self.n
        L, B = path.L, path.B
        start = n - _active_llr_level(l, n)
        for s in range(start, n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = f_operation(L[j, s], L[j + branch_size, s])
                else:
                    L[j, s + 1] = g_operation(
                        L[j - branch_size, s], L[j, s], B[j - branch_size, s + 1]
                    )

    def _update_bits(self, path, l):
        N, n = self.N, self.n
        B = path.B
        if l < N // 2:
            return
        start = n - _active_bit_level(l, n)
        for s in range(n, start, -1):
            block_size = 2 ** s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                    B[j, s - 1] = B[j, s]

    def _branch_path(self, path, l, bit, llr_ch):
        pen = 0.0 if (0 if llr_ch[l, self.n] >= 0 else 1) == bit else abs(llr_ch[l, self.n])
        # use actual llr from path after update
        new = _Path(self.N, self.n, path.L[:, 0])
        new.L[:, :] = path.L
        new.B[:, :] = path.B
        new.pm = path.pm
        new.u[:] = path.u
        return new, pen

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [_Path(self.N, self.n, llr_ch)]

        for i in range(self.N):
            l = _bit_reversed_index(i, self.n)
            candidates = []

            for path in paths:
                self._update_llrs(path, l)
                llr = path.L[l, self.n]

                if l in self.frozen_set:
                    bit = 0
                    pen = 0.0 if llr >= 0 else abs(llr)
                    new = _Path(self.N, self.n, llr_ch)
                    new.L[:, :] = path.L
                    new.B[:, :] = path.B
                    new.pm = path.pm + pen
                    new.u[:] = path.u
                    new.B[l, self.n] = 0
                    new.u[l] = 0
                    self._update_bits(new, l)
                    candidates.append(new)
                else:
                    for bit in (0, 1):
                        pen = 0.0 if (0 if llr >= 0 else 1) == bit else abs(llr)
                        new = _Path(self.N, self.n, llr_ch)
                        new.L[:, :] = path.L
                        new.B[:, :] = path.B
                        new.pm = path.pm + pen
                        new.u[:] = path.u
                        new.B[l, self.n] = bit
                        new.u[l] = bit
                        self._update_bits(new, l)
                        candidates.append(new)

            candidates.sort(key=lambda p: p.pm)
            paths = candidates[: self.list_size]

        info_positions = sorted(set(range(self.N)) - self.frozen_set)

        if self.crc_length > 0:

            def _crc_ok(uvec):
                msg = uvec[info_positions]
                return crc_check(msg, self.crc_length)

            valid = [p for p in paths if _crc_ok(p.u)]
            best = min(valid if valid else paths, key=lambda p: p.pm)
        else:
            best = min(paths, key=lambda p: p.pm)

        return best.u.astype(int), best.pm
