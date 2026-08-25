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
    _lower_llr,
    _prepare_channel_llr,
    _upper_llr,
)


def _crc_poly(crc_length):
    if crc_length == 8:
        return 0x07
    if crc_length == 16:
        return 0x8005
    raise ValueError("crc_length must be 8 or 16")


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=np.int8)
    poly = _crc_poly(crc_length)
    mask = (1 << crc_length) - 1
    reg = 0
    for bit in info_bits:
        reg ^= int(bit) << (crc_length - 1)
        for _ in range(crc_length):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & mask
            else:
                reg = (reg << 1) & mask
    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=np.int8
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。"""
    bits = np.asarray(bits, dtype=np.int8)
    if len(bits) < crc_length:
        return False
    expected = crc_encode(bits[:-crc_length], crc_length)
    return np.array_equal(bits, expected)


class _Path:
    __slots__ = ("pm", "L", "B")

    def __init__(self, N, n, llr_ch):
        self.pm = 0.0
        self.L = np.zeros((N, n + 1), dtype=np.float64)
        self.B = np.zeros((N, n + 1), dtype=np.int8)
        self.L[:, 0] = llr_ch


class SCLDecoder:
    """SCL 译码器（含 Lazy Copy 优化）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length

    def _update_llrs(self, paths, l):
        n = self.n
        N = self.N
        for path in paths:
            for s in range(n - _active_llr_level(l, n), n):
                block_size = 1 << (s + 1)
                branch_size = block_size // 2
                for j in range(l, N, block_size):
                    if j % block_size < branch_size:
                        path.L[j, s + 1] = _upper_llr(
                            path.L[j, s], path.L[j + branch_size, s]
                        )
                    else:
                        top_bit = path.B[j - branch_size, s + 1]
                        path.L[j, s + 1] = _lower_llr(
                            path.L[j, s], path.L[j - branch_size, s], top_bit
                        )

    def _update_bits(self, paths, l):
        if l < self.N // 2:
            return
        n = self.n
        N = self.N
        for path in paths:
            for s in range(n, n - _active_bit_level(l, n), -1):
                block_size = 1 << s
                branch_size = block_size // 2
                for j in range(l, -1, -block_size):
                    if j % block_size >= branch_size:
                        path.B[j - branch_size, s - 1] = (
                            path.B[j, s] ^ path.B[j - branch_size, s]
                        )
                        path.B[j, s - 1] = path.B[j, s]

    def decode(self, llr_ch):
        llr_ch = _prepare_channel_llr(llr_ch)
        N, n = self.N, self.n
        frozen = self.frozen_bits

        paths = [_Path(N, n, llr_ch)]

        for i in range(N):
            l = _bit_reversed(i, n)
            self._update_llrs(paths, l)

            candidates = []
            for pid, path in enumerate(paths):
                llr = path.L[l, n]
                if frozen[l]:
                    penalty = 0.0 if llr >= 0 else abs(llr)
                    candidates.append((path.pm + penalty, pid, 0))
                else:
                    for u in (0, 1):
                        expected = 0 if llr >= 0 else 1
                        penalty = 0.0 if u == expected else abs(llr)
                        candidates.append((path.pm + penalty, pid, u))

            candidates.sort(key=lambda x: x[0])
            candidates = candidates[: self.list_size]

            new_paths = []
            for pm, pid, u in candidates:
                path = _Path(N, n, llr_ch)
                path.pm = pm
                path.L[:] = paths[pid].L
                path.B[:] = paths[pid].B
                path.B[l, n] = 0 if frozen[l] else u
                new_paths.append(path)

            self._update_bits(new_paths, l)
            paths = new_paths

        if self.crc_length > 0:
            info_idx = np.where(~self.frozen_bits)[0]
            passed = [
                p
                for p in paths
                if crc_check(p.B[info_idx, n], self.crc_length)
            ]
            pool = passed if passed else paths
            best = min(pool, key=lambda p: p.pm)
        else:
            best = min(paths, key=lambda p: p.pm)

        return best.B[:, n].astype(int), best.pm
