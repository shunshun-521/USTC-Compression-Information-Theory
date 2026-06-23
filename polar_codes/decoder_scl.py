"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math

import numpy as np

from decoder_sc import (
    _active_bit_level,
    _active_llr_level,
    _bit_reversed_int,
    _channel_llr_to_decoder,
    f_operation,
    g_operation,
    sc_decode,
)


CRC_POLYNOMIALS = {
    8: 0x07,
    16: 0x8005,
}


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=int).flatten()
    poly = CRC_POLYNOMIALS[crc_length]
    reg = 0
    for bit in info_bits:
        reg ^= int(bit) << (crc_length - 1)
        for _ in range(8):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)
    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits 是否包含正确的 CRC。"""
    if crc_length <= 0:
        return True
    expected = crc_encode(bits[:-crc_length], crc_length)
    return np.array_equal(bits, expected)


def _path_metric_update(pm, llr_val, u_val):
    hard = 0 if llr_val >= 0 else 1
    if u_val != hard:
        return pm + abs(llr_val)
    return pm


class _Path:
    __slots__ = ("L", "B", "pm")

    def __init__(self, N, n, llr):
        self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.B = np.full((N, n + 1), np.nan)
        self.L[:, 0] = llr
        self.pm = 0.0


class SCLDecoder:
    """SCL 译码器（Lazy Copy：路径分裂时复制数组）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits).astype(bool)
        self.list_size = max(1, list_size)
        self.crc_length = crc_length
        self.frozen_set = set(np.where(self.frozen_bits)[0])

    def _update_llrs(self, path, l):
        for s in range(self.n - _active_llr_level(l, self.n), self.n):
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    path.L[j, s + 1] = f_operation(path.L[j, s], path.L[j + branch_size, s])
                else:
                    top_bit = int(path.B[j - branch_size, s + 1])
                    path.L[j, s + 1] = g_operation(
                        path.L[j - branch_size, s], path.L[j, s], top_bit
                    )

    def _update_bits(self, path, l, bit):
        path.B[l, self.n] = bit
        if l < self.N // 2:
            return
        for s in range(self.n, self.n - _active_bit_level(l, self.n), -1):
            block_size = 1 << s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    path.B[j - branch_size, s - 1] = int(path.B[j, s]) ^ int(path.B[j - branch_size, s])
                    path.B[j, s - 1] = path.B[j, s]

    def decode(self, llr_ch):
        if self.list_size == 1 and self.crc_length == 0:
            return sc_decode(llr_ch, self.frozen_bits), 0.0

        llr = _channel_llr_to_decoder(np.asarray(llr_ch, dtype=np.float64))
        paths = [_Path(self.N, self.n, llr)]

        for phi in range(self.N):
            l = _bit_reversed_int(phi, self.n)
            candidates = []

            for path in paths:
                self._update_llrs(path, l)
                llr_val = path.L[l, self.n]

                if l in self.frozen_set:
                    new_path = _Path(self.N, self.n, llr)
                    new_path.L = path.L.copy()
                    new_path.B = path.B.copy()
                    new_path.pm = _path_metric_update(path.pm, llr_val, 0)
                    self._update_bits(new_path, l, 0)
                    candidates.append(new_path)
                else:
                    for u_val in (0, 1):
                        new_path = _Path(self.N, self.n, llr)
                        new_path.L = path.L.copy()
                        new_path.B = path.B.copy()
                        new_path.pm = _path_metric_update(path.pm, llr_val, u_val)
                        self._update_bits(new_path, l, u_val)
                        candidates.append(new_path)

            candidates.sort(key=lambda p: p.pm)
            paths = candidates[: self.list_size]

        if self.crc_length > 0:
            info_positions = np.where(~self.frozen_bits)[0]
            crc_paths = [
                p
                for p in paths
                if crc_check(p.B[:, self.n].astype(int)[info_positions], self.crc_length)
            ]
            if crc_paths:
                paths = crc_paths

        best = min(paths, key=lambda p: p.pm)
        return best.B[:, self.n].astype(int), best.pm
