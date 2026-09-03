"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math

import numpy as np

from decoder_sc import (
    _active_bit_level,
    _active_llr_level,
    _frozen_index_set,
    f_operation,
    g_operation,
)
from encoder import bit_reversed


CRC_POLYNOMIALS = {
    8: 0x07,
    16: 0x8005,
}


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    poly = CRC_POLYNOMIALS[crc_length]
    info_bits = np.asarray(info_bits, dtype=int)
    reg = 0
    for bit in info_bits:
        reg <<= 1
        reg |= int(bit)
        if reg & (1 << crc_length):
            reg ^= poly
    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。"""
    if crc_length == 0:
        return True
    poly = CRC_POLYNOMIALS[crc_length]
    bits = np.asarray(bits, dtype=int)
    reg = 0
    for bit in bits:
        reg <<= 1
        reg |= int(bit)
        if reg & (1 << crc_length):
            reg ^= poly
    return reg == 0


class _Path:
    __slots__ = ("pm", "L", "B", "u_hat")

    def __init__(self, N, n):
        self.pm = 0.0
        self.L = np.zeros((N, n + 1), dtype=np.float64)
        self.B = np.zeros((N, n + 1), dtype=np.int8)
        self.u_hat = np.zeros(N, dtype=int)


class SCLDecoder:
    """SCL 译码器（Permuted SCD + Lazy Copy）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.frozen_set = _frozen_index_set(frozen_bits)
        self.list_size = list_size
        self.crc_length = crc_length

    def _copy_path(self, src):
        dst = _Path(self.N, self.n)
        dst.pm = src.pm
        dst.L = src.L.copy()
        dst.B = src.B.copy()
        dst.u_hat = src.u_hat.copy()
        return dst

    def _update_llrs(self, path, l):
        for s in range(self.n - _active_llr_level(l, self.n), self.n):
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    path.L[j, s + 1] = f_operation(path.L[j, s], path.L[j + branch_size, s])
                else:
                    top_bit = path.B[j - branch_size, s + 1]
                    path.L[j, s + 1] = g_operation(
                        path.L[j - branch_size, s], path.L[j, s], top_bit
                    )

    def _update_bits(self, path, l):
        if l < self.N // 2:
            return
        for s in range(self.n, self.n - _active_bit_level(l, self.n), -1):
            block_size = 1 << s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    path.B[j - branch_size, s - 1] = path.B[j, s] ^ path.B[j - branch_size, s]
                    path.B[j, s - 1] = path.B[j, s]

    def _pm_penalty(self, llr_val, bit):
        hard = 0 if llr_val >= 0 else 1
        return 0.0 if bit == hard else abs(llr_val)

    def decode(self, llr_ch):
        """主译码函数。返回 u_hat, pm"""
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = []
        root = _Path(self.N, self.n)
        root.L[:, 0] = llr_ch
        paths.append(root)

        for phase in range(self.N):
            l = bit_reversed(phase, self.n)
            candidates = []

            for path in paths:
                self._update_llrs(path, l)
                cur_llr = path.L[l, self.n]

                if l in self.frozen_set:
                    new_path = self._copy_path(path)
                    new_path.pm += self._pm_penalty(cur_llr, 0)
                    new_path.B[l, self.n] = 0
                    new_path.u_hat[l] = 0
                    self._update_bits(new_path, l)
                    candidates.append(new_path)
                else:
                    for bit in (0, 1):
                        new_path = self._copy_path(path)
                        new_path.pm += self._pm_penalty(cur_llr, bit)
                        new_path.B[l, self.n] = bit
                        new_path.u_hat[l] = bit
                        self._update_bits(new_path, l)
                        candidates.append(new_path)

            candidates.sort(key=lambda p: p.pm)
            paths = candidates[: self.list_size]

        paths.sort(key=lambda p: p.pm)
        if self.crc_length > 0:
            valid = [p for p in paths if crc_check(p.u_hat, self.crc_length)]
            best = valid[0] if valid else paths[0]
        else:
            best = paths[0]

        return best.u_hat.copy(), best.pm
