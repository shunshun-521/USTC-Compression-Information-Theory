"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math

import numpy as np

from decoder_sc import (
    _active_bit_level,
    _active_llr_level,
    _bit_reversed_scalar,
    f_operation,
    g_operation,
)


CRC_POLYS = {
    8: 0x07,
    16: 0x8005,
}


def _poly_div_crc(message_bits, crc_length):
    poly = CRC_POLYS[crc_length]
    divisor = (1 << crc_length) | poly
    poly_bits = [(divisor >> i) & 1 for i in range(crc_length, -1, -1)]
    reg = list(np.asarray(message_bits, dtype=np.int8)) + [0] * crc_length
    for i in range(len(message_bits)):
        if reg[i]:
            for j, pbit in enumerate(poly_bits):
                if pbit:
                    reg[i + j] ^= 1
    return np.array(reg[-crc_length:], dtype=np.int8)


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=np.int8)
    crc_bits = _poly_div_crc(info_bits, crc_length)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。"""
    poly = CRC_POLYS[crc_length]
    divisor = (1 << crc_length) | poly
    poly_bits = [(divisor >> i) & 1 for i in range(crc_length, -1, -1)]
    reg = list(np.asarray(bits, dtype=np.int8))
    for i in range(len(bits) - crc_length):
        if reg[i]:
            for j, pbit in enumerate(poly_bits):
                if pbit:
                    reg[i + j] ^= 1
    return all(x == 0 for x in reg[-crc_length:])


class Path:
    """单条 SCL 路径。"""

    __slots__ = ("pm", "L", "B", "u_hat")

    def __init__(self, N, n, llr_ch):
        self.pm = 0.0
        self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.B = np.zeros((N, n + 1), dtype=np.int8)
        self.u_hat = np.zeros(N, dtype=np.int8)
        self.L[:, 0] = llr_ch

    def copy(self):
        p = Path.__new__(Path)
        p.pm = self.pm
        p.L = self.L.copy()
        p.B = self.B.copy()
        p.u_hat = self.u_hat.copy()
        return p


class SCLDecoder:
    """SCL 译码器（Permuted SCD + Lazy Copy）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]

    def _update_llrs(self, path, leaf):
        for s in range(self.n - _active_llr_level(leaf, self.n), self.n):
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(leaf, self.N, block_size):
                if j % block_size < branch_size:
                    path.L[j, s + 1] = f_operation(path.L[j, s], path.L[j + branch_size, s])
                else:
                    path.L[j, s + 1] = g_operation(
                        path.L[j - branch_size, s],
                        path.L[j, s],
                        path.B[j - branch_size, s + 1],
                    )

    def _update_bits(self, path, leaf):
        if leaf < self.N // 2:
            return
        for s in range(self.n, self.n - _active_bit_level(leaf, self.n), -1):
            block_size = 1 << s
            branch_size = block_size // 2
            for j in range(leaf, -1, -block_size):
                if j % block_size >= branch_size:
                    path.B[j - branch_size, s - 1] = path.B[j, s] ^ path.B[j - branch_size, s]
                    path.B[j, s - 1] = path.B[j, s]

    def _pm_penalty(self, llr, bit):
        hard = 0 if llr >= 0 else 1
        return 0.0 if bit == hard else abs(llr)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [Path(self.N, self.n, llr_ch)]

        for i in range(self.N):
            leaf = _bit_reversed_scalar(i, self.n)
            candidates = []

            for path in paths:
                self._update_llrs(path, leaf)
                llr = path.L[leaf, self.n]

                if leaf in self.frozen_set:
                    new_path = path.copy()
                    new_path.pm += self._pm_penalty(llr, 0)
                    new_path.B[leaf, self.n] = 0
                    new_path.u_hat[leaf] = 0
                    self._update_bits(new_path, leaf)
                    candidates.append(new_path)
                else:
                    for bit in (0, 1):
                        new_path = path.copy()
                        new_path.pm += self._pm_penalty(llr, bit)
                        new_path.B[leaf, self.n] = bit
                        new_path.u_hat[leaf] = bit
                        self._update_bits(new_path, leaf)
                        candidates.append(new_path)

            candidates.sort(key=lambda p: p.pm)
            paths = candidates[: self.list_size]

        if self.crc_length > 0:
            valid = []
            for p in paths:
                info_bits = p.u_hat[self.info_indices]
                if crc_check(info_bits, self.crc_length):
                    valid.append(p)
            best = min(valid, key=lambda p: p.pm) if valid else paths[0]
        else:
            best = paths[0]

        return best.u_hat.copy(), best.pm
