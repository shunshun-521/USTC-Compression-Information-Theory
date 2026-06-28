"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np

from decoder_sc import (
    active_bit_level,
    active_llr_level,
    f_operation,
    lower_llr,
)
from encoder import bit_reversed_index

CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=np.int8)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY

    reg = 0
    for bit in info_bits:
        reg ^= int(bit) << (crc_length - 1)
        for _ in range(crc_length):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)

    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=np.int8,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC。"""
    bits = np.asarray(bits, dtype=np.int8)
    encoded = crc_encode(bits[:-crc_length], crc_length)
    return np.array_equal(encoded, bits)


class Path:
    """单条 SCL 路径（Lazy Copy）。"""

    __slots__ = ("L", "B", "pm", "u_hat", "copied")

    def __init__(self, n, N, llr_ch, parent=None):
        self.copied = parent is None
        if parent is None:
            self.L = np.zeros((N, n + 1), dtype=np.float64)
            self.B = np.zeros((N, n + 1), dtype=np.int8)
            self.L[:, 0] = llr_ch
            self.u_hat = np.zeros(N, dtype=np.int8)
            self.pm = 0.0
        else:
            self.L = parent.L
            self.B = parent.B
            self.u_hat = parent.u_hat.copy()
            self.pm = parent.pm

    def ensure_copy(self):
        if not self.copied:
            self.L = self.L.copy()
            self.B = self.B.copy()
            self.copied = True


class SCLDecoder:
    """SCL 译码器（含 Lazy Copy 优化）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_indices = set(np.where(self.frozen_bits)[0])
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_positions = np.where(~self.frozen_bits)[0]
        self.decode_order = [bit_reversed_index(i, self.n) for i in range(N)]

    def _update_llr(self, path, l):
        path.ensure_copy()
        for s in range(self.n - active_llr_level(l, self.n), self.n):
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    path.L[j, s + 1] = f_operation(
                        path.L[j, s], path.L[j + branch_size, s]
                    )
                else:
                    path.L[j, s + 1] = lower_llr(
                        path.L[j, s],
                        path.L[j - branch_size, s],
                        path.B[j - branch_size, s + 1],
                    )
        return path.L[l, self.n]

    def _update_bits(self, path, l, bit):
        path.ensure_copy()
        path.u_hat[l] = bit
        path.B[l, self.n] = bit
        if l >= self.N // 2:
            for s in range(self.n, self.n - active_bit_level(l, self.n), -1):
                block_size = 1 << s
                branch_size = block_size // 2
                for j in range(l, -1, -block_size):
                    if j % block_size >= branch_size:
                        path.B[j - branch_size, s - 1] = (
                            path.B[j, s] ^ path.B[j - branch_size, s]
                        )
                        path.B[j, s - 1] = path.B[j, s]

    @staticmethod
    def _pm_penalty(llr, bit):
        hard = 0 if llr >= 0 else 1
        return 0.0 if bit == hard else abs(llr)

    def decode(self, llr_ch):
        """主译码函数。"""
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [Path(self.n, self.N, llr_ch)]

        for l in self.decode_order:
            new_paths = []
            for path in paths:
                llr = self._update_llr(path, l)

                if l in self.frozen_indices:
                    child = Path(self.n, self.N, llr_ch, parent=path)
                    child.pm = path.pm + self._pm_penalty(llr, 0)
                    self._update_bits(child, l, 0)
                    new_paths.append(child)
                else:
                    for bit in (0, 1):
                        child = Path(self.n, self.N, llr_ch, parent=path)
                        child.pm = path.pm + self._pm_penalty(llr, bit)
                        self._update_bits(child, l, bit)
                        new_paths.append(child)

            new_paths.sort(key=lambda p: p.pm)
            paths = new_paths[: self.list_size]

        if self.crc_length > 0:
            valid = [
                p
                for p in paths
                if crc_check(p.u_hat[self.info_positions], self.crc_length)
            ]
            best = min(valid, key=lambda p: p.pm) if valid else paths[0]
        else:
            best = paths[0]

        return best.u_hat.astype(int), best.pm
