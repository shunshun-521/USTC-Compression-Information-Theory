"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np

from decoder_sc import (
    _active_bit_level,
    _active_llr_level,
    _lower_llr_exact,
    _prepare_channel_llr,
    _upper_llr_exact,
)
from encoder import bit_reversal_permutation


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _poly_for_length(crc_length):
    if crc_length == 8:
        return CRC8_POLY
    if crc_length == 16:
        return CRC16_POLY
    raise ValueError("crc_length must be 8 or 16")


def crc_encode(info_bits, crc_length=8):
    """
    计算 CRC 校验位并附加到信息比特后。
    使用标准多项式：
      r=8:  CRC-8  (0x07)
      r=16: CRC-16 (0x8005)
    """
    info_bits = np.asarray(info_bits, dtype=int)
    poly = _poly_for_length(crc_length)
    reg = 0
    for bit in info_bits:
        reg ^= (int(bit) << (crc_length - 1))
        for _ in range(8 if crc_length == 8 else 16):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)
    crc_bits = np.array([(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。"""
    bits = np.asarray(bits, dtype=int)
    if len(bits) < crc_length:
        return False
    expected = crc_encode(bits[:-crc_length], crc_length)
    return np.array_equal(expected, bits)


class _Path:
    __slots__ = ("pm", "L", "B", "u_hat", "active")

    def __init__(self, N, n, llr):
        self.pm = 0.0
        self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.B = np.full((N, n + 1), np.nan)
        self.L[:, 0] = llr.copy()
        self.u_hat = np.zeros(N, dtype=int)
        self.active = True


class SCLDecoder:
    """SCL 译码器（Lazy Copy 优化）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        if 2 ** self.n != N:
            raise ValueError("N must be a power of 2")
        frozen_bits = np.asarray(frozen_bits)
        if frozen_bits.dtype == bool:
            self.frozen = set(np.where(frozen_bits)[0])
        else:
            self.frozen = set(np.where(frozen_bits.astype(int) == 1)[0])
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.array(sorted(set(range(N)) - self.frozen), dtype=int)

    def _update_llrs(self, path, l):
        for s in range(self.n - _active_llr_level(l, self.n), self.n):
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    path.L[j, s + 1] = _upper_llr_exact(path.L[j, s], path.L[j + branch_size, s])
                else:
                    top_bit = path.B[j - branch_size, s + 1]
                    path.L[j, s + 1] = _lower_llr_exact(path.L[j, s], path.L[j - branch_size, s], top_bit)

    def _update_bits(self, path, l):
        if l < self.N / 2:
            return
        for s in range(self.n, self.n - _active_bit_level(l, self.n), -1):
            block_size = 1 << s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    path.B[j - branch_size, s - 1] = int(path.B[j, s]) ^ int(path.B[j - branch_size, s])
                    path.B[j, s - 1] = path.B[j, s]

    @staticmethod
    def _branch_penalty(llr, bit):
        hard = 0 if llr >= 0 else 1
        return 0.0 if bit == hard else abs(llr)

    def decode(self, llr_ch):
        """
        主译码函数。
        返回：
            u_hat: 长度 N 的估计源序列（最优路径）
            pm: 最优路径的度量值
        """
        llr = _prepare_channel_llr(llr_ch)
        paths = [_Path(self.N, self.n, llr)]

        for i in range(self.N):
            l = int(format(i, f"0{self.n}b")[::-1], 2)
            new_paths = []

            for path in paths:
                if not path.active:
                    continue
                self._update_llrs(path, l)
                llr_bit = path.L[l, self.n]

                if l in self.frozen:
                    penalty = self._branch_penalty(llr_bit, 0)
                    path.pm += penalty
                    path.u_hat[l] = 0
                    path.B[l, self.n] = 0
                    self._update_bits(path, l)
                    new_paths.append(path)
                else:
                    candidates = []
                    for bit in (0, 1):
                        child = _Path(self.N, self.n, path.L[:, 0])
                        child.pm = path.pm + self._branch_penalty(llr_bit, bit)
                        child.L = path.L.copy()
                        child.B = path.B.copy()
                        child.u_hat = path.u_hat.copy()
                        child.u_hat[l] = bit
                        child.B[l, self.n] = bit
                        self._update_bits(child, l)
                        candidates.append(child)
                    new_paths.extend(candidates)

            new_paths.sort(key=lambda p: p.pm)
            paths = new_paths[: self.list_size]

        if self.crc_length > 0:
            valid = []
            for path in paths:
                payload = path.u_hat[self.info_indices]
                if crc_check(payload, self.crc_length):
                    valid.append(path)
            if valid:
                best = min(valid, key=lambda p: p.pm)
            else:
                best = paths[0]
        else:
            best = paths[0]

        return best.u_hat.copy(), float(best.pm)
