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
    hard_decision,
    active_llr_level,
    active_bit_level,
)


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_divisor(crc_length):
    """返回含 x^n 项的完整生成多项式（位宽 crc_length+1）"""
    if crc_length == 8:
        return 0b100000111  # x^8 + x^2 + x + 1
    return 0b11000000000000101  # CRC-16: x^16 + x^15 + x^2 + 1


def _crc_remainder(bits, crc_length):
    """标准 CRC 余数（MSB 先行，bit-by-bit 长除法）"""
    divisor = _crc_divisor(crc_length)
    reg = 0
    for bit in bits:
        reg ^= int(bit) << crc_length
        for _ in range(8):
            if reg & (1 << crc_length):
                reg ^= divisor
            reg <<= 1
    return reg >> crc_length


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=int)
    padded = np.concatenate([info_bits, np.zeros(crc_length, dtype=int)])
    remainder = _crc_remainder(padded, crc_length)
    crc_bits = np.array(
        [(remainder >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits 是否通过 CRC 校验"""
    bits = np.asarray(bits, dtype=int)
    return _crc_remainder(bits, crc_length) == 0


class Path:
    """单条译码路径（Lazy Copy）"""

    __slots__ = ("L", "B", "pm", "u_hat", "active")

    def __init__(self, N, n):
        self.L = np.zeros((N, n + 1), dtype=np.float64)
        self.B = np.zeros((N, n + 1), dtype=int)
        self.pm = 0.0
        self.u_hat = np.zeros(N, dtype=int)
        self.active = True


class SCLDecoder:
    """SCL 译码器（含 Lazy Copy 优化）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0, info_indices=None):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.list_size = list_size
        self.crc_length = crc_length
        if info_indices is None:
            self.info_indices = np.array(
                [i for i in range(N) if i not in self.frozen_set], dtype=int
            )
        else:
            self.info_indices = np.asarray(info_indices, dtype=int)

    def _update_llrs(self, path, l):
        for s in range(self.n - active_llr_level(l, self.n), self.n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    path.L[j, s + 1] = f_operation(path.L[j, s], path.L[j + branch_size, s])
                else:
                    path.L[j, s + 1] = g_operation(
                        path.L[j - branch_size, s],
                        path.L[j, s],
                        int(path.B[j - branch_size, s + 1]),
                    )

    def _update_bits(self, path, l):
        if l < self.N // 2:
            return
        for s in range(self.n, self.n - active_bit_level(l, self.n), -1):
            block_size = 2 ** s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    path.B[j - branch_size, s - 1] = int(path.B[j, s]) ^ int(
                        path.B[j - branch_size, s]
                    )
                    path.B[j, s - 1] = path.B[j, s]

    def _pm_penalty(self, llr, bit):
        """路径度量惩罚：与 LLR 符号不一致时加 |LLR|"""
        preferred = 0 if llr >= 0 else 1
        return 0.0 if bit == preferred else abs(llr)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [Path(self.N, self.n)]
        paths[0].L[:, 0] = llr_ch.copy()

        for i in range(self.N):
            l = bit_reversed(i, self.n)
            new_paths = []

            for path in paths:
                if not path.active:
                    continue
                self._update_llrs(path, l)
                llr = path.L[l, self.n]

                if l in self.frozen_set:
                    penalty = self._pm_penalty(llr, 0)
                    path.pm += penalty
                    path.B[l, self.n] = 0
                    path.u_hat[l] = 0
                    self._update_bits(path, l)
                    new_paths.append(path)
                else:
                    for bit in (0, 1):
                        child = Path(self.N, self.n)
                        child.L = path.L.copy()
                        child.B = path.B.copy()
                        child.pm = path.pm + self._pm_penalty(llr, bit)
                        child.u_hat = path.u_hat.copy()
                        child.B[l, self.n] = bit
                        child.u_hat[l] = bit
                        self._update_bits(child, l)
                        new_paths.append(child)

            new_paths.sort(key=lambda p: p.pm)
            paths = new_paths[: self.list_size]

        valid_paths = paths
        if self.crc_length > 0:
            crc_ok = []
            for p in paths:
                info_bits = p.u_hat[self.info_indices]
                if crc_check(info_bits, self.crc_length):
                    crc_ok.append(p)
            if crc_ok:
                valid_paths = crc_ok

        best = min(valid_paths, key=lambda p: p.pm)
        return best.u_hat.astype(int), best.pm
