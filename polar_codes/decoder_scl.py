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
    _prepare_llr,
    f_operation,
    g_operation,
    sc_decode,
)
from encoder import polar_encode


# ==================== CRC 工具 ====================

_CRC8_POLY = 0x07
_CRC16_POLY = 0x8005


def crc_encode(info_bits, crc_length=8):
    """
    计算 CRC 校验位并附加到信息比特后。
    CRC-8: 0x07; CRC-16: 0x8005
    """
    info_bits = np.asarray(info_bits, dtype=int)
    poly = _CRC8_POLY if crc_length == 8 else _CRC16_POLY
    mask = (1 << crc_length) - 1
    msb = 1 << (crc_length - 1)

    reg = 0
    for bit in info_bits:
        reg ^= int(bit) << (crc_length - 1)
        for _ in range(1):
            if reg & msb:
                reg = ((reg << 1) ^ poly) & mask
            else:
                reg = (reg << 1) & mask

    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC"""
    bits = np.asarray(bits, dtype=int)
    if len(bits) < crc_length:
        return False
    recomputed = crc_encode(bits[:-crc_length], crc_length)
    return np.array_equal(bits, recomputed)


def _pm_penalty(llr, bit):
    """路径度量惩罚：与 LLR 符号不一致时加 |LLR|"""
    hard = 0 if llr >= 0 else 1
    return 0.0 if bit == hard else abs(llr)


class _Path:
    """单条 SCL 路径（Lazy Copy）"""
    __slots__ = ("L", "B", "pm", "active")

    def __init__(self, N, n, llr_tree):
        self.L = np.zeros((N, n + 1), dtype=np.float64)
        self.L[:, 0] = llr_tree
        self.B = np.zeros((N, n + 1), dtype=int)
        self.pm = 0.0
        self.active = True

    def copy(self):
        p = _Path.__new__(_Path)
        p.L = self.L.copy()
        p.B = self.B.copy()
        p.pm = self.pm
        p.active = True
        return p


class SCLDecoder:
    """SCL 译码器（含 Lazy Copy 优化）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.list_size = list_size
        self.crc_length = crc_length

    def _update_llrs(self, path, l):
        for s in range(self.n - _active_llr_level(l, self.n), self.n):
            block_size = 1 << (s + 1)
            branch_size = block_size >> 1
            j = l
            while j < self.N:
                if j % block_size < branch_size:
                    path.L[j, s + 1] = f_operation(
                        path.L[j, s], path.L[j + branch_size, s]
                    )
                else:
                    path.L[j, s + 1] = g_operation(
                        path.L[j - branch_size, s],
                        path.L[j, s],
                        path.B[j - branch_size, s + 1],
                    )
                j += block_size

    def _update_bits(self, path, l):
        if l < self.N // 2:
            return
        for s in range(self.n, self.n - _active_bit_level(l, self.n), -1):
            block_size = 1 << s
            branch_size = block_size >> 1
            j = l
            while j >= 0:
                if j % block_size >= branch_size:
                    path.B[j - branch_size, s - 1] = (
                        path.B[j, s] ^ path.B[j - branch_size, s]
                    )
                    path.B[j, s - 1] = path.B[j, s]
                j -= block_size

    def decode(self, llr_ch):
        """
        主译码函数。
        返回：(u_hat, pm)
        """
        if self.list_size == 1 and self.crc_length == 0:
            u_hat = sc_decode(llr_ch, self.frozen_bits)
            return u_hat, 0.0

        llr_tree = _prepare_llr(llr_ch)
        paths = [_Path(self.N, self.n, llr_tree)]

        for i in range(self.N):
            l = _bit_reversed(i, self.n)
            candidates = []

            for path in paths:
                if not path.active:
                    continue
                self._update_llrs(path, l)
                llr = path.L[l, self.n]

                if l in self.frozen_set:
                    new_path = path.copy()
                    new_path.pm += _pm_penalty(llr, 0)
                    new_path.B[l, self.n] = 0
                    self._update_bits(new_path, l)
                    candidates.append(new_path)
                else:
                    for bit in (0, 1):
                        new_path = path.copy()
                        new_path.pm += _pm_penalty(llr, bit)
                        new_path.B[l, self.n] = bit
                        self._update_bits(new_path, l)
                        candidates.append(new_path)

            candidates.sort(key=lambda p: p.pm)
            paths = candidates[: self.list_size]

        best_path = paths[0]
        u_hat = best_path.B[:, self.n].astype(int)

        if self.crc_length > 0:
            info_positions = sorted(
                i for i in range(self.N) if i not in self.frozen_set
            )
            crc_pass = []
            for path in paths:
                u = path.B[:, self.n].astype(int)
                info_bits = u[info_positions]
                if crc_check(info_bits, self.crc_length):
                    crc_pass.append(path)
            if crc_pass:
                best_path = min(crc_pass, key=lambda p: p.pm)

        u_hat = best_path.B[:, self.n].astype(int)
        return u_hat, best_path.pm
