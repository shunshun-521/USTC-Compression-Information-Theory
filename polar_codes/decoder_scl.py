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
    f_operation,
    g_operation,
)


# CRC 多项式
_CRC_POLY = {8: 0x07, 16: 0x8005}


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后"""
    poly = _CRC_POLY[crc_length]
    info_bits = np.asarray(info_bits, dtype=int)
    reg = 0
    for bit in info_bits:
        reg ^= (bit << (crc_length - 1))
        for _ in range(8 if crc_length == 8 else 16):
            if crc_length == 8:
                if reg & 0x80:
                    reg = ((reg << 1) ^ poly) & 0xFF
                else:
                    reg = (reg << 1) & 0xFF
            else:
                if reg & 0x8000:
                    reg = ((reg << 1) ^ poly) & 0xFFFF
                else:
                    reg = (reg << 1) & 0xFFFF
    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC"""
    if crc_length == 0:
        return True
    expected = crc_encode(bits[:-crc_length], crc_length)
    return np.array_equal(bits[-crc_length:], expected[-crc_length:])


class _Path:
    """单条译码路径（Lazy Copy）"""
    __slots__ = ("L", "B", "pm", "parent", "copy_L", "copy_B")

    def __init__(self, N, n, llr_ch):
        self.L = np.zeros((N, n + 1), dtype=np.float64)
        self.B = np.zeros((N, n + 1), dtype=int)
        self.L[:, 0] = llr_ch
        self.pm = 0.0
        self.parent = None
        self.copy_L = False
        self.copy_B = False

    def ensure_L(self):
        if self.copy_L and self.parent is not None:
            self.L = self.parent.L.copy()
            self.copy_L = False

    def ensure_B(self):
        if self.copy_B and self.parent is not None:
            self.B = self.parent.B.copy()
            self.copy_B = False


class SCLDecoder:
    """SCL 译码器（含 Lazy Copy 优化）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0, info_indices=None):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = info_indices

    def _update_llrs(self, path, l):
        path.ensure_L()
        for s in range(self.n - _active_llr_level(l, self.n), self.n):
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    path.L[j, s + 1] = f_operation(path.L[j, s], path.L[j + branch_size, s])
                else:
                    path.L[j, s + 1] = g_operation(
                        path.L[j - branch_size, s], path.L[j, s],
                        path.B[j - branch_size, s + 1],
                    )

    def _update_bits(self, path, l):
        path.ensure_B()
        if l < self.N // 2:
            return
        for s in range(self.n, self.n - _active_bit_level(l, self.n), -1):
            block_size = 1 << s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    path.B[j - branch_size, s - 1] = (
                        path.B[j, s] + path.B[j - branch_size, s]
                    ) % 2
                    path.B[j, s - 1] = path.B[j, s]

    def _pm_penalty(self, llr, bit):
        hard = 0 if llr >= 0 else 1
        return 0.0 if bit == hard else abs(llr)

    def decode(self, llr_ch):
        """主译码函数，返回 (u_hat, pm)"""
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [_Path(self.N, self.n, llr_ch)]

        for i in range(self.N):
            l = _bit_reversed(i, self.n)
            candidates = []

            for path in paths:
                self._update_llrs(path, l)
                llr = path.L[l, self.n]

                if self.frozen_bits[l]:
                    pen = self._pm_penalty(llr, 0)
                    path.ensure_B()
                    path.B[l, self.n] = 0
                    path.pm += pen
                    self._update_bits(path, l)
                    candidates.append(path)
                else:
                    for bit in (0, 1):
                        child = _Path(self.N, self.n, llr_ch)
                        child.parent = path
                        child.copy_L = True
                        child.copy_B = True
                        child.pm = path.pm + self._pm_penalty(llr, bit)
                        child.ensure_L()
                        child.ensure_B()
                        child.L[l, self.n] = llr
                        child.B[l, self.n] = bit
                        self._update_bits(child, l)
                        candidates.append(child)

            candidates.sort(key=lambda p: p.pm)
            paths = candidates[: self.list_size]

        # 选择最优路径
        best = None
        if self.crc_length > 0 and self.info_indices is not None:
            for path in sorted(paths, key=lambda p: p.pm):
                u_hat = path.B[:, self.n].astype(int)
                payload = u_hat[self.info_indices]
                if crc_check(payload, self.crc_length):
                    best = path
                    break
        if best is None:
            best = min(paths, key=lambda p: p.pm)

        u_hat = best.B[:, self.n].astype(int)
        return u_hat, best.pm
