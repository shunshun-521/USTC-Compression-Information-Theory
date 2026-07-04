"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np

from decoder_sc import (
    f_boxplus,
    _lower_llr,
    _bit_reversed,
    _active_llr_level,
    _active_bit_level,
)


_CRC8_POLY = 0x07
_CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, crc_length):
    reg = 0
    for bit in bits:
        reg ^= int(bit) << (crc_length - 1)
        for _ in range(crc_length):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)
    return reg


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=int).ravel()
    poly = _CRC8_POLY if crc_length == 8 else _CRC16_POLY

    reg = 0
    for bit in info_bits:
        reg ^= int(bit) << (crc_length - 1)
        for _ in range(crc_length):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)

    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。"""
    bits = np.asarray(bits, dtype=int).ravel()
    if len(bits) < crc_length:
        return False
    poly = _CRC8_POLY if crc_length == 8 else _CRC16_POLY
    return _crc_remainder(bits, poly, crc_length) == 0


class _Path:
    """单条 SCL 译码路径（Lazy Copy）。"""

    __slots__ = ("pm", "u_hat", "L", "B")

    def __init__(self, n, N):
        self.pm = 0.0
        self.u_hat = np.zeros(N, dtype=int)
        self.L = np.zeros((N, n + 1), dtype=np.float64)
        self.B = np.full((N, n + 1), np.nan)

    def copy(self):
        new = _Path(self.L.shape[1] - 1, len(self.u_hat))
        new.pm = self.pm
        new.u_hat = self.u_hat.copy()
        new.L = self.L.copy()
        new.B = self.B.copy()
        return new


class SCLDecoder:
    """SCL 译码器（含 Lazy Copy 优化）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]
        self.process_order = [_bit_reversed(phi, self.n) for phi in range(N)]

    @staticmethod
    def _pm_penalty(llr, bit):
        hard = 0 if llr >= 0 else 1
        return 0.0 if bit == hard else abs(llr)

    def _update_llrs(self, path, l):
        n = self.n
        for s in range(n - _active_llr_level(l, n), n):
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    path.L[j, s + 1] = f_boxplus(path.L[j, s], path.L[j + branch_size, s])
                else:
                    path.L[j, s + 1] = _lower_llr(
                        path.L[j, s],
                        path.L[j - branch_size, s],
                        path.B[j - branch_size, s + 1],
                    )

    def _update_bits(self, path, l):
        if l < self.N // 2:
            return
        n = self.n
        for s in range(n, n - _active_bit_level(l, n), -1):
            block_size = 1 << s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    path.B[j - branch_size, s - 1] = int(path.B[j, s]) ^ int(
                        path.B[j - branch_size, s]
                    )
                    path.B[j, s - 1] = path.B[j, s]

    def decode(self, llr_ch):
        """主译码函数，返回 u_hat, pm。"""
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = []
        init = _Path(self.n, self.N)
        init.L[:, 0] = llr_ch
        paths.append(init)

        for l in self.process_order:
            phi = _bit_reversed(l, self.n)
            new_paths = []

            for path in paths:
                self._update_llrs(path, l)
                llr0 = path.L[l, self.n]

                if self.frozen_bits[phi]:
                    child = path.copy()
                    child.pm += self._pm_penalty(llr0, 0)
                    child.u_hat[phi] = 0
                    child.B[l, self.n] = 0
                    self._update_bits(child, l)
                    new_paths.append(child)
                else:
                    for bit in (0, 1):
                        child = path.copy()
                        child.pm += self._pm_penalty(llr0, bit)
                        child.u_hat[phi] = bit
                        child.B[l, self.n] = bit
                        self._update_bits(child, l)
                        new_paths.append(child)

            new_paths.sort(key=lambda p: p.pm)
            paths = new_paths[: self.list_size]

        crc_pass = []
        for p in paths:
            if self.crc_length > 0:
                info_bits = p.u_hat[self.info_indices]
                if crc_check(info_bits, self.crc_length):
                    crc_pass.append(p)
            else:
                crc_pass.append(p)

        best = min(crc_pass or paths, key=lambda p: p.pm)
        return best.u_hat.copy(), best.pm
