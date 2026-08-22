"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np
from encoder import bit_reversed
from decoder_sc import (
    active_llr_level, active_bit_level, upper_llr, lower_llr,
)


def crc_encode(info_bits, crc_length=8):
    """
    计算 CRC 校验位并附加到信息比特后。
    CRC-8: 0x07 (x^8+x^2+x+1), CRC-16: 0x8005
    """
    info_bits = np.asarray(info_bits, dtype=np.int8)
    if crc_length == 8:
        poly = 0x07
    elif crc_length == 16:
        poly = 0x8005
    else:
        raise ValueError("crc_length must be 8 or 16")

    reg = 0
    for bit in info_bits:
        reg ^= (int(bit) << (crc_length - 1))
        for _ in range(8):
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
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC"""
    if crc_length == 0:
        return True
    bits = np.asarray(bits, dtype=np.int8)
    recomputed = crc_encode(bits[:-crc_length], crc_length)
    return np.array_equal(recomputed[-crc_length:], bits[-crc_length:])


class Path:
    """单条 SCL 路径（Lazy Copy 引用父路径 LLR/比特数组）"""

    __slots__ = ("L", "B", "pm", "u_hat", "parent", "llr_ref", "bit_ref")

    def __init__(self, N, n, parent=None):
        self.parent = parent
        if parent is None:
            self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
            self.B = np.full((N, n + 1), np.nan)
            self.llr_ref = None
            self.bit_ref = None
        else:
            self.L = parent.L
            self.B = parent.B
            self.llr_ref = parent
            self.bit_ref = parent
        self.pm = 0.0
        self.u_hat = np.zeros(N, dtype=int)

    def copy_on_write(self):
        """分裂时复制路径状态"""
        child = Path(self.L.shape[0], int(math.log2(self.L.shape[0])), parent=self)
        child.L = self.L.copy()
        child.B = self.B.copy()
        child.llr_ref = child
        child.bit_ref = child
        child.pm = self.pm
        child.u_hat = self.u_hat.copy()
        return child


class SCLDecoder:
    """SCL 译码器（Lazy Copy 优化）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length

    def _update_llrs(self, path, l):
        n = self.n
        for s in range(n - active_llr_level(l, n), n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    path.L[j, s + 1] = upper_llr(
                        path.L[j, s], path.L[j + branch_size, s]
                    )
                else:
                    path.L[j, s + 1] = lower_llr(
                        path.L[j, s], path.L[j - branch_size, s],
                        int(path.B[j - branch_size, s + 1]),
                    )

    def _update_bits(self, path, l):
        n = self.n
        if l < self.N / 2:
            return
        for s in range(n, n - active_bit_level(l, n), -1):
            block_size = 2 ** s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    path.B[j - branch_size, s - 1] = (
                        int(path.B[j, s]) ^ int(path.B[j - branch_size, s])
                    )
                    path.B[j, s - 1] = path.B[j, s]

    def _path_penalty(self, llr_val, u_bit):
        """路径度量惩罚：与 LLR 符号不一致时加 |LLR|"""
        hard = 0 if llr_val >= 0 else 1
        return 0.0 if u_bit == hard else abs(llr_val)

    def decode(self, llr_ch):
        """
        主译码函数。
        返回：(u_hat, pm)
        """
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N, n = self.N, self.n

        paths = [Path(N, n)]
        paths[0].L[:, 0] = llr_ch

        for phi in range(N):
            l = bit_reversed(phi, n)
            new_paths = []

            for path in paths:
                self._update_llrs(path, l)
                llr_val = path.L[l, n]

                if self.frozen_bits[l]:
                    pen = self._path_penalty(llr_val, 0)
                    child = path.copy_on_write()
                    child.pm += pen
                    child.u_hat[l] = 0
                    child.B[l, n] = 0
                    self._update_bits(child, l)
                    new_paths.append(child)
                else:
                    for u_bit in (0, 1):
                        child = path.copy_on_write()
                        child.pm += self._path_penalty(llr_val, u_bit)
                        child.u_hat[l] = u_bit
                        child.B[l, n] = u_bit
                        self._update_bits(child, l)
                        new_paths.append(child)

            new_paths.sort(key=lambda p: p.pm)
            paths = new_paths[:self.list_size]

        paths.sort(key=lambda p: p.pm)

        if self.crc_length > 0:
            info_idx = np.where(~self.frozen_bits)[0]
            for path in paths:
                info_bits = path.u_hat[info_idx]
                if crc_check(info_bits, self.crc_length):
                    return path.u_hat, path.pm

        best = paths[0]
        return best.u_hat, best.pm
