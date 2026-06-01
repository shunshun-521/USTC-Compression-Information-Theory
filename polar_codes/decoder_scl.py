"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np

from decoder_sc import (
    active_bit_level,
    active_llr_level,
    bit_reversed_index,
    lower_llr,
    upper_llr,
)


def crc_encode(info_bits, crc_length=8):
    """
    CRC 编码：将校验位附加在信息比特之后。
    r=8:  CRC-8  (0x07)
    r=16: CRC-16 (0x8005)
    """
    info_bits = np.asarray(info_bits, dtype=int)
    if crc_length == 8:
        poly = 0x07
    elif crc_length == 16:
        poly = 0x8005
    else:
        raise ValueError("crc_length must be 8 or 16")

    reg = 0
    for bit in info_bits:
        reg ^= int(bit) << (crc_length - 1)
        for _ in range(8 if crc_length == 8 else 1):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)

    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC 是否正确"""
    bits = np.asarray(bits, dtype=int)
    return np.array_equal(
        bits[-crc_length:], crc_encode(bits[:-crc_length], crc_length)[-crc_length:]
    )


class PathState:
    """单条 SCL 路径的 L/B 状态"""

    __slots__ = ("pm", "L", "B", "u_hat")

    def __init__(self, N, n, llr_ch):
        self.pm = 0.0
        self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.B = np.full((N, n + 1), np.nan)
        self.L[:, 0] = llr_ch.copy()
        self.u_hat = np.zeros(N, dtype=int)


class SCLDecoder:
    """SCL 译码器（每条路径独立 L/B，Lazy Copy 通过复制路径状态实现）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.L_size = list_size
        self.crc_length = crc_length
        self.info_positions = np.where(~self.frozen_bits)[0]

    def _pm_update(self, pm, llr, u):
        hard = 0 if llr >= 0 else 1
        if u != hard:
            pm += abs(llr)
        return pm

    def _update_llrs(self, path, l):
        n = self.n
        N = self.N
        for s in range(n - active_llr_level(l, n), n):
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(l, N, block_size):
                if j % block_size < branch_size:
                    path.L[j, s + 1] = upper_llr(path.L[j, s], path.L[j + branch_size, s])
                else:
                    top_bit = path.B[j - branch_size, s + 1]
                    path.L[j, s + 1] = lower_llr(
                        path.L[j, s], path.L[j - branch_size, s], top_bit
                    )

    def _update_bits(self, path, l):
        n = self.n
        N = self.N
        if l < N // 2:
            return
        for s in range(n, n - active_bit_level(l, n), -1):
            block_size = 1 << s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    path.B[j - branch_size, s - 1] = int(path.B[j, s]) ^ int(
                        path.B[j - branch_size, s]
                    )
                    path.B[j, s - 1] = path.B[j, s]

    def decode(self, llr_ch):
        """SCL 译码，返回 (u_hat, pm)"""
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N = self.N
        n = self.n

        paths = [PathState(N, n, llr_ch)]

        for i in range(N):
            l = bit_reversed_index(i, n)
            new_paths = []

            for path in paths:
                self._update_llrs(path, l)
                llr = path.L[l, n]

                if l in self.frozen_set:
                    path.pm = self._pm_update(path.pm, llr, 0)
                    path.u_hat[l] = 0
                    path.B[l, n] = 0
                    self._update_bits(path, l)
                    new_paths.append(path)
                else:
                    for u in (0, 1):
                        child = PathState(N, n, llr_ch)
                        child.L = path.L.copy()
                        child.B = path.B.copy()
                        child.u_hat = path.u_hat.copy()
                        child.pm = self._pm_update(path.pm, llr, u)
                        child.u_hat[l] = u
                        child.B[l, n] = u
                        self._update_bits(child, l)
                        new_paths.append(child)

            new_paths.sort(key=lambda p: p.pm)
            paths = new_paths[: self.L_size]

        paths.sort(key=lambda p: p.pm)

        if self.crc_length > 0:
            valid = []
            for p in paths:
                info_bits = p.u_hat[self.info_positions]
                if crc_check(info_bits, self.crc_length):
                    valid.append(p)
            if valid:
                paths = sorted(valid, key=lambda p: p.pm)

        best = paths[0]
        return best.u_hat.copy(), best.pm
