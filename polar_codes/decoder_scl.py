"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL），基于 Permuted SC 结构
"""
import numpy as np
import math

from decoder_sc import (
    f_operation,
    g_operation,
    _bit_reversed,
    _active_llr_level,
    _active_bit_level,
)


def _crc_poly(crc_length):
    if crc_length == 8:
        return 0x07
    if crc_length == 16:
        return 0x8005
    raise ValueError(f"Unsupported CRC length: {crc_length}")


def _crc_update(reg, bit, crc_length, poly):
    reg ^= int(bit) << (crc_length - 1)
    if reg & (1 << (crc_length - 1)):
        reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
    else:
        reg = (reg << 1) & ((1 << crc_length) - 1)
    return reg


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=np.int8)
    poly = _crc_poly(crc_length)
    reg = 0
    for bit in info_bits:
        reg = _crc_update(reg, bit, crc_length, poly)
    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=np.int8,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC"""
    bits = np.asarray(bits, dtype=np.int8)
    if crc_length == 0:
        return True
    poly = _crc_poly(crc_length)
    reg = 0
    for bit in bits:
        reg = _crc_update(reg, bit, crc_length, poly)
    return reg == 0


class Path:
    __slots__ = ("pm", "L", "B")

    def __init__(self, N, n, llr_ch=None):
        self.pm = 0.0
        self.L = np.zeros((N, n + 1), dtype=np.float64)
        self.B = np.zeros((N, n + 1), dtype=np.int8)
        if llr_ch is not None:
            self.L[:, 0] = llr_ch

    def copy(self):
        child = Path(self.L.shape[0], self.L.shape[1] - 1)
        child.pm = self.pm
        child.L = self.L.copy()
        child.B = self.B.copy()
        return child


class SCLDecoder:
    """SCL 译码器"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length

    def _update_llrs(self, path, l):
        start_s = self.n - _active_llr_level(l, self.n)
        for s in range(start_s, self.n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    top_llr = path.L[j, s]
                    btm_llr = path.L[j + branch_size, s]
                    path.L[j, s + 1] = f_operation(top_llr, btm_llr)
                else:
                    btm_llr = path.L[j, s]
                    top_llr = path.L[j - branch_size, s]
                    top_bit = path.B[j - branch_size, s + 1]
                    path.L[j, s + 1] = g_operation(top_llr, btm_llr, top_bit)

    def _update_bits(self, path, l):
        if l < self.N // 2:
            return
        end_s = self.n - _active_bit_level(l, self.n)
        for s in range(self.n, end_s, -1):
            block_size = 2 ** s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    path.B[j - branch_size, s - 1] = (
                        path.B[j, s] ^ path.B[j - branch_size, s]
                    )
                    path.B[j, s - 1] = path.B[j, s]

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [Path(self.N, self.n, llr_ch)]

        for i in range(self.N):
            l = _bit_reversed(i, self.n)
            new_paths = []

            for path in paths:
                self._update_llrs(path, l)
                llr_val = path.L[l, self.n]

                if self.frozen_bits[l]:
                    child = path.copy()
                    penalty = abs(llr_val) if llr_val < 0 else 0.0
                    child.pm += penalty
                    child.B[l, self.n] = 0
                    self._update_bits(child, l)
                    new_paths.append(child)
                else:
                    for u_bit in (0, 1):
                        child = path.copy()
                        hard = 0 if llr_val >= 0 else 1
                        if u_bit != hard:
                            child.pm += abs(llr_val)
                        child.B[l, self.n] = u_bit
                        self._update_bits(child, l)
                        new_paths.append(child)

            new_paths.sort(key=lambda p: p.pm)
            paths = new_paths[: self.list_size]

        paths.sort(key=lambda p: p.pm)
        u_hat = paths[0].B[:, self.n].astype(int)

        if self.crc_length > 0:
            info_idx = np.where(~self.frozen_bits)[0]
            for path in paths:
                candidate = path.B[:, self.n].astype(int)
                if crc_check(candidate[info_idx], self.crc_length):
                    return candidate.copy(), path.pm

        return u_hat, paths[0].pm
