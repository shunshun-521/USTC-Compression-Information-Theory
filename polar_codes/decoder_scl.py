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
    _logdomain_sum,
)


CRC_POLYS = {8: 0x07, 16: 0x8005}


def _crc_remainder(bits, crc_length):
    poly = CRC_POLYS[crc_length]
    reg = 0
    for bit in bits:
        reg ^= int(bit) << (crc_length - 1)
        if reg & (1 << (crc_length - 1)):
            reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
        else:
            reg = (reg << 1) & ((1 << crc_length) - 1)
    return reg


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=np.int8)
    remainder = _crc_remainder(info_bits, crc_length)
    crc_bits = np.array(
        [(remainder >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=np.int8,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。"""
    bits = np.asarray(bits, dtype=np.int8)
    expected = crc_encode(bits[:-crc_length], crc_length)
    return np.array_equal(bits[-crc_length:], expected[-crc_length:])


class Path:
    __slots__ = ("pm", "L", "B", "u_hat")

    def __init__(self, N, n):
        self.pm = 0.0
        self.L = np.zeros((N, n + 1), dtype=np.float64)
        self.B = np.zeros((N, n + 1), dtype=np.int8)
        self.u_hat = np.zeros(N, dtype=np.int8)


class SCLDecoder:
    """SCL 译码器（Lazy Copy：路径分裂时复制数组）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]

    def _upper_llr(self, l1, l2):
        if np.isinf(l1) and not np.isinf(l2):
            return l2
        if not np.isinf(l1) and np.isinf(l2):
            return l1
        if np.isinf(l1) and np.isinf(l2):
            return np.inf
        return _logdomain_sum(l1 + l2, 0.0) - _logdomain_sum(l1, l2)

    def _lower_llr(self, l1, l2, b):
        if b == 0:
            if np.isinf(l1) or np.isinf(l2):
                return np.inf
            return l1 + l2
        return l1 - l2

    def _update_llrs(self, path, l):
        for s in range(self.n - _active_llr_level(l, self.n), self.n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    path.L[j, s + 1] = self._upper_llr(
                        path.L[j, s], path.L[j + branch_size, s]
                    )
                else:
                    path.L[j, s + 1] = self._lower_llr(
                        path.L[j, s],
                        path.L[j - branch_size, s],
                        path.B[j - branch_size, s + 1],
                    )

    def _update_bits(self, path, l, bit):
        path.u_hat[l] = bit
        path.B[l, self.n] = bit
        if l < self.N / 2:
            return
        for s in range(self.n, self.n - _active_bit_level(l, self.n), -1):
            block_size = 2 ** s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    path.B[j - branch_size, s - 1] = int(path.B[j, s]) ^ int(
                        path.B[j - branch_size, s]
                    )
                    path.B[j, s - 1] = path.B[j, s]

    def _pm_penalty(self, llr, bit):
        hard = 0 if llr >= 0 else 1
        return 0.0 if bit == hard else abs(llr)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [Path(self.N, self.n)]
        paths[0].L[:, 0] = llr_ch

        for i in range(self.N):
            l = _bit_reversed(i, self.n)
            candidates = []

            for path in paths:
                self._update_llrs(path, l)
                llr = path.L[l, self.n]
                if self.frozen_bits[l]:
                    pm_new = path.pm + self._pm_penalty(llr, 0)
                    candidates.append((pm_new, path, 0))
                else:
                    for bit in (0, 1):
                        pm_new = path.pm + self._pm_penalty(llr, bit)
                        candidates.append((pm_new, path, bit))

            candidates.sort(key=lambda x: x[0])
            selected = candidates[: self.list_size]

            new_paths = []
            for pm_new, parent, bit in selected:
                child = Path(self.N, self.n)
                child.pm = pm_new
                child.L[:] = parent.L
                child.B[:] = parent.B
                child.u_hat[:] = parent.u_hat
                self._update_bits(child, l, bit)
                new_paths.append(child)
            paths = new_paths

        if self.crc_length > 0:
            valid = []
            for path in paths:
                info_bits = path.u_hat[self.info_indices]
                if crc_check(info_bits, self.crc_length):
                    valid.append(path)
            pool = valid if valid else paths
        else:
            pool = paths

        best = min(pool, key=lambda p: p.pm)
        return best.u_hat.copy(), best.pm
