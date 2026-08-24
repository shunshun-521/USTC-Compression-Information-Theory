"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from encoder import bit_reversed
from decoder_sc import (
    active_bit_level,
    active_llr_level,
    f_operation,
    g_operation,
)


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=np.int8)
    if crc_length == 8:
        poly = 0x07
    elif crc_length == 16:
        poly = 0x8005
    else:
        raise ValueError("crc_length must be 8 or 16")

    reg = 0
    mask = (1 << crc_length) - 1
    for bit in info_bits:
        reg ^= int(bit) << (crc_length - 1)
        for _ in range(8):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & mask
            else:
                reg = (reg << 1) & mask

    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=np.int8,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC"""
    bits = np.asarray(bits, dtype=np.int8)
    if crc_length == 8:
        poly = 0x07
    elif crc_length == 16:
        poly = 0x8005
    else:
        raise ValueError("crc_length must be 8 or 16")

    reg = 0
    mask = (1 << crc_length) - 1
    for bit in bits:
        reg ^= int(bit) << (crc_length - 1)
        for _ in range(8):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & mask
            else:
                reg = (reg << 1) & mask
    return reg == 0


class Path:
    """SCL 译码路径"""

    def __init__(self, N, n):
        self.N = N
        self.n = n
        self.L = np.zeros((N, n + 1), dtype=np.float64)
        self.B = np.zeros((N, n + 1), dtype=np.int8)
        self.u_hat = np.zeros(N, dtype=np.int8)
        self.pm = 0.0

    def copy(self):
        p = Path(self.N, self.n)
        p.L[:] = self.L
        p.B[:] = self.B
        p.u_hat[:] = self.u_hat
        p.pm = self.pm
        return p


class SCLDecoder:
    """SCL 译码器（Lazy Copy 优化）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length

    def _update_llrs(self, path, l):
        for s in range(self.n - active_llr_level(l, self.n), self.n):
            block_size = 1 << (s + 1)
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
        for s in range(self.n, self.n - active_bit_level(l, self.n), -1):
            block_size = 1 << s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    path.B[j - branch_size, s - 1] = int(path.B[j, s]) ^ int(
                        path.B[j - branch_size, s]
                    )
                    path.B[j, s - 1] = path.B[j, s]

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [Path(self.N, self.n)]
        paths[0].L[:, 0] = llr_ch

        for i in range(self.N):
            l = bit_reversed(i, self.n)
            for path in paths:
                self._update_llrs(path, l)

            new_paths = []
            if self.frozen_bits[l]:
                for path in paths:
                    llr = path.L[l, self.n]
                    if llr < 0:
                        path.pm += abs(llr)
                    path.u_hat[l] = 0
                    path.B[l, self.n] = 0
                    self._update_bits(path, l)
                    new_paths.append(path)
            else:
                for path in paths:
                    llr = path.L[l, self.n]
                    for u_bit in (0, 1):
                        child = path.copy()
                        child.u_hat[l] = u_bit
                        child.B[l, self.n] = u_bit
                        hard = 0 if llr >= 0 else 1
                        if u_bit != hard:
                            child.pm += abs(llr)
                        self._update_bits(child, l)
                        new_paths.append(child)

            new_paths.sort(key=lambda p: p.pm)
            paths = new_paths[:self.list_size]

        crc_valid = []
        for path in paths:
            if self.crc_length > 0:
                info_bits = path.u_hat[~self.frozen_bits]
                if crc_check(info_bits, self.crc_length):
                    crc_valid.append(path)
            else:
                crc_valid.append(path)

        best = min(crc_valid if crc_valid else paths, key=lambda p: p.pm)
        return best.u_hat.copy(), best.pm
