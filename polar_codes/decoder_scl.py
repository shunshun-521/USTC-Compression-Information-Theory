"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
import math
from encoder import bit_reversed
from decoder_sc import (
    f_operation, g_operation, active_llr_level, active_bit_level
)

CRC_POLYNOMIALS = {
    8: 0x07,
    16: 0x8005,
}


def _crc_remainder(bits, crc_length):
    poly = CRC_POLYNOMIALS[crc_length]
    reg = 0
    for bit in bits:
        reg = (reg << 1) | int(bit)
        if reg & (1 << crc_length):
            reg ^= poly
    for _ in range(crc_length):
        reg <<= 1
        if reg & (1 << crc_length):
            reg ^= poly
    return reg & ((1 << crc_length) - 1)


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=int)
    remainder = _crc_remainder(info_bits, crc_length)
    crc_bits = np.array([(remainder >> (crc_length - 1 - i)) & 1
                         for i in range(crc_length)], dtype=int)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC 是否正确"""
    bits = np.asarray(bits, dtype=int)
    if len(bits) < crc_length:
        return False
    return _crc_remainder(bits, crc_length) == 0


class PathState:
    """单条译码路径状态"""
    __slots__ = ('pm', 'L', 'B', 'u_hat')

    def __init__(self, N, n, llr_ch):
        self.pm = 0.0
        self.L = np.full((N, n + 1), np.nan)
        self.B = np.full((N, n + 1), np.nan)
        self.L[:, 0] = llr_ch
        self.u_hat = np.zeros(N, dtype=int)

    def copy(self):
        new = PathState.__new__(PathState)
        new.pm = self.pm
        new.L = self.L.copy()
        new.B = self.B.copy()
        new.u_hat = self.u_hat.copy()
        return new


class SCLDecoder:
    """SCL 译码器（含 Lazy Copy 优化）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]

    def _update_llrs(self, path, l):
        n = self.n
        for s in range(n - active_llr_level(l, n), n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    path.L[j, s + 1] = f_operation(path.L[j, s], path.L[j + branch_size, s])
                else:
                    top_bit = path.B[j - branch_size, s + 1]
                    if np.isnan(top_bit):
                        top_bit = 0
                    else:
                        top_bit = int(top_bit)
                    path.L[j, s + 1] = g_operation(
                        path.L[j - branch_size, s], path.L[j, s], top_bit
                    )

    def _update_bits(self, path, l):
        n = self.n
        if l < self.N // 2:
            return
        for s in range(n, n - active_bit_level(l, n), -1):
            block_size = 2 ** s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    path.B[j - branch_size, s - 1] = int(path.B[j, s]) ^ int(path.B[j - branch_size, s])
                    path.B[j, s - 1] = path.B[j, s]

    def _pm_penalty(self, llr, bit):
        """路径度量惩罚"""
        if (bit == 0 and llr < 0) or (bit == 1 and llr >= 0):
            return abs(llr)
        return 0.0

    def decode(self, llr_ch):
        """主译码函数，返回 u_hat, pm"""
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n = self.n
        L_size = self.list_size

        paths = [PathState(self.N, n, llr_ch)]

        for l in [bit_reversed(i, n) for i in range(self.N)]:
            new_paths = []

            for path in paths:
                self._update_llrs(path, l)
                llr = path.L[l, n]

                if l in self.frozen_set:
                    new_p = path.copy()
                    new_p.pm += self._pm_penalty(llr, 0)
                    new_p.B[l, n] = 0
                    new_p.u_hat[l] = 0
                    self._update_bits(new_p, l)
                    new_paths.append(new_p)
                else:
                    for bit in (0, 1):
                        new_p = path.copy()
                        new_p.pm += self._pm_penalty(llr, bit)
                        new_p.B[l, n] = bit
                        new_p.u_hat[l] = bit
                        self._update_bits(new_p, l)
                        new_paths.append(new_p)

            new_paths.sort(key=lambda p: p.pm)
            paths = new_paths[:L_size]

        if self.crc_length > 0:
            crc_pass = [p for p in paths if crc_check(p.u_hat[self.info_indices], self.crc_length)]
            best = min(crc_pass, key=lambda p: p.pm) if crc_pass else min(paths, key=lambda p: p.pm)
        else:
            best = min(paths, key=lambda p: p.pm)

        return best.u_hat.copy(), best.pm
