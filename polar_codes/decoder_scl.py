"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np
from encoder import bit_reversed
from decoder_sc import (
    f_operation, g_operation,
    _active_llr_level, _active_bit_level,
)


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, crc_length):
    reg = 0
    for bit in bits:
        reg = ((reg << 1) | int(bit)) & ((1 << (crc_length + 1)) - 1)
        if reg & (1 << crc_length):
            reg ^= poly
    for _ in range(crc_length):
        reg = (reg << 1) & ((1 << (crc_length + 1)) - 1)
        if reg & (1 << crc_length):
            reg ^= poly
    return reg & ((1 << crc_length) - 1)


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=np.int32)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    remainder = _crc_remainder(info_bits, poly, crc_length)
    crc_bits = np.array([(remainder >> (crc_length - 1 - i)) & 1
                         for i in range(crc_length)], dtype=np.int32)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC 是否正确"""
    bits = np.asarray(bits, dtype=np.int32)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    return _crc_remainder(bits, poly, crc_length) == 0


class Path:
    """SCL 单条路径"""

    __slots__ = ('L', 'C', 'pm', 'u_hat', 'N', 'n')

    def __init__(self, N, n):
        self.N = N
        self.n = n
        self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.C = np.zeros((N, n + 1), dtype=np.int32)
        self.pm = 0.0
        self.u_hat = np.zeros(N, dtype=np.int32)

    def copy_from(self, other):
        self.L = other.L.copy()
        self.C = other.C.copy()
        self.pm = other.pm
        self.u_hat = other.u_hat.copy()


class SCLDecoder:
    """SCL 译码器"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.info_indices = np.where(~self.frozen_bits)[0]

    def _update_llr(self, path, l):
        for s in range(self.n - _active_llr_level(l, self.n), self.n):
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    path.L[j, s + 1] = f_operation(path.L[j, s], path.L[j + branch_size, s])
                else:
                    path.L[j, s + 1] = g_operation(
                        path.L[j - branch_size, s], path.L[j, s],
                        path.C[j - branch_size, s + 1]
                    )

    def _update_bits(self, path, l):
        if l < self.N // 2:
            return
        for s in range(self.n, self.n - _active_bit_level(l, self.n), -1):
            block_size = 1 << s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    path.C[j - branch_size, s - 1] = path.C[j, s] ^ path.C[j - branch_size, s]
                    path.C[j, s - 1] = path.C[j, s]

    def _pm_penalty(self, llr, bit):
        hard = 0 if llr >= 0 else 1
        return 0.0 if bit == hard else abs(llr)

    def decode(self, llr_ch):
        """主译码函数"""
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N = self.N
        n = self.n
        L_size = self.list_size

        paths = []
        p0 = Path(N, n)
        p0.L[:, 0] = llr_ch
        paths.append(p0)

        for phi in range(N):
            l = bit_reversed(phi, n)
            candidates = []

            for path in paths:
                self._update_llr(path, l)
                llr = path.L[l, n]

                if l in self.frozen_set:
                    new_path = Path(N, n)
                    new_path.copy_from(path)
                    new_path.pm += self._pm_penalty(llr, 0)
                    new_path.C[l, n] = 0
                    new_path.u_hat[l] = 0
                    self._update_bits(new_path, l)
                    candidates.append(new_path)
                else:
                    for bit in (0, 1):
                        new_path = Path(N, n)
                        new_path.copy_from(path)
                        new_path.pm += self._pm_penalty(llr, bit)
                        new_path.C[l, n] = bit
                        new_path.u_hat[l] = bit
                        self._update_bits(new_path, l)
                        candidates.append(new_path)

            candidates.sort(key=lambda p: p.pm)
            paths = candidates[:L_size]

        if self.crc_length > 0:
            crc_passed = []
            for path in paths:
                info_bits = path.u_hat[self.info_indices]
                if crc_check(info_bits, self.crc_length):
                    crc_passed.append(path)
            pool = crc_passed if crc_passed else paths
        else:
            pool = paths

        best_path = min(pool, key=lambda p: p.pm)
        return best_path.u_hat, best_path.pm
