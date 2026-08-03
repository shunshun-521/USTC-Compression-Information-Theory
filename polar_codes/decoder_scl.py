"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
import math
from encoder import bit_reversal_permutation
from decoder_sc import (
    _upper_llr, _lower_llr, _active_llr_level, _active_bit_level
)


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    mask = (1 << crc_length) - 1
    reg = 0
    for bit in info_bits:
        msb = (reg >> (crc_length - 1)) & 1
        reg = ((reg << 1) | int(bit)) & mask
        if msb ^ int(bit):
            reg ^= poly & mask
    for _ in range(crc_length):
        msb = (reg >> (crc_length - 1)) & 1
        reg = (reg << 1) & mask
        if msb:
            reg ^= poly & mask
    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC。"""
    expected = crc_encode(bits[:-crc_length], crc_length)
    return np.array_equal(bits[-crc_length:], expected[-crc_length:])


class PathState:
    """单条 SCL 路径状态（Lazy Copy）。"""
    __slots__ = ('pm', 'L', 'B', 'u_hat')

    def __init__(self, N, n, llr_ch):
        self.pm = 0.0
        self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.B = np.full((N, n + 1), np.nan)
        self.L[:, 0] = llr_ch.copy()
        self.u_hat = np.zeros(N, dtype=int)

    def copy(self):
        p = PathState.__new__(PathState)
        p.pm = self.pm
        p.L = self.L.copy()
        p.B = self.B.copy()
        p.u_hat = self.u_hat.copy()
        return p


class SCLDecoder:
    """SCL 译码器（含 Lazy Copy 优化）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.list_size = list_size
        self.crc_length = crc_length
        self.br = bit_reversal_permutation(N)

    def _update_llrs(self, paths, l):
        n = self.n
        N = self.N
        for path in paths:
            for s in range(n - _active_llr_level(l, n), n):
                block_size = 2 ** (s + 1)
                branch_size = block_size // 2
                for j in range(l, N, block_size):
                    if j % block_size < branch_size:
                        path.L[j, s + 1] = _upper_llr(path.L[j, s], path.L[j + branch_size, s])
                    else:
                        top_bit = int(path.B[j - branch_size, s + 1])
                        path.L[j, s + 1] = _lower_llr(path.L[j, s], path.L[j - branch_size, s], top_bit)

    def _update_bits(self, paths, l):
        n = self.n
        N = self.N
        if l < N // 2:
            return
        for path in paths:
            for s in range(n, n - _active_bit_level(l, n), -1):
                block_size = 2 ** s
                branch_size = block_size // 2
                for j in range(l, -1, -block_size):
                    if j % block_size >= branch_size:
                        path.B[j - branch_size, s - 1] = int(path.B[j, s]) ^ int(path.B[j - branch_size, s])
                        path.B[j, s - 1] = path.B[j, s]

    def decode(self, llr_ch):
        """主译码函数。"""
        N = self.N
        n = self.n
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [PathState(N, n, llr_ch)]

        for idx in range(N):
            l = self.br[idx]
            self._update_llrs(paths, l)

            new_paths = []
            for path in paths:
                llr_val = path.L[l, n]
                if l in self.frozen_set:
                    penalty = 0.0 if llr_val >= 0 else abs(llr_val)
                    p = path.copy()
                    p.pm += penalty
                    p.B[l, n] = 0
                    p.u_hat[l] = 0
                    new_paths.append(p)
                else:
                    for bit_val in (0, 1):
                        hard = 0 if llr_val >= 0 else 1
                        penalty = 0.0 if bit_val == hard else abs(llr_val)
                        p = path.copy()
                        p.pm += penalty
                        p.B[l, n] = bit_val
                        p.u_hat[l] = bit_val
                        new_paths.append(p)

            new_paths.sort(key=lambda p: p.pm)
            paths = new_paths[:self.list_size]
            self._update_bits(paths, l)

        if self.crc_length > 0:
            valid = [p for p in paths if crc_check(p.u_hat, self.crc_length)]
            best = min(valid, key=lambda p: p.pm) if valid else min(paths, key=lambda p: p.pm)
        else:
            best = min(paths, key=lambda p: p.pm)

        return best.u_hat, best.pm
