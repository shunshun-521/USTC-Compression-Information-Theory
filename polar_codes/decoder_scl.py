"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math

import numpy as np

from decoder_sc import (
    _active_bit_level,
    _active_llr_level,
    _bit_reversed_index,
    _frozen_indices_from_mask,
    _sc_decode_core,
    f_operation,
    g_operation,
    precompute_sc_indices,
)
from encoder import bit_reversal_permutation

CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY

    if crc_length == 8:
        reg = 0
        for bit in info_bits:
            reg ^= int(bit) << 7
            for _ in range(8):
                if reg & 0x80:
                    reg = ((reg << 1) ^ (poly << 1)) & 0xFF
                else:
                    reg = (reg << 1) & 0xFF
        crc_bits = np.array([(reg >> (7 - i)) & 1 for i in range(8)], dtype=int)
    else:
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
    """检验 CRC 是否正确。"""
    bits = np.asarray(bits, dtype=int)
    info = bits[:-crc_length]
    expected = crc_encode(info, crc_length)[-crc_length:]
    return np.array_equal(bits[-crc_length:], expected)


class Path:
    """SCL 译码路径。"""

    __slots__ = ('pm', 'u_hat', 'L', 'B')

    def __init__(self, N, n, llr):
        self.pm = 0.0
        self.u_hat = np.zeros(N, dtype=int)
        self.L = np.zeros((N, n + 1), dtype=np.float64)
        self.B = np.zeros((N, n + 1), dtype=int)
        self.L[:, 0] = llr


class SCLDecoder:
    """SCL 译码器。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_set = _frozen_indices_from_mask(self.frozen_bits)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]
        self.rev = bit_reversal_permutation(N)

    def _pm_update(self, pm, llr, u):
        u_hard = 0 if llr >= 0 else 1
        return pm if u == u_hard else pm + abs(llr)

    def _compute_llrs(self, path, l):
        for s in range(self.n - _active_llr_level(l, self.n), self.n):
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    path.L[j, s + 1] = f_operation(path.L[j, s], path.L[j + branch_size, s])
                else:
                    path.L[j, s + 1] = g_operation(
                        path.L[j - branch_size, s], path.L[j, s], path.B[j - branch_size, s + 1]
                    )
        return path.L[l, self.n]

    def _update_bits(self, path, l):
        if l < self.N / 2:
            return
        for s in range(self.n, self.n - _active_bit_level(l, self.n), -1):
            block_size = 1 << s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    path.B[j - branch_size, s - 1] = int(path.B[j, s]) ^ int(path.B[j - branch_size, s])
                    path.B[j, s - 1] = path.B[j, s]

    def decode(self, llr_ch):
        """主译码函数，返回 (u_hat, pm)。"""
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        llr = llr_ch[self.rev]

        paths = [Path(self.N, self.n, llr)]
        decode_order = [_bit_reversed_index(i, self.n) for i in range(self.N)]

        for l in decode_order:
            new_paths = []
            for path in paths:
                llr_val = self._compute_llrs(path, l)

                if l in self.frozen_set:
                    p = Path(self.N, self.n, llr)
                    p.pm = self._pm_update(path.pm, llr_val, 0)
                    p.u_hat = path.u_hat.copy()
                    p.L = path.L.copy()
                    p.B = path.B.copy()
                    p.u_hat[l] = 0
                    p.B[l, self.n] = 0
                    self._update_bits(p, l)
                    new_paths.append(p)
                else:
                    for u in (0, 1):
                        p = Path(self.N, self.n, llr)
                        p.pm = self._pm_update(path.pm, llr_val, u)
                        p.u_hat = path.u_hat.copy()
                        p.L = path.L.copy()
                        p.B = path.B.copy()
                        p.u_hat[l] = u
                        p.B[l, self.n] = u
                        self._update_bits(p, l)
                        new_paths.append(p)

            new_paths.sort(key=lambda p: p.pm)
            paths = new_paths[: self.list_size]

        if self.crc_length > 0:
            valid = [p for p in paths if crc_check(p.u_hat[self.info_indices], self.crc_length)]
            best = min(valid, key=lambda p: p.pm) if valid else min(paths, key=lambda p: p.pm)
        else:
            best = min(paths, key=lambda p: p.pm)

        return best.u_hat.copy(), best.pm
