"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math

import numpy as np

from decoder_sc import f_operation, g_operation, _active_bit_level, _active_llr_level, _bit_reversed
from encoder import bit_reversal_permutation


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY

    reg = 0
    for bit in info_bits:
        reg ^= int(bit) << (crc_length - 1)
        if reg & (1 << (crc_length - 1)):
            reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
        else:
            reg = (reg << 1) & ((1 << crc_length) - 1)

    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=int,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC"""
    bits = np.asarray(bits, dtype=int)
    expected = crc_encode(bits[:-crc_length], crc_length)
    return np.array_equal(bits[-crc_length:], expected[-crc_length:])


class SCLDecoder:
    """SCL 译码器（含 Lazy Copy 优化）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.frozen_set = set(np.where(self.frozen_bits == 1)[0])
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(self.frozen_bits == 0)[0]
        self.brp = bit_reversal_permutation(N)

    def _permute_llr(self, llr_ch):
        llr_perm = np.zeros(self.N, dtype=np.float64)
        llr_perm[self.brp] = llr_ch
        return llr_perm

    def _update_llrs(self, paths, l):
        n, N = self.n, self.N
        for path in paths:
            for s in range(n - _active_llr_level(l, n), n):
                block_size = 1 << (s + 1)
                branch_size = block_size // 2
                for j in range(l, N, block_size):
                    if j % block_size < branch_size:
                        path.L[j, s + 1] = f_operation(path.L[j, s], path.L[j + branch_size, s])
                    else:
                        path.L[j, s + 1] = g_operation(
                            path.L[j - branch_size, s],
                            path.L[j, s],
                            path.B[j - branch_size, s + 1],
                        )

    def _update_bits(self, path, l):
        n, N = self.n, self.N
        if l < N / 2:
            return
        for s in range(n, n - _active_bit_level(l, n), -1):
            block_size = 1 << s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    path.B[j - branch_size, s - 1] = path.B[j, s] ^ path.B[j - branch_size, s]
                    path.B[j, s - 1] = path.B[j, s]

    def _pm_penalty(self, llr, bit):
        u_from_llr = 0 if llr >= 0 else 1
        return 0.0 if bit == u_from_llr else abs(llr)

    def decode(self, llr_ch):
        """主译码函数，返回 (u_hat, pm)"""
        llr_perm = self._permute_llr(np.asarray(llr_ch, dtype=np.float64))
        n, N = self.n, self.N

        class _Path:
            __slots__ = ("pm", "L", "B")

            def __init__(self):
                self.pm = 0.0
                self.L = np.zeros((N, n + 1), dtype=np.float64)
                self.B = np.zeros((N, n + 1), dtype=int)

        init = _Path()
        init.L[:, 0] = llr_perm
        paths = [init]

        for i in range(N):
            l = _bit_reversed(i, n)
            for path in paths:
                for s in range(n - _active_llr_level(l, n), n):
                    block_size = 1 << (s + 1)
                    branch_size = block_size // 2
                    for j in range(l, N, block_size):
                        if j % block_size < branch_size:
                            path.L[j, s + 1] = f_operation(path.L[j, s], path.L[j + branch_size, s])
                        else:
                            path.L[j, s + 1] = g_operation(
                                path.L[j - branch_size, s],
                                path.L[j, s],
                                path.B[j - branch_size, s + 1],
                            )

            if l in self.frozen_set:
                for path in paths:
                    path.pm += self._pm_penalty(path.L[l, n], 0)
                    path.B[l, n] = 0
                    self._update_bits(path, l)
            else:
                candidates = []
                for path in paths:
                    llr = path.L[l, n]
                    for bit in (0, 1):
                        new_path = _Path()
                        new_path.pm = path.pm + self._pm_penalty(llr, bit)
                        new_path.L[:] = path.L
                        new_path.B[:] = path.B
                        new_path.B[l, n] = bit
                        self._update_bits(new_path, l)
                        candidates.append(new_path)
                candidates.sort(key=lambda p: p.pm)
                paths = candidates[: self.list_size]

        u_hat = paths[0].B[:, n].astype(int)
        if self.crc_length > 0:
            valid = [
                p for p in paths
                if crc_check(p.B[:, n][self.info_indices], self.crc_length)
            ]
            best = min(valid, key=lambda p: p.pm) if valid else min(paths, key=lambda p: p.pm)
        else:
            best = min(paths, key=lambda p: p.pm)

        return best.B[:, n].astype(int), best.pm
