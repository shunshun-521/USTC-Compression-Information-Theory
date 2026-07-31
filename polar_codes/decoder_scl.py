"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math

import numpy as np

from decoder_sc import (
    f_operation,
    g_operation,
    precompute_sc_indices,
    _active_bit_level,
    _active_llr_level,
    _bit_reversed,
)
from encoder import bit_reversal_permutation


def _crc_poly(crc_length):
    if crc_length == 8:
        return 0x07
    if crc_length == 16:
        return 0x8005
    raise ValueError(f'Unsupported CRC length: {crc_length}')


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=np.int8)
    poly = _crc_poly(crc_length)
    reg = 0
    for bit in info_bits:
        reg <<= 1
        reg |= int(bit)
        if reg & (1 << crc_length):
            reg ^= poly
    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=np.int8,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。"""
    if crc_length == 0:
        return True
    expected = crc_encode(bits[:-crc_length], crc_length)
    return np.array_equal(bits[-crc_length:], expected[-crc_length:])


class _Path:
    """单条 SCL 译码路径（Lazy Copy）。"""

    __slots__ = ('pm', 'u_hat', 'L', 'B', 'parent', 'copied')

    def __init__(self, N, n, llr_ch):
        self.pm = 0.0
        self.u_hat = np.zeros(N, dtype=np.int8)
        self.L = np.zeros((N, n + 1), dtype=np.float64)
        self.B = np.zeros((N, n + 1), dtype=np.int8)
        self.L[:, 0] = llr_ch
        self.parent = None
        self.copied = False

    def fork(self):
        child = _Path.__new__(_Path)
        child.pm = self.pm
        child.u_hat = self.u_hat.copy()
        child.L = self.L
        child.B = self.B
        child.parent = self
        child.copied = False
        return child

    def ensure_copy(self):
        if self.parent is not None and not self.copied:
            self.L = self.L.copy()
            self.B = self.B.copy()
            self.copied = True


class SCLDecoder:
    """SCL 译码器（含 Lazy Copy 优化）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.decode_order = [_bit_reversed(i, self.n) for i in range(N)]
        self.info_indices = np.where(~self.frozen_bits)[0]

    def _update_llrs(self, path, l):
        n = self.n
        for s in range(n - _active_llr_level(l, n), n):
            block_size = 1 << (s + 1)
            branch_size = block_size >> 1
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    path.L[j, s + 1] = f_operation(
                        path.L[j, s], path.L[j + branch_size, s]
                    )
                else:
                    top_bit = path.B[j - branch_size, s + 1]
                    path.L[j, s + 1] = g_operation(
                        path.L[j - branch_size, s], path.L[j, s], top_bit
                    )

    def _propagate_bits(self, path, l):
        if l < self.N // 2:
            return
        n = self.n
        for s in range(n, n - _active_bit_level(l, n), -1):
            block_size = 1 << s
            branch_size = block_size >> 1
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    path.B[j - branch_size, s - 1] = (
                        path.B[j, s] ^ path.B[j - branch_size, s]
                    )
                    path.B[j, s - 1] = path.B[j, s]

    def decode(self, llr_ch):
        """主译码函数。"""
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        rev = bit_reversal_permutation(self.N)
        llr_internal = llr_ch[rev]

        paths = [_Path(self.N, self.n, llr_internal)]

        for l in self.decode_order:
            new_paths = []

            for path in paths:
                path.ensure_copy()
                self._update_llrs(path, l)
                llr = path.L[l, self.n]

                if self.frozen_bits[l]:
                    path.u_hat[l] = 0
                    if llr < 0:
                        path.pm += abs(llr)
                    path.B[l, self.n] = 0
                    self._propagate_bits(path, l)
                    new_paths.append(path)
                else:
                    for bit in (0, 1):
                        child = path.fork()
                        child.ensure_copy()
                        child.u_hat[l] = bit
                        child.B[l, self.n] = bit
                        hard = 0 if llr >= 0 else 1
                        if bit != hard:
                            child.pm += abs(llr)
                        self._propagate_bits(child, l)
                        new_paths.append(child)

            new_paths.sort(key=lambda p: p.pm)
            paths = new_paths[:self.list_size]

        crc_valid = []
        for path in paths:
            if self.crc_length > 0:
                info_bits = path.u_hat[self.info_indices]
                crc_valid.append(crc_check(info_bits, self.crc_length))
            else:
                crc_valid.append(True)

        candidates = [p for p, ok in zip(paths, crc_valid) if ok] if any(crc_valid) else paths
        best = min(candidates, key=lambda p: p.pm)
        return best.u_hat.copy(), best.pm
