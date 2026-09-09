"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np

from decoder_sc import (
    f_operation,
    g_operation,
    sc_decode,
    _bit_reversed,
    _active_llr_level,
    _active_bit_level,
)


_CRC8_POLY = 0x07
_CRC16_POLY = 0x8005


def _crc_process(bits, crc_length):
    poly = _CRC8_POLY if crc_length == 8 else _CRC16_POLY
    reg = 0
    for bit in bits:
        msb = (reg >> (crc_length - 1)) & 1
        reg = ((reg << 1) | int(bit)) & ((1 << crc_length) - 1)
        if msb:
            reg ^= poly
    return reg


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=int)
    remainder = _crc_process(list(info_bits) + [0] * crc_length, crc_length)
    crc_bits = np.array(
        [(remainder >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=int,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits 是否通过 CRC 校验。"""
    return _crc_process(np.asarray(bits, dtype=int), crc_length) == 0


class _PathState:
    """单条 SCL 路径，维护 L/B 矩阵。"""

    __slots__ = ("L", "B", "pm")

    def __init__(self, N, n, llr_ch):
        self.L = np.zeros((N, n + 1), dtype=np.float64)
        self.B = np.zeros((N, n + 1), dtype=int)
        self.L[:, 0] = llr_ch
        self.pm = 0.0

    def clone(self):
        p = _PathState.__new__(_PathState)
        p.L = self.L.copy()
        p.B = self.B.copy()
        p.pm = self.pm
        return p


def _update_llrs_path(L, B, l, n, N):
    for s in range(n - _active_llr_level(l, n), n):
        block_size = 1 << (s + 1)
        branch_size = block_size // 2
        for j in range(l, N, block_size):
            if j % block_size < branch_size:
                L[j, s + 1] = f_operation(L[j, s], L[j + branch_size, s])
            else:
                L[j, s + 1] = g_operation(
                    L[j, s], L[j - branch_size, s], B[j - branch_size, s + 1]
                )


def _update_bits_path(B, l, n, N):
    if l < N // 2:
        return
    for s in range(n, n - _active_bit_level(l, n), -1):
        block_size = 1 << s
        branch_size = block_size // 2
        for j in range(l, -1, -block_size):
            if j % block_size >= branch_size:
                B[j - branch_size, s - 1] = (
                    int(B[j, s]) ^ int(B[j - branch_size, s])
                )
                B[j, s - 1] = B[j, s]


class SCLDecoder:
    """SCL 译码器（含 Lazy Copy 优化）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.frozen_set = set(np.where(self.frozen_bits)[0])

    def _path_metric_penalty(self, llr, bit):
        hard = 0 if llr >= 0 else 1
        return 0.0 if bit == hard else abs(llr)

    def decode(self, llr_ch):
        """主译码函数，返回 (u_hat, pm)。"""
        llr_ch = np.asarray(llr_ch, dtype=np.float64)

        if self.list_size == 1 and self.crc_length == 0:
            return sc_decode(llr_ch, self.frozen_bits), 0.0

        paths = [_PathState(self.N, self.n, llr_ch)]

        for i in range(self.N):
            l = _bit_reversed(i, self.n)
            candidates = []

            for path in paths:
                _update_llrs_path(path.L, path.B, l, self.n, self.N)
                llr = path.L[l, self.n]

                if l in self.frozen_set:
                    child = path.clone()
                    child.pm += self._path_metric_penalty(llr, 0)
                    child.B[l, self.n] = 0
                    _update_bits_path(child.B, l, self.n, self.N)
                    candidates.append(child)
                else:
                    for bit in (0, 1):
                        child = path.clone()
                        child.pm += self._path_metric_penalty(llr, bit)
                        child.B[l, self.n] = bit
                        _update_bits_path(child.B, l, self.n, self.N)
                        candidates.append(child)

            candidates.sort(key=lambda p: p.pm)
            paths = candidates[: self.list_size]

        if self.crc_length > 0:
            info_idx = np.where(~self.frozen_bits)[0]
            valid = [
                p for p in paths
                if crc_check(p.B[:, self.n][info_idx], self.crc_length)
            ]
            best = min(valid if valid else paths, key=lambda p: p.pm)
        else:
            best = min(paths, key=lambda p: p.pm)

        return best.B[:, self.n].astype(int).copy(), best.pm
