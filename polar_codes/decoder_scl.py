"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
import math

from encoder import bit_reversal_permutation
from decoder_sc import (
    bit_reversed_index,
    _active_bit_level,
    _active_llr_level,
    f_operation,
    g_operation,
    sc_decode,
)

# ==================== CRC 工具 ====================

_CRC8_POLY = 0x07
_CRC16_POLY = 0x8005


def _crc_remainder(bits, crc_length):
    poly = _CRC8_POLY if crc_length == 8 else _CRC16_POLY
    reg = 0
    high = 1 << (crc_length - 1)
    mask = (1 << crc_length) - 1
    for bit in np.asarray(bits, dtype=int):
        reg ^= int(bit) << (crc_length - 1)
        for _ in range(crc_length):
            if reg & high:
                reg = ((reg << 1) ^ poly) & mask
            else:
                reg = (reg << 1) & mask
    return reg


def crc_encode(info_bits, crc_length=8):
    info_bits = np.asarray(info_bits, dtype=int)
    rem = _crc_remainder(info_bits, crc_length)
    crc_bits = np.array(
        [(rem >> i) & 1 for i in range(crc_length - 1, -1, -1)], dtype=int
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=int)
    return _crc_remainder(bits, crc_length) == 0


def _pm_update(pm, llr, u):
    hard = 0 if llr >= 0 else 1
    return pm if u == hard else pm + abs(llr)


class _Path:
    __slots__ = ("L", "B", "pm", "u_hat")

    def __init__(self, N, n, llr_bf):
        self.L = np.zeros((N, n + 1), dtype=np.float64)
        self.B = np.zeros((N, n + 1), dtype=int)
        self.L[:, 0] = llr_bf
        self.pm = 0.0
        self.u_hat = np.zeros(N, dtype=int)


class SCLDecoder:
    """SCL 译码器（路径维护 L/B 数组，Lazy Copy 浅拷贝）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.L_size = list_size
        self.crc_length = crc_length
        self.info_positions = np.where(self.frozen_bits == 0)[0]
        self.frozen_set = set(np.where(self.frozen_bits == 1)[0])
        self.br = bit_reversal_permutation(N)
        self.decode_order = [bit_reversed_index(i, self.n) for i in range(N)]

    def _update_llrs(self, path, l):
        N, n = self.N, self.n
        for s in range(n - _active_llr_level(l, n), n):
            block = 2 ** (s + 1)
            branch = block // 2
            for j in range(l, N, block):
                if j % block < branch:
                    path.L[j, s + 1] = f_operation(
                        path.L[j, s], path.L[j + branch, s]
                    )
                else:
                    path.L[j, s + 1] = g_operation(
                        path.L[j - branch, s],
                        path.L[j, s],
                        path.B[j - branch, s + 1],
                    )

    def _update_bits(self, path, l):
        N, n = self.N, self.n
        if l < N // 2:
            return
        for s in range(n, n - _active_bit_level(l, n), -1):
            block = 2**s
            branch = block // 2
            for j in range(l, -1, -block):
                if j % block >= branch:
                    path.B[j - branch, s - 1] = (
                        path.B[j, s] ^ path.B[j - branch, s]
                    )
                    path.B[j, s - 1] = path.B[j, s]

    def _fork(self, path):
        child = _Path(self.N, self.n, path.L[:, 0])
        child.L = path.L.copy()
        child.B = path.B.copy()
        child.pm = path.pm
        child.u_hat = path.u_hat.copy()
        return child

    def decode(self, llr_ch):
        if self.L_size == 1 and self.crc_length == 0:
            u = sc_decode(llr_ch, self.frozen_bits)
            return u, 0.0

        llr_bf = np.asarray(llr_ch, dtype=np.float64)[self.br]
        paths = [_Path(self.N, self.n, llr_bf)]

        for l in self.decode_order:
            new_paths = []
            for path in paths:
                self._update_llrs(path, l)
                llr_phi = path.L[l, self.n]

                if l in self.frozen_set:
                    path.pm = _pm_update(path.pm, llr_phi, 0)
                    path.u_hat[l] = 0
                    path.B[l, self.n] = 0
                    self._update_bits(path, l)
                    new_paths.append(path)
                else:
                    for u in (0, 1):
                        child = self._fork(path)
                        child.pm = _pm_update(child.pm, llr_phi, u)
                        child.u_hat[l] = u
                        child.B[l, self.n] = u
                        self._update_bits(child, l)
                        new_paths.append(child)

            new_paths.sort(key=lambda p: p.pm)
            paths = new_paths[: self.L_size]

        best = paths[0]
        if self.crc_length > 0:
            K_info = len(self.info_positions) - self.crc_length
            valid = []
            for p in paths:
                payload = p.u_hat[self.info_positions]
                if crc_check(payload, self.crc_length):
                    valid.append(p)
            if valid:
                best = min(valid, key=lambda p: p.pm)

        return best.u_hat.copy(), best.pm
