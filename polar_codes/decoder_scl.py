"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np

from decoder_sc import (
    f_operation,
    g_operation,
    _active_bit_level,
    _active_llr_level,
    _align_channel_llrs,
    _bit_reversed_index,
    sc_decode,
)

CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_update(reg, bit, poly, crc_length):
    reg ^= int(bit)
    for _ in range(crc_length):
        if reg & 1:
            reg = (reg >> 1) ^ poly
        else:
            reg >>= 1
    return reg


def crc_encode(info_bits, crc_length=8):
    """
    计算 CRC 校验位并附加到信息比特后。
    CRC-8: 0x07; CRC-16: 0x8005（LSB 优先反射形式）
    """
    info_bits = np.asarray(info_bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    reg = 0
    for bit in info_bits:
        reg = _crc_update(reg, bit, poly, crc_length)
    crc_bits = np.array([(reg >> i) & 1 for i in range(crc_length)], dtype=int)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC"""
    bits = np.asarray(bits, dtype=int)
    if len(bits) < crc_length:
        return False
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    reg = 0
    for bit in bits:
        reg = _crc_update(reg, bit, poly, crc_length)
    return reg == 0


class _Path:
    __slots__ = ("pm", "B", "u_hat")

    def __init__(self, N, n):
        self.pm = 0.0
        self.B = np.full((N, n + 1), np.nan)
        self.u_hat = np.zeros(N, dtype=int)


class SCLDecoder:
    """SCL 译码器"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]

    @staticmethod
    def _metric(pm, llr, u):
        if (u == 0 and llr >= 0) or (u == 1 and llr < 0):
            return pm
        return pm + abs(llr)

    def _update_llrs(self, L, B, l):
        n = self.n
        for s in range(n - _active_llr_level(l, n), n):
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = f_operation(L[j, s], L[j + branch_size, s])
                else:
                    L[j, s + 1] = g_operation(
                        L[j - branch_size, s], L[j, s], B[j - branch_size, s + 1]
                    )

    def _update_bits(self, B, l):
        if l < self.N / 2:
            return
        n = self.n
        for s in range(n, n - _active_bit_level(l, n), -1):
            block_size = 1 << s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                    B[j, s - 1] = B[j, s]

    def _leaf_llr(self, llr_ch, B, l):
        L = np.full((self.N, self.n + 1), np.nan, dtype=np.float64)
        L[:, 0] = llr_ch
        self._update_llrs(L, B, l)
        return L[l, self.n]

    def decode(self, llr_ch, bit_reversed_codeword=True):
        if self.list_size == 1 and self.crc_length == 0:
            u_hat = sc_decode(llr_ch, self.frozen_bits, bit_reversed_codeword)
            return u_hat, 0.0

        llr_ch = _align_channel_llrs(llr_ch, bit_reversed_codeword)
        paths = [_Path(self.N, self.n)]

        for phi in range(self.N):
            l = _bit_reversed_index(phi, self.n)
            candidates = []

            for path in paths:
                llr_leaf = self._leaf_llr(llr_ch, path.B, l)
                if l in self.frozen_set:
                    u = 0
                    pm = self._metric(path.pm, llr_leaf, u)
                    candidates.append((pm, path, u))
                else:
                    for u in (0, 1):
                        pm = self._metric(path.pm, llr_leaf, u)
                        candidates.append((pm, path, u))

            candidates.sort(key=lambda x: x[0])
            candidates = candidates[: self.list_size]

            new_paths = []
            for pm, parent, u in candidates:
                child = _Path(self.N, self.n)
                child.pm = pm
                child.B = parent.B.copy()
                child.u_hat = parent.u_hat.copy()
                child.B[l, self.n] = u
                child.u_hat[l] = u
                self._update_bits(child.B, l)
                new_paths.append(child)
            paths = new_paths

        return self._select_best(paths)

    def _select_best(self, paths):
        if self.crc_length > 0:
            valid = []
            for path in paths:
                info_bits = path.u_hat[self.info_indices]
                if crc_check(info_bits, self.crc_length):
                    valid.append(path)
            if valid:
                best = min(valid, key=lambda p: p.pm)
                return best.u_hat.copy(), best.pm

        best = min(paths, key=lambda p: p.pm)
        return best.u_hat.copy(), best.pm
