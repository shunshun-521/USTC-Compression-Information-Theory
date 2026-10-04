"""
极化码 SCL（串行抵消列表）译码器，支持 CRC 辅助（CA-SCL）
"""
import copy
import numpy as np
from encoder import bit_reversal_permutation
from decoder_sc import upper_llr, lower_llr, _bit_reversed, _active_llr_level, _active_bit_level

CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_digest(info_bits, crc_length, poly):
    """按位串行 CRC（多项式 poly，长度 crc_length）"""
    mask = (1 << crc_length) - 1
    reg = 0
    for b in info_bits:
        reg ^= int(b) << (crc_length - 1)
        for _ in range(crc_length):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & mask
            else:
                reg = (reg << 1) & mask
    return reg


def crc_encode(info_bits, crc_length=8):
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    info_bits = np.asarray(info_bits, dtype=int)
    reg = _crc_digest(info_bits, crc_length, poly)
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
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    bits = np.asarray(bits, dtype=int)
    info = bits[:-crc_length]
    expected = crc_encode(info, crc_length)[-crc_length:]
    return np.array_equal(bits[-crc_length:], expected)


class PathState:
    def __init__(self, N, n):
        self.pm = 0.0
        self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.B = np.zeros((N, n + 1), dtype=np.int8)
        self.u_hat = np.zeros(N, dtype=int)


class SCLDecoder:
    """SCL 译码器"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0, info_indices=None):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen = np.asarray(frozen_bits, dtype=int).astype(bool)
        self.frozen_set = set(np.where(self.frozen)[0])
        self.Lsize = list_size
        self.crc_length = crc_length
        self.info_indices = (
            np.asarray(info_indices, dtype=int) if info_indices is not None else None
        )
        self.br = bit_reversal_permutation(N)
        self.decode_order = [_bit_reversed(i, self.n) for i in range(N)]

    def _pm_add(self, pm, llr, u):
        u_hard = 0 if llr >= 0 else 1
        if u != u_hard:
            pm += abs(llr)
        return pm

    def _update_llrs(self, path, l):
        for s in range(self.n - _active_llr_level(l, self.n), self.n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    path.L[j, s + 1] = upper_llr(path.L[j, s], path.L[j + branch_size, s])
                else:
                    path.L[j, s + 1] = lower_llr(
                        path.L[j, s],
                        path.L[j - branch_size, s],
                        int(path.B[j - branch_size, s + 1]),
                    )

    def _update_bits(self, path, l):
        if l < self.N / 2:
            return
        for s in range(self.n, self.n - _active_bit_level(l, self.n), -1):
            block_size = 2 ** s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    path.B[j - branch_size, s - 1] = int(path.B[j, s]) ^ int(
                        path.B[j - branch_size, s]
                    )
                    path.B[j, s - 1] = path.B[j, s]

    def decode(self, llr_ch):
        llr_perm = np.asarray(llr_ch, dtype=np.float64)[self.br]
        paths = [PathState(self.N, self.n)]
        paths[0].L[:, 0] = llr_perm

        for l in self.decode_order:
            new_paths = []
            for path in paths:
                self._update_llrs(path, l)
                llr = path.L[l, self.n]
                if l in self.frozen_set:
                    p2 = copy.deepcopy(path)
                    p2.pm = self._pm_add(path.pm, llr, 0)
                    p2.B[l, self.n] = 0
                    p2.u_hat[l] = 0
                    self._update_bits(p2, l)
                    new_paths.append(p2)
                else:
                    for u in (0, 1):
                        p2 = copy.deepcopy(path)
                        p2.pm = self._pm_add(path.pm, llr, u)
                        p2.B[l, self.n] = u
                        p2.u_hat[l] = u
                        self._update_bits(p2, l)
                        new_paths.append(p2)
            new_paths.sort(key=lambda p: p.pm)
            paths = new_paths[: self.Lsize]

        paths.sort(key=lambda p: p.pm)
        if self.crc_length > 0 and self.info_indices is not None:
            for p in paths:
                block = p.u_hat[self.info_indices]
                if crc_check(block, self.crc_length):
                    return p.u_hat, p.pm
        best = paths[0]
        return best.u_hat, best.pm
