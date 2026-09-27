"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from decoder_sc import (
    SCDecoderNonRecursive,
    bit_reversed_index,
    active_llr_level,
    active_bit_level,
    upper_llr,
    lower_llr,
)


def _crc_poly(crc_length):
    if crc_length == 8:
        return 0x07
    if crc_length == 16:
        return 0x8005
    raise ValueError("crc_length must be 8 or 16")


def crc_encode(info_bits, crc_length=8):
    """CRC 附加到信息比特末尾（MSB 优先）"""
    info_bits = np.asarray(info_bits, dtype=np.int8)
    poly = _crc_poly(crc_length)
    reg = 0
    for b in info_bits:
        reg ^= int(b) << (crc_length - 1)
        for _ in range(8):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) & ((1 << crc_length) - 1)) ^ poly
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)
    crc_bits = np.array([(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """校验 bits 末尾 crc 是否与前面信息匹配"""
    bits = np.asarray(bits, dtype=np.int8)
    return np.array_equal(crc_encode(bits[:-crc_length], crc_length)[-crc_length:], bits[-crc_length:])


class _Path:
    __slots__ = ("pm", "L", "B", "u")

    def __init__(self, N, n, llr_ch):
        self.pm = 0.0
        self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.B = np.zeros((N, n + 1), dtype=np.float64)
        self.L[:, 0] = llr_ch
        self.u = np.zeros(N, dtype=int)


class SCLDecoder:
    """SCL 译码器（路径复制实现，适用于中等 N/L）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_set = set(np.where(np.asarray(frozen_bits, dtype=int) == 1)[0])
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = sorted(set(range(N)) - self.frozen_set)

    def _update_llrs_path(self, path, l):
        for s in range(self.n - active_llr_level(l, self.n), self.n):
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

    def _update_bits_path(self, path, l):
        if l < self.N / 2:
            return
        for s in range(self.n, self.n - active_bit_level(l, self.n), -1):
            block_size = 2 ** s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    path.B[j - branch_size, s - 1] = int(path.B[j, s]) ^ int(
                        path.B[j - branch_size, s]
                    )
                    path.B[j, s - 1] = path.B[j, s]

    def _clone_path(self, path):
        new = _Path(self.N, self.n, path.L[:, 0])
        new.pm = path.pm
        new.L = path.L.copy()
        new.B = path.B.copy()
        new.u = path.u.copy()
        return new

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [_Path(self.N, self.n, llr_ch)]

        for i in range(self.N):
            l = bit_reversed_index(i, self.n)
            candidates = []
            for path in paths:
                self._update_llrs_path(path, l)
                llr_bit = path.L[l, self.n]
                if l in self.frozen_set:
                    penalty = 0.0 if llr_bit >= 0 else abs(llr_bit)
                    np_path = self._clone_path(path)
                    np_path.pm += penalty
                    np_path.B[l, self.n] = 0
                    np_path.u[l] = 0
                    self._update_bits_path(np_path, l)
                    candidates.append(np_path)
                else:
                    for bit in (0, 1):
                        cp = self._clone_path(path)
                        expected = 0 if llr_bit >= 0 else 1
                        if bit != expected:
                            cp.pm += abs(llr_bit)
                        cp.B[l, self.n] = bit
                        cp.u[l] = bit
                        self._update_bits_path(cp, l)
                        candidates.append(cp)
            candidates.sort(key=lambda p: p.pm)
            paths = candidates[: self.list_size]

        if self.crc_length > 0:
            valid = []
            for p in paths:
                info_bits = p.u[self.info_indices]
                if crc_check(info_bits, self.crc_length):
                    valid.append(p)
            if valid:
                paths = valid

        best = min(paths, key=lambda p: p.pm)
        return best.u.copy(), best.pm


def scl_equivalent_sc(llr, frozen_bits):
    """L=1 的 SCL 应与 SC 一致（用于单元测试）"""
    dec = SCLDecoder(len(llr), frozen_bits, list_size=1, crc_length=0)
    u, _ = dec.decode(llr)
    return u
