"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from decoder_sc import (
    f_operation,
    g_operation,
    _bit_reversed_scalar,
    _active_llr_level,
    _active_bit_level,
)
from encoder import bit_reversal_permutation


def crc_encode(info_bits, crc_length=8):
    """CRC-8 (0x07) 或 CRC-16 (0x8005)"""
    info_bits = np.asarray(info_bits, dtype=np.int8)
    if crc_length == 8:
        poly = 0x07
    elif crc_length == 16:
        poly = 0x8005
    else:
        raise ValueError("crc_length must be 8 or 16")
    reg = 0
    for b in info_bits:
        reg ^= int(b) << (crc_length - 1)
        for _ in range(8):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)
    crc_bits = np.array([(reg >> i) & 1 for i in range(crc_length - 1, -1, -1)], dtype=int)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=np.int8)
    return np.array_equal(crc_encode(bits[:-crc_length], crc_length)[-crc_length:], bits[-crc_length:])


class _Path:
    __slots__ = ("pm", "L", "B", "u_hat")

    def __init__(self, N, n):
        self.pm = 0.0
        self.L = np.zeros((N, n + 1), dtype=np.float64)
        self.B = np.zeros((N, n + 1), dtype=np.int8)
        self.u_hat = np.zeros(N, dtype=int)


class SCLDecoder:
    """SCL 译码器（路径复制实现）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.list_size = list_size
        self.crc_length = crc_length
        frozen_bits = np.asarray(frozen_bits)
        self.frozen_set = set(np.where(frozen_bits.astype(bool))[0])
        self.rev = bit_reversal_permutation(N)

    def _advance_llr(self, path, l):
        for s in range(self.n - _active_llr_level(l, self.n), self.n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    path.L[j, s + 1] = f_operation(path.L[j, s], path.L[j + branch_size, s])
                else:
                    path.L[j, s + 1] = g_operation(
                        path.L[j - branch_size, s], path.L[j, s], path.B[j - branch_size, s + 1]
                    )

    def _advance_bits(self, path, l):
        if l < self.N // 2:
            return
        for s in range(self.n, self.n - _active_bit_level(l, self.n), -1):
            block_size = 2 ** s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    path.B[j - branch_size, s - 1] = (path.B[j, s] ^ path.B[j - branch_size, s]) & 1
                    path.B[j, s - 1] = path.B[j, s]

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        llr = llr_ch[self.rev].copy()
        paths = [_Path(self.N, self.n)]
        paths[0].L[:, 0] = llr

        decode_order = [_bit_reversed_scalar(i, self.n) for i in range(self.N)]

        for l in decode_order:
            for p in paths:
                self._advance_llr(p, l)
            new_paths = []
            for p in paths:
                llr_bit = p.L[l, self.n]
                if l in self.frozen_set:
                    pm = p.pm + (abs(llr_bit) if llr_bit < 0 else 0.0)
                    child = _Path(self.N, self.n)
                    child.pm = pm
                    child.L = p.L.copy()
                    child.B = p.B.copy()
                    child.u_hat = p.u_hat.copy()
                    child.B[l, self.n] = 0
                    child.u_hat[l] = 0
                    self._advance_bits(child, l)
                    new_paths.append(child)
                else:
                    for bit in (0, 1):
                        pm = p.pm + (0.0 if (bit == 0 and llr_bit >= 0) or (bit == 1 and llr_bit < 0) else abs(llr_bit))
                        child = _Path(self.N, self.n)
                        child.pm = pm
                        child.L = p.L.copy()
                        child.B = p.B.copy()
                        child.u_hat = p.u_hat.copy()
                        child.B[l, self.n] = bit
                        child.u_hat[l] = bit
                        self._advance_bits(child, l)
                        new_paths.append(child)
            new_paths.sort(key=lambda x: x.pm)
            paths = new_paths[: self.list_size]

        info_pos = sorted(set(range(self.N)) - self.frozen_set)
        crc_ok = [
            p
            for p in paths
            if self.crc_length == 0 or crc_check(p.u_hat[info_pos], self.crc_length)
        ]
        best = min(crc_ok if crc_ok else paths, key=lambda x: x.pm)
        return best.u_hat.copy(), best.pm
