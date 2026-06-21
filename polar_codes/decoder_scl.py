"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np

from decoder_sc import (
    _active_bit_level,
    _active_llr_level,
    _bit_reversed,
    _path_metric_penalty,
    _update_bits,
    _update_llrs,
)
from encoder import bit_reversal_permutation


CRC_POLYS = {8: 0x07, 16: 0x8005}


def _crc_remainder(bits, crc_length):
    poly = CRC_POLYS[crc_length]
    reg = 0
    for bit in bits:
        reg ^= int(bit) << (crc_length - 1)
        for _ in range(crc_length):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)
    return reg


def crc_encode(info_bits, crc_length=8):
    info_bits = np.asarray(info_bits, dtype=int)
    remainder = _crc_remainder(info_bits, crc_length)
    crc_bits = np.array(
        [(remainder >> i) & 1 for i in range(crc_length - 1, -1, -1)], dtype=int
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=int)
    if len(bits) < crc_length:
        return False
    return np.array_equal(bits, crc_encode(bits[:-crc_length], crc_length))


class _Path:
    __slots__ = ("L", "B", "pm", "owned")

    def __init__(self, n, N, llr_ch=None, br=None):
        self.L = np.zeros((N, n + 1), dtype=np.float64)
        self.B = np.zeros((N, n + 1), dtype=np.int32)
        if llr_ch is not None:
            self.L[:, 0] = llr_ch[br]
        self.pm = 0.0
        self.owned = True

    def fork(self):
        child = _Path.__new__(_Path)
        child.L = self.L
        child.B = self.B
        child.pm = self.pm
        child.owned = False
        self.owned = False
        return child

    def make_writable(self):
        if not self.owned:
            self.L = self.L.copy()
            self.B = self.B.copy()
            self.owned = True


class SCLDecoder:
    """SCL 译码器（含 Lazy Copy 优化）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]
        self.br = bit_reversal_permutation(N)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [_Path(self.n, self.N, llr_ch, self.br)]

        for i in range(self.N):
            l = _bit_reversed(i, self.n)
            for path in paths:
                path.make_writable()
                _update_llrs(path.L, path.B, l, self.n, self.N)

            if self.frozen_bits[l]:
                for path in paths:
                    path.pm += _path_metric_penalty(path.L[l, self.n], 0)
                    path.B[l, self.n] = 0
            else:
                candidates = []
                for path in paths:
                    llr_val = path.L[l, self.n]
                    for bit in (0, 1):
                        child = path.fork()
                        child.pm += _path_metric_penalty(llr_val, bit)
                        child.make_writable()
                        child.B[l, self.n] = bit
                        candidates.append(child)
                candidates.sort(key=lambda p: p.pm)
                paths = candidates[: self.list_size]

            for path in paths:
                path.make_writable()
                _update_bits(path.B, l, self.n, self.N)

        if self.crc_length > 0:
            valid = []
            for path in paths:
                payload = path.B[:, self.n][self.info_indices]
                if crc_check(payload, self.crc_length):
                    valid.append(path)
            best = min(valid or paths, key=lambda p: p.pm)
        else:
            best = min(paths, key=lambda p: p.pm)

        u_hat = best.B[:, self.n].astype(np.int32)
        return u_hat, best.pm
