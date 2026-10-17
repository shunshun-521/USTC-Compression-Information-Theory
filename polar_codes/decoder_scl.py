"""
极化码 SCL（串行抵消列表）译码器，支持 CRC 辅助（CA-SCL）
"""
import copy
import math
import numpy as np

from decoder_sc import (
    _active_bit_level,
    _active_llr_level,
    _bit_reversed,
    _lower_llr,
    _update_bits,
    _update_llrs,
    _upper_llr,
)


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 并附加到信息比特末尾"""
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

    crc_bits = np.array([(reg >> i) & 1 for i in range(crc_length - 1, -1, -1)], dtype=np.int8)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """校验 CRC"""
    bits = np.asarray(bits, dtype=np.int8)
    expected = crc_encode(bits[:-crc_length], crc_length)
    return np.array_equal(expected, bits[-crc_length:])


class _Path:
    __slots__ = ("L", "B", "pm", "u_hat")

    def __init__(self, N, n):
        self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.B = np.full((N, n + 1), np.nan)
        self.pm = 0.0
        self.u_hat = np.zeros(N, dtype=np.int8)


class SCLDecoder:
    """SCL 译码器（Lazy Copy：路径分裂时复制 LLR/比特树）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_set = set(int(i) for i in np.where(self.frozen_bits)[0])
        self.L_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]

    def _path_llr_at_root(self, path, l):
        return path.L[l, self.n]

    def _extend_path(self, path, l, bit):
        llr = self._path_llr_at_root(path, l)
        penalty = 0.0 if (bit == 0 and llr >= 0) or (bit == 1 and llr < 0) else abs(llr)
        new_path = _Path(self.N, self.n)
        new_path.L = path.L.copy()
        new_path.B = path.B.copy()
        new_path.pm = path.pm + penalty
        new_path.u_hat = path.u_hat.copy()
        new_path.u_hat[l] = bit
        new_path.B[l, self.n] = bit
        _update_bits(new_path.B, l, self.n, self.N)
        return new_path

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [_Path(self.N, self.n)]
        paths[0].L[:, 0] = llr_ch

        decode_order = [_bit_reversed(i, self.n) for i in range(self.N)]

        for l in decode_order:
            for p in paths:
                _update_llrs(p.L, p.B, l, self.n)

            if l in self.frozen_set:
                new_paths = []
                for p in paths:
                    llr = self._path_llr_at_root(p, l)
                    if llr < 0:
                        p.pm += abs(llr)
                    p.u_hat[l] = 0
                    p.B[l, self.n] = 0
                    _update_bits(p.B, l, self.n, self.N)
                    new_paths.append(p)
                paths = new_paths
            else:
                candidates = []
                for p in paths:
                    for bit in (0, 1):
                        candidates.append(self._extend_path(p, l, bit))
                candidates.sort(key=lambda x: x.pm)
                paths = candidates[: self.L_size]

        paths.sort(key=lambda x: x.pm)
        if self.crc_length > 0:
            for p in paths:
                if crc_check(p.u_hat[self.info_indices], self.crc_length):
                    return p.u_hat.copy(), p.pm
        return paths[0].u_hat.copy(), paths[0].pm
