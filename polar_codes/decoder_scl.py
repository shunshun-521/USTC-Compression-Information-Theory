"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from decoder_sc import (
    bit_reversed,
    f_operation,
    g_operation,
    _update_llr,
    _update_bits,
)


def _crc_poly(crc_length):
    if crc_length == 8:
        return 0x07
    if crc_length == 16:
        return 0x8005
    raise ValueError("crc_length must be 8 or 16")


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 并附加到信息比特后（MSB-first LFSR）"""
    info_bits = np.asarray(info_bits, dtype=np.int8)
    poly = _crc_poly(crc_length)
    reg = 0
    for bit in info_bits:
        reg ^= int(bit) << (crc_length - 1)
        if reg & (1 << (crc_length - 1)):
            reg = ((reg << 1) & ((1 << crc_length) - 1)) ^ poly
        else:
            reg = (reg << 1) & ((1 << crc_length) - 1)
    crc_bits = [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)]
    return np.concatenate([info_bits, np.array(crc_bits, dtype=np.int8)])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=np.int8)
    expected = crc_encode(bits[:-crc_length], crc_length)
    return np.array_equal(expected, bits)


class _Path:
    __slots__ = ("L", "B", "pm", "u_hat")

    def __init__(self, N, n, llr_ch):
        self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.B = np.zeros((N, n + 1), dtype=np.int8)
        self.L[:, n] = llr_ch.copy()
        self.pm = 0.0
        self.u_hat = np.zeros(N, dtype=np.int8)


class SCLDecoder:
    """SCL 译码器（路径复制实现，列表规模适中时足够高效）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.frozen_set = set(np.where(self.frozen_bits)[0])

    def _llr_at_root(self, path, l):
        _update_llr(path.L, path.B, l, self.n)
        return path.L[l, 0]

    def _advance_bits(self, path, l):
        _update_bits(path.B, l, self.n)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [_Path(self.N, self.n, llr_ch)]

        for i in range(self.N):
            l = bit_reversed(i, self.n)
            candidates = []

            for p_idx, path in enumerate(paths):
                llr = self._llr_at_root(path, l)
                if np.isnan(llr):
                    llr = 0.0

                if l in self.frozen_set:
                    pen = abs(llr) if llr < 0 else 0.0
                    new_path = self._clone(path)
                    new_path.pm += pen
                    new_path.u_hat[l] = 0
                    new_path.B[l, 0] = 0
                    self._advance_bits(new_path, l)
                    candidates.append(new_path)
                else:
                    for bit in (0, 1):
                        new_path = self._clone(path)
                        if bit == 0:
                            pen = abs(llr) if llr < 0 else 0.0
                        else:
                            pen = abs(llr) if llr >= 0 else 0.0
                        new_path.pm += pen
                        new_path.u_hat[l] = bit
                        new_path.B[l, 0] = bit
                        self._advance_bits(new_path, l)
                        candidates.append(new_path)

            candidates.sort(key=lambda p: p.pm)
            paths = candidates[: self.list_size]

        if self.crc_length > 0:
            valid = [p for p in paths if crc_check(p.u_hat, self.crc_length)]
            best = min(valid if valid else paths, key=lambda p: p.pm)
        else:
            best = min(paths, key=lambda p: p.pm)

        return best.u_hat.astype(int), best.pm

    def _clone(self, path):
        q = _Path(self.N, self.n, np.zeros(self.N))
        q.L = path.L.copy()
        q.B = path.B.copy()
        q.pm = path.pm
        q.u_hat = path.u_hat.copy()
        return q
