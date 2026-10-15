"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np

from decoder_sc import (
    _bit_reversed,
    update_llrs_mcba,
    update_bits_mcba,
)

_CRC_POLY = {8: 0x07, 16: 0x8005}


def _crc_bits_remainder(bits, crc_length):
    poly = _CRC_POLY[crc_length]
    mask = (1 << crc_length) - 1
    top = 1 << (crc_length - 1)
    reg = 0
    for b in np.asarray(bits, dtype=np.int_).ravel():
        reg ^= int(b) << (crc_length - 1)
        if reg & top:
            reg = ((reg << 1) ^ poly) & mask
        else:
            reg = (reg << 1) & mask
    return reg


def crc_encode(info_bits, crc_length=8):
    info_bits = np.asarray(info_bits, dtype=np.int_)
    rem = _crc_bits_remainder(info_bits, crc_length)
    crc_bits = np.array(
        [(rem >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=np.int_
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=np.int_)
    if len(bits) < crc_length:
        return False
    payload = bits[:-crc_length]
    expected = crc_encode(payload, crc_length)[-crc_length:]
    return np.array_equal(bits[-crc_length:], expected)


class _Path:
    __slots__ = ("L", "B", "pm", "u_hat")

    def __init__(self, N, n, llr_ch=None, other=None):
        if other is None:
            self.L = np.zeros((N, n + 1), dtype=np.float64)
            self.B = np.zeros((N, n + 1), dtype=np.int_)
            if llr_ch is not None:
                self.L[:, 0] = llr_ch.copy()
            self.pm = 0.0
            self.u_hat = np.zeros(N, dtype=np.int_)
        else:
            self.L = other.L.copy()
            self.B = other.B.copy()
            self.pm = other.pm
            self.u_hat = other.u_hat.copy()


class SCLDecoder:
    """SCL 译码器"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0, info_indices=None):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=np.int_)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = None if info_indices is None else np.asarray(info_indices, dtype=int)
        self.frozen_set = set(np.where(self.frozen_bits)[0])

    @staticmethod
    def _penalty(llr, u_bit):
        hard = 0 if llr >= 0 else 1
        return abs(llr) if u_bit != hard else 0.0

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [_Path(self.N, self.n, llr_ch=llr_ch)]

        for phi in range(self.N):
            l = _bit_reversed(phi, self.n)
            new_paths = []
            for path in paths:
                update_llrs_mcba(path.L, path.B, l, self.n)
                llr = path.L[l, self.n]

                if l in self.frozen_set:
                    child = _Path(self.N, self.n, other=path)
                    child.pm += self._penalty(llr, 0)
                    child.B[l, self.n] = 0
                    child.u_hat[l] = 0
                    update_bits_mcba(child.B, l, self.n)
                    new_paths.append(child)
                else:
                    for u_bit in (0, 1):
                        child = _Path(self.N, self.n, other=path)
                        child.pm += self._penalty(llr, u_bit)
                        child.B[l, self.n] = u_bit
                        child.u_hat[l] = u_bit
                        update_bits_mcba(child.B, l, self.n)
                        new_paths.append(child)

            new_paths.sort(key=lambda p: p.pm)
            paths = new_paths[: self.list_size]

        if self.crc_length > 0:
            def _crc_ok(p):
                bits = p.u_hat[self.info_indices] if self.info_indices is not None else p.u_hat
                return crc_check(bits, self.crc_length)

            valid = [p for p in paths if _crc_ok(p)]
            pool = valid if valid else paths
        else:
            pool = paths

        best = min(pool, key=lambda p: p.pm)
        return best.u_hat.copy(), best.pm
