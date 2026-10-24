"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np

from encoder import bit_reversal_permutation
from decoder_sc import (
    bit_reversed_index,
    _update_llrs,
    _update_bits,
)


def _crc_poly(crc_length):
    if crc_length == 8:
        return 0x07
    if crc_length == 16:
        return 0x8005
    raise ValueError("crc_length must be 8 or 16")


def _crc_run(bits, crc_length):
    poly = _crc_poly(crc_length)
    width = 8 if crc_length == 8 else 16
    mask = (1 << width) - 1
    top = 1 << (width - 1)
    crc = 0
    for b in bits:
        crc ^= int(b) << (width - 1)
        for _ in range(width):
            if crc & top:
                crc = ((crc << 1) ^ poly) & mask
            else:
                crc = (crc << 1) & mask
    return crc


def crc_encode(info_bits, crc_length=8):
    info_bits = np.asarray(info_bits, dtype=int)
    rem = _crc_run(info_bits, crc_length)
    width = 8 if crc_length == 8 else 16
    crc_bits = np.array(
        [(rem >> (width - 1 - i)) & 1 for i in range(crc_length)], dtype=int
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=int)
    if len(bits) < crc_length:
        return False
    payload = bits[:-crc_length]
    expected = crc_encode(payload, crc_length)[-crc_length:]
    return np.array_equal(bits[-crc_length:], expected)


class _Path:
    __slots__ = ("pm", "L", "B")

    def __init__(self, N, n, llr_ch):
        self.pm = 0.0
        self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.B = np.full((N, n + 1), np.nan)
        br = bit_reversal_permutation(N)
        self.L[:, 0] = llr_ch[br]


class SCLDecoder:
    """SCL 译码器（每条路径独立 L/B 数组）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0, info_indices=None):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = (
            np.sort(np.asarray(info_indices, dtype=int))
            if info_indices is not None
            else np.sort(np.where(~self.frozen_bits)[0])
        )

    @staticmethod
    def _pm_penalty(llr, u):
        v = 0 if llr >= 0 else 1
        return 0.0 if u == v else abs(llr)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [_Path(self.N, self.n, llr_ch)]

        for i in range(self.N):
            l = bit_reversed_index(i, self.n)
            candidates = []

            for path in paths:
                _update_llrs(path.L, path.B, l, self.n, self.N)
                cur = path.L[l, self.n]
                if self.frozen_bits[l]:
                    pm = path.pm + self._pm_penalty(cur, 0)
                    child = _Path(self.N, self.n, llr_ch)
                    child.pm = pm
                    child.L = path.L.copy()
                    child.B = path.B.copy()
                    child.B[l, self.n] = 0
                    candidates.append(child)
                else:
                    for u in (0, 1):
                        pm = path.pm + self._pm_penalty(cur, u)
                        child = _Path(self.N, self.n, llr_ch)
                        child.pm = pm
                        child.L = path.L.copy()
                        child.B = path.B.copy()
                        child.B[l, self.n] = u
                        candidates.append(child)

            candidates.sort(key=lambda p: p.pm)
            paths = candidates[: self.list_size]

            for path in paths:
                _update_bits(path.B, l, self.n, self.N)

        if self.crc_length > 0:
            passed = []
            for p in paths:
                u_full = p.B[:, self.n].astype(int)
                if crc_check(u_full[self.info_indices], self.crc_length):
                    passed.append(p)
            best = min(passed if passed else paths, key=lambda p: p.pm)
        else:
            best = min(paths, key=lambda p: p.pm)

        return best.B[:, self.n].astype(int), best.pm
