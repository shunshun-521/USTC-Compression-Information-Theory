"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np

from decoder_sc import (
    active_bit_level,
    active_llr_level,
    bit_reversed,
    lower_llr,
    upper_llr,
    _update_bits,
    _update_llrs,
)


def _crc_poly(crc_length):
    if crc_length == 8:
        return 0x07
    if crc_length == 16:
        return 0x8005
    raise ValueError("crc_length must be 8 or 16")


def _run_crc_reg(info_bits, crc_length):
    poly = _crc_poly(crc_length)
    mask = (1 << crc_length) - 1
    top = 1 << (crc_length - 1)
    reg = 0
    for b in info_bits:
        if int(b):
            reg ^= top
        if reg & top:
            reg = ((reg << 1) & mask) ^ poly
        else:
            reg = (reg << 1) & mask
    return reg


def crc_encode(info_bits, crc_length=8):
    info_bits = np.asarray(info_bits, dtype=int).ravel()
    reg = _run_crc_reg(info_bits, crc_length)
    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=int).ravel()
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
        self.B = np.zeros((N, n + 1), dtype=int)
        self.L[:, 0] = llr_ch


class SCLDecoder:
    """SCL 译码器（Permuted SC + 路径列表）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0, info_indices=None):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = info_indices
        self.frozen_set = set(np.where(self.frozen_bits)[0])

    def _penalty(self, llr, u):
        hard = 0 if llr >= 0.0 else 1
        return 0.0 if u == hard else abs(llr)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N, n, L = self.N, self.n, self.list_size

        paths = [_Path(N, n, llr_ch.copy())]

        for i in range(N):
            l = bit_reversed(i, n)
            candidates = []

            for path in paths:
                _update_llrs(path.L, path.B, l, n)
                cur_llr = path.L[l, n]

                if l in self.frozen_set:
                    pm = path.pm + self._penalty(cur_llr, 0)
                    p2 = _Path(N, n, llr_ch)
                    p2.pm = pm
                    p2.L = path.L.copy()
                    p2.B = path.B.copy()
                    p2.B[l, n] = 0
                    _update_bits(p2.B, l, n)
                    candidates.append(p2)
                else:
                    for u in (0, 1):
                        pm = path.pm + self._penalty(cur_llr, u)
                        p2 = _Path(N, n, llr_ch)
                        p2.pm = pm
                        p2.L = path.L.copy()
                        p2.B = path.B.copy()
                        p2.B[l, n] = u
                        _update_bits(p2.B, l, n)
                        candidates.append(p2)

            candidates.sort(key=lambda p: p.pm)
            paths = candidates[:L]

        if self.crc_length > 0 and self.info_indices is not None:
            good = []
            for p in paths:
                info = p.B[:, n][self.info_indices]
                if crc_check(info, self.crc_length):
                    good.append(p)
            if good:
                paths = good

        best = min(paths, key=lambda p: p.pm)
        return best.B[:, n].astype(int), best.pm
