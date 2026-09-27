"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from decoder_sc import (
    sc_decode,
    reorder_channel_llr,
    _frozen_bool,
    bit_reversed,
    _update_llrs,
    _update_bits,
)


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, crc_length):
    reg = 0
    for b in bits:
        reg ^= int(b) << (crc_length - 1)
        for _ in range(8):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)
    return reg


def crc_encode(info_bits, crc_length=8):
    info_bits = np.asarray(info_bits, dtype=int).ravel()
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    rem = _crc_remainder(info_bits, poly, crc_length)
    crc_bits = np.array(
        [(rem >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=int).ravel()
    if len(bits) < crc_length:
        return False
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    return _crc_remainder(bits, poly, crc_length) == 0


class _Path:
    __slots__ = ("pm", "B", "u_hat")

    def __init__(self, n, N):
        self.pm = 0.0
        self.B = np.full((N, n + 1), np.nan)
        self.u_hat = np.zeros(N, dtype=int)


class SCLDecoder:
    """SCL 译码器"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = _frozen_bool(frozen_bits)
        self.frozen_idx = set(np.where(self.frozen_bits)[0])
        self.list_size = list_size
        self.crc_length = crc_length

    def _pm_penalty(self, llr, u):
        u_hard = 0 if llr >= 0 else 1
        return 0.0 if u == u_hard else abs(llr)

    def decode(self, llr_ch):
        if self.list_size == 1 and self.crc_length == 0:
            u_hat = sc_decode(llr_ch, self.frozen_bits)
            return u_hat, 0.0

        llr_ch = reorder_channel_llr(llr_ch)
        n = self.n
        N = self.N

        paths = [_Path(n, N)]

        for i in range(N):
            l = bit_reversed(i, n)
            candidates = []
            for pidx, path in enumerate(paths):
                L = np.full((N, n + 1), np.nan, dtype=np.float64)
                L[:, 0] = llr_ch
                _update_llrs(L, path.B, l, n, N)
                llr = L[l, n]
                if l in self.frozen_idx:
                    candidates.append((path.pm + self._pm_penalty(llr, 0), pidx, 0))
                else:
                    for u in (0, 1):
                        candidates.append(
                            (path.pm + self._pm_penalty(llr, u), pidx, u)
                        )

            candidates.sort(key=lambda t: t[0])
            candidates = candidates[: self.list_size]

            new_paths = []
            for new_pm, parent_idx, u in candidates:
                parent = paths[parent_idx]
                child = _Path(n, N)
                child.pm = new_pm
                child.B = parent.B.copy()
                child.u_hat = parent.u_hat.copy()
                child.B[l, n] = u
                child.u_hat[l] = u
                _update_bits(child.B, l, n, N)
                new_paths.append(child)
            paths = new_paths

        if self.crc_length > 0:
            valid = [
                p
                for p in paths
                if crc_check(p.u_hat[~self.frozen_bits], self.crc_length)
            ]
            pool = valid if valid else paths
        else:
            pool = paths

        best = min(pool, key=lambda p: p.pm)
        return best.u_hat.copy(), best.pm
