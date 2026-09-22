"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL），Permuted SC
"""
import numpy as np
from decoder_sc import (
    _bit_reversed,
    _update_bits,
    _update_llrs,
)


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, crc_length):
    reg = 0
    for bit in bits:
        reg <<= 1
        reg |= int(bit)
        if reg & (1 << crc_length):
            reg ^= poly
    mask = (1 << crc_length) - 1
    return reg & mask


def crc_encode(info_bits, crc_length=8):
    info_bits = np.asarray(info_bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    remainder = _crc_remainder(info_bits, poly, crc_length)
    crc_bits = np.array(
        [(remainder >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=int,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=int)
    if len(bits) < crc_length:
        return False
    expected = crc_encode(bits[:-crc_length], crc_length)
    return np.array_equal(bits[-crc_length:], expected[-crc_length:])


class Path:
    __slots__ = ("pm", "L", "B", "u_hat")

    def __init__(self, n, N):
        self.pm = 0.0
        self.L = np.zeros((N, n + 1), dtype=np.float64)
        self.B = np.zeros((N, n + 1), dtype=np.float64)
        self.u_hat = np.zeros(N, dtype=int)

    def copy(self):
        p = Path.__new__(Path)
        p.pm = self.pm
        p.L = self.L.copy()
        p.B = self.B.copy()
        p.u_hat = self.u_hat.copy()
        return p


class SCLDecoder:
    def __init__(self, N, frozen_bits, list_size=4, crc_length=0, info_indices=None):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = info_indices

    def _pm_penalty(self, llr, u):
        u_hard = 0 if llr >= 0 else 1
        return 0.0 if u == u_hard else abs(llr)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N = self.N
        n = self.n

        paths = [Path(n, N)]
        paths[0].L[:, 0] = llr_ch

        for phi in range(N):
            l = _bit_reversed(phi, n)
            candidates = []
            for path in paths:
                _update_llrs(path.L, path.B, l, n, N)
                llr = path.L[l, n]

                if self.frozen_bits[l]:
                    new_path = path.copy()
                    new_path.pm += self._pm_penalty(llr, 0)
                    new_path.B[l, n] = 0
                    new_path.u_hat[l] = 0
                    _update_bits(new_path.B, l, n, N)
                    candidates.append(new_path)
                else:
                    for u in (0, 1):
                        new_path = path.copy()
                        new_path.pm += self._pm_penalty(llr, u)
                        new_path.B[l, n] = u
                        new_path.u_hat[l] = u
                        _update_bits(new_path.B, l, n, N)
                        candidates.append(new_path)

            candidates.sort(key=lambda p: p.pm)
            paths = candidates[: self.list_size]

        best_all = min(paths, key=lambda p: p.pm)
        if self.crc_length > 0:
            def _crc_ok(path):
                bits = path.u_hat[self.info_indices] if self.info_indices is not None else path.u_hat
                return crc_check(bits, self.crc_length)

            crc_paths = [p for p in paths if _crc_ok(p)]
            if crc_paths:
                best = min(crc_paths, key=lambda p: p.pm)
                return best.u_hat.copy(), best.pm

        return best_all.u_hat.copy(), best_all.pm
