"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np
from decoder_sc import f_operation, g_operation, _update_llrs, _update_bits
from encoder import bit_reversed, bit_reversal_permutation


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, r):
    bits = list(bits)
    for i in range(len(bits)):
        if bits[i]:
            for j in range(r + 1):
                if poly & (1 << (r - j)):
                    k = i + j
                    if k < len(bits):
                        bits[k] ^= 1
    return bits[-r:]


def crc_encode(info_bits, crc_length=8):
    info_bits = np.asarray(info_bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    padded = list(info_bits) + [0] * crc_length
    rem = _crc_remainder(padded, poly, crc_length)
    return np.array(list(info_bits) + rem, dtype=int)


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    rem = _crc_remainder(bits, poly, crc_length)
    return all(x == 0 for x in rem)


class _Path:
    __slots__ = ("L", "B", "pm", "u_hat")

    def __init__(self, N, n):
        self.L = np.zeros((N, n + 1), dtype=np.float64)
        self.B = np.zeros((N, n + 1), dtype=np.int8)
        self.pm = 0.0
        self.u_hat = np.zeros(N, dtype=int)

    def copy(self):
        p = _Path(self.L.shape[0], self.L.shape[1] - 1)
        p.L = self.L.copy()
        p.B = self.B.copy()
        p.pm = self.pm
        p.u_hat = self.u_hat.copy()
        return p


class SCLDecoder:
    """SCL 译码器"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(self.frozen_bits == 0)[0]

    def _pm_penalty(self, llr_val, bit):
        hard = 0 if llr_val >= 0 else 1
        return 0.0 if bit == hard else abs(llr_val)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        br = bit_reversal_permutation(self.N)
        llr_ch = llr_ch[br]
        paths = [_Path(self.N, self.n)]
        paths[0].L[:, 0] = llr_ch

        for phi in range(self.N):
            l = bit_reversed(phi, self.n)

            for path in paths:
                _update_llrs(path.L, path.B, l, self.n, self.N)

            if self.frozen_bits[phi]:
                new_paths = []
                for path in paths:
                    np_ = path.copy()
                    np_.pm += self._pm_penalty(path.L[l, self.n], 0)
                    np_.B[l, self.n] = 0
                    np_.u_hat[phi] = 0
                    _update_bits(np_.B, l, self.n, self.N)
                    new_paths.append(np_)
                paths = new_paths
            else:
                candidates = []
                for path in paths:
                    llr_val = path.L[l, self.n]
                    for bit in (0, 1):
                        np_ = path.copy()
                        np_.pm += self._pm_penalty(llr_val, bit)
                        np_.B[l, self.n] = bit
                        np_.u_hat[phi] = bit
                        _update_bits(np_.B, l, self.n, self.N)
                        candidates.append(np_)
                candidates.sort(key=lambda p: p.pm)
                paths = candidates[: self.list_size]

        if self.crc_length > 0:
            valid = [
                p
                for p in paths
                if crc_check(p.u_hat[self.info_indices], self.crc_length)
            ]
            best = min(valid if valid else paths, key=lambda p: p.pm)
        else:
            best = min(paths, key=lambda p: p.pm)

        return best.u_hat, best.pm
