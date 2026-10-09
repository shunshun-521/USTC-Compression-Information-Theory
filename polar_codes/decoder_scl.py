"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np

from encoder import bit_reversal_permutation
from decoder_bp import _f_min_sum
from decoder_sc import sc_decode


_CRC8_POLY = 0x07
_CRC16_POLY = 0x8005


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
    poly = _CRC8_POLY if crc_length == 8 else _CRC16_POLY
    rem = _crc_remainder(info_bits, poly, crc_length)
    crc_bits = np.array(
        [(rem >> i) & 1 for i in range(crc_length - 1, -1, -1)], dtype=int
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=int).ravel()
    poly = _CRC8_POLY if crc_length == 8 else _CRC16_POLY
    return _crc_remainder(bits, poly, crc_length) == 0


class _PathState:
    __slots__ = ("L", "R", "pm", "u_hat")

    def __init__(self, n, N):
        self.L = np.zeros((N, n + 1), dtype=np.float64)
        self.R = np.zeros((N, n + 1), dtype=np.float64)
        self.pm = 0.0
        self.u_hat = np.zeros(N, dtype=int)


class SCLDecoder:
    """SCL 译码器（因子图串行调度 + 路径列表）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.alpha = alpha
        self.LARGE = 1e6
        self.br = bit_reversal_permutation(N)

    def _init_path(self, llr):
        p = _PathState(self.n, self.N)
        p.L[:, self.n] = llr
        p.R[self.frozen_bits, 0] = self.LARGE
        return p

    def _update_L(self, path):
        N, n = self.N, self.n
        for j in range(n - 1, -1, -1):
            s = 2 ** j
            for i in range(0, N, 2 * s):
                for k in range(s):
                    i0 = i + k
                    i1 = i + k + s
                    path.L[i0, j] = _f_min_sum(
                        path.R[i0, j] + path.L[i1, j + 1],
                        path.L[i0, j + 1],
                        self.alpha,
                    )
                    path.L[i1, j] = (
                        _f_min_sum(path.R[i0, j], path.L[i0, j + 1], self.alpha)
                        + path.L[i1, j + 1]
                    )

    def _update_R(self, path):
        N, n = self.N, self.n
        for j in range(1, n + 1):
            s = 2 ** (j - 1)
            for i in range(0, N, 2 * s):
                for k in range(s):
                    i0 = i + k
                    i1 = i + k + s
                    path.R[i0, j] = _f_min_sum(
                        path.R[i1, j] + path.L[i1, j],
                        path.R[i0, j - 1],
                        self.alpha,
                    )
                    path.R[i1, j] = (
                        _f_min_sum(path.R[i0, j - 1], path.L[i0, j], self.alpha)
                        + path.R[i1, j - 1]
                    )

    def _pm_penalty(self, llr, u):
        u_hard = 0 if llr >= 0 else 1
        return 0.0 if u == u_hard else abs(llr)

    def decode(self, llr_ch):
        if self.list_size == 1 and self.crc_length == 0:
            return sc_decode(llr_ch, self.frozen_bits, self.alpha), 0.0

        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        llr = llr_ch[self.br]
        paths = [self._init_path(llr)]

        for phi in range(self.N):
            candidates = []
            for path in paths:
                self._update_L(path)
                llr_phi = path.L[phi, 0] + path.R[phi, 0]
                if self.frozen_bits[phi]:
                    path.pm += self._pm_penalty(llr_phi, 0)
                    path.u_hat[phi] = 0
                    path.R[phi, 0] = self.LARGE
                    self._update_R(path)
                    candidates.append(path)
                else:
                    for u in (0, 1):
                        cp = _PathState(self.n, self.N)
                        cp.L[:] = path.L
                        cp.R[:] = path.R
                        cp.u_hat[:] = path.u_hat
                        cp.pm = path.pm + self._pm_penalty(llr_phi, u)
                        cp.u_hat[phi] = u
                        cp.R[phi, 0] = self.LARGE if u == 0 else -self.LARGE
                        self._update_R(cp)
                        candidates.append(cp)

            candidates.sort(key=lambda p: p.pm)
            paths = candidates[: self.list_size]

        paths.sort(key=lambda p: p.pm)
        if self.crc_length > 0:
            info_positions = np.where(~self.frozen_bits)[0]
            for p in paths:
                bits = p.u_hat[info_positions]
                if crc_check(bits, self.crc_length):
                    return p.u_hat.copy(), p.pm
        best = paths[0]
        return best.u_hat.copy(), best.pm
