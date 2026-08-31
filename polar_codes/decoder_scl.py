"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL），基于 Permuted SCD
"""
import math
import numpy as np
from decoder_sc import (
    f_operation,
    g_operation,
    _bit_reversed,
    _active_llr_level,
    _active_bit_level,
    _update_llrs_pscd,
    _update_bits_pscd,
)


def _crc_poly(crc_length):
    if crc_length == 8:
        return 0x07
    if crc_length == 16:
        return 0x8005
    raise ValueError(f"Unsupported CRC length: {crc_length}")


def crc_encode(info_bits, crc_length=8):
    """CRC-8: 0x07; CRC-16: 0x8005"""
    info_bits = np.asarray(info_bits, dtype=np.int8)
    poly = _crc_poly(crc_length)
    reg = 0
    for bit in info_bits:
        reg ^= int(bit) << (crc_length - 1)
        for _ in range(crc_length):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)
    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=np.int8,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC"""
    if crc_length == 0:
        return True
    bits = np.asarray(bits, dtype=np.int8)
    expected = crc_encode(bits[:-crc_length], crc_length)
    return np.array_equal(bits[-crc_length:], expected[-crc_length:])


class PathState:
    """单条 SCL 路径状态"""

    __slots__ = ("L", "B", "pm", "u_hat")

    def __init__(self, N, n, llr_ch):
        self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.B = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.pm = 0.0
        self.u_hat = np.zeros(N, dtype=int)
        self.L[:, 0] = llr_ch

    def copy_state(self):
        new = PathState.__new__(PathState)
        new.L = self.L.copy()
        new.B = self.B.copy()
        new.pm = self.pm
        new.u_hat = self.u_hat.copy()
        return new


class SCLDecoder:
    """SCL 译码器（Lazy Copy）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0, info_indices=None):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        if info_indices is None:
            self.info_indices = np.where(~self.frozen_bits)[0]
        else:
            self.info_indices = np.asarray(info_indices, dtype=int)

    def _path_metric_penalty(self, llr, u):
        hard = 0 if llr >= 0 else 1
        return 0.0 if u == hard else abs(llr)

    def decode(self, llr_ch):
        if self.list_size == 1 and self.crc_length == 0:
            from decoder_sc import sc_decode

            u_hat = sc_decode(llr_ch, self.frozen_bits)
            return u_hat, 0.0

        from channel import reorder_channel_llr

        llr_ch = reorder_channel_llr(np.asarray(llr_ch, dtype=np.float64))
        paths = [PathState(self.N, self.n, llr_ch)]

        for phi in range(self.N):
            l = _bit_reversed(phi, self.n)
            candidates = []

            for path in paths:
                current = path.copy_state()
                _update_llrs_pscd(current.L, current.B, l, self.n)
                llr = current.L[l, self.n]

                if self.frozen_bits[l]:
                    current.pm += self._path_metric_penalty(llr, 0)
                    current.B[l, self.n] = 0
                    current.u_hat[phi] = 0
                    _update_bits_pscd(current.B, l, self.n, self.N)
                    candidates.append(current)
                else:
                    c0 = current.copy_state()
                    c1 = current.copy_state()
                    c0.pm += self._path_metric_penalty(llr, 0)
                    c1.pm += self._path_metric_penalty(llr, 1)
                    c0.B[l, self.n] = 0
                    c1.B[l, self.n] = 1
                    c0.u_hat[phi] = 0
                    c1.u_hat[phi] = 1
                    _update_bits_pscd(c0.B, l, self.n, self.N)
                    _update_bits_pscd(c1.B, l, self.n, self.N)
                    candidates.extend([c0, c1])

            candidates.sort(key=lambda p: p.pm)
            paths = candidates[: self.list_size]

        best = None
        if self.crc_length > 0:
            valid = [
                p
                for p in paths
                if crc_check(p.u_hat[self.info_indices], self.crc_length)
            ]
            if valid:
                best = min(valid, key=lambda p: p.pm)
        if best is None:
            best = min(paths, key=lambda p: p.pm)
        return best.u_hat.copy(), best.pm
