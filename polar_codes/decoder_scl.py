"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np

from decoder_sc import (
    _SCDState,
    active_bit_level,
    active_llr_level,
    f_operation,
    g_operation,
)
from encoder import bit_reversed


POLY8 = np.array([1, 0, 0, 0, 0, 0, 1, 1, 1], dtype=np.int8)
POLY16 = np.array([1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], dtype=np.int8)


def _crc_remainder(bits, poly):
    reg = np.zeros(len(poly) - 1, dtype=np.int8)
    for bit in bits:
        feedback = bit ^ reg[0]
        reg[:-1] = reg[1:]
        reg[-1] = 0
        if feedback:
            reg ^= poly[1:]
    return reg


def crc_encode(info_bits, crc_length=8):
    info_bits = np.asarray(info_bits, dtype=np.int8)
    poly = POLY8 if crc_length == 8 else POLY16
    remainder = _crc_remainder(info_bits, poly)
    return np.concatenate([info_bits, remainder])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=np.int8)
    poly = POLY8 if crc_length == 8 else POLY16
    payload = bits[:-crc_length]
    expected = _crc_remainder(payload, poly)
    return np.array_equal(expected, bits[-crc_length:])


def _pm_update(pm, llr, u):
    hard = 0 if llr >= 0 else 1
    if u != hard:
        pm += abs(llr)
    return pm


class _SCLPath:
    __slots__ = ("state", "pm", "u_hat")

    def __init__(self, N, n, frozen_bits):
        self.state = _SCDState(N, frozen_bits)
        self.pm = 0.0
        self.u_hat = np.zeros(N, dtype=np.int8)

    def copy(self):
        n = self.state.n
        new_path = _SCLPath(self.state.N, n, self.state.frozen_bits)
        new_path.state.L[:] = self.state.L
        new_path.state.B[:] = self.state.B
        new_path.pm = self.pm
        new_path.u_hat[:] = self.u_hat
        return new_path


class SCLDecoder:
    """SCL 译码器（PSCD 风格）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [_SCLPath(self.N, self.n, self.frozen_bits)]
        paths[0].state.L[:, 0] = llr_ch.copy()

        for i in range(self.N):
            l = bit_reversed(i, self.n)
            new_paths = []

            for path in paths:
                path.state.update_llrs(l)
                llr_val = path.state.L[l, self.n]

                if self.frozen_bits[l]:
                    path.pm = _pm_update(path.pm, llr_val, 0)
                    path.state.B[l, self.n] = 0
                    path.u_hat[l] = 0
                    path.state.update_bits(l)
                    new_paths.append(path)
                else:
                    for u in (0, 1):
                        child = path.copy()
                        child.pm = _pm_update(child.pm, llr_val, u)
                        child.state.B[l, self.n] = u
                        child.u_hat[l] = u
                        child.state.update_bits(l)
                        new_paths.append(child)

            new_paths.sort(key=lambda p: p.pm)
            paths = new_paths[: self.list_size]

        if self.crc_length > 0:
            valid = []
            for path in paths:
                info_bits = path.u_hat[self.info_indices]
                if crc_check(info_bits, self.crc_length):
                    valid.append(path)
            if valid:
                paths = valid

        best = min(paths, key=lambda p: p.pm)
        return best.u_hat.astype(int), best.pm
