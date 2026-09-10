"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np
from decoder_sc import (
    _bit_reversed,
    _natural_to_tree_frozen,
    update_llrs_path,
    update_bits_path,
)


_CRC8_GEN = [1, 0, 0, 0, 0, 0, 1, 1, 1]
_CRC16_GEN = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1]


def _gf2_remainder(msg, gen):
    """GF(2) 多项式除法余数（MSB first）"""
    msg = [int(b) for b in msg]
    gen = list(gen)
    n = len(gen)
    for i in range(len(msg) - n + 1):
        if msg[i] == 1:
            for j in range(n):
                msg[i + j] ^= gen[j]
    return msg[-(n - 1):]


def crc_encode(info_bits, crc_length=8):
    info_bits = np.asarray(info_bits, dtype=int)
    gen = _CRC8_GEN if crc_length == 8 else _CRC16_GEN
    msg = list(info_bits) + [0] * crc_length
    remainder = _gf2_remainder(msg, gen)
    return np.concatenate([info_bits, np.array(remainder, dtype=int)])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=int)
    gen = _CRC8_GEN if crc_length == 8 else _CRC16_GEN
    remainder = _gf2_remainder(bits, gen)
    return all(r == 0 for r in remainder)


class _SCLPath:
    __slots__ = ("pm", "u_hat", "L", "B")

    def __init__(self, N, n):
        self.pm = 0.0
        self.u_hat = np.zeros(N, dtype=int)
        self.L = np.zeros((N, n + 1), dtype=np.float64)
        self.B = np.zeros((N, n + 1), dtype=int)

    def copy(self):
        new = _SCLPath(self.u_hat.shape[0], self.L.shape[1] - 1)
        new.pm = self.pm
        new.u_hat = self.u_hat.copy()
        new.L = self.L.copy()
        new.B = self.B.copy()
        return new


class SCLDecoder:
    """SCL 译码器（Lazy Copy）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_tree = _natural_to_tree_frozen(self.frozen_bits)
        self.info_positions = np.where(~self.frozen_bits)[0]
        self.list_size = list_size
        self.crc_length = crc_length

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [_SCLPath(self.N, self.n)]
        paths[0].L[:, 0] = llr_ch

        for phi in range(self.N):
            l = _bit_reversed(phi, self.n)
            candidates = []

            for path in paths:
                llr = update_llrs_path(path.L, path.B, l, self.n)

                if self.frozen_bits[phi]:
                    new_path = path.copy()
                    if llr < 0:
                        new_path.pm += abs(llr)
                    new_path.u_hat[phi] = 0
                    update_bits_path(new_path.B, l, self.n, 0)
                    candidates.append(new_path)
                else:
                    for u_bit in (0, 1):
                        new_path = path.copy()
                        expected = 0 if llr >= 0 else 1
                        if u_bit != expected:
                            new_path.pm += abs(llr)
                        new_path.u_hat[phi] = u_bit
                        update_bits_path(new_path.B, l, self.n, u_bit)
                        candidates.append(new_path)

            candidates.sort(key=lambda p: p.pm)
            paths = candidates[: self.list_size]

        if self.crc_length > 0:
            valid = [
                p for p in paths
                if crc_check(p.u_hat[self.info_positions], self.crc_length)
            ]
            best = min(valid if valid else paths, key=lambda p: p.pm)
        else:
            best = min(paths, key=lambda p: p.pm)

        return best.u_hat.copy(), best.pm
