"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
import math

from encoder import bit_reversal_permutation
from decoder_sc import (
    f_operation,
    g_operation,
    _active_llr_level,
    _active_bit_level,
    _update_llrs,
    _update_bits,
    _bit_reversed,
)


def crc_encode(info_bits, crc_length=8):
    """CRC-8 (0x07) 或 CRC-16 (0x8005)"""
    info_bits = np.asarray(info_bits, dtype=np.int8)
    if crc_length == 8:
        poly, mask, width = 0x07, 0xFF, 8
    elif crc_length == 16:
        poly, mask, width = 0x8005, 0xFFFF, 16
    else:
        raise ValueError("crc_length must be 8 or 16")

    reg = 0
    top = 1 << (width - 1)
    for bit in info_bits:
        feedback = ((reg >> (width - 1)) & 1) ^ int(bit)
        reg = ((reg << 1) & mask) ^ (poly if feedback else 0)

    crc_bits = np.array(
        [(reg >> i) & 1 for i in range(width - 1, -1, -1)], dtype=np.int8
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=np.int8)
    if crc_length == 0:
        return True
    payload = bits[:-crc_length]
    expected = crc_encode(payload, crc_length)[-crc_length:]
    return np.array_equal(bits[-crc_length:], expected)


class _Path:
    __slots__ = ("L", "B", "pm", "parent", "u_hat")

    def __init__(self, N, n, llr_ch, parent=None):
        self.parent = parent
        if parent is None:
            self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
            self.B = np.zeros((N, n + 1), dtype=np.int8)
            self.L[:, 0] = llr_ch
            self.pm = 0.0
            self.u_hat = np.zeros(N, dtype=np.int8)
        else:
            self.L = parent.L
            self.B = parent.B.copy()
            self.pm = parent.pm
            self.u_hat = parent.u_hat.copy()


class SCLDecoder:
    """SCL 译码器（Lazy Copy：分裂时复制 B 与 u_hat）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_positions = np.where(~self.frozen_bits)[0]

    def _pm_update(self, pm, llr_val, u_bit):
        penalty = 0.0 if (u_bit == 0 and llr_val >= 0) or (u_bit == 1 and llr_val < 0) else abs(llr_val)
        return pm + penalty

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [_Path(self.N, self.n, llr_ch)]

        for i in range(self.N):
            l = _bit_reversed(i, self.n)
            new_paths = []

            for path in paths:
                _update_llrs(path.L, path.B, l, self.n, self.N)
                llr_leaf = path.L[l, self.n]

                if l in self.frozen_set:
                    u_bit = 0
                    path.pm = self._pm_update(path.pm, llr_leaf, u_bit)
                    path.B[l, self.n] = u_bit
                    path.u_hat[l] = u_bit
                    _update_bits(path.B, l, self.n, self.N)
                    new_paths.append(path)
                else:
                    for u_bit in (0, 1):
                        child = _Path(self.N, self.n, llr_ch, parent=path)
                        child.pm = self._pm_update(path.pm, llr_leaf, u_bit)
                        child.B[l, self.n] = u_bit
                        child.u_hat[l] = u_bit
                        _update_bits(child.B, l, self.n, self.N)
                        new_paths.append(child)

            new_paths.sort(key=lambda p: p.pm)
            paths = new_paths[: self.list_size]

        if self.crc_length > 0:
            valid = [p for p in paths if crc_check(p.u_hat[self.info_positions], self.crc_length)]
            chosen = valid[0] if valid else paths[0]
        else:
            chosen = paths[0]

        return chosen.u_hat.copy(), chosen.pm
