"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL），与 SC 共用 L/B 矩阵结构
"""
import copy
import numpy as np
from encoder import bit_reversed_index
from decoder_sc import (
    active_bit_level,
    active_llr_level,
    lower_llr,
    upper_llr,
    _update_bits,
    _update_llr,
)

CRC_POLYS = {8: 0x07, 16: 0x8005}


def crc_encode(info_bits, crc_length=8):
    info_bits = np.asarray(info_bits, dtype=int)
    poly = CRC_POLYS.get(crc_length)
    if poly is None:
        raise ValueError(f"Unsupported CRC length: {crc_length}")

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
        dtype=int,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    if crc_length == 0:
        return True
    bits = np.asarray(bits, dtype=int)
    return np.array_equal(bits, crc_encode(bits[:-crc_length], crc_length))


class _Path:
    __slots__ = ("L", "B", "pm")

    def __init__(self, N, n, llr_ch):
        self.L = np.zeros((N, n + 1), dtype=np.float64)
        self.B = np.zeros((N, n + 1), dtype=np.int8)
        self.L[:, 0] = llr_ch
        self.pm = 0.0


class SCLDecoder:
    def __init__(self, N, frozen_bits, list_size=4, crc_length=0, info_indices=None):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = (
            np.asarray(info_indices, dtype=int)
            if info_indices is not None
            else np.flatnonzero(~self.frozen_bits)
        )

    def _pm_penalty(self, llr_val, u_bit):
        u_from_llr = 0 if llr_val >= 0 else 1
        return 0.0 if u_bit == u_from_llr else abs(llr_val)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n, N = self.n, self.N
        decode_order = [bit_reversed_index(i, n) for i in range(N)]

        paths = [_Path(N, n, llr_ch)]

        for l in decode_order:
            expanded = []
            for path in paths:
                _update_llr(path.L, path.B, l, n)
                llr_l = path.L[l, n]

                if self.frozen_bits[l]:
                    path.pm += self._pm_penalty(llr_l, 0)
                    path.B[l, n] = 0
                    _update_bits(path.B, l, n)
                    expanded.append(path)
                else:
                    for bit in (0, 1):
                        p = copy.deepcopy(path)
                        p.pm += self._pm_penalty(llr_l, bit)
                        p.B[l, n] = bit
                        _update_bits(p.B, l, n)
                        expanded.append(p)

            paths = sorted(expanded, key=lambda p: p.pm)[:self.list_size]

        best_crc = None
        best_any = paths[0]

        for p in paths:
            if p.pm < best_any.pm:
                best_any = p
            if self.crc_length > 0:
                info_bits = p.B[:, n][self.info_indices]
                if crc_check(info_bits, self.crc_length):
                    if best_crc is None or p.pm < best_crc.pm:
                        best_crc = p

        chosen = best_crc if best_crc is not None else best_any
        return chosen.B[:, n].astype(int).copy(), chosen.pm
