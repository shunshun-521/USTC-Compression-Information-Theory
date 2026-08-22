"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np
from encoder import bit_reversal_permutation
from decoder_sc import (
    upper_llr, lower_llr, _bit_reversed,
    _active_llr_level, _active_bit_level,
    _update_llrs, _update_bits,
)


CRC_POLYS = {
    8: 0x07,
    16: 0x8005,
}


def _gf2_crc_remainder(bits, crc_length=8):
    """GF(2) 多项式长除法余数，生成多项式 x^crc_length + ... + 1"""
    poly = CRC_POLYS[crc_length] | (1 << crc_length)
    msg = list(map(int, bits)) + [0] * crc_length
    for i in range(len(bits)):
        if msg[i]:
            for j in range(crc_length + 1):
                if (poly >> (crc_length - j)) & 1 and i + j < len(msg):
                    msg[i + j] ^= 1
    return np.array(msg[len(bits):len(bits) + crc_length], dtype=int)


def crc_encode(info_bits, crc_length=8):
    info_bits = np.asarray(info_bits, dtype=int)
    crc_bits = _gf2_crc_remainder(info_bits, crc_length)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=int)
    poly = CRC_POLYS[crc_length] | (1 << crc_length)
    msg = list(map(int, bits))
    for i in range(len(msg) - crc_length):
        if msg[i]:
            for j in range(crc_length + 1):
                if (poly >> (crc_length - j)) & 1 and i + j < len(msg):
                    msg[i + j] ^= 1
    return sum(msg[-crc_length:]) == 0


class Path:
    def __init__(self, N, n):
        self.N = N
        self.n = n
        self.L = np.zeros((N, n + 1), dtype=np.float64)
        self.B = np.zeros((N, n + 1), dtype=np.int8)
        self.pm = 0.0

    def copy(self):
        p = Path(self.N, self.n)
        p.L = self.L.copy()
        p.B = self.B.copy()
        p.pm = self.pm
        return p


class SCLDecoder:
    """SCL 译码器"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.list_size = list_size
        self.crc_length = crc_length
        self.br = bit_reversal_permutation(N)
        self.decode_order = [_bit_reversed(i, self.n) for i in range(N)]

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        llr = llr_ch[self.br]

        paths = [Path(self.N, self.n)]
        paths[0].L[:, 0] = llr

        for l in self.decode_order:
            new_paths = []
            for path in paths:
                _update_llrs(path.L, path.B, l, self.n, self.N)
                llr_val = path.L[l, self.n]

                if self.frozen_bits[l]:
                    if llr_val < 0:
                        path.pm += abs(llr_val)
                    path.B[l, self.n] = 0
                    _update_bits(path.B, l, self.n, self.N)
                    new_paths.append(path)
                else:
                    for u in (0, 1):
                        p = path.copy()
                        if u == 1 and llr_val >= 0:
                            p.pm += abs(llr_val)
                        elif u == 0 and llr_val < 0:
                            p.pm += abs(llr_val)
                        p.B[l, self.n] = u
                        _update_bits(p.B, l, self.n, self.N)
                        new_paths.append(p)

            new_paths.sort(key=lambda p: p.pm)
            paths = new_paths[:self.list_size]

        crc_pass = []
        for p in paths:
            u_hat = p.B[:, self.n].astype(int)
            if self.crc_length > 0:
                info_part = u_hat[~self.frozen_bits.astype(bool)]
                if crc_check(info_part, self.crc_length):
                    crc_pass.append(p)
            else:
                crc_pass.append(p)

        best = min(crc_pass if crc_pass else paths, key=lambda p: p.pm)
        return best.B[:, self.n].astype(int), best.pm
