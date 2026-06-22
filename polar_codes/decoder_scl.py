"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np
from encoder import bit_reversal_permutation
from decoder_sc import (
    precompute_sc_indices,
    _update_llrs,
    _update_bits,
)


_CRC8_POLY = 0x07
_CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, crc_length):
    reg = 0
    top = 1 << (crc_length - 1)
    mask = (1 << crc_length) - 1
    for bit in bits:
        reg ^= int(bit) << (crc_length - 1)
        if reg & top:
            reg = ((reg << 1) ^ poly) & mask
        else:
            reg = (reg << 1) & mask
    return reg


def crc_encode(info_bits, crc_length=8):
    info_bits = np.asarray(info_bits, dtype=int)
    poly = _CRC8_POLY if crc_length == 8 else _CRC16_POLY
    remainder = _crc_remainder(info_bits, poly, crc_length)
    crc_bits = np.array(
        [(remainder >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=int,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=int)
    poly = _CRC8_POLY if crc_length == 8 else _CRC16_POLY
    return _crc_remainder(bits, poly, crc_length) == 0


class Path:
    __slots__ = ("pm", "u_hat", "L", "B")

    def __init__(self, n, N, llr_ch=None):
        self.pm = 0.0
        self.u_hat = np.zeros(N, dtype=int)
        self.L = np.zeros((N, n + 1), dtype=np.float64)
        self.B = np.zeros((N, n + 1), dtype=np.int32)
        if llr_ch is not None:
            self.L[:, 0] = llr_ch

    def copy(self):
        new_path = Path(self.L.shape[1] - 1, self.L.shape[0])
        new_path.pm = self.pm
        new_path.u_hat = self.u_hat.copy()
        new_path.L = self.L.copy()
        new_path.B = self.B.copy()
        return new_path


class SCLDecoder:
    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(self.frozen_bits == 0)[0]
        self.decode_order, _, _ = precompute_sc_indices(N)
        self.br = bit_reversal_permutation(N)

    @staticmethod
    def _path_metric_update(pm, llr, u_bit):
        v = 0 if llr >= 0 else 1
        if u_bit != v:
            pm += abs(llr)
        return pm

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)[self.br]
        paths = [Path(self.n, self.N, llr_ch)]

        for l in self.decode_order:
            expanded = []
            for path in paths:
                _update_llrs(path.L, path.B, l, self.n, self.N)
                llr = path.L[l, self.n]
                bit_opts = [0] if self.frozen_bits[l] else [0, 1]
                for u_bit in bit_opts:
                    new_path = path.copy()
                    new_path.pm = self._path_metric_update(new_path.pm, llr, u_bit)
                    new_path.B[l, self.n] = u_bit
                    new_path.u_hat[l] = u_bit
                    _update_bits(new_path.B, l, self.n, self.N)
                    expanded.append(new_path)

            expanded.sort(key=lambda p: p.pm)
            paths = expanded[: self.list_size]

        u_candidates = [p.u_hat.copy() for p in paths]
        pms = [p.pm for p in paths]

        if self.crc_length > 0:
            valid = []
            for u_hat, pm in zip(u_candidates, pms):
                info_bits = u_hat[self.info_indices]
                if crc_check(info_bits, self.crc_length):
                    valid.append((pm, u_hat))
            if valid:
                valid.sort(key=lambda x: x[0])
                return valid[0][1], valid[0][0]

        best_idx = int(np.argmin(pms))
        return u_candidates[best_idx], pms[best_idx]
