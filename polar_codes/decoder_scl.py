"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np

from encoder import bit_reversal_permutation
from decoder_sc import (
    _active_bit_level,
    _active_llr_level,
    _bit_reversed,
    _update_bits,
    _update_llrs,
)


CRC8_POLY = 0x107   # x^8 + x^2 + x + 1
CRC16_POLY = 0x11021  # CRC-16-IBM


def _crc_mod(msg_bits, poly, crc_len):
    reg = list(map(int, msg_bits)) + [0] * crc_len
    plen = poly.bit_length()
    for i in range(len(msg_bits)):
        if reg[i]:
            for j in range(plen):
                if poly & (1 << (plen - 1 - j)):
                    k = i + j
                    if k < len(reg):
                        reg[k] ^= 1
    return np.array(reg[-crc_len:], dtype=int)


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=int).ravel()
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    remainder = _crc_mod(info_bits, poly, crc_length)
    return np.concatenate([info_bits, remainder])


def crc_check(bits, crc_length=8):
    """检验 bits 末尾 CRC 是否正确。"""
    bits = np.asarray(bits, dtype=int).ravel()
    if len(bits) < crc_length:
        return False
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    expected = _crc_mod(bits[:-crc_length], poly, crc_length)
    return np.array_equal(expected, bits[-crc_length:])


class _Path:
    __slots__ = ("L", "B", "pm", "u_hat")

    def __init__(self, N, n, llr):
        self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.B = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.L[:, 0] = llr
        self.pm = 0.0
        self.u_hat = np.zeros(N, dtype=int)


class SCLDecoder:
    """SCL 译码器（Permuted SCD + Lazy Copy）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.frozen_set = set(np.where(self.frozen_bits == 1)[0])
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(self.frozen_bits == 0)[0]

    @staticmethod
    def _pm_update(pm, llr_val, u_bit):
        hard = 0 if llr_val >= 0 else 1
        if u_bit != hard:
            pm += abs(llr_val)
        return pm

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N, n = self.N, self.n
        br = bit_reversal_permutation(N)
        llr = llr_ch[br]

        paths = [_Path(N, n, llr)]

        for i in range(N):
            l = _bit_reversed(i, n)
            candidates = []

            for path in paths:
                _update_llrs(path.L, path.B, l, n, N)
                llr_val = path.L[l, n]

                if l in self.frozen_set:
                    path.pm = self._pm_update(path.pm, llr_val, 0)
                    path.u_hat[l] = 0
                    path.B[l, n] = 0
                    _update_bits(path.B, l, n, N)
                    candidates.append(path)
                    continue

                for u_bit in (0, 1):
                    p = _Path(N, n, llr)
                    p.L = path.L.copy()
                    p.B = path.B.copy()
                    p.pm = self._pm_update(path.pm, llr_val, u_bit)
                    p.u_hat = path.u_hat.copy()
                    p.u_hat[l] = u_bit
                    p.B[l, n] = u_bit
                    _update_bits(p.B, l, n, N)
                    candidates.append(p)

            candidates.sort(key=lambda p: p.pm)
            paths = candidates[: self.list_size]

        if self.crc_length > 0:
            valid = [
                p
                for p in paths
                if crc_check(p.u_hat[self.info_indices], self.crc_length)
            ]
            if valid:
                paths = valid

        best = min(paths, key=lambda p: p.pm)
        return best.u_hat.copy(), best.pm
