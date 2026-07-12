"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from encoder import bit_reversal_permutation, bit_reversed
from decoder_sc import (
    f_operation,
    g_operation,
    _active_llr_level,
    _active_bit_level,
    _permute_channel_llr,
    _update_llrs,
    _update_bits,
)

_CRC8_POLY_BITS = [1, 0, 0, 0, 0, 0, 1, 1, 1]
_CRC16_POLY_BITS = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1]


def _crc_remainder(bits, poly_bits, crc_length):
    msg = list(map(int, bits)) + [0] * crc_length
    for i in range(len(bits)):
        if msg[i]:
            for j in range(len(poly_bits)):
                msg[i + j] ^= poly_bits[j]
    return np.array(msg[len(bits):len(bits) + crc_length], dtype=int)


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=int)
    poly = _CRC8_POLY_BITS if crc_length == 8 else _CRC16_POLY_BITS
    crc_bits = _crc_remainder(info_bits, poly, crc_length)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC 是否正确"""
    bits = np.asarray(bits, dtype=int)
    poly = _CRC8_POLY_BITS if crc_length == 8 else _CRC16_POLY_BITS
    msg = list(map(int, bits))
    n = len(bits) - crc_length
    for i in range(n):
        if msg[i]:
            for j in range(len(poly)):
                if i + j < len(msg):
                    msg[i + j] ^= poly[j]
    return sum(msg[-crc_length:]) == 0


class _Path:
    __slots__ = ("pm", "L", "B", "u_hat")

    def __init__(self, N, n, llr):
        self.pm = 0.0
        self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.B = np.full((N, n + 1), np.nan)
        self.L[:, 0] = llr
        self.u_hat = np.zeros(N, dtype=int)

    def copy(self):
        import copy
        new = copy.copy(self)
        new.L = self.L.copy()
        new.B = self.B.copy()
        new.u_hat = self.u_hat.copy()
        return new


class SCLDecoder:
    """SCL 译码器（Lazy Copy）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.frozen_set = set(np.where(self.frozen_bits == 1)[0])
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(self.frozen_bits == 0)[0]

    def _pm_penalty(self, llr_val, u_val):
        hard = 0 if llr_val >= 0 else 1
        return 0.0 if u_val == hard else abs(llr_val)

    def decode(self, llr_ch):
        llr = _permute_channel_llr(llr_ch, self.N)
        paths = [_Path(self.N, self.n, llr)]

        for phi in range(self.N):
            l = bit_reversed(phi, self.n)
            candidates = []

            for path in paths:
                _update_llrs(path.L, path.B, l, self.n, self.N)
                llr_val = path.L[l, self.n]

                if l in self.frozen_set:
                    new_path = path.copy()
                    new_path.pm += self._pm_penalty(llr_val, 0)
                    new_path.u_hat[l] = 0
                    new_path.B[l, self.n] = 0
                    _update_bits(new_path.B, l, self.n, self.N)
                    candidates.append(new_path)
                else:
                    for u_val in (0, 1):
                        new_path = path.copy()
                        new_path.pm += self._pm_penalty(llr_val, u_val)
                        new_path.u_hat[l] = u_val
                        new_path.B[l, self.n] = u_val
                        _update_bits(new_path.B, l, self.n, self.N)
                        candidates.append(new_path)

            candidates.sort(key=lambda p: p.pm)
            paths = candidates[: self.list_size]

        if self.crc_length > 0:
            valid = [
                p for p in paths
                if crc_check(p.u_hat[self.info_indices], self.crc_length)
            ]
            best = min(valid or paths, key=lambda p: p.pm)
        else:
            best = min(paths, key=lambda p: p.pm)

        return best.u_hat.copy(), best.pm
