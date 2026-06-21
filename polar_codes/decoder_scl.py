"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np

from decoder_sc import (
    _active_bit_level,
    _active_llr_level,
    _bit_reversed_index,
    _update_bits_pscd,
    _update_llrs_pscd,
    sc_decode,
)
from encoder import bit_reversal_permutation


CRC8_POLY = [1, 0, 0, 0, 0, 0, 1, 1, 1]
CRC16_POLY = [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1]


def _crc_poly_div(msg, gen):
    msg = list(map(int, msg))
    r = len(gen) - 1
    reg = msg + [0] * r
    n = len(msg)
    for i in range(n):
        if reg[i]:
            for j, p in enumerate(gen):
                if p:
                    reg[i + j] ^= 1
    return np.array(reg[n : n + r], dtype=int)


def _crc_poly_check(bits, gen):
    bits = list(map(int, bits))
    r = len(gen) - 1
    reg = bits[:]
    n = len(bits) - r
    for i in range(n):
        if reg[i]:
            for j, p in enumerate(gen):
                if p:
                    reg[i + j] ^= 1
    return all(x == 0 for x in reg[n:])


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=int)
    gen = CRC8_POLY if crc_length == 8 else CRC16_POLY
    crc_bits = _crc_poly_div(info_bits, gen)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits 是否通过 CRC。"""
    bits = np.asarray(bits, dtype=int)
    gen = CRC8_POLY if crc_length == 8 else CRC16_POLY
    return _crc_poly_check(bits, gen)


class _PathState:
    __slots__ = ("pm", "L", "B")

    def __init__(self, N, n, llr_ch):
        self.pm = 0.0
        self.L = np.zeros((N, n + 1), dtype=np.float64)
        self.B = np.zeros((N, n + 1), dtype=np.int8)
        br = bit_reversal_permutation(N)
        self.L[:, 0] = llr_ch[br]


class SCLDecoder:
    """SCL 译码器（Lazy Copy：路径分裂时复制 PSCD 状态）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length

    @staticmethod
    def _metric_penalty(llr, bit):
        hard = 0 if llr >= 0 else 1
        return 0.0 if hard == bit else abs(llr)

    def _clone(self, path):
        new_path = _PathState(self.N, self.n, np.zeros(self.N))
        new_path.pm = path.pm
        new_path.L = path.L.copy()
        new_path.B = path.B.copy()
        return new_path

    def decode(self, llr_ch):
        """主译码函数。"""
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        if self.list_size == 1:
            return sc_decode(llr_ch, self.frozen_bits), 0.0

        paths = [_PathState(self.N, self.n, llr_ch)]

        for phi in range(self.N):
            l = _bit_reversed_index(phi, self.n)
            candidates = []

            for path in paths:
                _update_llrs_pscd(path.L, path.B, l, self.n)
                llr = path.L[l, self.n]

                if self.frozen_bits[l]:
                    path.pm += self._metric_penalty(llr, 0)
                    path.B[l, self.n] = 0
                    _update_bits_pscd(path.B, l, self.n, self.N)
                    candidates.append(path)
                else:
                    for bit in (0, 1):
                        new_path = self._clone(path)
                        new_path.pm += self._metric_penalty(llr, bit)
                        new_path.B[l, self.n] = bit
                        _update_bits_pscd(new_path.B, l, self.n, self.N)
                        candidates.append(new_path)

            candidates.sort(key=lambda p: p.pm)
            paths = candidates[: self.list_size]

        if self.crc_length > 0:
            info_positions = np.where(~self.frozen_bits)[0]
            valid = []
            for path in paths:
                bits = path.B[:, self.n].astype(int)[info_positions]
                if crc_check(bits, self.crc_length):
                    valid.append(path)
            pool = valid if valid else paths
        else:
            pool = paths

        best = min(pool, key=lambda p: p.pm)
        return best.B[:, self.n].astype(int), best.pm
