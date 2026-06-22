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
    _update_bits,
    _update_llrs,
    sc_decode,
)


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_division(info_bits, poly, crc_length):
    reg = np.zeros(crc_length, dtype=np.int8)
    poly_bits = [
        (crc_length - 1 - i) for i in range(crc_length + 1) if (poly >> i) & 1
    ]
    for bit in info_bits:
        feedback = bit ^ reg[0]
        reg[:-1] = reg[1:]
        reg[-1] = 0
        if feedback:
            for p in poly_bits[1:]:
                reg[p] ^= 1
    return reg


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=np.int8)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    crc_bits = _crc_division(info_bits, poly, crc_length)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。"""
    bits = np.asarray(bits, dtype=np.int8)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    remainder = _crc_division(bits, poly, crc_length)
    return not np.any(remainder)


class _Path:
    __slots__ = ("L", "B", "pm", "u_hat")

    def __init__(self, n, N):
        self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.B = np.zeros((N, n + 1), dtype=np.int8)
        self.pm = 0.0
        self.u_hat = np.zeros(N, dtype=np.int8)


class SCLDecoder:
    """SCL 译码器（PSCD 结构 + 路径复制）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0, info_positions=None):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        if info_positions is None:
            self.info_positions = np.where(~self.frozen_bits)[0]
        else:
            self.info_positions = np.asarray(info_positions, dtype=np.int64)
        self.frozen_set = set(np.where(self.frozen_bits)[0])

    def _path_metric_penalty(self, llr, bit):
        hard = 0 if llr >= 0 else 1
        return 0.0 if bit == hard else abs(llr)

    def decode(self, llr_ch):
        if self.list_size == 1 and self.crc_length == 0:
            u_hat = sc_decode(llr_ch, self.frozen_bits)
            return u_hat, 0.0

        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [_Path(self.n, self.N)]
        paths[0].L[:, 0] = llr_ch

        for phi in range(self.N):
            l = _bit_reversed_index(phi, self.n)
            candidates = []

            for path in paths:
                _update_llrs(path.L, path.B, l, self.n)
                llr_bit = path.L[l, self.n]

                if l in self.frozen_set:
                    new_path = self._clone_path(path)
                    new_path.pm += self._path_metric_penalty(llr_bit, 0)
                    new_path.u_hat[l] = 0
                    new_path.B[l, self.n] = 0
                    _update_bits(new_path.B, l, self.n)
                    candidates.append(new_path)
                else:
                    for bit in (0, 1):
                        new_path = self._clone_path(path)
                        new_path.pm += self._path_metric_penalty(llr_bit, bit)
                        new_path.u_hat[l] = bit
                        new_path.B[l, self.n] = bit
                        _update_bits(new_path.B, l, self.n)
                        candidates.append(new_path)

            candidates.sort(key=lambda p: p.pm)
            paths = candidates[: self.list_size]

        best = self._select_best_path(paths)
        return best.u_hat.copy(), best.pm

    def _clone_path(self, path):
        new_path = _Path(self.n, self.N)
        new_path.L = path.L.copy()
        new_path.B = path.B.copy()
        new_path.pm = path.pm
        new_path.u_hat = path.u_hat.copy()
        return new_path

    def _select_best_path(self, paths):
        if self.crc_length > 0:
            crc_ok = []
            for p in paths:
                info_bits = p.u_hat[self.info_positions]
                if crc_check(info_bits, self.crc_length):
                    crc_ok.append(p)
            if crc_ok:
                return min(crc_ok, key=lambda p: p.pm)
        return min(paths, key=lambda p: p.pm)
