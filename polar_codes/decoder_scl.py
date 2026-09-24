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
    f_operation,
    g_operation,
    precompute_sc_indices,
)
from encoder import bit_reversal_permutation


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, crc_length):
    crc = 0
    mask = (1 << crc_length) - 1
    for bit in bits:
        msb = (crc >> (crc_length - 1)) & 1
        crc = (crc << 1) & mask
        if msb ^ int(bit):
            crc ^= poly
    return crc


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    remainder = _crc_remainder(info_bits, poly, crc_length)
    crc_bits = np.array(
        [(remainder >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=int,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits 是否通过 CRC"""
    bits = np.asarray(bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    return _crc_remainder(bits, poly, crc_length) == 0


class Path:
    """SCL 单条路径"""

    __slots__ = ("pm", "u_hat", "L", "B")

    def __init__(self, N, n):
        self.pm = 0.0
        self.u_hat = np.zeros(N, dtype=int)
        self.L = np.zeros((N, n + 1), dtype=np.float64)
        self.B = np.zeros((N, n + 1), dtype=np.float64)

    def copy(self):
        new_path = Path(len(self.u_hat), int(math.log2(len(self.u_hat))))
        new_path.pm = self.pm
        new_path.u_hat = self.u_hat.copy()
        new_path.L = self.L.copy()
        new_path.B = self.B.copy()
        return new_path


class SCLDecoder:
    """SCL 译码器（Lazy Copy）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.rev = bit_reversal_permutation(N)
        self.info_indices = np.where(~self.frozen_bits)[0]
        precompute_sc_indices(N)

    def decode(self, llr_ch):
        paths = [Path(self.N, self.n)]
        paths[0].L[self.rev, 0] = llr_ch

        for phi in range(self.N):
            l = _bit_reversed_index(phi, self.n)
            new_paths = []

            for path in paths:
                _update_llrs(path.L, path.B, l, self.n)
                llr_val = path.L[l, self.n]

                if self.frozen_bits[l]:
                    p = path.copy()
                    if llr_val < 0:
                        p.pm += abs(llr_val)
                    p.u_hat[l] = 0
                    p.B[l, self.n] = 0
                    _update_bits(p.B, l, self.n, self.N)
                    new_paths.append(p)
                else:
                    for bit in (0, 1):
                        p = path.copy()
                        if bit == 1 and llr_val >= 0:
                            p.pm += abs(llr_val)
                        if bit == 0 and llr_val < 0:
                            p.pm += abs(llr_val)
                        p.u_hat[l] = bit
                        p.B[l, self.n] = bit
                        _update_bits(p.B, l, self.n, self.N)
                        new_paths.append(p)

            new_paths.sort(key=lambda p: p.pm)
            paths = new_paths[: self.list_size]

        best_path = paths[0]
        if self.crc_length > 0:
            for path in paths:
                info_bits = path.u_hat[self.info_indices]
                if crc_check(info_bits, self.crc_length):
                    best_path = path
                    break

        return best_path.u_hat, best_path.pm
