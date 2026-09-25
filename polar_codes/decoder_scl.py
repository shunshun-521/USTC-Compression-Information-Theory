"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math

import numpy as np

from decoder_sc import (
    _active_bit_level,
    _active_llr_level,
    _bit_reversed,
    _lower_llr,
    _update_bits,
    _update_llrs,
)


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_process(bits, poly, crc_length):
    """LFSR 按位 CRC，返回最终寄存器值。"""
    reg = 0
    mask = (1 << crc_length) - 1
    for bit in bits:
        msb = (reg >> (crc_length - 1)) & 1
        reg = (reg << 1) & mask
        if bit:
            reg ^= 1
        if msb:
            reg ^= poly
    return reg


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    padded = np.concatenate([info_bits, np.zeros(crc_length, dtype=int)])
    remainder = _crc_process(padded, poly, crc_length)
    crc_bits = np.array(
        [(remainder >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=int,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。"""
    bits = np.asarray(bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    return _crc_process(bits, poly, crc_length) == 0


class _Path:
    __slots__ = ("pm", "u_hat", "L", "B")

    def __init__(self, N, n, llr_perm=None, other=None):
        self.pm = 0.0
        self.u_hat = np.zeros(N, dtype=np.int8)
        self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.B = np.full((N, n + 1), np.nan)
        if other is not None:
            self.pm = other.pm
            self.u_hat[:] = other.u_hat
            self.L[:] = other.L
            self.B[:] = other.B
        elif llr_perm is not None:
            self.L[:, 0] = llr_perm


class SCLDecoder:
    """SCL 译码器（基于 Permuted SCD）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.decode_order = [_bit_reversed(i, self.n) for i in range(N)]

    def _prepare_llr(self, llr_ch):
        rev = np.array([_bit_reversed(i, self.n) for i in range(self.N)])
        return llr_ch[rev]

    @staticmethod
    def _pm_update(pm, llr_val, u_bit):
        hard = 0 if llr_val >= 0 else 1
        if u_bit != hard:
            pm += abs(llr_val)
        return pm

    def decode(self, llr_ch):
        llr_perm = self._prepare_llr(llr_ch)
        paths = [_Path(self.N, self.n, llr_perm=llr_perm)]

        for l in self.decode_order:
            new_paths = []
            for path in paths:
                _update_llrs(path.L, path.B, l, self.n, self.N)
                llr_val = path.L[l, self.n]

                if l in self.frozen_set:
                    p = _Path(self.N, self.n, other=path)
                    p.pm = self._pm_update(p.pm, llr_val, 0)
                    p.B[l, self.n] = 0
                    p.u_hat[l] = 0
                    _update_bits(p.B, l, self.n, self.N)
                    new_paths.append(p)
                else:
                    for u_bit in (0, 1):
                        p = _Path(self.N, self.n, other=path)
                        p.pm = self._pm_update(p.pm, llr_val, u_bit)
                        p.B[l, self.n] = u_bit
                        p.u_hat[l] = u_bit
                        _update_bits(p.B, l, self.n, self.N)
                        new_paths.append(p)

            new_paths.sort(key=lambda p: p.pm)
            paths = new_paths[: self.list_size]

        if self.crc_length > 0:
            valid = [
                p for p in paths
                if crc_check(p.u_hat[self.info_indices], self.crc_length)
            ]
            best = min(valid, key=lambda p: p.pm) if valid else paths[0]
        else:
            best = paths[0]

        return best.u_hat.astype(int), best.pm
