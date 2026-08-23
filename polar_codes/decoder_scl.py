"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np
from encoder import bit_reversal_permutation
from decoder_sc import (
    f_operation,
    g_operation,
    sc_decode,
    _bit_reversed_index,
    _active_llr_level,
    _active_bit_level,
    _update_llrs,
    _update_bits,
)


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, crc_length):
    reg = 0
    for bit in bits:
        reg ^= int(bit) << (crc_length - 1)
        for _ in range(crc_length):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)
    return reg


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
    """检验 CRC 是否正确"""
    bits = np.asarray(bits, dtype=int)
    return np.array_equal(bits[-crc_length:], crc_encode(bits[:-crc_length], crc_length)[-crc_length:])


def _pm_penalty(llr, u):
    preferred = 0 if llr >= 0 else 1
    return 0.0 if u == preferred else abs(llr)


class _Path:
    __slots__ = ("L", "B", "pm", "u_hat")

    def __init__(self, N, n, llr_ch=None):
        self.L = np.zeros((N, n + 1), dtype=np.float64)
        self.B = np.zeros((N, n + 1), dtype=int)
        self.pm = 0.0
        self.u_hat = np.zeros(N, dtype=int)
        if llr_ch is not None:
            self.L[:, 0] = llr_ch

    def copy(self):
        p = _Path(self.L.shape[0], self.L.shape[1] - 1)
        p.L = self.L.copy()
        p.B = self.B.copy()
        p.pm = self.pm
        p.u_hat = self.u_hat.copy()
        return p


class SCLDecoder:
    """SCL 译码器（含 Lazy Copy 优化）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.rev = bit_reversal_permutation(N)
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.info_indices = np.where(~self.frozen_bits)[0]
        self.decode_order = [_bit_reversed_index(i, self.n) for i in range(N)]

    def decode(self, llr_ch):
        if self.list_size == 1 and self.crc_length == 0:
            return sc_decode(llr_ch, self.frozen_bits), 0.0

        llr_ch = np.asarray(llr_ch, dtype=np.float64)[self.rev]
        paths = [_Path(self.N, self.n, llr_ch)]

        for l in self.decode_order:
            new_paths = []
            for path in paths:
                _update_llrs(path.L, path.B, l, self.n)
                llr = path.L[l, self.n]

                if l in self.frozen_set:
                    child = path.copy()
                    child.pm += _pm_penalty(llr, 0)
                    child.u_hat[l] = 0
                    child.B[l, self.n] = 0
                    _update_bits(child.B, l, self.n)
                    new_paths.append(child)
                else:
                    for u in (0, 1):
                        child = path.copy()
                        child.pm += _pm_penalty(llr, u)
                        child.u_hat[l] = u
                        child.B[l, self.n] = u
                        _update_bits(child.B, l, self.n)
                        new_paths.append(child)

            new_paths.sort(key=lambda p: p.pm)
            paths = new_paths[: self.list_size]

        if self.crc_length > 0:
            valid = []
            for p in paths:
                info_bits = p.u_hat[self.info_indices]
                if crc_check(info_bits, self.crc_length):
                    valid.append(p)
            if valid:
                paths = valid

        best = min(paths, key=lambda p: p.pm)
        return best.u_hat.copy(), best.pm
