"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math

import numpy as np

from decoder_sc import (
    _active_bit_level,
    _active_llr_level,
    _br_scalar,
    _prepare_channel_llr,
    _update_bits,
    _update_llrs,
    f_operation,
    g_operation,
    sc_decode,
)


# ==================== CRC 工具 ====================

_CRC8_POLY = 0x07
_CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, crc_length):
    reg = 0
    mask = (1 << crc_length) - 1
    top = 1 << crc_length
    for bit in bits:
        reg <<= 1
        reg |= int(bit)
        if reg & top:
            reg ^= poly
    return reg & mask


def crc_encode(info_bits, crc_length=8):
    """
    计算 CRC 校验位并附加到信息比特后。
    """
    info_bits = np.asarray(info_bits, dtype=int)
    if crc_length == 8:
        poly = _CRC8_POLY
    elif crc_length == 16:
        poly = _CRC16_POLY
    else:
        raise ValueError("crc_length must be 8 or 16")

    remainder = _crc_remainder(info_bits, poly, crc_length)
    crc_bits = np.array(
        [(remainder >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=int,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。"""
    bits = np.asarray(bits, dtype=int)
    if crc_length == 8:
        poly = _CRC8_POLY
    elif crc_length == 16:
        poly = _CRC16_POLY
    else:
        raise ValueError("crc_length must be 8 or 16")

    expected = crc_encode(bits[:-crc_length], crc_length)
    return np.array_equal(bits, expected)


# ==================== SCL 译码器 ====================


class _Path:
    __slots__ = ("pm", "L", "B")

    def __init__(self, N, n, llr_ch):
        self.pm = 0.0
        self.L = np.zeros((N, n + 1), dtype=np.float64)
        self.B = np.zeros((N, n + 1), dtype=np.int32)
        self.L[:, 0] = llr_ch


class SCLDecoder:
    """
    SCL 译码器（Lazy Copy：路径分裂时复制 L/B 数组）。
    """

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.frozen_set = set(np.where(self.frozen_bits)[0])

    @staticmethod
    def _penalty(llr, bit):
        hard = 0 if llr >= 0 else 1
        return 0.0 if bit == hard else abs(llr)

    def _clone(self, path):
        new_path = _Path(self.N, self.n, path.L[:, 0])
        new_path.pm = path.pm
        new_path.L = path.L.copy()
        new_path.B = path.B.copy()
        return new_path

    def decode(self, llr_ch):
        """
        主译码函数。
        返回 (u_hat, pm)
        """
        if self.list_size == 1:
            return sc_decode(llr_ch, self.frozen_bits), 0.0

        llr_ch = _prepare_channel_llr(llr_ch)
        paths = [_Path(self.N, self.n, llr_ch)]

        for phi in range(self.N):
            l = _br_scalar(phi, self.n)
            candidates = []

            for path in paths:
                _update_llrs(path.L, path.B, l, self.n)
                cur_llr = path.L[l, self.n]

                if l in self.frozen_set:
                    new_path = self._clone(path)
                    new_path.pm += self._penalty(cur_llr, 0)
                    new_path.B[l, self.n] = 0
                    _update_bits(new_path.B, l, self.n, self.N)
                    candidates.append(new_path)
                else:
                    for bit in (0, 1):
                        new_path = self._clone(path)
                        new_path.pm += self._penalty(cur_llr, bit)
                        new_path.B[l, self.n] = bit
                        _update_bits(new_path.B, l, self.n, self.N)
                        candidates.append(new_path)

            candidates.sort(key=lambda p: p.pm)
            paths = candidates[: self.list_size]

        best = paths[0]
        if self.crc_length > 0:
            info_positions = np.where(~self.frozen_bits)[0]
            valid = []
            for path in paths:
                info_bits = path.B[:, self.n][info_positions]
                if crc_check(info_bits, self.crc_length):
                    valid.append(path)
            if valid:
                best = min(valid, key=lambda p: p.pm)

        return best.B[:, self.n].astype(int), best.pm
