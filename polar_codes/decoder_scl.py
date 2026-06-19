"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math

import numpy as np

from decoder_sc import (
    _active_bit_level,
    _active_llr_level,
    _update_bits,
    _update_llrs,
    sc_decode,
)
from encoder import bit_reversed

# ==================== CRC 工具 ====================

_CRC8_POLY = 0x07
_CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, crc_length):
    reg = 0
    for bit in bits:
        reg ^= int(bit) << (crc_length - 1)
        for _ in range(8 if crc_length <= 8 else 1):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)
    return reg


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=int)
    poly = _CRC8_POLY if crc_length == 8 else _CRC16_POLY
    remainder = _crc_remainder(info_bits, poly, crc_length)
    crc_bits = np.array(
        [(remainder >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=int,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。"""
    bits = np.asarray(bits, dtype=int)
    poly = _CRC8_POLY if crc_length == 8 else _CRC16_POLY
    payload = bits[:-crc_length]
    expected = crc_encode(payload, crc_length)[-crc_length:]
    return np.array_equal(bits[-crc_length:], expected)


# ==================== SCL 译码器 ====================


class Path:
    """单条 SCL 路径。"""

    __slots__ = ("L", "B", "pm", "u_hat")

    def __init__(self, N, n):
        self.L = np.zeros((N, n + 1), dtype=np.float64)
        self.B = np.zeros((N, n + 1), dtype=int)
        self.pm = 0.0
        self.u_hat = np.zeros(N, dtype=int)

    def copy(self):
        new_path = Path(self.L.shape[0], self.L.shape[1] - 1)
        new_path.L = self.L.copy()
        new_path.B = self.B.copy()
        new_path.pm = self.pm
        new_path.u_hat = self.u_hat.copy()
        return new_path


class SCLDecoder:
    """SCL 译码器。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0, info_indices=None):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.list_size = list_size
        self.crc_length = crc_length
        self.frozen_set = set(np.where(self.frozen_bits == 1)[0])
        if info_indices is not None:
            self.sorted_info = np.sort(np.asarray(info_indices, dtype=int))
        else:
            self.sorted_info = np.sort(np.where(self.frozen_bits == 0)[0])

    def _crc_valid(self, u_hat):
        payload = u_hat[self.sorted_info[: self.sorted_info.size - self.crc_length]]
        crc_part = u_hat[
            self.sorted_info[self.sorted_info.size - self.crc_length :]
        ]
        return crc_check(np.concatenate([payload, crc_part]), self.crc_length)

    @staticmethod
    def _branch_penalty(llr, bit):
        hard = 0 if llr >= 0 else 1
        return 0.0 if bit == hard else abs(llr)

    def decode(self, llr_ch):
        """主译码函数。"""
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        if self.list_size == 1:
            u_hat = sc_decode(llr_ch, self.frozen_bits)
            return u_hat, 0.0

        path = Path(self.N, self.n)
        path.L[:, 0] = llr_ch
        paths = [path]

        for phi in range(self.N):
            l = bit_reversed(phi, self.n)
            active = []

            for p in paths:
                _update_llrs(p.L, p.B, l, self.n, self.N)
                llr = p.L[l, self.n]

                if l in self.frozen_set:
                    p.pm += self._branch_penalty(llr, 0)
                    p.B[l, self.n] = 0
                    p.u_hat[phi] = 0
                    _update_bits(p.B, l, self.n, self.N)
                    active.append(p)
                else:
                    for bit in (0, 1):
                        new_p = p.copy()
                        new_p.pm += self._branch_penalty(llr, bit)
                        new_p.B[l, self.n] = bit
                        new_p.u_hat[phi] = bit
                        _update_bits(new_p.B, l, self.n, self.N)
                        active.append(new_p)

            active.sort(key=lambda item: item.pm)
            paths = active[: self.list_size]

        if self.crc_length > 0:
            valid = [p for p in paths if self._crc_valid(p.u_hat)]
            chosen = min(valid, key=lambda p: p.pm) if valid else min(
                paths, key=lambda p: p.pm
            )
        else:
            chosen = min(paths, key=lambda p: p.pm)

        return chosen.u_hat.copy(), chosen.pm
