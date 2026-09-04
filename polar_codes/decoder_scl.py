"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np
from decoder_sc import (
    _SCDCore,
    _bit_reversed,
    _map_channel_llr,
    g_operation,
)


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc8_remainder(bits):
    reg = 0
    for bit in bits:
        reg ^= int(bit) << 7
        for _ in range(8):
            if reg & 0x80:
                reg = ((reg << 1) ^ (CRC8_POLY << 1)) & 0xFF
            else:
                reg = (reg << 1) & 0xFF
    return reg


def _crc16_remainder(bits):
    reg = 0
    for bit in bits:
        reg ^= int(bit) << 15
        if reg & 0x8000:
            reg = ((reg << 1) ^ CRC16_POLY) & 0xFFFF
        else:
            reg = (reg << 1) & 0xFFFF
    return reg


def crc_encode(info_bits, crc_length=8):
    """
    计算 CRC 校验位并附加到信息比特后。
    """
    info_bits = np.asarray(info_bits, dtype=np.int8)
    if crc_length == 8:
        padded = np.concatenate([info_bits, np.zeros(8, dtype=np.int8)])
        remainder = _crc8_remainder(padded)
        crc_bits = np.array([(remainder >> (7 - i)) & 1 for i in range(8)], dtype=np.int8)
    elif crc_length == 16:
        padded = np.concatenate([info_bits, np.zeros(16, dtype=np.int8)])
        remainder = _crc16_remainder(padded)
        crc_bits = np.array([(remainder >> (15 - i)) & 1 for i in range(16)], dtype=np.int8)
    else:
        raise ValueError("crc_length must be 8 or 16")
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """
    检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。
    """
    bits = np.asarray(bits, dtype=np.int8)
    expected = crc_encode(bits[:-crc_length], crc_length)
    return np.array_equal(bits, expected)


class _Path:
    __slots__ = ("core", "pm", "u_hat")

    def __init__(self, N, frozen_bits):
        self.core = _SCDCore(N, frozen_bits, use_minsum=True)
        self.pm = 0.0
        self.u_hat = np.zeros(N, dtype=np.int8)


class SCLDecoder:
    """
    SCL 译码器（含 Lazy Copy 优化）。
    """

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.frozen_set = set(np.where(self.frozen_bits == 1)[0])
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_mask = self.frozen_bits == 0

    def _copy_path(self, path):
        new_path = _Path(self.N, self.frozen_bits)
        new_path.core.L = path.core.L.copy()
        new_path.core.B = path.core.B.copy()
        new_path.pm = path.pm
        new_path.u_hat = path.u_hat.copy()
        return new_path

    def _pm_penalty(self, llr, bit):
        hard = 0 if llr >= 0 else 1
        return 0.0 if bit == hard else abs(llr)

    def decode(self, llr_ch):
        """
        主译码函数。
        返回：u_hat, pm
        """
        llr = _map_channel_llr(llr_ch)
        paths = [_Path(self.N, self.frozen_bits)]
        paths[0].core.L[:, 0] = llr

        for l in [_bit_reversed(i, self.n) for i in range(self.N)]:
            new_paths = []
            for path in paths:
                path.core.update_llrs(l)
                llr_val = path.core.L[l, self.n]

                if l in self.frozen_set:
                    p = self._copy_path(path)
                    p.pm += self._pm_penalty(llr_val, 0)
                    p.u_hat[l] = 0
                    p.core.B[l, self.n] = 0
                    p.core.update_bits(l)
                    new_paths.append(p)
                else:
                    for bit in (0, 1):
                        p = self._copy_path(path)
                        p.pm += self._pm_penalty(llr_val, bit)
                        p.u_hat[l] = bit
                        p.core.B[l, self.n] = bit
                        p.core.update_bits(l)
                        new_paths.append(p)

            new_paths.sort(key=lambda p: p.pm)
            paths = new_paths[: self.list_size]

        if self.crc_length > 0:
            valid = []
            for p in paths:
                info_bits = p.u_hat[self.info_mask]
                if crc_check(info_bits, self.crc_length):
                    valid.append(p)
            if valid:
                paths = valid

        best = min(paths, key=lambda p: p.pm)
        return best.u_hat.copy(), best.pm
