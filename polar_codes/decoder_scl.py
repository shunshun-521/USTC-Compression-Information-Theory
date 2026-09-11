"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math

import numpy as np

from decoder_sc import _channel_llr_to_decode, _update_bits, _update_llrs, _bit_reversed


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_division(info_bits, poly, crc_length):
    """GF(2) 长除法计算 CRC 校验位"""
    bits = np.concatenate([info_bits.astype(np.int8), np.zeros(crc_length, dtype=np.int8)])
    for i in range(len(info_bits)):
        if bits[i] == 1:
            for j in range(crc_length + 1):
                if (poly >> (crc_length - j)) & 1:
                    bits[i + j] ^= 1
    return bits[-crc_length:]


def crc_encode(info_bits, crc_length=8):
    """
    计算 CRC 校验位并附加到信息比特后。
    """
    info_bits = np.asarray(info_bits, dtype=np.int8)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    crc_bits = _crc_division(info_bits, poly, crc_length)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """
    检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。
    """
    if crc_length == 0:
        return True
    bits = np.asarray(bits, dtype=np.int8)
    info = bits[:-crc_length]
    expected = crc_encode(info, crc_length)
    return np.array_equal(bits, expected)


class _Path:
    """单条 SCL 路径"""

    __slots__ = ("L", "B", "pm", "u_hat")

    def __init__(self, N, n):
        self.L = np.zeros((N, n + 1), dtype=np.float64)
        self.B = np.zeros((N, n + 1), dtype=np.int8)
        self.pm = 0.0
        self.u_hat = np.zeros(N, dtype=int)


class SCLDecoder:
    """
    SCL 译码器。
    """

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length

    def _pm_penalty(self, llr, u_bit):
        u_from_llr = 0 if llr >= 0 else 1
        return 0.0 if u_bit == u_from_llr else abs(llr)

    def decode(self, llr_ch):
        """
        主译码函数。
        返回：(u_hat, pm)
        """
        llr_ch = _channel_llr_to_decode(np.asarray(llr_ch, dtype=np.float64))
        N = self.N
        n = self.n
        decode_order = [_bit_reversed(phi, n) for phi in range(N)]

        paths = [_Path(N, n)]
        paths[0].L[:, 0] = llr_ch

        for l in decode_order:
            new_paths = []

            for path in paths:
                _update_llrs(path.L, path.B, l, n)
                llr = path.L[l, n]

                if self.frozen_bits[l]:
                    child = _Path(N, n)
                    child.L = path.L.copy()
                    child.B = path.B.copy()
                    child.u_hat = path.u_hat.copy()
                    child.pm = path.pm + self._pm_penalty(llr, 0)
                    child.u_hat[l] = 0
                    child.B[l, n] = 0
                    _update_bits(child.B, l, n, N)
                    new_paths.append(child)
                else:
                    for u_bit in (0, 1):
                        child = _Path(N, n)
                        child.L = path.L.copy()
                        child.B = path.B.copy()
                        child.u_hat = path.u_hat.copy()
                        child.pm = path.pm + self._pm_penalty(llr, u_bit)
                        child.u_hat[l] = u_bit
                        child.B[l, n] = u_bit
                        _update_bits(child.B, l, n, N)
                        new_paths.append(child)

            new_paths.sort(key=lambda p: p.pm)
            paths = new_paths[: self.list_size]

        if not paths:
            return np.zeros(N, dtype=int), 0.0

        if self.crc_length > 0:
            valid = [p for p in paths if crc_check(p.u_hat, self.crc_length)]
            best = min(valid if valid else paths, key=lambda p: p.pm)
        else:
            best = min(paths, key=lambda p: p.pm)

        return best.u_hat.copy(), best.pm
