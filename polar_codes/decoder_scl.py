"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math

import numpy as np

from decoder_sc import _bit_reversed, _update_bits, _update_llr


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, crc_length):
    reg = 0
    mask = (1 << crc_length) - 1
    for b in bits:
        msb = (reg >> (crc_length - 1)) & 1
        reg = ((reg << 1) | int(b)) & mask
        if msb:
            reg ^= poly
    return reg


def crc_encode(info_bits, crc_length=8):
    info_bits = np.asarray(info_bits, dtype=np.int8)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    rem = _crc_remainder(list(info_bits) + [0] * crc_length, poly, crc_length)
    crc_bits = np.array(
        [(rem >> i) & 1 for i in range(crc_length - 1, -1, -1)], dtype=np.int8
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=np.int8)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    return _crc_remainder(bits, poly, crc_length) == 0


class _Path:
    __slots__ = ("pm", "L", "B", "u_hat")

    def __init__(self, N, n, llr_ch):
        self.pm = 0.0
        self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.B = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.L[:, n] = llr_ch
        self.u_hat = np.zeros(N, dtype=np.int8)


class SCLDecoder:
    """SCL 译码器（与 SC 相同的 LLR/比特更新规则）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.L = list_size
        self.crc_length = crc_length
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.info_indices = np.where(~self.frozen_bits)[0]

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [_Path(self.N, self.n, llr_ch)]

        for phi in range(self.N):
            l = _bit_reversed(phi, self.n)
            new_paths = []

            for path in paths:
                _update_llr(path.L, path.B, l, self.n)
                llr0 = path.L[l, 0]

                if l in self.frozen_set:
                    pen = abs(llr0) if llr0 < 0 else 0.0
                    path.pm += pen
                    path.B[l, 0] = 0
                    path.u_hat[l] = 0
                    _update_bits(path.B, l, self.n)
                    new_paths.append(path)
                else:
                    for bit in (0, 1):
                        child = _Path(self.N, self.n, llr_ch)
                        child.pm = path.pm
                        child.L = path.L.copy()
                        child.B = path.B.copy()
                        child.u_hat = path.u_hat.copy()
                        pen = (
                            0.0
                            if (bit == 0 and llr0 >= 0) or (bit == 1 and llr0 < 0)
                            else abs(llr0)
                        )
                        child.pm += pen
                        child.B[l, 0] = bit
                        child.u_hat[l] = bit
                        _update_bits(child.B, l, self.n)
                        new_paths.append(child)

            new_paths.sort(key=lambda p: p.pm)
            paths = new_paths[: self.L]

        paths.sort(key=lambda p: p.pm)
        if self.crc_length > 0:
            for path in paths:
                info_bits = path.u_hat[self.info_indices]
                if crc_check(info_bits, self.crc_length):
                    return path.u_hat.copy(), path.pm
        best = paths[0]
        return best.u_hat.copy(), best.pm
