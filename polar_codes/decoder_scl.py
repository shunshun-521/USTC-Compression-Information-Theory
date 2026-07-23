"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import copy
import math

import numpy as np

from decoder_sc import SCState, bitreversed, lowerconv, f_operation


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_divide(info_bits, poly, crc_length):
    reg = [0] * crc_length
    poly_bits = [(poly >> i) & 1 for i in range(crc_length - 1, -1, -1)]
    for bit in info_bits:
        msb = reg[0]
        reg = reg[1:] + [bit ^ msb]
        if msb:
            reg = [reg[i] ^ poly_bits[i + 1] for i in range(crc_length)]
    return np.array(reg, dtype=int)


def crc_encode(info_bits, crc_length=8):
    info_bits = np.asarray(info_bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    remainder = _crc_divide(info_bits, poly, crc_length)
    return np.concatenate([info_bits, remainder])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=int)
    if len(bits) < crc_length:
        return False
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    remainder = _crc_divide(bits, poly, crc_length)
    return np.all(remainder == 0)


class Path(SCState):
    __slots__ = ("pm", "u_hat")

    def __init__(self, N):
        super().__init__(N)
        self.pm = 0.0
        self.u_hat = np.zeros(N, dtype=int)


class SCLDecoder:
    """SCL 译码器。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(self.frozen_bits == 0)[0]

    def _metric_penalty(self, llr, u_bit):
        hard = 0 if llr >= 0 else 1
        return 0.0 if u_bit == hard else abs(llr)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [Path(self.N)]
        paths[0].set_channel(llr_ch)

        for j in range(self.N):
            i = bitreversed(j, self.n)
            candidates = []

            for path in paths:
                path.update_llrs(i)
                llr = path.llrs[0]

                if self.frozen_bits[j]:
                    new_path = copy.deepcopy(path)
                    new_path.pm += self._metric_penalty(llr, 0)
                    new_path.u_hat[j] = 0
                    new_path.decoded_bit = 0
                    new_path.update_bits(i)
                    candidates.append(new_path)
                else:
                    for u_bit in (0, 1):
                        new_path = copy.deepcopy(path)
                        new_path.pm += self._metric_penalty(llr, u_bit)
                        new_path.u_hat[j] = u_bit
                        new_path.decoded_bit = u_bit
                        new_path.update_bits(i)
                        candidates.append(new_path)

            candidates.sort(key=lambda p: p.pm)
            paths = candidates[: self.list_size]

        crc_pass = []
        for path in paths:
            if self.crc_length > 0:
                info_bits = path.u_hat[self.info_indices]
                if crc_check(info_bits, self.crc_length):
                    crc_pass.append(path)
            else:
                crc_pass.append(path)

        best = min(crc_pass or paths, key=lambda p: p.pm)
        return best.u_hat.copy(), best.pm
