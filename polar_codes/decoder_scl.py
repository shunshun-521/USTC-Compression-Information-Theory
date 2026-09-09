"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import copy
import numpy as np
from decoder_sc import (
    _SCDCore,
    _bit_reversed,
    _active_llr_level,
    _active_bit_level,
    sc_decode,
)


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, crc_length):
    reg = 0
    mask = (1 << crc_length) - 1
    top = 1 << (crc_length - 1)
    for bit in bits:
        reg ^= int(bit) << (crc_length - 1)
        for _ in range(crc_length):
            if reg & top:
                reg = ((reg << 1) ^ poly) & mask
            else:
                reg = (reg << 1) & mask
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
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC"""
    bits = np.asarray(bits, dtype=int)
    if crc_length <= 0:
        return True
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    payload = bits[:-crc_length]
    expected = crc_encode(payload, crc_length=crc_length)[-crc_length:]
    return np.array_equal(bits[-crc_length:], expected)


class SCLDecoder:
    """SCL 译码器"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen = set(np.where(self.frozen_bits)[0].tolist())
        self.list_size = list_size
        self.crc_length = crc_length

    def _path_metric_penalty(self, llr, bit):
        hard = 0 if llr >= 0 else 1
        return 0.0 if hard == bit else abs(llr)

    def decode(self, llr_ch):
        if self.list_size == 1 and self.crc_length == 0:
            u_hat = sc_decode(llr_ch, self.frozen_bits)
            return u_hat, 0.0

        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [{
            'pm': 0.0,
            'core': _SCDCore(self.N, self.frozen_bits),
        }]
        paths[0]['core'].L[:, 0] = llr_ch

        for i in range(self.N):
            l = _bit_reversed(i, self.n)
            candidates = []

            for path in paths:
                core = path['core']
                core._update_llrs(l)
                llr = core.L[l, self.n]

                if l in self.frozen:
                    penalty = self._path_metric_penalty(llr, 0)
                    core.B[l, self.n] = 0
                    core._update_bits(l)
                    candidates.append({
                        'pm': path['pm'] + penalty,
                        'core': core,
                    })
                else:
                    for bit in (0, 1):
                        new_core = copy.deepcopy(core)
                        new_core.B[l, self.n] = bit
                        new_core._update_bits(l)
                        candidates.append({
                            'pm': path['pm'] + self._path_metric_penalty(llr, bit),
                            'core': new_core,
                        })

            candidates.sort(key=lambda p: p['pm'])
            paths = candidates[: self.list_size]

        if self.crc_length > 0:
            valid = []
            for path in paths:
                u_hat = path['core'].B[:, self.n].astype(int)
                if crc_check(u_hat, self.crc_length):
                    valid.append(path)
            best = min(valid if valid else paths, key=lambda p: p['pm'])
        else:
            best = min(paths, key=lambda p: p['pm'])

        u_hat = best['core'].B[:, self.n].astype(int)
        return u_hat, best['pm']
