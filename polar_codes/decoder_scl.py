"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np
from decoder_sc import (
    _prepare_llr,
    _update_bits,
    _update_llrs,
    bit_reversed_index,
)


def _crc_polynomial(crc_length):
    if crc_length == 8:
        return 0x07
    if crc_length == 16:
        return 0x8005
    raise ValueError("crc_length must be 8 or 16")


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后"""
    poly = _crc_polynomial(crc_length)
    info_bits = np.asarray(info_bits, dtype=np.int8)
    reg = 0
    topbit = 1 << (crc_length - 1)
    mask = (1 << crc_length) - 1

    for bit in info_bits:
        reg ^= int(bit) << (crc_length - 1)
        if reg & topbit:
            reg = ((reg << 1) ^ poly) & mask
        else:
            reg = (reg << 1) & mask

    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=np.int8,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC 是否正确"""
    if crc_length == 0:
        return True
    poly = _crc_polynomial(crc_length)
    bits = np.asarray(bits, dtype=np.int8)
    reg = 0
    topbit = 1 << (crc_length - 1)
    mask = (1 << crc_length) - 1

    for bit in bits:
        reg ^= int(bit) << (crc_length - 1)
        if reg & topbit:
            reg = ((reg << 1) ^ poly) & mask
        else:
            reg = (reg << 1) & mask
    return reg == 0


class SCLDecoder:
    """SCL 译码器"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0, info_indices=None):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        if info_indices is None:
            self.info_indices = np.where(~self.frozen_bits)[0]
        else:
            self.info_indices = np.asarray(info_indices, dtype=int)

    @staticmethod
    def _pm_update(pm, llr, u):
        hard = 0 if llr >= 0 else 1
        if u != hard:
            pm += abs(llr)
        return pm

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n = self.n
        N = self.N
        Lsz = self.list_size

        llr0 = _prepare_llr(llr_ch, N)
        paths = [{
            'pm': 0.0,
            'L': np.zeros((N, n + 1), dtype=np.float64),
            'B': np.zeros((N, n + 1), dtype=int),
        }]
        paths[0]['L'][:, 0] = llr0

        for i in range(N):
            l = bit_reversed_index(i, n)
            new_paths = []

            for path in paths:
                _update_llrs(path['L'], path['B'], l, n)
                llr = path['L'][l, n]

                if self.frozen_bits[l]:
                    child = {
                        'pm': self._pm_update(path['pm'], llr, 0),
                        'L': path['L'].copy(),
                        'B': path['B'].copy(),
                    }
                    child['B'][l, n] = 0
                    _update_bits(child['B'], l, n, N)
                    new_paths.append(child)
                else:
                    for u in (0, 1):
                        child = {
                            'pm': self._pm_update(path['pm'], llr, u),
                            'L': path['L'].copy(),
                            'B': path['B'].copy(),
                        }
                        child['B'][l, n] = u
                        _update_bits(child['B'], l, n, N)
                        new_paths.append(child)

            new_paths.sort(key=lambda p: p['pm'])
            paths = new_paths[:Lsz]

        if self.crc_length > 0:
            valid = []
            for p in paths:
                info_bits = p['B'][:, n][self.info_indices]
                if crc_check(info_bits, self.crc_length):
                    valid.append(p)
            best = min(valid if valid else paths, key=lambda p: p['pm'])
        else:
            best = min(paths, key=lambda p: p['pm'])

        return best['B'][:, n], best['pm']
