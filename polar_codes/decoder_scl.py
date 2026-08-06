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
    _update_bits,
    _update_llrs,
    f_operation,
)


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, crc_length):
    reg = 0
    for bit in bits:
        reg = (reg << 1) | int(bit)
        if reg & (1 << crc_length):
            reg ^= poly
    return reg & ((1 << crc_length) - 1)


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    remainder = _crc_remainder(info_bits, poly, crc_length)
    crc_bits = np.array(
        [(remainder >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=int,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC 是否正确。"""
    bits = np.asarray(bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    return _crc_remainder(bits, poly, crc_length) == 0


class SCLDecoder:
    """SCL 译码器。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]

    def _path_metric_penalty(self, llr, bit):
        hard = 0 if llr >= 0 else 1
        return 0.0 if bit == hard else abs(llr)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N, n = self.N, self.n

        paths = [{
            'pm': 0.0,
            'L': np.zeros((N, n + 1), dtype=np.float64),
            'B': np.zeros((N, n + 1), dtype=np.int8),
            'u': np.zeros(N, dtype=int),
        }]
        paths[0]['L'][:, 0] = llr_ch

        for i in range(N):
            l = _bit_reversed(i, n)
            new_paths = []

            for path in paths:
                _update_llrs(path['L'], path['B'], l, n, N)
                llr = path['L'][l, n]

                if self.frozen_bits[l]:
                    pm = path['pm'] + self._path_metric_penalty(llr, 0)
                    p = {
                        'pm': pm,
                        'L': path['L'].copy(),
                        'B': path['B'].copy(),
                        'u': path['u'].copy(),
                    }
                    p['u'][l] = 0
                    p['B'][l, n] = 0
                    _update_bits(p['B'], l, n, N)
                    new_paths.append(p)
                else:
                    for bit in (0, 1):
                        pm = path['pm'] + self._path_metric_penalty(llr, bit)
                        p = {
                            'pm': pm,
                            'L': path['L'].copy(),
                            'B': path['B'].copy(),
                            'u': path['u'].copy(),
                        }
                        p['u'][l] = bit
                        p['B'][l, n] = bit
                        _update_bits(p['B'], l, n, N)
                        new_paths.append(p)

            new_paths.sort(key=lambda p: p['pm'])
            paths = new_paths[: self.list_size]

        if self.crc_length > 0:
            valid = []
            for p in paths:
                info = p['u'][self.info_indices]
                if crc_check(info, self.crc_length):
                    valid.append(p)
            best = min(valid, key=lambda p: p['pm']) if valid else paths[0]
        else:
            best = paths[0]

        return best['u'].copy(), best['pm']
