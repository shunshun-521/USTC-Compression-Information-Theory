"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np
from encoder import bit_reversed
from decoder_sc import (
    f_operation, g_operation, _active_llr_level, _active_bit_level,
    _update_llrs, _update_bits,
)


CRC_POLYNOMIALS = {8: 0x07, 16: 0x8005}


def _crc_remainder(bits, crc_length):
    poly = CRC_POLYNOMIALS[crc_length]
    reg = 0
    for bit in bits:
        reg <<= 1
        reg |= int(bit)
        if reg & (1 << crc_length):
            reg ^= poly
    return reg


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=int)
    remainder = _crc_remainder(info_bits, crc_length)
    crc_bits = np.array(
        [(remainder >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=int,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits 的 CRC 是否正确"""
    bits = np.asarray(bits, dtype=int)
    return _crc_remainder(bits, crc_length) == 0


def _path_metric_update(pm, llr, u):
    if u == (0 if llr >= 0 else 1):
        return pm
    return pm + abs(llr)


class SCLDecoder:
    """SCL 译码器"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.decode_order = [bit_reversed(i, self.n) for i in range(N)]
        self.info_indices = np.where(~self.frozen_bits)[0]

    def _new_path(self, llr_ch):
        L = np.zeros((self.N, self.n + 1), dtype=np.float64)
        B = np.zeros((self.N, self.n + 1), dtype=int)
        L[:, 0] = llr_ch
        return {'L': L, 'B': B, 'pm': 0.0, 'u_hat': np.zeros(self.N, dtype=int)}

    def _copy_path(self, path):
        return {
            'L': path['L'].copy(),
            'B': path['B'].copy(),
            'pm': path['pm'],
            'u_hat': path['u_hat'].copy(),
        }

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [self._new_path(llr_ch)]

        for l in self.decode_order:
            candidates = []
            for path in paths:
                _update_llrs(path['L'], path['B'], l, self.n, self.N)
                llr = path['L'][l, self.n]

                if self.frozen_bits[l]:
                    new_path = self._copy_path(path)
                    new_path['pm'] = _path_metric_update(path['pm'], llr, 0)
                    new_path['B'][l, self.n] = 0
                    new_path['u_hat'][l] = 0
                    _update_bits(new_path['B'], l, self.n, self.N)
                    candidates.append(new_path)
                else:
                    for u in (0, 1):
                        new_path = self._copy_path(path)
                        new_path['pm'] = _path_metric_update(path['pm'], llr, u)
                        new_path['B'][l, self.n] = u
                        new_path['u_hat'][l] = u
                        _update_bits(new_path['B'], l, self.n, self.N)
                        candidates.append(new_path)

            candidates.sort(key=lambda p: p['pm'])
            paths = candidates[: self.list_size]

        if self.crc_length > 0:
            valid = [
                p for p in paths
                if crc_check(p['u_hat'][self.info_indices], self.crc_length)
            ]
            if valid:
                paths = valid

        best = min(paths, key=lambda p: p['pm'])
        return best['u_hat'], best['pm']
