"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math

import numpy as np

from decoder_sc import (
    bit_reversed,
    sc_decode,
    _update_llrs,
    _update_bits,
)


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_remainder(bits, crc_length=8):
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    register = 0
    mask = (1 << crc_length) - 1
    for bit in bits:
        msb = (register >> (crc_length - 1)) & 1
        register = (register << 1) & mask
        register |= int(bit)
        if msb:
            register ^= poly
    return register


def crc_encode(info_bits, crc_length=8):
    info_bits = np.asarray(info_bits, dtype=np.int8)
    remainder = _crc_remainder(
        np.concatenate([info_bits, np.zeros(crc_length, dtype=np.int8)]),
        crc_length,
    )
    crc_bits = np.array(
        [(remainder >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=np.int8,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    if crc_length == 0:
        return True
    return _crc_remainder(bits, crc_length) == 0


class SCLDecoder:
    """SCL 译码器"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.info_indices = np.where(~self.frozen_bits)[0]
        self.decode_order = [bit_reversed(i, self.n) for i in range(N)]

    def _path_metric_update(self, pm, llr, u):
        hard = 0 if llr >= 0 else 1
        if u != hard:
            pm += abs(llr)
        return pm

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)

        if self.list_size == 1 and self.crc_length == 0:
            return sc_decode(llr_ch, self.frozen_bits), 0.0

        paths = [{
            'L': np.zeros((self.N, self.n + 1), dtype=np.float64),
            'B': np.zeros((self.N, self.n + 1), dtype=np.int8),
            'pm': 0.0,
        }]
        paths[0]['L'][:, 0] = llr_ch.copy()

        for l in self.decode_order:
            candidates = []
            is_frozen = l in self.frozen_set

            for path in paths:
                _update_llrs(path['L'], path['B'], l, self.n, self.N)
                cur_llr = path['L'][l, self.n]

                if is_frozen:
                    new_path = {
                        'L': path['L'].copy(),
                        'B': path['B'].copy(),
                        'pm': self._path_metric_update(path['pm'], cur_llr, 0),
                    }
                    new_path['B'][l, self.n] = 0
                    _update_bits(new_path['B'], l, self.n, self.N)
                    candidates.append(new_path)
                else:
                    for u in (0, 1):
                        new_path = {
                            'L': path['L'].copy(),
                            'B': path['B'].copy(),
                            'pm': self._path_metric_update(path['pm'], cur_llr, u),
                        }
                        new_path['B'][l, self.n] = u
                        _update_bits(new_path['B'], l, self.n, self.N)
                        candidates.append(new_path)

            candidates.sort(key=lambda p: p['pm'])
            paths = candidates[: self.list_size]

        crc_pass = []
        for p in paths:
            u_hat = p['B'][:, self.n].astype(np.int8)
            if self.crc_length > 0:
                info_bits = u_hat[self.info_indices]
                if crc_check(info_bits, self.crc_length):
                    crc_pass.append((u_hat, p['pm']))
            else:
                crc_pass.append((u_hat, p['pm']))

        if crc_pass:
            best = min(crc_pass, key=lambda x: x[1])
        else:
            best = (paths[0]['B'][:, self.n].astype(np.int8), paths[0]['pm'])

        return best[0], best[1]
