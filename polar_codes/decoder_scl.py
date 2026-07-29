"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from decoder_sc import (
    _update_llrs, _update_bits, _active_llr_level, _active_bit_level, _bit_reversed
)

CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_process(bits, crc_length):
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    poly_full = poly | (1 << crc_length)
    reg = 0
    for b in bits:
        reg = (reg << 1) | int(b)
        if reg & (1 << crc_length):
            reg ^= poly_full
    return reg


def crc_encode(info_bits, crc_length=8):
    info_bits = np.asarray(info_bits, dtype=int)
    msg = list(info_bits) + [0] * crc_length
    remainder = _crc_process(msg, crc_length) & ((1 << crc_length) - 1)
    crc_bits = np.array(
        [(remainder >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=int,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=int)
    return _crc_process(bits, crc_length) == 0


def _path_metric_update(pm, llr, u):
    hard = 0 if llr >= 0 else 1
    if u != hard:
        pm += abs(llr)
    return pm


class SCLDecoder:
    """SCL 译码器。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        frozen_bits = np.asarray(frozen_bits)
        self.frozen_bits = frozen_bits.astype(bool) if frozen_bits.dtype != bool else frozen_bits
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]
        self.decode_order = [_bit_reversed(i, self.n) for i in range(N)]

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N, n = self.N, self.n

        from encoder import bit_reversal_permutation
        rev = bit_reversal_permutation(N)
        llr_mapped = llr_ch[rev]

        paths = [{
            'L': np.zeros((N, n + 1), dtype=np.float64),
            'B': np.zeros((N, n + 1), dtype=int),
            'pm': 0.0,
            'u_hat': np.zeros(N, dtype=int),
        }]
        paths[0]['L'][:, 0] = llr_mapped

        for l in self.decode_order:
            candidates = []
            for path in paths:
                _update_llrs(path['L'], path['B'], l, n, N)
                cur_llr = path['L'][l, n]

                if l in self.frozen_set:
                    new_path = {
                        'L': path['L'].copy(),
                        'B': path['B'].copy(),
                        'pm': _path_metric_update(path['pm'], cur_llr, 0),
                        'u_hat': path['u_hat'].copy(),
                    }
                    new_path['B'][l, n] = 0
                    new_path['u_hat'][l] = 0
                    _update_bits(new_path['B'], l, n, N)
                    candidates.append(new_path)
                else:
                    for u in (0, 1):
                        new_path = {
                            'L': path['L'].copy(),
                            'B': path['B'].copy(),
                            'pm': _path_metric_update(path['pm'], cur_llr, u),
                            'u_hat': path['u_hat'].copy(),
                        }
                        new_path['B'][l, n] = u
                        new_path['u_hat'][l] = u
                        _update_bits(new_path['B'], l, n, N)
                        candidates.append(new_path)

            candidates.sort(key=lambda p: p['pm'])
            paths = candidates[: self.list_size]

        if self.crc_length > 0:
            crc_pass = [p for p in paths if self._check_crc(p['u_hat'])]
            best = min(crc_pass if crc_pass else paths, key=lambda p: p['pm'])
        else:
            best = min(paths, key=lambda p: p['pm'])

        return best['u_hat'].copy(), best['pm']

    def _check_crc(self, u_hat):
        info_bits = u_hat[self.info_indices]
        return crc_check(info_bits, self.crc_length)
