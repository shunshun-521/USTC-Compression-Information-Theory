"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
import math
from decoder_sc import _update_llrs, _update_bits


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def crc_encode(info_bits, crc_length=8):
    info_bits = np.asarray(info_bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    mask = (1 << crc_length) - 1
    reg = 0
    for bit in info_bits:
        reg ^= int(bit) << (crc_length - 1)
        for _ in range(crc_length):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & mask
            else:
                reg = (reg << 1) & mask
    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=int)
    expected = crc_encode(bits[:-crc_length], crc_length)
    return np.array_equal(bits[-crc_length:], expected[-crc_length:])


class SCLDecoder:
    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]

    def _path_penalty(self, llr, u):
        return 0.0 if u == (0 if llr >= 0 else 1) else abs(llr)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N, n = self.N, self.n
        from encoder import bit_reversal_permutation
        from decoder_sc import _bit_reversed_index
        rev = bit_reversal_permutation(N)
        llr_perm = llr_ch[rev]

        paths = [{'pm': 0.0, 'L': np.zeros((N, n + 1)), 'B': np.zeros((N, n + 1), dtype=np.int32)}]
        paths[0]['L'][:, 0] = llr_perm

        for i in range(N):
            l = _bit_reversed_index(i, n)
            candidates = []
            for path in paths:
                _update_llrs(path['L'], path['B'], l, n, N)
                cur_llr = path['L'][l, n]
                if self.frozen_bits[l]:
                    np_ = {'pm': path['pm'] + self._path_penalty(cur_llr, 0),
                           'L': path['L'].copy(), 'B': path['B'].copy()}
                    np_['B'][l, n] = 0
                    _update_bits(np_['B'], l, n, N)
                    candidates.append(np_)
                else:
                    for u in (0, 1):
                        np_ = {'pm': path['pm'] + self._path_penalty(cur_llr, u),
                               'L': path['L'].copy(), 'B': path['B'].copy()}
                        np_['B'][l, n] = u
                        _update_bits(np_['B'], l, n, N)
                        candidates.append(np_)
            candidates.sort(key=lambda p: p['pm'])
            paths = candidates[:self.list_size]

        crc_pass = []
        for p in paths:
            u_hat = p['B'][:, n].astype(int)
            if self.crc_length == 0 or crc_check(u_hat[self.info_indices], self.crc_length):
                crc_pass.append((p, u_hat))
        best_p, best_u = min(crc_pass, key=lambda x: x[0]['pm']) if crc_pass else (
            min(paths, key=lambda p: p['pm']),
            min(paths, key=lambda p: p['pm'])['B'][:, n].astype(int),
        )
        return best_u, best_p['pm']
