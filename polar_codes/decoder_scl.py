"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np
from encoder import bit_reversed
from decoder_sc import _update_llrs, _update_bits, f_operation


_CRC_POLYS = {8: 0x07, 16: 0x8005}


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=np.int32)
    poly = _CRC_POLYS[crc_length]
    reg = 0
    for b in info_bits:
        msb = (reg >> (crc_length - 1)) & 1
        reg = ((reg << 1) | int(b)) & ((1 << crc_length) - 1)
        if msb ^ int(b):
            reg ^= poly
    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=np.int32,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC"""
    bits = np.asarray(bits, dtype=np.int32)
    if len(bits) < crc_length:
        return False
    return np.array_equal(
        bits[-crc_length:],
        crc_encode(bits[:-crc_length], crc_length)[-crc_length:],
    )


class SCLDecoder:
    """SCL 译码器（Lazy Copy 优化）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.decode_order = [bit_reversed(i, self.n) for i in range(N)]

    @staticmethod
    def _pm_penalty(llr, u):
        hard = 0 if llr >= 0 else 1
        return 0.0 if u == hard else abs(llr)

    def _new_path(self, llr_ch):
        L = np.full((self.N, self.n + 1), np.nan, dtype=np.float64)
        B = np.zeros((self.N, self.n + 1), dtype=np.int32)
        L[:, 0] = llr_ch
        return {'pm': 0.0, 'L': L, 'B': B, 'L_refs': 1, 'B_refs': 1}

    def _fork(self, path):
        new_path = {
            'pm': path['pm'],
            'L': path['L'],
            'B': path['B'],
            'L_refs': path['L_refs'] + 1,
            'B_refs': path['B_refs'] + 1,
        }
        path['L_refs'] += 1
        path['B_refs'] += 1
        return new_path

    def _cow_l(self, path):
        if path['L_refs'] > 1:
            path['L'] = path['L'].copy()
            path['L_refs'] = 1

    def _cow_b(self, path):
        if path['B_refs'] > 1:
            path['B'] = path['B'].copy()
            path['B_refs'] = 1

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [self._new_path(llr_ch)]

        for l in self.decode_order:
            candidates = []
            for path in paths:
                self._cow_l(path)
                self._cow_b(path)
                _update_llrs(path['L'], path['B'], l, self.n)
                llr = path['L'][l, self.n]

                if l in self.frozen_set:
                    p = self._fork(path)
                    p['pm'] += self._pm_penalty(llr, 0)
                    self._cow_b(p)
                    p['B'][l, self.n] = 0
                    _update_bits(p['B'], l, self.n, self.N)
                    candidates.append(p)
                else:
                    for u in (0, 1):
                        p = self._fork(path)
                        p['pm'] += self._pm_penalty(llr, u)
                        self._cow_b(p)
                        p['B'][l, self.n] = u
                        _update_bits(p['B'], l, self.n, self.N)
                        candidates.append(p)

            candidates.sort(key=lambda p: p['pm'])
            paths = candidates[: self.list_size]

        paths.sort(key=lambda p: p['pm'])

        if self.crc_length > 0:
            valid = []
            for p in paths:
                u_hat = p['B'][:, self.n].astype(np.int32)
                if crc_check(u_hat, self.crc_length):
                    valid.append(p)
            best = valid[0] if valid else paths[0]
        else:
            best = paths[0]

        return best['B'][:, self.n].astype(np.int32), best['pm']
