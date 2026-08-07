"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from encoder import bit_reversal_permutation
from decoder_sc import (
    f_operation,
    g_operation,
    _bit_reversed_int,
    _active_llr_level,
    _active_bit_level,
    _frozen_indices_from_mask,
)


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, crc_length):
    """逐比特 CRC 余数（MSB first）。"""
    crc = 0
    top = 1 << (crc_length - 1)
    mask = (1 << crc_length) - 1
    for b in bits:
        if ((crc >> (crc_length - 1)) ^ int(b)) & 1:
            crc = ((crc << 1) ^ poly) & mask
        else:
            crc = (crc << 1) & mask
    return crc


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
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。"""
    bits = np.asarray(bits, dtype=int)
    if len(bits) < crc_length:
        return False
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    return _crc_remainder(bits, poly, crc_length) == 0


class SCLDecoder:
    """SCL 译码器（Lazy Copy）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_set = set(_frozen_indices_from_mask(frozen_bits))
        self.info_indices = np.array(
            sorted([i for i in range(N) if i not in self.frozen_set]), dtype=int
        )
        self.list_size = list_size
        self.crc_length = crc_length
        self.brp = bit_reversal_permutation(N)

    def _new_path(self, llr_br):
        L = np.full((self.N, self.n + 1), np.nan, dtype=np.float64)
        B = np.full((self.N, self.n + 1), np.nan)
        L[:, 0] = llr_br
        return {'L': L, 'B': B, 'pm': 0.0}

    def _copy_path(self, path):
        return {
            'L': path['L'].copy(),
            'B': path['B'].copy(),
            'pm': path['pm'],
        }

    def _update_llrs(self, path, l):
        L, B = path['L'], path['B']
        for s in range(self.n - _active_llr_level(l, self.n), self.n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = float(f_operation(L[j, s], L[j + branch_size, s]))
                else:
                    top_bit = int(B[j - branch_size, s + 1])
                    btm = L[j, s]
                    top = L[j - branch_size, s]
                    if top_bit == 0:
                        L[j, s + 1] = btm + top
                    else:
                        L[j, s + 1] = btm - top

    def _update_bits(self, path, l, bit):
        B = path['B']
        B[l, self.n] = bit
        if l >= self.N / 2:
            for s in range(self.n, self.n - _active_bit_level(l, self.n), -1):
                block_size = 2 ** s
                branch_size = block_size // 2
                for j in range(l, -1, -block_size):
                    if j % block_size >= branch_size:
                        B[j - branch_size, s - 1] = int(B[j, s]) ^ int(
                            B[j - branch_size, s]
                        )
                        B[j, s - 1] = B[j, s]

    def _pm_penalty(self, llr, bit):
        hard = 0 if llr >= 0 else 1
        return 0.0 if bit == hard else abs(llr)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        llr_br = llr_ch[self.brp]
        paths = [self._new_path(llr_br)]
        decode_order = [_bit_reversed_int(i, self.n) for i in range(self.N)]

        for l in decode_order:
            candidates = []
            for path in paths:
                self._update_llrs(path, l)
                llr_bit = path['L'][l, self.n]

                if l in self.frozen_set:
                    new_path = self._copy_path(path)
                    new_path['pm'] += self._pm_penalty(llr_bit, 0)
                    self._update_bits(new_path, l, 0)
                    candidates.append(new_path)
                else:
                    for bit in (0, 1):
                        new_path = self._copy_path(path)
                        new_path['pm'] += self._pm_penalty(llr_bit, bit)
                        self._update_bits(new_path, l, bit)
                        candidates.append(new_path)

            candidates.sort(key=lambda p: p['pm'])
            paths = candidates[:self.list_size]

        def u_from_path(p):
            return p['B'][:, self.n].astype(int)

        best = None
        if self.crc_length > 0:
            crc_ok = [
                p for p in paths
                if crc_check(u_from_path(p)[self.info_indices], self.crc_length)
            ]
            if crc_ok:
                best = min(crc_ok, key=lambda p: p['pm'])

        if best is None:
            best = min(paths, key=lambda p: p['pm'])

        return u_from_path(best), best['pm']
