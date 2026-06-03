"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from _pp_utils import bit_reversed
from _pp_decoder_utils import (
    active_llr_level,
    active_bit_level,
    upper_llr,
    lower_llr,
    hard_decision,
)


_CRC8_POLY = 0x07
_CRC16_POLY = 0x8005


def crc_encode(info_bits, crc_length=8):
    info_bits = np.asarray(info_bits, dtype=int).ravel()
    if crc_length == 8:
        poly = _CRC8_POLY
        reg = 0
        for b in info_bits:
            reg ^= int(b) << 7
            for _ in range(8):
                if reg & 0x80:
                    reg = ((reg << 1) ^ poly) & 0xFF
                else:
                    reg = (reg << 1) & 0xFF
        crc_bits = np.array([(reg >> (7 - i)) & 1 for i in range(8)], dtype=int)
    elif crc_length == 16:
        poly = _CRC16_POLY
        reg = 0
        for b in info_bits:
            reg ^= int(b) << 15
            for _ in range(16):
                if reg & 0x8000:
                    reg = ((reg << 1) ^ poly) & 0xFFFF
                else:
                    reg = (reg << 1) & 0xFFFF
        crc_bits = np.array([(reg >> (15 - i)) & 1 for i in range(16)], dtype=int)
    else:
        raise ValueError("crc_length must be 8 or 16")
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=int).ravel()
    if crc_length == 8:
        poly = _CRC8_POLY
        reg = 0
        for b in bits:
            reg ^= int(b) << 7
            for _ in range(8):
                if reg & 0x80:
                    reg = ((reg << 1) ^ poly) & 0xFF
                else:
                    reg = (reg << 1) & 0xFF
        return reg == 0
    if crc_length == 16:
        poly = _CRC16_POLY
        reg = 0
        for b in bits:
            reg ^= int(b) << 15
            for _ in range(16):
                if reg & 0x8000:
                    reg = ((reg << 1) ^ poly) & 0xFFFF
                else:
                    reg = (reg << 1) & 0xFFFF
        return reg == 0
    raise ValueError("crc_length must be 8 or 16")


def _update_llrs(L, B, l, n, N):
    for s in range(n - active_llr_level(l, n), n):
        block_size = 1 << (s + 1)
        branch_size = block_size // 2
        for j in range(l, N, block_size):
            if j % block_size < branch_size:
                L[j, s + 1] = upper_llr(L[j, s], L[j + branch_size, s])
            else:
                top_bit = B[j - branch_size, s + 1]
                L[j, s + 1] = lower_llr(
                    L[j - branch_size, s], L[j, s], top_bit
                )


def _update_bits(B, l, n, N):
    if l < N // 2:
        return
    for s in range(n, n - active_bit_level(l, n), -1):
        block_size = 1 << s
        branch_size = block_size // 2
        for j in range(l, -1, -block_size):
            if j % block_size >= branch_size:
                B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                B[j, s - 1] = B[j, s]


def _pm_penalty(llr_val, u_bit):
    hard = 0 if llr_val >= 0 else 1
    return 0.0 if u_bit == hard else abs(llr_val)


class SCLDecoder:
    """SCL 译码器（Lazy Copy：路径分裂时复制 L/B 数组）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0, info_indices=None):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_set = set(np.where(frozen_bits)[0])
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = info_indices

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        if self.list_size == 1:
            from decoder_sc import sc_decode
            frozen = np.zeros(self.N, dtype=bool)
            frozen[list(self.frozen_set)] = True
            u = sc_decode(llr_ch, frozen)
            return u, 0.0

        N, n = self.N, self.n

        paths = [{
            'pm': 0.0,
            'L': np.full((N, n + 1), np.nan, dtype=np.float64),
            'B': np.full((N, n + 1), np.nan),
        }]
        paths[0]['L'][:, 0] = llr_ch

        for i in range(N):
            l = bit_reversed(i, n)
            new_paths = []
            for path in paths:
                _update_llrs(path['L'], path['B'], l, n, N)
                llr0 = path['L'][l, n]
                if np.isnan(llr0):
                    llr0 = 0.0

                if l in self.frozen_set:
                    path['pm'] += _pm_penalty(llr0, 0)
                    path['B'][l, n] = 0
                    _update_bits(path['B'], l, n, N)
                    new_paths.append(path)
                else:
                    for u_bit in (0, 1):
                        cp = {
                            'pm': path['pm'] + _pm_penalty(llr0, u_bit),
                            'L': path['L'].copy(),
                            'B': path['B'].copy(),
                        }
                        cp['B'][l, n] = u_bit
                        _update_bits(cp['B'], l, n, N)
                        new_paths.append(cp)

            new_paths.sort(key=lambda p: p['pm'])
            paths = new_paths[: self.list_size]

        if self.crc_length > 0 and self.info_indices is not None:
            valid = []
            for p in paths:
                u = p['B'][:, n].astype(int)
                if crc_check(u[self.info_indices], self.crc_length):
                    valid.append(p)
            best = min(valid, key=lambda p: p['pm']) if valid else min(paths, key=lambda p: p['pm'])
        else:
            best = min(paths, key=lambda p: p['pm'])

        return best['B'][:, n].astype(int), best['pm']
