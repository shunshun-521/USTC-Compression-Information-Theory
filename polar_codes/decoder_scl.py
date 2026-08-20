"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from decoder_sc import (
    f_operation, g_operation, bit_reversed, bit_reversal_permutation,
    active_llr_level, active_bit_level,
)


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, crc_length):
    reg = 0
    for b in bits:
        reg ^= int(b) << (crc_length - 1)
        for _ in range(crc_length):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)
    return reg


def crc_encode(info_bits, crc_length=8):
    info_bits = np.asarray(info_bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    remainder = _crc_remainder(info_bits, poly, crc_length)
    crc_bits = np.array(
        [(remainder >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    return _crc_remainder(bits, poly, crc_length) == 0


def _pm_update(pm, llr, u):
    hard = 0 if llr >= 0 else 1
    if u != hard:
        pm += abs(llr)
    return pm


class SCLDecoder:
    """SCL 译码器"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]

    def _new_path(self, llr_ch=None):
        path = {
            'L': np.zeros((self.N, self.n + 1), dtype=np.float64),
            'B': np.zeros((self.N, self.n + 1), dtype=np.int32),
            'pm': 0.0,
            'u_hat': np.zeros(self.N, dtype=np.int32),
        }
        if llr_ch is not None:
            path['L'][:, 0] = llr_ch
        return path

    def _compute_llr(self, path, idx):
        for s in range(self.n - active_llr_level(idx, self.n), self.n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(idx, self.N, block_size):
                if j % block_size < branch_size:
                    path['L'][j, s + 1] = f_operation(
                        path['L'][j, s], path['L'][j + branch_size, s]
                    )
                else:
                    path['L'][j, s + 1] = g_operation(
                        path['L'][j - branch_size, s],
                        path['L'][j, s],
                        path['B'][j - branch_size, s + 1],
                    )

    def _propagate_bits(self, path, idx, u_val):
        path['B'][idx, self.n] = u_val
        if idx >= self.N // 2:
            for s in range(self.n, self.n - active_bit_level(idx, self.n), -1):
                block_size = 2 ** s
                branch_size = block_size // 2
                for j in range(idx, -1, -block_size):
                    if j % block_size >= branch_size:
                        path['B'][j - branch_size, s - 1] = (
                            path['B'][j, s] ^ path['B'][j - branch_size, s]
                        )
                        path['B'][j, s - 1] = path['B'][j, s]

    def _copy_path(self, path):
        return {
            'L': path['L'].copy(),
            'B': path['B'].copy(),
            'pm': path['pm'],
            'u_hat': path['u_hat'].copy(),
        }

    def decode(self, llr_ch):
        rev = bit_reversal_permutation(self.N)
        llr_internal = llr_ch[rev]
        paths = [self._new_path(llr_internal)]
        decode_order = [bit_reversed(i, self.n) for i in range(self.N)]

        for idx in decode_order:
            candidates = []
            for path in paths:
                self._compute_llr(path, idx)
                llr = path['L'][idx, self.n]

                if self.frozen_bits[idx]:
                    child = self._copy_path(path)
                    child['pm'] = _pm_update(child['pm'], llr, 0)
                    child['u_hat'][idx] = 0
                    self._propagate_bits(child, idx, 0)
                    candidates.append(child)
                else:
                    for u in (0, 1):
                        child = self._copy_path(path)
                        child['pm'] = _pm_update(child['pm'], llr, u)
                        child['u_hat'][idx] = u
                        self._propagate_bits(child, idx, u)
                        candidates.append(child)

            candidates.sort(key=lambda p: p['pm'])
            paths = candidates[: self.list_size]

        best_crc = None
        if self.crc_length > 0:
            for p in paths:
                info_bits = p['u_hat'][self.info_indices]
                if crc_check(info_bits, self.crc_length):
                    if best_crc is None or p['pm'] < best_crc['pm']:
                        best_crc = p

        best = best_crc if best_crc is not None else paths[0]
        return best['u_hat'].copy(), best['pm']
