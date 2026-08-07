"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import copy

import numpy as np

from decoder_sc import _active_bit_level, _active_llr_level, lower_llr, upper_llr
from encoder import bit_reversed


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=np.int8)
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

    for _ in range(crc_length):
        for _ in range(crc_length):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & mask
            else:
                reg = (reg << 1) & mask

    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=np.int8,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。"""
    bits = np.asarray(bits, dtype=np.int8)
    if len(bits) < crc_length:
        return False
    expected = crc_encode(bits[:-crc_length], crc_length)
    return np.array_equal(bits[-crc_length:], expected[-crc_length:])


def _pm_update(pm, llr, bit):
    hard = 0 if llr >= 0 else 1
    if bit == hard:
        return pm
    return pm + abs(llr)


def _init_path(llr_ch, n):
    N = len(llr_ch)
    return {
        'pm': 0.0,
        'L': np.full((N, n + 1), np.nan, dtype=np.float64),
        'B': np.full((N, n + 1), np.nan),
        'u_hat': np.zeros(N, dtype=np.int8),
    }


def _path_update_llrs(path, l, n, N):
    L = path['L']
    B = path['B']
    for s in range(n - _active_llr_level(l, n), n):
        block_size = 1 << (s + 1)
        branch_size = block_size // 2
        for j in range(l, N, block_size):
            if j % block_size < branch_size:
                L[j, s + 1] = upper_llr(L[j, s], L[j + branch_size, s])
            else:
                L[j, s + 1] = lower_llr(
                    L[j, s], L[j - branch_size, s], int(B[j - branch_size, s + 1])
                )


def _path_update_bits(path, l, n, N):
    if l < N // 2:
        return
    B = path['B']
    for s in range(n, n - _active_bit_level(l, n), -1):
        block_size = 1 << s
        branch_size = block_size // 2
        for j in range(l, -1, -block_size):
            if j % block_size >= branch_size:
                B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                B[j, s - 1] = B[j, s]


class SCLDecoder:
    """SCL 译码器（含 Lazy Copy 优化）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0, info_indices=None):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = info_indices

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n = self.n
        N = self.N
        L_size = self.list_size

        paths = [_init_path(llr_ch, n)]
        paths[0]['L'][:, 0] = llr_ch

        for i in range(N):
            l = bit_reversed(i, n)
            candidates = []

            for path in paths:
                _path_update_llrs(path, l, n, N)
                llr = path['L'][l, n]

                if l in self.frozen_set:
                    child = copy.deepcopy(path)
                    child['pm'] = _pm_update(path['pm'], llr, 0)
                    child['B'][l, n] = 0
                    child['u_hat'][l] = 0
                    _path_update_bits(child, l, n, N)
                    candidates.append(child)
                else:
                    for bit in (0, 1):
                        child = copy.deepcopy(path)
                        child['pm'] = _pm_update(path['pm'], llr, bit)
                        child['B'][l, n] = bit
                        child['u_hat'][l] = bit
                        _path_update_bits(child, l, n, N)
                        candidates.append(child)

            candidates.sort(key=lambda p: p['pm'])
            paths = candidates[:L_size]

        if self.crc_length > 0:
            valid = []
            for p in paths:
                bits = p['u_hat'][self.info_indices] if self.info_indices is not None else p['u_hat']
                if crc_check(bits, self.crc_length):
                    valid.append(p)
            if valid:
                paths = valid

        best = min(paths, key=lambda p: p['pm'])
        return best['u_hat'], best['pm']
