"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from decoder_sc import (
    _bit_reversed,
    _active_llr_level,
    _active_bit_level,
    _prepare_channel_llr,
    f_operation,
    g_operation,
)


_CRC_POLY = {8: 0x07, 16: 0x8005}


def _crc_remainder(bits, crc_length):
    poly = _CRC_POLY[crc_length]
    reg = 0
    for b in bits:
        reg = (reg << 1) | int(b)
        if reg & (1 << crc_length):
            reg ^= poly
    return reg & ((1 << crc_length) - 1)


def crc_encode(info_bits, crc_length=8):
    info_bits = np.asarray(info_bits, dtype=np.int8)
    remainder = _crc_remainder(info_bits, crc_length)
    crc_bits = np.array(
        [(remainder >> i) & 1 for i in range(crc_length - 1, -1, -1)],
        dtype=np.int8,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=np.int8)
    if len(bits) < crc_length:
        return False
    remainder = _crc_remainder(bits[:-crc_length], crc_length)
    received = 0
    for b in bits[-crc_length:]:
        received = (received << 1) | int(b)
    return remainder == received


def _update_llrs_for_bit(L, B, l, n, N):
    """对给定 B 状态，更新 L 至可判决比特 l。"""
    for s in range(n - _active_llr_level(l, n), n):
        block_size = 1 << (s + 1)
        branch_size = block_size // 2
        for j in range(l, N, block_size):
            if j % block_size < branch_size:
                L[j, s + 1] = f_operation(L[j, s], L[j + branch_size, s])
            else:
                L[j, s + 1] = g_operation(
                    L[j - branch_size, s], L[j, s],
                    B[j - branch_size, s + 1],
                )


def _update_bits(B, l, n, N):
    if l < N // 2:
        return
    for s in range(n, n - _active_bit_level(l, n), -1):
        block_size = 1 << s
        branch_size = block_size // 2
        for j in range(l, -1, -block_size):
            if j % block_size >= branch_size:
                B[j - branch_size, s - 1] = (
                    int(B[j, s]) ^ int(B[j - branch_size, s])
                )
                B[j, s - 1] = B[j, s]


class SCLDecoder:
    """SCL 译码器（基于 Permuted SC）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.info_positions = np.where(~self.frozen_bits)[0]

    def _pm_penalty(self, llr, u):
        u_hard = 0 if llr >= 0 else 1
        return 0.0 if u == u_hard else abs(llr)

    def decode(self, llr_ch):
        N, n = self.N, self.n
        channel_llr = _prepare_channel_llr(llr_ch)

        paths = [{
            'pm': 0.0,
            'B': np.zeros((N, n + 1), dtype=np.int8),
            'L': np.full((N, n + 1), np.nan, dtype=np.float64),
            'u_hat': np.zeros(N, dtype=np.int8),
        }]
        paths[0]['L'][:, 0] = channel_llr

        for i in range(N):
            l = _bit_reversed(i, n)
            new_paths = []

            for p in paths:
                _update_llrs_for_bit(p['L'], p['B'], l, n, N)
                cur_llr = p['L'][l, n]

                if l in self.frozen_set:
                    pen = self._pm_penalty(cur_llr, 0)
                    p['pm'] += pen
                    p['B'][l, n] = 0
                    p['u_hat'][l] = 0
                    _update_bits(p['B'], l, n, N)
                    new_paths.append(p)
                else:
                    for u_cand in (0, 1):
                        child = {
                            'pm': p['pm'] + self._pm_penalty(cur_llr, u_cand),
                            'B': p['B'].copy(),
                            'L': p['L'].copy(),
                            'u_hat': p['u_hat'].copy(),
                        }
                        child['B'][l, n] = u_cand
                        child['u_hat'][l] = u_cand
                        _update_bits(child['B'], l, n, N)
                        new_paths.append(child)

            order = np.argsort([p['pm'] for p in new_paths])[:self.list_size]
            paths = [new_paths[j] for j in order]

        if self.crc_length > 0:
            crc_ok = [
                i for i, p in enumerate(paths)
                if crc_check(p['u_hat'][self.info_positions], self.crc_length)
            ]
            best = (
                min(crc_ok, key=lambda i: paths[i]['pm'])
                if crc_ok
                else int(np.argmin([p['pm'] for p in paths]))
            )
        else:
            best = int(np.argmin([p['pm'] for p in paths]))

        return paths[best]['u_hat'], paths[best]['pm']
