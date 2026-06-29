"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import copy
import numpy as np

from decoder_sc import _prepare_channel_llr
from decoder_utils import (
    active_bit_level,
    active_llr_level,
    bit_reversed,
    f_minsum,
    g_llr,
    hard_decision,
)


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, crc_length):
    reg = 0
    for bit in bits:
        reg ^= (int(bit) << (crc_length - 1))
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
        [(remainder >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=int,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    remainder = _crc_remainder(bits[:-crc_length], poly, crc_length)
    expected = bits[-crc_length:]
    actual = np.array(
        [(remainder >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=int,
    )
    return np.array_equal(expected, actual)


class SCLDecoder:
    """SCL 译码器（Lazy Copy）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]

    def _new_path(self, llr_ch):
        L = np.zeros((self.N, self.n + 1), dtype=np.float64)
        B = np.full((self.N, self.n + 1), np.nan)
        L[:, 0] = llr_ch
        return {'pm': 0.0, 'L': L, 'B': B, 'u_hat': np.zeros(self.N, dtype=int)}

    def _copy_path(self, path):
        child = {
            'pm': path['pm'],
            'L': path['L'].copy(),
            'B': path['B'].copy(),
            'u_hat': path['u_hat'].copy(),
        }
        return child

    def _update_llrs(self, path, l):
        L, B = path['L'], path['B']
        for s in range(self.n - active_llr_level(l, self.n), self.n):
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = f_minsum(L[j, s], L[j + branch_size, s])
                else:
                    B_top = int(B[j - branch_size, s + 1])
                    L[j, s + 1] = g_llr(L[j, s], L[j - branch_size, s], B_top)

    def _update_bits(self, path, l):
        if l < self.N // 2:
            return
        B = path['B']
        for s in range(self.n, self.n - active_bit_level(l, self.n), -1):
            block_size = 1 << s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                    B[j, s - 1] = B[j, s]

    @staticmethod
    def _pm_penalty(llr, u):
        u_from_llr = hard_decision(llr)
        return 0.0 if u == u_from_llr else abs(llr)

    def decode(self, llr_ch):
        llr_ch = _prepare_channel_llr(llr_ch, self.N)
        paths = [self._new_path(llr_ch)]

        for i in range(self.N):
            l = bit_reversed(i, self.n)
            candidates = []

            for path in paths:
                self._update_llrs(path, l)
                llr = path['L'][l, self.n]

                if l in self.frozen_set:
                    child = self._copy_path(path)
                    child['pm'] += self._pm_penalty(llr, 0)
                    child['u_hat'][l] = 0
                    child['B'][l, self.n] = 0
                    self._update_bits(child, l)
                    candidates.append(child)
                else:
                    for u in (0, 1):
                        child = self._copy_path(path)
                        child['pm'] += self._pm_penalty(llr, u)
                        child['u_hat'][l] = u
                        child['B'][l, self.n] = u
                        self._update_bits(child, l)
                        candidates.append(child)

            candidates.sort(key=lambda p: p['pm'])
            paths = candidates[: self.list_size]

        if self.crc_length > 0:
            valid = []
            for path in paths:
                info_bits = path['u_hat'][self.info_indices]
                if crc_check(info_bits, self.crc_length):
                    valid.append(path)
            pool = valid if valid else paths
        else:
            pool = paths

        best = min(pool, key=lambda p: p['pm'])
        return best['u_hat'].copy(), best['pm']


if __name__ == "__main__":
    from construction import ga_construction
    from encoder import polar_encode
    from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma
    from decoder_sc import sc_decode

    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False

    rng = np.random.default_rng(0)
    sigma = eb_n0_to_sigma(5.0, K / N)
    for _ in range(20):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)
        u_sc = sc_decode(llr, frozen_bits)
        u_scl, _ = SCLDecoder(N, frozen_bits, list_size=1).decode(llr)
        assert np.array_equal(u_sc, u_scl), "L=1 SCL 应与 SC 等价"
    print("SCL L=1 等价 SC 校验通过")
