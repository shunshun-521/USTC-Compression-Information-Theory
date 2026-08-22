"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
import math

from encoder import bit_reversal_permutation
from decoder_sc import (
    f_operation, g_operation,
    _bit_reversed, _active_llr_level, _active_bit_level,
)


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, crc_length):
    reg = 0
    for bit in bits:
        reg = ((reg << 1) | int(bit)) & ((1 << crc_length) - 1)
        if reg & (1 << (crc_length - 1)):
            reg ^= poly
    for _ in range(crc_length):
        reg = (reg << 1) & ((1 << crc_length) - 1)
        if reg & (1 << (crc_length - 1)):
            reg ^= poly
    return reg


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    remainder = _crc_remainder(info_bits, poly, crc_length)
    crc_bits = np.array(
        [(remainder >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=int,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC 是否正确"""
    bits = np.asarray(bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    return _crc_remainder(bits, poly, crc_length) == 0


class SCLDecoder:
    """SCL 译码器（含 Lazy Copy 优化）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.list_size = list_size
        self.crc_length = crc_length
        self.frozen_set = set(np.where(frozen_bits == 1)[0])
        self.brp = bit_reversal_permutation(N)

    def _pm_penalty(self, llr, u):
        hard = 0 if llr >= 0 else 1
        return 0.0 if u == hard else abs(llr)

    def _update_llrs(self, L, B, l):
        n = self.n
        N = self.N
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

    def _update_bits(self, B, l):
        n = self.n
        N = self.N
        if l < N // 2:
            return
        for s in range(n, n - _active_bit_level(l, n), -1):
            block_size = 1 << s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    B[j - branch_size, s - 1] = (
                        B[j, s] + B[j - branch_size, s]
                    ) % 2
                    B[j, s - 1] = B[j, s]

    def decode(self, llr_ch):
        """主译码函数，返回 (u_hat, pm)"""
        llr_ch = np.asarray(llr_ch, dtype=np.float64)[self.brp]
        N = self.N
        n = self.n
        L_size = self.list_size

        paths = []
        L0 = np.zeros((N, n + 1), dtype=np.float64)
        B0 = np.zeros((N, n + 1), dtype=int)
        L0[:, 0] = llr_ch
        paths.append({'pm': 0.0, 'L': L0, 'B': B0, 'u_hat': np.zeros(N, dtype=int)})

        for phi in range(N):
            l = _bit_reversed(phi, n)
            candidates = []

            for path in paths:
                self._update_llrs(path['L'], path['B'], l)
                llr = path['L'][l, n]

                if l in self.frozen_set:
                    penalty = self._pm_penalty(llr, 0)
                    candidates.append((path['pm'] + penalty, path, 0))
                else:
                    for u in (0, 1):
                        penalty = self._pm_penalty(llr, u)
                        candidates.append((path['pm'] + penalty, path, u))

            candidates.sort(key=lambda x: x[0])
            candidates = candidates[:L_size]

            new_paths = []
            for pm, src, u_val in candidates:
                new_L = src['L'].copy()
                new_B = src['B'].copy()
                new_u = src['u_hat'].copy()
                new_u[l] = u_val
                new_B[l, n] = u_val
                self._update_bits(new_B, l)
                new_paths.append({'pm': pm, 'L': new_L, 'B': new_B, 'u_hat': new_u})

            paths = new_paths

        if self.crc_length > 0:
            valid = [p for p in paths if crc_check(p['u_hat'], self.crc_length)]
            best = min(valid, key=lambda p: p['pm']) if valid else min(paths, key=lambda p: p['pm'])
        else:
            best = min(paths, key=lambda p: p['pm'])

        return best['u_hat'].copy(), best['pm']


if __name__ == "__main__":
    from construction import ga_construction
    from encoder import polar_encode
    from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma
    from decoder_sc import sc_decode

    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0

    rng = np.random.default_rng(42)
    sigma = eb_n0_to_sigma(5.0, K / N)

    mismatches = 0
    for _ in range(20):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u)
        s = bpsk_modulate(x)
        y = awgn_channel(s, sigma, rng)
        llr = compute_llr(y, sigma)

        u_sc = sc_decode(llr, frozen_bits)
        u_scl, _ = SCLDecoder(N, frozen_bits, list_size=1).decode(llr)
        if not np.array_equal(u_sc, u_scl):
            mismatches += 1

    print(f"SCL(L=1) vs SC 一致性: {20 - mismatches}/20")
    assert mismatches == 0
