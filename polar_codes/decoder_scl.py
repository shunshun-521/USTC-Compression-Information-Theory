"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np

from encoder import bit_reversal_permutation
from decoder_sc import (
    _active_llr_level,
    _active_bit_level,
    _bit_reversed,
    _lower_llr,
    f_operation,
)


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=np.int8)
    poly = 0x07 if crc_length == 8 else 0x8005
    mask = (1 << crc_length) - 1

    reg = 0
    for bit in info_bits:
        fb = ((reg >> (crc_length - 1)) & 1) ^ int(bit)
        reg = (reg << 1) & mask
        if fb:
            reg ^= poly

    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。"""
    if crc_length == 0:
        return True

    poly = 0x07 if crc_length == 8 else 0x8005
    mask = (1 << crc_length) - 1
    reg = 0
    for bit in np.asarray(bits, dtype=np.int8):
        fb = ((reg >> (crc_length - 1)) & 1) ^ int(bit)
        reg = (reg << 1) & mask
        if fb:
            reg ^= poly
    return reg == 0


def _update_llrs(L, B, phi, n, N):
    for s in range(n - _active_llr_level(phi, n), n):
        block_size = 1 << (s + 1)
        branch_size = block_size // 2
        for j in range(phi, N, block_size):
            if j % block_size < branch_size:
                L[j, s + 1] = f_operation(L[j, s], L[j + branch_size, s])
            else:
                L[j, s + 1] = _lower_llr(
                    L[j, s], L[j - branch_size, s], int(B[j - branch_size, s + 1])
                )


def _update_bits(B, phi, n, N):
    if phi < N // 2:
        return
    for s in range(n, n - _active_bit_level(phi, n), -1):
        block_size = 1 << s
        branch_size = block_size // 2
        for j in range(phi, -1, -block_size):
            if j % block_size >= branch_size:
                B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                B[j, s - 1] = B[j, s]


class SCLDecoder:
    """SCL 译码器"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.rev = bit_reversal_permutation(N)
        self.decode_order = [_bit_reversed(i, self.n) for i in range(N)]
        self.info_indices = np.where(~self.frozen_bits)[0]

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N, n = self.N, self.n

        paths = [{
            'L': np.full((N, n + 1), np.nan, dtype=np.float64),
            'B': np.full((N, n + 1), np.nan, dtype=np.float64),
            'pm': 0.0,
            'u_hat': np.zeros(N, dtype=int),
        }]
        paths[0]['L'][:, 0] = llr_ch[self.rev]

        for phi in self.decode_order:
            candidates = []
            for path in paths:
                _update_llrs(path['L'], path['B'], phi, n, N)
                llr = path['L'][phi, n]

                if self.frozen_bits[phi]:
                    penalty = 0.0 if llr >= 0 else abs(llr)
                    candidates.append((path['pm'] + penalty, path, 0))
                else:
                    for bit in (0, 1):
                        hard = 0 if llr >= 0 else 1
                        penalty = 0.0 if bit == hard else abs(llr)
                        candidates.append((path['pm'] + penalty, path, bit))

            candidates.sort(key=lambda x: x[0])
            new_paths = []
            for pm, parent, bit in candidates[: self.list_size]:
                child = {
                    'L': parent['L'].copy(),
                    'B': parent['B'].copy(),
                    'pm': pm,
                    'u_hat': parent['u_hat'].copy(),
                }
                child['u_hat'][phi] = 0 if self.frozen_bits[phi] else bit
                child['B'][phi, n] = child['u_hat'][phi]
                _update_bits(child['B'], phi, n, N)
                new_paths.append(child)
            paths = new_paths

        if self.crc_length > 0:
            valid = [
                p for p in paths
                if crc_check(p['u_hat'][self.info_indices], self.crc_length)
            ]
            if valid:
                paths = valid

        best = min(paths, key=lambda p: p['pm'])
        return best['u_hat'].copy(), best['pm']


if __name__ == "__main__":
    from construction import ga_construction
    from encoder import polar_encode
    from channel import bpsk_modulate, compute_llr, eb_n0_to_sigma
    from decoder_sc import sc_decode

    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0

    rng = np.random.default_rng(1)
    sigma = eb_n0_to_sigma(8.0, K / N)
    mismatches = 0
    for _ in range(50):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, size=K)
        x = polar_encode(u)
        llr = compute_llr(bpsk_modulate(x) + rng.normal(0, sigma, N), sigma)
        u_sc = sc_decode(llr, frozen_bits)
        u_scl, _ = SCLDecoder(N, frozen_bits, list_size=1).decode(llr)
        if not np.array_equal(u_sc, u_scl):
            mismatches += 1
    print(f"SCL L=1 vs SC mismatches: {mismatches}/50")
