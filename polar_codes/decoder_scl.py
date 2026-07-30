"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
import math

from decoder_sc import (
    SCDecoder,
    active_llr_level,
    active_bit_level,
    upper_llr,
    lower_llr,
    hard_decision,
)
from encoder import bit_reversed


CRC_POLYNOMIALS = {8: 0x07, 16: 0x8005}


def crc_encode(info_bits, crc_length=8):
    info_bits = np.asarray(info_bits, dtype=int)
    poly = CRC_POLYNOMIALS[crc_length]
    reg = 0
    for bit in info_bits:
        reg = (reg << 1) | int(bit)
        if reg & (1 << crc_length):
            reg ^= poly
    for _ in range(crc_length):
        reg <<= 1
        if reg & (1 << crc_length):
            reg ^= poly
    crc_bits = np.array([(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=int)
    if len(bits) < crc_length:
        return False
    poly = CRC_POLYNOMIALS[crc_length]
    reg = 0
    for bit in bits:
        reg = (reg << 1) | int(bit)
        if reg & (1 << crc_length):
            reg ^= poly
    return reg == 0


class SCLDecoder:
    """SCL 译码器（Lazy Copy via array duplication per path）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.frozen = set(np.where(self.frozen_bits == 1)[0])
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(self.frozen_bits == 0)[0]

    def _path_llr(self, L, B, l):
        for s in range(self.n - active_llr_level(l, self.n), self.n):
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = upper_llr(L[j, s], L[j + branch_size, s])
                else:
                    L[j, s + 1] = lower_llr(L[j, s], L[j - branch_size, s], B[j - branch_size, s + 1])
        return L[l, self.n]

    def _path_bits(self, B, l, u_val):
        B[l, self.n] = u_val
        if l < self.N / 2:
            return
        for s in range(self.n, self.n - active_bit_level(l, self.n), -1):
            block_size = 1 << s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                    B[j, s - 1] = B[j, s]

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N, n, Lmax = self.N, self.n, self.list_size

        paths = [{
            "pm": 0.0,
            "L": np.full((N, n + 1), np.nan, dtype=np.float64),
            "B": np.full((N, n + 1), np.nan),
            "u": np.zeros(N, dtype=int),
        }]
        paths[0]["L"][:, 0] = llr_ch

        for phi in range(N):
            l = bit_reversed(phi, n)
            candidates = []
            for path in paths:
                L = path["L"].copy()
                B = path["B"].copy()
                llr = self._path_llr(L, B, l)
                if l in self.frozen:
                    penalty = 0.0 if llr >= 0 else abs(llr)
                    u_val = 0
                    candidates.append((path["pm"] + penalty, L, B, path["u"].copy(), l, u_val))
                else:
                    for u_val in (0, 1):
                        penalty = 0.0 if (llr >= 0 and u_val == 0) or (llr < 0 and u_val == 1) else abs(llr)
                        candidates.append((path["pm"] + penalty, L, B, path["u"].copy(), l, u_val))

            candidates.sort(key=lambda x: x[0])
            candidates = candidates[:Lmax]

            new_paths = []
            for pm, L, B, u, l_val, u_val in candidates:
                u[l_val] = u_val
                self._path_bits(B, l_val, u_val)
                new_paths.append({"pm": pm, "L": L, "B": B, "u": u})
            paths = new_paths

        best_idx = 0
        if self.crc_length > 0:
            valid = [i for i, p in enumerate(paths) if crc_check(p["u"][self.info_indices], self.crc_length)]
            if valid:
                best_idx = min(valid, key=lambda i: paths[i]["pm"])

        best = paths[best_idx]
        return best["u"], best["pm"]


def verify_scl_equals_sc(N=64, K=32, num_frames=50):
    from construction import ga_construction
    from encoder import polar_encode
    from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma
    from decoder_sc import sc_decode

    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    sigma = eb_n0_to_sigma(4.0, K / N)
    rng = np.random.default_rng(1)

    for _ in range(num_frames):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        llr = compute_llr(awgn_channel(bpsk_modulate(polar_encode(u)), sigma, rng), sigma)
        u_sc = sc_decode(llr, frozen_bits)
        u_scl, _ = SCLDecoder(N, frozen_bits, list_size=1).decode(llr)
        assert np.array_equal(u_sc, u_scl), "SCL L=1 != SC"
    return True


if __name__ == "__main__":
    verify_scl_equals_sc()
    print("SCL verification passed")
