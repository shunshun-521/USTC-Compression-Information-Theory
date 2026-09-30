"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from encoder import bit_reversal_permutation
from decoder_sc import (
    _active_bit_level,
    _active_llr_level,
    _bit_reversed_scalar,
    _lower_llr,
    _upper_llr,
    sc_decode,
)
_CRC8_POLY = 0x07
_CRC16_POLY = 0x8005


def crc_encode(info_bits, crc_length=8):
    info_bits = np.asarray(info_bits, dtype=int).ravel()
    poly = _CRC8_POLY if crc_length == 8 else _CRC16_POLY
    reg = 0
    mask = (1 << crc_length) - 1
    for b in info_bits:
        reg ^= int(b) << (crc_length - 1)
        for _ in range(1):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & mask
            else:
                reg = (reg << 1) & mask
    crc_bits = np.array([(reg >> i) & 1 for i in range(crc_length - 1, -1, -1)], dtype=int)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=int).ravel()
    poly = _CRC8_POLY if crc_length == 8 else _CRC16_POLY
    reg = 0
    mask = (1 << crc_length) - 1
    for b in bits:
        reg ^= int(b) << (crc_length - 1)
        for _ in range(1):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & mask
            else:
                reg = (reg << 1) & mask
    return reg == 0


class SCLDecoder:
    """SCL 译码器"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.L = list_size
        self.crc_length = crc_length
        self.rev = bit_reversal_permutation(N)
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.info_indices = np.where(~self.frozen_bits)[0]

    def _pm_add(self, pm, llr, u):
        if u == 0:
            return pm + (0.0 if llr >= 0 else abs(llr))
        return pm + (0.0 if llr < 0 else abs(llr))

    def _update_llrs(self, L, B, l):
        for s in range(self.n - _active_llr_level(l, self.n), self.n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = _upper_llr(L[j, s], L[j + branch_size, s])
                else:
                    L[j, s + 1] = _lower_llr(L[j, s], L[j - branch_size, s], B[j - branch_size, s + 1])

    def _update_bits(self, B, l):
        if l < self.N // 2:
            return
        for s in range(self.n, self.n - _active_bit_level(l, self.n), -1):
            block_size = 2 ** s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                    B[j, s - 1] = B[j, s]

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = []
        L0 = np.full((self.N, self.n + 1), np.nan, dtype=np.float64)
        B0 = np.zeros((self.N, self.n + 1), dtype=np.int8)
        L0[:, 0] = llr_ch[self.rev]
        paths.append({"pm": 0.0, "L": L0, "B": B0})

        for i in range(self.N):
            l = _bit_reversed_scalar(i, self.n)
            candidates = []
            for path in paths:
                self._update_llrs(path["L"], path["B"], l)
                llr_bit = path["L"][l, self.n]
                if l in self.frozen_set:
                    pm = self._pm_add(path["pm"], llr_bit, 0)
                    path["B"][l, self.n] = 0
                    path["pm"] = pm
                    self._update_bits(path["B"], l)
                    candidates.append(path)
                else:
                    for u in (0, 1):
                        Lc = path["L"].copy()
                        Bc = path["B"].copy()
                        Bc[l, self.n] = u
                        pm = self._pm_add(path["pm"], llr_bit, u)
                        self._update_bits(Bc, l)
                        candidates.append({"pm": pm, "L": Lc, "B": Bc})

            candidates.sort(key=lambda p: p["pm"])
            paths = candidates[: self.L]

        if self.crc_length > 0:
            valid = []
            for p in paths:
                payload = p["B"][:, self.n][self.info_indices]
                if crc_check(payload, self.crc_length):
                    valid.append(p)
            if valid:
                paths = valid

        best = min(paths, key=lambda p: p["pm"])
        return best["B"][:, self.n].astype(int), best["pm"]


def scl_equiv_sc_test(N=64, K=32, seed=1):
    from construction import ga_construction
    from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma
    from encoder import polar_encode

    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False
    sigma = eb_n0_to_sigma(4.0, K / N)
    rng = np.random.default_rng(seed)
    scl = SCLDecoder(N, frozen_bits, list_size=1)
    for _ in range(20):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        llr = compute_llr(awgn_channel(bpsk_modulate(polar_encode(u)), sigma, rng), sigma)
        u_sc = sc_decode(llr, frozen_bits)
        u_scl, _ = scl.decode(llr)
        if not np.array_equal(u_sc, u_scl):
            return False
    return True
