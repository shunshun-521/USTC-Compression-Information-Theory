"""
极化码 SCL（串行抵消列表）译码器，支持 CRC 辅助
"""
import copy

import numpy as np

from decoder_sc import sc_decode
from vendor_SCD import SCD


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, crc_len):
    mask = (1 << crc_len) - 1
    reg = 0
    for b in bits:
        reg ^= int(b) << (crc_len - 1)
        if reg & (1 << (crc_len - 1)):
            reg = ((reg << 1) ^ poly) & mask
        else:
            reg = (reg << 1) & mask
    return reg


def crc_encode(info_bits, crc_length=8):
    info_bits = np.asarray(info_bits, dtype=int).ravel()
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    rem = _crc_remainder(info_bits, poly, crc_length)
    crc_bits = np.array(
        [(rem >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=int).ravel()
    if len(bits) < crc_length:
        return False
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    return _crc_remainder(bits, poly, crc_length) == 0


class _PC:
    def __init__(self, N, n, frozen, likelihoods):
        self.N = N
        self.n = n
        self.frozen = frozen
        self.likelihoods = likelihoods


def _pm_update(pm, llr, u):
    u_hard = 0 if llr >= 0 else 1
    if u != u_hard:
        pm += abs(llr)
    return pm


class SCLDecoder:
    """SCL 译码器（路径级复制）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.L = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        if self.L == 1:
            u_hat = sc_decode(llr_ch, self.frozen_bits)
            return u_hat, 0.0

        from vendor_utils import bit_reversed
        from vendor_decoder_utils import active_bit_level, active_llr_level, hard_decision
        from vendor_decoder_utils import lower_llr, upper_llr

        N, n = self.N, self.n
        paths = [{"pm": 0.0, "L": np.full((N, n + 1), np.nan), "B": np.full((N, n + 1), np.nan)}]
        paths[0]["L"][:, 0] = llr_ch

        def update_llrs(state, l):
            L, B = state["L"], state["B"]
            for s in range(n - active_llr_level(l, n), n):
                block_size = 2 ** (s + 1)
                branch_size = block_size // 2
                for j in range(l, N, block_size):
                    if j % block_size < branch_size:
                        L[j, s + 1] = upper_llr(L[j, s], L[j + branch_size, s])
                    else:
                        top_bit = int(B[j - branch_size, s + 1])
                        L[j, s + 1] = lower_llr(L[j, s], L[j - branch_size, s], top_bit)

        def update_bits(state, l):
            if l < N / 2:
                return
            L, B = state["L"], state["B"]
            for s in range(n, n - active_bit_level(l, n), -1):
                block_size = 2 ** s
                branch_size = block_size // 2
                for j in range(l, -1, -block_size):
                    if j % block_size >= branch_size:
                        B[j - branch_size, s - 1] = int(B[j, s]) ^ int(
                            B[j - branch_size, s]
                        )
                        B[j, s - 1] = B[j, s]

        for i in range(N):
            l_idx = bit_reversed(i, n)
            new_paths = []
            for state in paths:
                update_llrs(state, l_idx)
                llr = state["L"][l_idx, n]
                if l_idx in self.frozen_set:
                    st = copy.deepcopy(state)
                    st["pm"] = _pm_update(st["pm"], llr, 0)
                    st["B"][l_idx, n] = 0
                    update_bits(st, l_idx)
                    new_paths.append(st)
                else:
                    for u in (0, 1):
                        st = copy.deepcopy(state)
                        st["pm"] = _pm_update(st["pm"], llr, u)
                        st["B"][l_idx, n] = u
                        update_bits(st, l_idx)
                        new_paths.append(st)
            new_paths.sort(key=lambda s: s["pm"])
            paths = new_paths[: self.L]

        paths.sort(key=lambda s: s["pm"])
        if self.crc_length > 0:
            for st in paths:
                u_hat = st["B"][:, n].astype(int)
                if crc_check(u_hat[self.info_indices], self.crc_length):
                    return u_hat, st["pm"]
        best = paths[0]
        return best["B"][:, n].astype(int), best["pm"]
