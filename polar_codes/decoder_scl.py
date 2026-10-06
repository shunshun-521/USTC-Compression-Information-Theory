"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from decoder_sc import (
    active_bit_level,
    active_llr_level,
    bit_reversed_int,
    f_operation,
    g_operation,
)
def _crc_poly(crc_length):
    if crc_length == 8:
        return 0x07
    if crc_length == 16:
        return 0x8005
    raise ValueError("crc_length must be 8 or 16")


def crc_encode(info_bits, crc_length=8):
    info_bits = np.asarray(info_bits, dtype=np.int8)
    poly = _crc_poly(crc_length)
    reg = 0
    for bit in info_bits:
        msb = (reg >> (crc_length - 1)) & 1
        reg = (reg << 1) & ((1 << crc_length) - 1)
        if bit ^ msb:
            reg ^= poly
    crc_bits = np.array([(reg >> i) & 1 for i in range(crc_length - 1, -1, -1)], dtype=np.int8)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=np.int8)
    poly = _crc_poly(crc_length)
    reg = 0
    for bit in bits:
        msb = (reg >> (crc_length - 1)) & 1
        reg = (reg << 1) & ((1 << crc_length) - 1)
        if bit ^ msb:
            reg ^= poly
    return reg == 0


def _pm_update(pm, llr, u):
    u_hard = 0 if llr >= 0 else 1
    if u == u_hard:
        return pm
    return pm + abs(llr)


def _update_llrs_path(L, B, l, n):
    for s in range(n - active_llr_level(l, n), n):
        block_size = 1 << (s + 1)
        branch_size = block_size // 2
        for j in range(l, L.shape[0], block_size):
            if j % block_size < branch_size:
                L[j, s + 1] = f_operation(L[j, s], L[j + branch_size, s])
            else:
                top_bit = B[j - branch_size, s + 1]
                L[j, s + 1] = g_operation(L[j - branch_size, s], L[j, s], top_bit)


def _update_bits_path(B, l, n, N):
    if l < N // 2:
        return
    for s in range(n, n - active_bit_level(l, n), -1):
        block_size = 1 << s
        branch_size = block_size // 2
        for j in range(l, -1, -block_size):
            if j % block_size >= branch_size:
                B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                B[j, s - 1] = B[j, s]


class SCLDecoder:
    """SCL 译码器"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.info_indices = np.where(~self.frozen_bits)[0]

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N = self.N
        n = self.n

        paths = [
            {
                "L": np.full((N, n + 1), np.nan, dtype=np.float64),
                "B": np.full((N, n + 1), np.nan),
                "pm": 0.0,
            }
        ]
        paths[0]["L"][:, 0] = llr_ch

        for i in range(N):
            l = bit_reversed_int(i, n)
            new_paths = []
            for path in paths:
                _update_llrs_path(path["L"], path["B"], l, n)
                llr_leaf = path["L"][l, n]
                if l in self.frozen_set:
                    cand = dict(path)
                    cand["L"] = path["L"].copy()
                    cand["B"] = path["B"].copy()
                    cand["B"][l, n] = 0
                    cand["pm"] = _pm_update(cand["pm"], llr_leaf, 0)
                    _update_bits_path(cand["B"], l, n, N)
                    new_paths.append(cand)
                else:
                    for u_bit in (0, 1):
                        cand = dict(path)
                        cand["L"] = path["L"].copy()
                        cand["B"] = path["B"].copy()
                        cand["B"][l, n] = u_bit
                        cand["pm"] = _pm_update(cand["pm"], llr_leaf, u_bit)
                        _update_bits_path(cand["B"], l, n, N)
                        new_paths.append(cand)
            new_paths.sort(key=lambda p: p["pm"])
            paths = new_paths[: self.list_size]

        best = paths[0]
        if self.crc_length > 0:
            ok = []
            for p in paths:
                u_hat = p["B"][:, n].astype(np.int8)
                info_bits = u_hat[self.info_indices]
                if crc_check(info_bits, self.crc_length):
                    ok.append(p)
            if ok:
                best = min(ok, key=lambda p: p["pm"])

        u_hat = best["B"][:, n].astype(np.int8)
        return u_hat, best["pm"]
