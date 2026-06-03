"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np

from decoder_sc import (
    _SCDCore,
    active_bit_level,
    active_llr_level,
    bit_reversed_index,
    hard_decision,
    lower_llr,
    upper_llr,
)

CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def crc_encode(info_bits, crc_length=8):
    info_bits = np.asarray(info_bits, dtype=int).ravel()
    if crc_length == 8:
        poly = CRC8_POLY
        rem = 0
        for b in info_bits:
            rem ^= int(b) << 7
            for _ in range(8):
                if rem & 0x80:
                    rem = ((rem << 1) ^ poly) & 0xFF
                else:
                    rem = (rem << 1) & 0xFF
        crc_bits = np.array([(rem >> (7 - i)) & 1 for i in range(8)], dtype=int)
    elif crc_length == 16:
        poly = CRC16_POLY
        rem = 0
        for b in info_bits:
            rem ^= int(b) << 15
            for _ in range(16):
                if rem & 0x8000:
                    rem = ((rem << 1) ^ poly) & 0xFFFF
                else:
                    rem = (rem << 1) & 0xFFFF
        crc_bits = np.array([(rem >> (15 - i)) & 1 for i in range(16)], dtype=int)
    else:
        raise ValueError("crc_length must be 8 or 16")
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=int).ravel()
    return np.array_equal(bits, crc_encode(bits[:-crc_length], crc_length))


def _pm_add(pm, llr, bit):
    hard = hard_decision(llr)
    if bit != hard:
        pm += abs(llr)
    return pm


class SCLDecoder:
    """SCL 译码器（基于 Permuted SC 的 L/B 树）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.list_size = list_size
        self.crc_length = crc_length
        frozen_bits = np.asarray(frozen_bits)
        self.frozen = set(int(i) for i in np.where(frozen_bits.astype(bool))[0])
        self.info_indices = np.where(~frozen_bits.astype(bool))[0]

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N, n, L_max = self.N, self.n, self.list_size

        paths = [
            {
                "L": np.full((N, n + 1), np.nan, dtype=np.float64),
                "B": np.zeros((N, n + 1), dtype=int),
                "pm": 0.0,
            }
        ]
        paths[0]["L"][:, 0] = llr_ch

        for i in range(N):
            l = bit_reversed_index(i, n)
            new_paths = []
            for path in paths:
                self._update_llrs(path, l)
                llr_leaf = path["L"][l, n]
                if l in self.frozen:
                    bit = 0
                    pm = _pm_add(path["pm"], llr_leaf, bit)
                    p2 = self._clone_path(path)
                    p2["B"][l, n] = bit
                    p2["pm"] = pm
                    self._update_bits(p2, l)
                    new_paths.append(p2)
                else:
                    for bit in (0, 1):
                        pm = _pm_add(path["pm"], llr_leaf, bit)
                        p2 = self._clone_path(path)
                        p2["B"][l, n] = bit
                        p2["pm"] = pm
                        self._update_bits(p2, l)
                        new_paths.append(p2)

            new_paths.sort(key=lambda p: p["pm"])
            paths = new_paths[:L_max]

        best = min(paths, key=lambda p: p["pm"])
        if self.crc_length > 0:
            valid = []
            for p in paths:
                u = p["B"][:, n].astype(int)
                bits = u[self.info_indices]
                if crc_check(bits, self.crc_length):
                    valid.append(p)
            if valid:
                best = min(valid, key=lambda p: p["pm"])

        u_hat = best["B"][:, n].astype(int)
        return u_hat, best["pm"]

    @staticmethod
    def _clone_path(path):
        return {
            "L": path["L"].copy(),
            "B": path["B"].copy(),
            "pm": path["pm"],
        }

    def _update_llrs(self, path, l):
        L = path["L"]
        B = path["B"]
        for s in range(self.n - active_llr_level(l, self.n), self.n):
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = upper_llr(L[j, s], L[j + branch_size, s])
                else:
                    top_bit = B[j - branch_size, s + 1]
                    L[j, s + 1] = lower_llr(L[j, s], L[j - branch_size, s], top_bit)

    def _update_bits(self, path, l):
        if l < self.N // 2:
            return
        B = path["B"]
        for s in range(self.n, self.n - active_bit_level(l, self.n), -1):
            block_size = 1 << s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                    B[j, s - 1] = B[j, s]
