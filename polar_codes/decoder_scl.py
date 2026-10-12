"""
极化码 SCL（串行抵消列表）译码器，支持 CRC 辅助（CA-SCL）
"""
import numpy as np

from decoder_sc import (
    _SCD,
    active_bit_level,
    active_llr_level,
    bit_reversed,
    lower_llr,
    upper_llr,
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
    mask = (1 << crc_length) - 1
    reg = 0
    for b in info_bits:
        reg ^= int(b) << (crc_length - 1)
        if reg & (1 << (crc_length - 1)):
            reg = ((reg << 1) ^ poly) & mask
        else:
            reg = (reg << 1) & mask
    crc_bits = np.array(
        [(reg >> i) & 1 for i in range(crc_length - 1, -1, -1)], dtype=np.int8
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=np.int8)
    return np.array_equal(bits[-crc_length:], crc_encode(bits[:-crc_length], crc_length)[-crc_length:])


class SCLDecoder:
    """SCL 译码器（路径复制）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.frozen = set(np.where(self.frozen_bits == 1)[0])
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(self.frozen_bits == 0)[0]

    def _update_llrs(self, L, B, l):
        for s in range(self.n - active_llr_level(l, self.n), self.n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = upper_llr(L[j, s], L[j + branch_size, s])
                else:
                    L[j, s + 1] = lower_llr(
                        L[j, s], L[j - branch_size, s], int(B[j - branch_size, s + 1])
                    )

    def _update_bits(self, B, l):
        if l < self.N / 2:
            return
        for s in range(self.n, self.n - active_bit_level(l, self.n), -1):
            block_size = 2 ** s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                    B[j, s - 1] = B[j, s]

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [
            {
                "pm": 0.0,
                "L": np.full((self.N, self.n + 1), np.nan, dtype=np.float64),
                "B": np.full((self.N, self.n + 1), np.nan),
            }
        ]
        paths[0]["L"][:, 0] = llr_ch

        for i in range(self.N):
            l = bit_reversed(i, self.n)
            is_frozen = l in self.frozen
            new_paths = []

            for path in paths:
                self._update_llrs(path["L"], path["B"], l)
                llr = path["L"][l, self.n]
                if is_frozen:
                    cand = [(path["pm"] + (abs(llr) if llr < 0 else 0.0), path, 0)]
                else:
                    cand = []
                    for u in (0, 1):
                        pen = 0.0 if (u == 0 and llr >= 0) or (u == 1 and llr < 0) else abs(llr)
                        cand.append((path["pm"] + pen, path, u))

                for pm, parent, u in cand:
                    child = {
                        "pm": pm,
                        "L": parent["L"].copy(),
                        "B": parent["B"].copy(),
                    }
                    child["B"][l, self.n] = u
                    self._update_bits(child["B"], l)
                    new_paths.append(child)

            new_paths.sort(key=lambda p: p["pm"])
            paths = new_paths[: self.list_size]

        paths.sort(key=lambda p: p["pm"])
        if self.crc_length > 0:
            for p in paths:
                u_hat = p["B"][:, self.n].astype(int)
                payload = u_hat[self.info_indices]
                if crc_check(payload, self.crc_length):
                    return u_hat, p["pm"]

        best = paths[0]
        return best["B"][:, self.n].astype(int), best["pm"]
