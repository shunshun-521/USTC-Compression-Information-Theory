"""
极化码 SCL（串行抵消列表）译码器，支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np

from decoder_utils import (
    active_bit_level,
    active_llr_level,
    bit_reversed_index,
    frozen_indices_from_mask,
    lower_llr,
    upper_llr,
)

CRC_POLYS = {8: 0x07, 16: 0x8005}


def crc_encode(info_bits, crc_length=8):
    info_bits = np.asarray(info_bits, dtype=int).ravel()
    poly = CRC_POLYS[crc_length]
    mask = (1 << crc_length) - 1
    top = 1 << (crc_length - 1)
    reg = 0
    for b in info_bits:
        reg ^= int(b) << (crc_length - 1)
        if reg & top:
            reg = ((reg << 1) ^ poly) & mask
        else:
            reg = (reg << 1) & mask
    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=int).ravel()
    if len(bits) < crc_length:
        return False
    return np.array_equal(bits[-crc_length:], crc_encode(bits[:-crc_length], crc_length)[-crc_length:])


class SCLDecoder:
    """SCL 译码器（路径级复制 L/B 数组）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.list_size = list_size
        self.crc_length = crc_length
        self.frozen_set = frozen_indices_from_mask(frozen_bits)
        self.info_indices = np.array(
            sorted(set(range(N)) - self.frozen_set), dtype=int
        )

    def _pm_add(self, pm, llr, u):
        u_hard = 0 if llr >= 0 else 1
        return pm + (0.0 if u == u_hard else abs(llr))

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n, N = self.n, self.N

        paths = [
            {
                "pm": 0.0,
                "L": np.full((N, n + 1), np.nan, dtype=np.float64),
                "B": np.full((N, n + 1), np.nan),
                "u": np.zeros(N, dtype=int),
            }
        ]
        paths[0]["L"][:, 0] = llr_ch

        for phase in range(N):
            l = bit_reversed_index(phase, n)
            new_paths = []

            for path in paths:
                L, B = path["L"], path["B"]
                for s in range(n - active_llr_level(l, n), n):
                    block_size = 2 ** (s + 1)
                    branch_size = block_size // 2
                    for j in range(l, N, block_size):
                        if j % block_size < branch_size:
                            L[j, s + 1] = upper_llr(L[j, s], L[j + branch_size, s])
                        else:
                            tb = int(B[j - branch_size, s + 1])
                            L[j, s + 1] = lower_llr(L[j, s], L[j - branch_size, s], tb)

                llr = L[l, n]
                if l in self.frozen_set:
                    p2 = {
                        "pm": self._pm_add(path["pm"], llr, 0),
                        "L": L.copy(),
                        "B": B.copy(),
                        "u": path["u"].copy(),
                    }
                    p2["u"][l] = 0
                    p2["B"][l, n] = 0
                    self._update_bits(p2["B"], l, n, N)
                    new_paths.append(p2)
                else:
                    for u in (0, 1):
                        p2 = {
                            "pm": self._pm_add(path["pm"], llr, u),
                            "L": L.copy(),
                            "B": B.copy(),
                            "u": path["u"].copy(),
                        }
                        p2["u"][l] = u
                        p2["B"][l, n] = u
                        self._update_bits(p2["B"], l, n, N)
                        new_paths.append(p2)

            new_paths.sort(key=lambda p: p["pm"])
            paths = new_paths[: self.list_size]

        if self.crc_length > 0:
            valid = []
            for p in paths:
                bits = p["u"][self.info_indices]
                if crc_check(bits, self.crc_length):
                    valid.append(p)
            best = min(valid if valid else paths, key=lambda p: p["pm"])
        else:
            best = min(paths, key=lambda p: p["pm"])

        return best["u"], best["pm"]

    @staticmethod
    def _update_bits(B, l, n, N):
        if l < N / 2:
            return
        for s in range(n, n - active_bit_level(l, n), -1):
            block_size = 2 ** s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                    B[j, s - 1] = B[j, s]
