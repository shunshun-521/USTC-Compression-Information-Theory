"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np

from decoder_sc import active_bit_level, active_llr_level, lower_llr, upper_llr
from encoder import bit_reversal_permutation


_CRC_POLY_BITS = {
    8: [1, 0, 0, 0, 0, 0, 1, 1, 1],  # CRC-8: x^8 + x^2 + x + 1
    16: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1],  # CRC-16-IBM
}


def _crc_mod2(msg_bits, poly_bits):
    msg = [int(b) for b in msg_bits]
    r = len(poly_bits) - 1
    for i in range(len(msg) - r):
        if msg[i] == 1:
            for j, p in enumerate(poly_bits):
                msg[i + j] ^= p
    return msg[-r:]


def crc_encode(info_bits, crc_length=8):
    poly = _CRC_POLY_BITS[crc_length]
    padded = np.concatenate(
        [np.asarray(info_bits, dtype=np.int8), np.zeros(crc_length, dtype=np.int8)]
    )
    remainder = _crc_mod2(padded, poly)
    return np.concatenate([np.asarray(info_bits, dtype=np.int8), np.array(remainder, dtype=np.int8)])


def crc_check(bits, crc_length=8):
    poly = _CRC_POLY_BITS[crc_length]
    remainder = _crc_mod2(bits, poly)
    return all(b == 0 for b in remainder)


class SCLDecoder:
    """SCL 译码器（置换 SC + 路径复制）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.br = bit_reversal_permutation(N)

    def _new_path(self, llr_ch):
        path = {
            "pm": 0.0,
            "L": np.zeros((self.N, self.n + 1), dtype=np.float64),
            "B": np.zeros((self.N, self.n + 1), dtype=np.float64),
        }
        path["L"][:, 0] = llr_ch[self.br]
        return path

    def _copy_path(self, path):
        return {
            "pm": path["pm"],
            "L": path["L"].copy(),
            "B": path["B"].copy(),
        }

    def _update_llrs(self, path, l):
        for s in range(self.n - active_llr_level(l, self.n), self.n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    path["L"][j, s + 1] = upper_llr(
                        path["L"][j, s], path["L"][j + branch_size, s]
                    )
                else:
                    path["L"][j, s + 1] = lower_llr(
                        path["L"][j, s],
                        path["L"][j - branch_size, s],
                        int(path["B"][j - branch_size, s + 1]),
                    )

    def _update_bits(self, path, l, bit):
        path["B"][l, self.n] = bit
        if l < self.N // 2:
            return
        for s in range(self.n, self.n - active_bit_level(l, self.n), -1):
            block_size = 2 ** s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    path["B"][j - branch_size, s - 1] = int(path["B"][j, s]) ^ int(
                        path["B"][j - branch_size, s]
                    )
                    path["B"][j, s - 1] = path["B"][j, s]

    def _pm_penalty(self, llr, bit):
        hard = 0 if llr >= 0 else 1
        return 0.0 if bit == hard else abs(llr)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [self._new_path(llr_ch)]

        for phi in range(self.N):
            l = int(self.br[phi])
            candidates = []
            for path in paths:
                self._update_llrs(path, l)
                llr = path["L"][l, self.n]

                if self.frozen_bits[l]:
                    new_path = self._copy_path(path)
                    new_path["pm"] += self._pm_penalty(llr, 0)
                    self._update_bits(new_path, l, 0)
                    candidates.append(new_path)
                else:
                    for bit in (0, 1):
                        new_path = self._copy_path(path)
                        new_path["pm"] += self._pm_penalty(llr, bit)
                        self._update_bits(new_path, l, bit)
                        candidates.append(new_path)

            candidates.sort(key=lambda p: p["pm"])
            paths = candidates[: self.list_size]

        if self.crc_length > 0:
            valid = [
                p
                for p in paths
                if crc_check(p["B"][:, self.n].astype(np.int8), self.crc_length)
            ]
            chosen = valid[0] if valid else paths[0]
        else:
            chosen = paths[0]

        u_hat = chosen["B"][:, self.n].astype(np.int8)
        return u_hat, chosen["pm"]
