"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np

from encoder import bit_reversed_index
from decoder_sc import (
    active_bit_level,
    active_llr_level,
    f_operation,
    g_operation,
)

CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, crc_length):
    mask = (1 << crc_length) - 1
    reg = 0
    for b in bits:
        reg ^= int(b) << (crc_length - 1)
        for _ in range(1):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & mask
            else:
                reg = (reg << 1) & mask
    return reg


def crc_encode(info_bits, crc_length=8):
    info_bits = np.asarray(info_bits, dtype=np.int8)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    padded = np.concatenate([info_bits, np.zeros(crc_length, dtype=np.int8)])
    rem = _crc_remainder(padded, poly, crc_length)
    crc_bits = np.array(
        [(rem >> i) & 1 for i in range(crc_length - 1, -1, -1)], dtype=np.int8
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=np.int8)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    return _crc_remainder(bits, poly, crc_length) == 0


class SCLDecoder:
    """SCL 译码器（路径复制）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_positions = np.where(self.frozen == 0)[0]

    def _branch_penalty(self, llr, u):
        u_hard = 0 if llr >= 0 else 1
        return 0.0 if u == u_hard else abs(llr)

    def _update_llrs(self, L, B, l):
        n, N = self.n, self.N
        for s in range(n - active_llr_level(l, n), n):
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(l, N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = f_operation(L[j, s], L[j + branch_size, s])
                else:
                    L[j, s + 1] = g_operation(
                        L[j - branch_size, s], L[j, s], B[j - branch_size, s + 1]
                    )

    def _update_bits(self, B, l):
        n, N = self.n, self.N
        if l < N // 2:
            return
        for s in range(n, n - active_bit_level(l, n), -1):
            block_size = 1 << s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    B[j - branch_size, s - 1] = int(B[j, s]) ^ int(
                        B[j - branch_size, s]
                    )
                    B[j, s - 1] = B[j, s]

    def decode(self, llr_ch):
        n, N = self.n, self.N
        frozen = self.frozen

        L0 = np.zeros((N, n + 1), dtype=np.float64)
        B0 = np.zeros((N, n + 1), dtype=np.float64)
        L0[:, 0] = np.asarray(llr_ch, dtype=np.float64)
        paths = [{"pm": 0.0, "L": L0, "B": B0}]

        for i in range(N):
            l = bit_reversed_index(i, n)
            candidates = []
            for path in paths:
                self._update_llrs(path["L"], path["B"], l)
                llr_bit = path["L"][l, n]
                if frozen[l]:
                    path["pm"] += self._branch_penalty(llr_bit, 0)
                    path["B"][l, n] = 0
                    self._update_bits(path["B"], l)
                    candidates.append((path["pm"], path))
                else:
                    for u in (0, 1):
                        new_L = path["L"].copy()
                        new_B = path["B"].copy()
                        pm = path["pm"] + self._branch_penalty(llr_bit, u)
                        new_B[l, n] = u
                        self._update_bits(new_B, l)
                        candidates.append((pm, {"pm": pm, "L": new_L, "B": new_B}))

            candidates.sort(key=lambda x: x[0])
            paths = []
            for pm, p in candidates[: self.list_size]:
                p["pm"] = pm
                paths.append(p)

        best = paths[0]
        if self.crc_length > 0:
            ok_paths = [
                p
                for p in paths
                if crc_check(
                    p["B"][:, n].astype(np.int8)[self.info_positions], self.crc_length
                )
            ]
            if ok_paths:
                best = min(ok_paths, key=lambda p: p["pm"])

        u_hat = best["B"][:, n].astype(np.int8)
        return u_hat, best["pm"]
