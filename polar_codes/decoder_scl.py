"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np

from decoder_sc import (
    bit_reversed,
    f_operation,
    g_operation,
    _active_bit_level,
    _active_llr_level,
)
from encoder import bit_reversal_permutation

CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, crc_length):
    mask = (1 << crc_length) - 1
    top = 1 << (crc_length - 1)
    reg = 0
    for b in bits:
        reg ^= int(b) << (crc_length - 1)
        if reg & top:
            reg = ((reg << 1) ^ poly) & mask
        else:
            reg = (reg << 1) & mask
    return reg


def crc_encode(info_bits, crc_length=8):
    info_bits = np.asarray(info_bits, dtype=np.int8)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    remainder = _crc_remainder(info_bits, poly, crc_length)
    crc_bits = np.array(
        [(remainder >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=np.int8,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=np.int8)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    return _crc_remainder(bits, poly, crc_length) == 0


def _update_llrs_path(L, B, l, n, N):
    for s in range(n - _active_llr_level(l, n), n):
        block_size = 2 ** (s + 1)
        branch_size = block_size // 2
        for j in range(l, N, block_size):
            if j % block_size < branch_size:
                L[j, s + 1] = f_operation(L[j, s], L[j + branch_size, s])
            else:
                L[j, s + 1] = g_operation(
                    L[j - branch_size, s], L[j, s], B[j - branch_size, s + 1]
                )


def _update_bits_path(B, l, n, N):
    if l < N / 2:
        return
    for s in range(n, n - _active_bit_level(l, n), -1):
        block_size = 2 ** s
        branch_size = block_size // 2
        for j in range(l, -1, -block_size):
            if j % block_size >= branch_size:
                B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                B[j, s - 1] = B[j, s]


class SCLDecoder:
    """SCL 译码器（路径复制在信息位分支时进行）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]

    def _pm_add(self, pm, llr, u):
        u_hard = 0 if llr >= 0 else 1
        return pm + (0.0 if u == u_hard else abs(llr))

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        llr_ch = llr_ch[bit_reversal_permutation(self.N)]
        N, n = self.N, self.n

        paths = [
            {
                "pm": 0.0,
                "L": np.zeros((N, n + 1), dtype=np.float64),
                "B": np.zeros((N, n + 1), dtype=np.int8),
            }
        ]
        paths[0]["L"][:, 0] = llr_ch

        for i in range(N):
            l = bit_reversed(i, n)
            new_paths = []
            for path in paths:
                _update_llrs_path(path["L"], path["B"], l, n, N)
                llr = path["L"][l, n]
                if self.frozen_bits[l]:
                    pm = self._pm_add(path["pm"], llr, 0)
                    p2 = {
                        "pm": pm,
                        "L": path["L"].copy(),
                        "B": path["B"].copy(),
                    }
                    p2["B"][l, n] = 0
                    _update_bits_path(p2["B"], l, n, N)
                    new_paths.append(p2)
                else:
                    for u in (0, 1):
                        pm = self._pm_add(path["pm"], llr, u)
                        p2 = {
                            "pm": pm,
                            "L": path["L"].copy(),
                            "B": path["B"].copy(),
                        }
                        p2["B"][l, n] = u
                        _update_bits_path(p2["B"], l, n, N)
                        new_paths.append(p2)

            new_paths.sort(key=lambda p: p["pm"])
            paths = new_paths[: self.list_size]

        best = min(paths, key=lambda p: p["pm"])
        u_hat = best["B"][:, n].astype(int)

        if self.crc_length > 0:
            valid = [
                p
                for p in paths
                if crc_check(p["B"][:, n][self.info_indices], self.crc_length)
            ]
            if valid:
                best = min(valid, key=lambda p: p["pm"])
                u_hat = best["B"][:, n].astype(int)

        return u_hat, best["pm"]
