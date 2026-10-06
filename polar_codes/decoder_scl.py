"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math

import numpy as np

from decoder_sc import (
    _active_bit_level,
    _active_llr_level,
    _bit_reversed,
    _lower_llr,
    _upper_llr,
    sc_decode_scd,
)

_CRC_POLY = {8: 0x07, 16: 0x8005}


def _crc_shift(crc, bit, crc_length, poly):
    msb = (crc >> (crc_length - 1)) & 1
    crc = (crc << 1) & ((1 << crc_length) - 1)
    if msb ^ int(bit):
        crc ^= poly
    return crc


def crc_encode(info_bits, crc_length=8):
    info_bits = np.asarray(info_bits, dtype=np.int8).ravel()
    if crc_length not in _CRC_POLY:
        raise ValueError("crc_length must be 8 or 16")
    poly = _CRC_POLY[crc_length]
    crc = 0
    for b in info_bits:
        crc = _crc_shift(crc, b, crc_length, poly)
    crc_bits = np.array(
        [(crc >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=np.int8
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=np.int8).ravel()
    if crc_length not in _CRC_POLY:
        raise ValueError("crc_length must be 8 or 16")
    poly = _CRC_POLY[crc_length]
    crc = 0
    for b in bits:
        crc = _crc_shift(crc, b, crc_length, poly)
    return crc == 0


def _pm_add(pm, llr, ubit):
    v = 0 if llr >= 0 else 1
    return pm + (0.0 if ubit == v else abs(llr))


def _update_llrs_path(L, B, l, n):
    for s in range(n - _active_llr_level(l, n), n):
        block_size = 2 ** (s + 1)
        branch_size = block_size // 2
        for j in range(l, L.shape[0], block_size):
            if j % block_size < branch_size:
                L[j, s + 1] = _upper_llr(L[j, s], L[j + branch_size, s])
            else:
                L[j, s + 1] = _lower_llr(
                    L[j, s], L[j - branch_size, s], int(B[j - branch_size, s + 1])
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
    """SCL 译码器（路径复制，L 较小时可用）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=np.int8)
        self.frozen = set(np.where(self.frozen_bits != 0)[0])
        self.L = max(1, int(list_size))
        self.crc_length = int(crc_length)
        self.info_pos = np.where(self.frozen_bits == 0)[0]

    def decode(self, llr_ch):
        if self.L == 1 and self.crc_length == 0:
            return sc_decode_scd(llr_ch, self.frozen_bits), 0.0

        llr = np.asarray(llr_ch, dtype=np.float64)
        N, n = self.N, self.n

        paths = [
            {
                "pm": 0.0,
                "L": np.full((N, n + 1), np.nan, dtype=np.float64),
                "B": np.zeros((N, n + 1)),
                "u": np.zeros(N, dtype=np.int8),
            }
        ]
        paths[0]["L"][:, 0] = llr

        for i in range(N):
            l = _bit_reversed(i, n)
            new_paths = []
            for path in paths:
                _update_llrs_path(path["L"], path["B"], l, n)
                cur_llr = path["L"][l, n]
                if l in self.frozen:
                    pm = _pm_add(path["pm"], cur_llr, 0)
                    child = {
                        "pm": pm,
                        "L": path["L"].copy(),
                        "B": path["B"].copy(),
                        "u": path["u"].copy(),
                    }
                    child["B"][l, n] = 0
                    child["u"][l] = 0
                    _update_bits_path(child["B"], l, n, N)
                    new_paths.append(child)
                else:
                    for ubit in (0, 1):
                        pm = _pm_add(path["pm"], cur_llr, ubit)
                        child = {
                            "pm": pm,
                            "L": path["L"].copy(),
                            "B": path["B"].copy(),
                            "u": path["u"].copy(),
                        }
                        child["B"][l, n] = ubit
                        child["u"][l] = ubit
                        _update_bits_path(child["B"], l, n, N)
                        new_paths.append(child)

            new_paths.sort(key=lambda p: p["pm"])
            paths = new_paths[: self.L]

        if self.crc_length > 0:
            passed = [
                p
                for p in paths
                if crc_check(p["u"][self.info_pos], self.crc_length)
            ]
            best = min(passed if passed else paths, key=lambda p: p["pm"])
        else:
            best = min(paths, key=lambda p: p["pm"])

        return best["u"].astype(int), best["pm"]
