"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np

from decoder_sc import (
    active_bit_level,
    active_llr_level,
    bit_reversed,
    lower_llr,
    upper_llr,
    _prepare_llr,
    _frozen_indices,
)


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, crc_length):
    reg = 0
    for bit in bits:
        reg = (reg << 1) | int(bit)
        if reg & (1 << crc_length):
            reg ^= poly
    return reg & ((1 << crc_length) - 1)


def crc_encode(info_bits, crc_length=8):
    info_bits = np.asarray(info_bits, dtype=np.int8)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    msg = np.concatenate([info_bits, np.zeros(crc_length, dtype=np.int8)])
    remainder = _crc_remainder(msg, poly, crc_length)
    crc_bits = np.array(
        [(remainder >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=np.int8,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=np.int8)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    return _crc_remainder(bits, poly, crc_length) == 0


class SCLDecoder:
    """SCL 译码器（Lazy Copy 优化）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_set = _frozen_indices(frozen_bits)
        self.info_indices = np.where(~np.asarray(frozen_bits, dtype=bool))[0]
        self.list_size = list_size
        self.crc_length = crc_length

    def _path_metric_update(self, pm, llr, bit):
        hard = 0 if llr >= 0 else 1
        if bit != hard:
            pm += abs(llr)
        return pm

    def _update_llrs(self, L, B, l):
        for s in range(self.n - active_llr_level(l, self.n), self.n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = upper_llr(L[j, s], L[j + branch_size, s])
                else:
                    L[j, s + 1] = lower_llr(
                        L[j, s], L[j - branch_size, s], B[j - branch_size, s + 1]
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
        llr_ch = _prepare_llr(llr_ch, self.N)
        N = self.N
        n = self.n

        paths = [
            {
                "pm": 0.0,
                "L": np.full((N, n + 1), np.nan, dtype=np.float64),
                "B": np.full((N, n + 1), np.nan),
                "u_hat": np.zeros(N, dtype=int),
            }
        ]
        paths[0]["L"][:, 0] = llr_ch

        decode_order = [bit_reversed(i, n) for i in range(N)]

        for l in decode_order:
            new_paths = []

            if l in self.frozen_set:
                for path in paths:
                    self._update_llrs(path["L"], path["B"], l)
                    current_llr = path["L"][l, n]
                    bit = 0
                    path["pm"] = self._path_metric_update(path["pm"], current_llr, bit)
                    path["B"][l, n] = 0
                    path["u_hat"][l] = 0
                    self._update_bits(path["B"], l)
                    new_paths.append(path)
            else:
                for path in paths:
                    self._update_llrs(path["L"], path["B"], l)
                    current_llr = path["L"][l, n]
                    for bit in (0, 1):
                        child = {
                            "pm": self._path_metric_update(path["pm"], current_llr, bit),
                            "L": path["L"].copy(),
                            "B": path["B"].copy(),
                            "u_hat": path["u_hat"].copy(),
                        }
                        child["B"][l, n] = bit
                        child["u_hat"][l] = bit
                        self._update_bits(child["B"], l)
                        new_paths.append(child)

            new_paths.sort(key=lambda p: p["pm"])
            paths = new_paths[: self.list_size]

        if self.crc_length > 0:
            valid = []
            for i, path in enumerate(paths):
                info = path["u_hat"][self.info_indices]
                if crc_check(info, self.crc_length):
                    valid.append(i)
            if valid:
                best = min(valid, key=lambda i: paths[i]["pm"])
            else:
                best = int(np.argmin([p["pm"] for p in paths]))
        else:
            best = int(np.argmin([p["pm"] for p in paths]))

        return paths[best]["u_hat"], paths[best]["pm"]
