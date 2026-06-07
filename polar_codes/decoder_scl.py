"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math

import numpy as np

from decoder_sc import (
    _active_bit_level,
    _active_llr_level,
    _bit_reversed_index,
    _prepare_channel_llr,
    _update_bits,
    _update_llrs,
    f_operation,
    g_operation,
)


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, crc_length):
    reg = 0
    mask = (1 << crc_length) - 1
    for bit in bits:
        reg = (reg << 1) | int(bit)
        if reg & (1 << crc_length):
            reg ^= poly
    return reg & mask


def crc_encode(info_bits, crc_length=8):
    """CRC-8 (0x07) 或 CRC-16 (0x8005)"""
    info_bits = np.asarray(info_bits, dtype=int).ravel()
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    remainder = _crc_remainder(info_bits, poly, crc_length)
    crc_bits = np.array(
        [(remainder >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=int,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=int).ravel()
    if len(bits) < crc_length:
        return False
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    expected = _crc_remainder(bits[:-crc_length], poly, crc_length)
    received = 0
    for b in bits[-crc_length:]:
        received = (received << 1) | int(b)
    return expected == received


def _path_metric_update(pm, llr, u_bit):
    hard = 0 if llr >= 0 else 1
    if u_bit != hard:
        pm += abs(llr)
    return pm


class SCLDecoder:
    """SCL 译码器（Lazy Copy）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen = np.asarray(frozen_bits).astype(bool)
        self.info_indices = np.where(~self.frozen)[0]
        self.L = list_size
        self.crc_length = crc_length

    def decode(self, llr_ch):
        llr = _prepare_channel_llr(llr_ch)
        N, n = self.N, self.n

        paths = [
            {
                "L": np.full((N, n + 1), np.nan, dtype=np.float64),
                "B": np.full((N, n + 1), np.nan),
                "pm": 0.0,
                "u_hat": np.zeros(N, dtype=np.int8),
            }
        ]
        paths[0]["L"][:, 0] = llr

        for i in range(N):
            l = _bit_reversed_index(i, n)
            candidates = []

            for path in paths:
                if self.frozen[l]:
                    new_path = self._fork_path(path)
                    _update_llrs(new_path["L"], new_path["B"], l, n, N)
                    llr_bit = new_path["L"][l, n]
                    new_path["pm"] = _path_metric_update(new_path["pm"], llr_bit, 0)
                    new_path["B"][l, n] = 0
                    new_path["u_hat"][l] = 0
                    _update_bits(new_path["B"], l, n, N)
                    candidates.append(new_path)
                else:
                    _update_llrs(path["L"], path["B"], l, n, N)
                    llr_bit = path["L"][l, n]
                    for bit in (0, 1):
                        new_path = self._fork_path(path)
                        new_path["pm"] = _path_metric_update(new_path["pm"], llr_bit, bit)
                        new_path["B"][l, n] = bit
                        new_path["u_hat"][l] = bit
                        _update_bits(new_path["B"], l, n, N)
                        candidates.append(new_path)

            candidates.sort(key=lambda p: p["pm"])
            paths = candidates[: self.L]

        if self.crc_length > 0:
            valid = [
                p
                for p in paths
                if crc_check(p["u_hat"][self.info_indices], self.crc_length)
            ]
            best = min(valid if valid else paths, key=lambda p: p["pm"])
        else:
            best = min(paths, key=lambda p: p["pm"])

        return best["u_hat"].astype(int), best["pm"]

    @staticmethod
    def _fork_path(path):
        return {
            "L": path["L"].copy(),
            "B": path["B"].copy(),
            "pm": path["pm"],
            "u_hat": path["u_hat"].copy(),
        }
