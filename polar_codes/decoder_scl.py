"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import copy
import math

import numpy as np

from decoder_sc import (
    _active_bit_level,
    _active_llr_level,
    _bit_reversed,
    _update_bits,
    _update_llrs,
)


def crc_encode(info_bits, crc_length=8):
    """CRC-8 (0x07) 或 CRC-16 (0x8005)"""
    info_bits = np.asarray(info_bits, dtype=int).ravel()
    if crc_length == 8:
        poly = 0x07
    elif crc_length == 16:
        poly = 0x8005
    else:
        raise ValueError("crc_length must be 8 or 16")

    reg = 0
    for b in info_bits:
        reg ^= int(b) << (crc_length - 1)
        for _ in range(crc_length):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)

    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=int).ravel()
    if len(bits) < crc_length:
        return False
    payload = bits[:-crc_length]
    expected = crc_encode(payload, crc_length)[-crc_length:]
    return np.array_equal(bits[-crc_length:], expected)


class SCLDecoder:
    """SCL 译码器（置换 SC + 路径复制）"""

    def __init__(
        self,
        N,
        frozen_bits,
        list_size=4,
        crc_length=0,
        info_indices=None,
    ):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.L = list_size
        self.crc_length = crc_length
        self.info_indices = (
            np.asarray(info_indices, dtype=int)
            if info_indices is not None
            else None
        )

    def _pm_update(self, pm, llr, u):
        hard = 0 if llr >= 0 else 1
        if u != hard:
            pm += abs(llr)
        return pm

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N = self.N
        n = self.n

        paths = []
        L0 = np.zeros((N, n + 1), dtype=np.float64)
        B0 = np.zeros((N, n + 1), dtype=np.int8)
        L0[:, 0] = llr_ch
        paths.append({"L": L0, "B": B0, "pm": 0.0, "u_hat": np.zeros(N, dtype=int)})

        for i in range(N):
            l = _bit_reversed(i, n)
            for p in paths:
                _update_llrs(p["L"], p["B"], l, n)

            llr_leaf = paths[0]["L"][l, n]
            new_paths = []

            if self.frozen_bits[i]:
                for p in paths:
                    cp = copy.deepcopy(p)
                    cp["pm"] = self._pm_update(cp["pm"], cp["L"][l, n], 0)
                    cp["u_hat"][i] = 0
                    cp["B"][l, n] = 0
                    _update_bits(cp["B"], l, n)
                    new_paths.append(cp)
            else:
                for p in paths:
                    for u in (0, 1):
                        cp = copy.deepcopy(p)
                        cp["pm"] = self._pm_update(cp["pm"], cp["L"][l, n], u)
                        cp["u_hat"][i] = u
                        cp["B"][l, n] = u
                        _update_bits(cp["B"], l, n)
                        new_paths.append(cp)

            new_paths.sort(key=lambda x: x["pm"])
            paths = new_paths[: self.L]

        paths.sort(key=lambda x: x["pm"])
        if self.crc_length > 0 and self.info_indices is not None:
            for p in paths:
                bits = p["u_hat"][self.info_indices]
                if crc_check(bits, self.crc_length):
                    return p["u_hat"], p["pm"]

        best = paths[0]
        return best["u_hat"], best["pm"]
