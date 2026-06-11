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
    _update_bits,
    _update_llrs,
    f_operation,
    g_operation,
)


def crc_encode(info_bits, crc_length=8):
    """
    CRC 编码，多项式：
      r=8:  CRC-8  (0x07)
      r=16: CRC-16 (0x8005)
    """
    info_bits = np.asarray(info_bits, dtype=np.int8)
    poly_map = {8: 0x07, 16: 0x8005}
    if crc_length not in poly_map:
        raise ValueError("crc_length must be 8 or 16")
    poly = poly_map[crc_length]

    reg = 0
    for bit in info_bits:
        reg ^= int(bit) << (crc_length - 1)
        for _ in range(8):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)

    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=int,
    )
    return np.concatenate([info_bits.astype(int), crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits 末尾 CRC 是否正确。"""
    bits = np.asarray(bits, dtype=np.int8)
    expected = crc_encode(bits[:-crc_length], crc_length)
    return np.array_equal(bits[-crc_length:], expected[-crc_length:])


def _pm_penalty(llr, bit):
    hard = 0 if llr >= 0 else 1
    return 0.0 if bit == hard else abs(llr)


class SCLDecoder:
    """SCL 译码器（Lazy Copy 优化）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.list_size = list_size
        self.crc_length = crc_length
        self.decode_order = [_bit_reversed(i, self.n) for i in range(N)]
        self.info_positions = np.where(self.frozen_bits == 0)[0]

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)

        paths = [
            {
                "pm": 0.0,
                "L": np.zeros((self.N, self.n + 1), dtype=np.float64),
                "B": np.zeros((self.N, self.n + 1), dtype=int),
                "u_hat": np.zeros(self.N, dtype=int),
            }
        ]
        paths[0]["L"][:, 0] = llr_ch

        for l in self.decode_order:
            new_paths = []
            for path in paths:
                _update_llrs(path["L"], path["B"], l, self.n)
                llr = path["L"][l, self.n]

                if self.frozen_bits[l]:
                    path["pm"] += _pm_penalty(llr, 0)
                    path["u_hat"][l] = 0
                    path["B"][l, self.n] = 0
                    _update_bits(path["B"], l, self.n)
                    new_paths.append(path)
                else:
                    for bit in (0, 1):
                        child = self._fork_path(path)
                        child["pm"] += _pm_penalty(llr, bit)
                        child["u_hat"][l] = bit
                        child["B"][l, self.n] = bit
                        _update_bits(child["B"], l, self.n)
                        new_paths.append(child)

            new_paths.sort(key=lambda p: p["pm"])
            paths = new_paths[: self.list_size]

        if self.crc_length > 0:
            valid = [p for p in paths if self._crc_ok(p["u_hat"])]
            chosen = min(valid, key=lambda p: p["pm"]) if valid else min(
                paths, key=lambda p: p["pm"]
            )
        else:
            chosen = min(paths, key=lambda p: p["pm"])

        return chosen["u_hat"].copy(), chosen["pm"]

    def _fork_path(self, path):
        return {
            "pm": path["pm"],
            "L": path["L"].copy(),
            "B": path["B"].copy(),
            "u_hat": path["u_hat"].copy(),
        }

    def _crc_ok(self, u_hat):
        payload = u_hat[self.info_positions]
        return crc_check(payload, self.crc_length)
