"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math

import numpy as np

from decoder_sc import f_operation, g_operation, sc_decode_recursive
from encoder import polar_encode_partial


def _crc_poly(crc_length):
    if crc_length == 8:
        return 0x07
    if crc_length == 16:
        return 0x8005
    raise ValueError(f"Unsupported CRC length: {crc_length}")


def crc_encode(info_bits, crc_length=8):
    """CRC-8 (0x07) / CRC-16 (0x8005), MSB-first。"""
    info_bits = np.asarray(info_bits, dtype=np.int8)
    poly = _crc_poly(crc_length)
    reg = 0
    for bit in info_bits:
        reg ^= int(bit) << (crc_length - 1)
        if reg & (1 << (crc_length - 1)):
            reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
        else:
            reg = (reg << 1) & ((1 << crc_length) - 1)
    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=np.int8
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    if crc_length == 0:
        return True
    expected = crc_encode(bits[:-crc_length], crc_length)
    return np.array_equal(bits[-crc_length:], expected[-crc_length:])


def _pm_update(pm, llr, u):
    hard = 0 if llr >= 0 else 1
    return pm + (0.0 if u == hard else abs(llr))


class SCLDecoder:
    """SCL 译码器（递归路径扩展，L=1 时等价于 SC）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        if self.list_size == 1 and self.crc_length == 0:
            return sc_decode_recursive(llr_ch, self.frozen_bits), 0.0

        paths = [{"u_hat": np.zeros(self.N, dtype=int), "pm": 0.0}]
        self._decode_paths(llr_ch, paths, 0, self.n)
        return self._select_best(paths)

    def _decode_paths(self, llr_node, paths, bit_offset, depth):
        m = len(llr_node)
        if m == 1:
            idx = bit_offset
            new_paths = []
            for path in paths:
                llr = llr_node[0]
                if self.frozen_bits[idx]:
                    p = {"u_hat": path["u_hat"].copy(), "pm": _pm_update(path["pm"], llr, 0)}
                    p["u_hat"][idx] = 0
                    new_paths.append(p)
                else:
                    for u in (0, 1):
                        p = {"u_hat": path["u_hat"].copy(), "pm": _pm_update(path["pm"], llr, u)}
                        p["u_hat"][idx] = u
                        new_paths.append(p)
            paths[:] = sorted(new_paths, key=lambda p: p["pm"])[: self.list_size]
            return

        half = m // 2
        llr_left = f_operation(llr_node[:half], llr_node[half:])
        self._decode_paths(llr_left, paths, bit_offset, depth - 1)

        expanded = []
        for path in paths:
            u_left = path["u_hat"][bit_offset : bit_offset + half]
            encoded_left = polar_encode_partial(u_left)
            llr_right = g_operation(llr_node[:half], llr_node[half:], encoded_left)
            expanded.append(
                {
                    "u_hat": path["u_hat"].copy(),
                    "pm": path["pm"],
                    "llr_right": llr_right,
                }
            )

        all_right = []
        for item in expanded:
            sub = [{"u_hat": item["u_hat"].copy(), "pm": item["pm"]}]
            self._decode_paths(item["llr_right"], sub, bit_offset + half, depth - 1)
            all_right.extend(sub)
        paths[:] = sorted(all_right, key=lambda p: p["pm"])[: self.list_size]

    def _select_best(self, paths):
        if self.crc_length > 0:
            valid = [
                p
                for p in paths
                if crc_check(p["u_hat"][self.info_indices], self.crc_length)
            ]
            pool = valid if valid else paths
        else:
            pool = paths
        best = min(pool, key=lambda p: p["pm"])
        return best["u_hat"].copy(), best["pm"]
