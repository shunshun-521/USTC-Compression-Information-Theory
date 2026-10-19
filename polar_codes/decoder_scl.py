"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np
from decoder_sc import f_operation, g_operation, sc_decode

_CRC8_POLY = 0x07
_CRC16_POLY = 0x8005


def _crc_bits_remainder(bits, poly, crc_len):
    reg = 0
    mask = (1 << crc_len) - 1
    top = 1 << (crc_len - 1)
    for b in bits:
        reg ^= int(b) << (crc_len - 1)
        for _ in range(8):
            if reg & top:
                reg = ((reg << 1) ^ poly) & mask
            else:
                reg = (reg << 1) & mask
    return reg


def crc_encode(info_bits, crc_length=8):
    info_bits = np.asarray(info_bits, dtype=int).ravel()
    poly = _CRC8_POLY if crc_length == 8 else _CRC16_POLY
    rem = _crc_bits_remainder(info_bits, poly, crc_length)
    crc_bits = np.array(
        [(rem >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=int).ravel()
    poly = _CRC8_POLY if crc_length == 8 else _CRC16_POLY
    return _crc_bits_remainder(bits, poly, crc_length) == 0


def _merge_paths(left_pm, left_bits, right_pm, right_bits):
    pm = left_pm + right_pm
    bits = left_bits + right_bits
    return pm, bits


def _scl_decode_recursive(llr, frozen_bits, list_size):
    """递归 SCL 译码，返回 (pm, bits) 列表。"""
    llr = np.asarray(llr, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr)
    n = int(math.log2(N))

    def decode_node(llr_node, depth, bit_offset):
        length = len(llr_node)
        if length == 1:
            idx = bit_offset
            L = llr_node[0]
            paths = []
            if frozen_bits[idx]:
                penalty = abs(L) if L < 0 else 0.0
                paths.append((penalty, [0]))
            else:
                hard = 0 if L >= 0 else 1
                paths.append((0.0 if hard == 0 else abs(L), [0]))
                paths.append((0.0 if hard == 1 else abs(L), [1]))
            paths.sort(key=lambda x: x[0])
            return paths[:list_size]

        half = length // 2
        L1 = llr_node[:half]
        L2 = llr_node[half:]
        left_llr = f_operation(L1, L2)
        left_paths = decode_node(left_llr, depth + 1, bit_offset)

        merged = []
        for pm_l, bits_l in left_paths:
            u_left = np.array(bits_l, dtype=int)
            right_llr = g_operation(L1, L2, u_left)
            right_paths = decode_node(right_llr, depth + 1, bit_offset + half)
            for pm_r, bits_r in right_paths:
                pm, bits = _merge_paths(pm_l, bits_l, pm_r, bits_r)
                merged.append((pm, bits))

        merged.sort(key=lambda x: x[0])
        return merged[:list_size]

    return decode_node(llr, 0, 0)


class SCLDecoder:
    """SCL 译码器（递归列表实现）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.list_size = list_size
        self.crc_length = crc_length

    def decode(self, llr_ch):
        paths = _scl_decode_recursive(
            llr_ch, self.frozen_bits.astype(bool), self.list_size
        )
        candidates = []
        for pm, bits in paths:
            u_hat = np.zeros(self.N, dtype=int)
            u_hat[: len(bits)] = bits
            candidates.append((pm, u_hat))

        if self.crc_length > 0:
            valid = [c for c in candidates if crc_check(c[1], self.crc_length)]
            if valid:
                best = min(valid, key=lambda x: x[0])
            else:
                best = min(candidates, key=lambda x: x[0])
        else:
            best = min(candidates, key=lambda x: x[0])

        return best[1], best[0]


def scl_equivalent_to_sc(N, frozen_bits, llr_ch):
    u_scl, _ = SCLDecoder(N, frozen_bits, list_size=1, crc_length=0).decode(llr_ch)
    u_sc = sc_decode(llr_ch, frozen_bits)
    return np.array_equal(u_scl, u_sc)
