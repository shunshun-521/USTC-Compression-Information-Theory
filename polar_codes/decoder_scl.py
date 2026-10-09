"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math

import numpy as np

from decoder_sc import f_operation, g_operation, sc_decode

_CRC8_POLY = 0x07
_CRC16_POLY = 0x8005


def _crc_remainder(info_bits, poly, crc_length):
    """按 MSB 先行、反射 CRC 计算余数（CRC-8: poly=0x07）。"""
    reg = 0
    mask = (1 << crc_length) - 1
    top = 1 << (crc_length - 1)
    stream = np.concatenate([info_bits, np.zeros(crc_length, dtype=np.int8)])
    for b in stream:
        reg ^= int(b) << (crc_length - 1)
        if reg & top:
            reg = ((reg << 1) ^ poly) & mask
        else:
            reg = (reg << 1) & mask
    return reg


def crc_encode(info_bits, crc_length=8):
    info_bits = np.asarray(info_bits, dtype=np.int8)
    poly = _CRC8_POLY if crc_length == 8 else _CRC16_POLY
    rem = _crc_remainder(info_bits, poly, crc_length)
    crc_bits = np.array(
        [(rem >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=np.int8
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=np.int8)
    poly = _CRC8_POLY if crc_length == 8 else _CRC16_POLY
    payload = bits[:-crc_length]
    expected = crc_encode(payload, crc_length)
    return np.array_equal(bits, expected)


def _path_metric(llr_leaf, bit):
    pred = 0 if llr_leaf >= 0 else 1
    return abs(llr_leaf) if bit != pred else 0.0


def _decode_return(y, depth, node, nv, n_depth):
    if depth == n_depth - 1:
        return np.array([nv[node]], dtype=np.int8)
    half = len(y) // 2
    L1, L2 = y[:half], y[half:]
    arr1 = _decode_return(f_operation(L1, L2), depth + 1, 2 * node, nv, n_depth)
    right = g_operation(L1, L2, arr1)
    arr2 = _decode_return(right, depth + 1, 2 * node + 1, nv, n_depth)
    return np.concatenate([(arr1 + arr2) % 2, arr2])


class SCLDecoder:
    """SCL 译码器（列表路径与 SC 递归结构一致）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n_depth = int(math.log2(N)) + 1
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = max(1, list_size)
        self.crc_length = crc_length

    def _scl_rec(self, y, depth, node, paths):
        if depth == self.n_depth - 1:
            expanded = []
            for nv, pm in paths:
                if self.frozen_bits[node]:
                    bit = 0
                    pm2 = pm + _path_metric(y[0], 0)
                    nv2 = nv.copy()
                    nv2[node] = bit
                    expanded.append((nv2, pm2))
                else:
                    for bit in (0, 1):
                        pm2 = pm + _path_metric(y[0], bit)
                        nv2 = nv.copy()
                        nv2[node] = bit
                        expanded.append((nv2, pm2))
            expanded.sort(key=lambda x: x[1])
            return expanded[: self.list_size]

        half = len(y) // 2
        L1, L2 = y[:half], y[half:]
        left_llr = f_operation(L1, L2)
        paths = self._scl_rec(left_llr, depth + 1, 2 * node, paths)
        merged = []
        for nv, pm in paths:
            arr1 = _decode_return(left_llr, depth + 1, 2 * node, nv, self.n_depth)
            right_llr = g_operation(L1, L2, arr1)
            sub = self._scl_rec(right_llr, depth + 1, 2 * node + 1, [(nv, pm)])
            merged.extend(sub)
        merged.sort(key=lambda x: x[1])
        return merged[: self.list_size]

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        if self.list_size == 1:
            u = sc_decode(llr_ch, self.frozen_bits)
            return u, 0.0

        paths = self._scl_rec(llr_ch, 0, 0, [(np.zeros(self.N, dtype=np.int8), 0.0)])
        info_positions = np.where(~self.frozen_bits)[0]

        if self.crc_length > 0:
            valid = []
            for nv, pm in paths:
                if crc_check(nv[info_positions], self.crc_length):
                    valid.append((nv, pm))
            if valid:
                best = min(valid, key=lambda x: x[1])
            else:
                best = paths[0]
        else:
            best = paths[0]

        return best[0].copy(), best[1]
