"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import copy
import math

import numpy as np

from decoder_sc import (
    f_operation,
    g_operation,
    _llr_to_bit,
    _LLR_MAX,
    sc_decode_recursive,
)

CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, crc_length):
    reg = 0
    for b in bits:
        reg ^= int(b) << (crc_length - 1)
        for _ in range(8):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)
    return reg


def crc_encode(info_bits, crc_length=8):
    info_bits = np.asarray(info_bits, dtype=int).ravel()
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    rem = _crc_remainder(info_bits, poly, crc_length)
    crc_bits = np.array(
        [(rem >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=int).ravel()
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    return _crc_remainder(bits, poly, crc_length) == 0


class _Path:
    __slots__ = ("pm", "u_hat", "_llr_cache")

    def __init__(self, N):
        self.pm = 0.0
        self.u_hat = np.zeros(N, dtype=int)
        self._llr_cache = {}


def _pm_penalty(llr, u):
    return 0.0 if u == _llr_to_bit(llr, False) else abs(llr)


def _upward_bits(u_hat_seg):
    n = len(u_hat_seg)
    if n == 1:
        return u_hat_seg.copy()
    half = n // 2
    up_l = _upward_bits(u_hat_seg[:half])
    up_r = _upward_bits(u_hat_seg[half:])
    return np.concatenate([np.bitwise_xor(up_l, up_r), up_r])


def _bit_llr(llr, frozen, u_prefix, bit_index, offset=0):
    """在已知 u_prefix[:bit_index] 时，计算 u[bit_index] 的 LLR"""
    n = len(llr)
    if n == 1:
        return float(llr[0])

    half = n // 2
    left_end = offset + half - 1
    if bit_index <= left_end:
        L_left = f_operation(llr[:half], llr[half:])
        return _bit_llr(L_left, frozen[:half], u_prefix, bit_index, offset)
    else:
        u_left = u_prefix[offset : offset + half]
        u_left_up = _upward_bits(u_left)
        L_right = g_operation(llr[:half], llr[half:], u_left_up)
        return _bit_llr(
            L_right, frozen[half:], u_prefix, bit_index, offset + half
        )


class SCLDecoder:
    """按比特顺序的 SCL 译码（每步保留 L 条路径）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length

    def decode(self, llr_ch):
        llr_ch = np.clip(np.asarray(llr_ch, dtype=np.float64), -_LLR_MAX, _LLR_MAX)
        if self.list_size == 1:
            return sc_decode_recursive(llr_ch, self.frozen_bits), 0.0

        paths = [_Path(self.N)]
        for phi in range(self.N):
            new_paths = []
            for path in paths:
                if phi not in path._llr_cache:
                    path._llr_cache[phi] = _bit_llr(
                        llr_ch, self.frozen_bits, path.u_hat, phi
                    )
                llr_phi = path._llr_cache[phi]
                if self.frozen_bits[phi]:
                    path.pm += _pm_penalty(llr_phi, 0)
                    path.u_hat[phi] = 0
                    new_paths.append(path)
                else:
                    for u in (0, 1):
                        p = copy.deepcopy(path)
                        p.pm += _pm_penalty(llr_phi, u)
                        p.u_hat[phi] = u
                        new_paths.append(p)
            new_paths.sort(key=lambda p: p.pm)
            paths = new_paths[: self.list_size]

        paths.sort(key=lambda p: p.pm)
        best = paths[0]

        if self.crc_length > 0:
            info_mask = ~self.frozen_bits
            for p in paths:
                if crc_check(p.u_hat[info_mask], self.crc_length):
                    best = p
                    break

        return best.u_hat.copy(), best.pm
