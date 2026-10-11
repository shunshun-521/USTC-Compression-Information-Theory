"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np
from decoder_sc import f_operation, g_operation

_CRC_POLY = {8: 0x07, 16: 0x8005}


def _crc_remainder(bits, poly, order):
    reg = 0
    for b in bits:
        reg ^= int(b) << (order - 1)
        for _ in range(8):
            if reg & (1 << (order - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << order) - 1)
            else:
                reg = (reg << 1) & ((1 << order) - 1)
    return reg


def crc_encode(info_bits, crc_length=8):
    info_bits = np.asarray(info_bits, dtype=int).ravel()
    if crc_length not in _CRC_POLY:
        raise ValueError("crc_length must be 8 or 16")
    poly = _CRC_POLY[crc_length]
    rem = _crc_remainder(info_bits, poly, crc_length)
    crc_bits = np.array([(rem >> i) & 1 for i in range(crc_length - 1, -1, -1)], dtype=int)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=int).ravel()
    if crc_length not in _CRC_POLY:
        raise ValueError("crc_length must be 8 or 16")
    return _crc_remainder(bits, _CRC_POLY[crc_length], crc_length) == 0


def _partial_up(u_seg):
    """与 SC 递归返回的 u_up 一致的 stage 组合。"""
    u_seg = np.asarray(u_seg, dtype=int)
    if len(u_seg) == 1:
        return u_seg, u_seg.astype(np.float64)
    half = len(u_seg) // 2
    u1, up1 = _partial_up(u_seg[:half])
    u2, up2 = _partial_up(u_seg[half:])
    u = np.concatenate([u1, u2])
    up_left = (up1.astype(int) ^ up2.astype(int)).astype(np.float64)
    up = np.concatenate([up_left, up2])
    return u, up


def _bit_llr(phi, llr, u_prefix, frozen_bits):
    """计算第 phi 个比特的 SC LLR（已知 u_prefix[0:phi]）。"""
    N = len(llr)

    def walk(seg_llr, offset, length):
        if length == 1:
            return float(seg_llr[0])
        half = length // 2
        if phi < offset + half:
            left_llr = f_operation(seg_llr[:half], seg_llr[half:])
            return walk(left_llr, offset, half)
        u_left = u_prefix[offset : offset + half]
        _, u_up = _partial_up(u_left)
        right_llr = g_operation(seg_llr[:half], seg_llr[half:], u_up)
        return walk(right_llr, offset + half, half)

    return walk(llr, 0, N)


class SCLDecoder:
    """SCL 译码器（按比特顺序分裂路径）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0, info_indices=None):
        self.N = N
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.L = int(list_size)
        self.crc_length = int(crc_length)
        self.info_indices = (
            None if info_indices is None else np.asarray(info_indices, dtype=int)
        )

    def _pm_add(self, pm, llr, u):
        v = 0 if llr >= 0 else 1
        return pm + (0.0 if u == v else abs(llr))

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [{"pm": 0.0, "u": np.zeros(self.N, dtype=int)}]

        for phi in range(self.N):
            new_paths = []
            for st in paths:
                llr_bit = _bit_llr(phi, llr_ch, st["u"], self.frozen_bits)
                if self.frozen_bits[phi]:
                    pm = self._pm_add(st["pm"], llr_bit, 0)
                    u2 = st["u"].copy()
                    new_paths.append({"pm": pm, "u": u2})
                else:
                    for bit in (0, 1):
                        pm = self._pm_add(st["pm"], llr_bit, bit)
                        u2 = st["u"].copy()
                        u2[phi] = bit
                        new_paths.append({"pm": pm, "u": u2})
            new_paths.sort(key=lambda p: p["pm"])
            paths = new_paths[: self.L]

        best = min(paths, key=lambda p: p["pm"])
        if self.crc_length > 0:
            valid = []
            for p in paths:
                bits = (
                    p["u"][self.info_indices]
                    if self.info_indices is not None
                    else p["u"]
                )
                if crc_check(bits, self.crc_length):
                    valid.append(p)
            if valid:
                best = min(valid, key=lambda p: p["pm"])
        return best["u"], best["pm"]
