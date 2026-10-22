"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from decoder_sc import f_operation, g_operation, sc_decode_recursive

_CRC8_POLY = 0x07
_CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, reg_bits):
    reg = 0
    for b in bits:
        reg <<= 1
        reg |= int(b)
        mask = 1 << reg_bits
        if reg & mask:
            reg ^= poly << 1
            reg &= mask - 1
    return reg


def crc_encode(info_bits, crc_length=8):
    info_bits = np.asarray(info_bits, dtype=int).ravel()
    if crc_length == 8:
        poly, r = _CRC8_POLY, 8
    elif crc_length == 16:
        poly, r = _CRC16_POLY, 16
    else:
        raise ValueError("crc_length must be 8 or 16")
    rem = _crc_remainder(info_bits, poly, r)
    crc_bits = np.array([(rem >> (r - 1 - i)) & 1 for i in range(r)], dtype=int)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=int).ravel()
    if crc_length == 8:
        poly, r = _CRC8_POLY, 8
    elif crc_length == 16:
        poly, r = _CRC16_POLY, 16
    else:
        raise ValueError("crc_length must be 8 or 16")
    return _crc_remainder(bits, poly, r) == 0


def _leaf_llr(llr_ch, frozen_ind, u_prefix, phi):
    """在 u_prefix[0:phi] 已确定时，计算位 phi 的叶 LLR（内部 LLR 约定）"""
    llr = -np.asarray(llr_ch, dtype=np.float64)
    frozen = np.asarray(frozen_ind, dtype=np.float64)

    def walk(node, fr, offset, target):
        n = fr.size
        if n == 1:
            return node[0]
        h = n // 2
        l1, l2 = node[:h], node[h:]
        f1, f2 = fr[:h], fr[h:]
        up = f_operation(l1, l2)
        if target < offset + h:
            return walk(up, f1, offset, target)
        u_seg = u_prefix[offset : offset + h].astype(np.float64)
        low = g_operation(l1, l2, u_seg)
        return walk(low, f2, offset + h, target)

    return walk(llr, frozen, 0, phi)


def _pm_penalty(llr_leaf, u_bit):
    hard = 1 if llr_leaf < 0 else 0
    if llr_leaf == 0:
        hard = 1
    return 0.0 if u_bit == hard else abs(llr_leaf)


class SCLDecoder:
    """SCL 译码器"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        if N & (N - 1):
            raise ValueError("N must be power of 2")
        self.N = N
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_ind = self.frozen_bits.astype(np.float64)
        self.list_size = list_size
        self.crc_length = crc_length

    def decode(self, llr_ch):
        if self.list_size == 1:
            u = sc_decode_recursive(llr_ch, self.frozen_bits)
            return u, 0.0

        paths = [(0.0, np.zeros(self.N, dtype=int))]
        for phi in range(self.N):
            new_paths = []
            for pm, u in paths:
                if self.frozen_bits[phi]:
                    leaf = _leaf_llr(llr_ch, self.frozen_ind, u, phi)
                    u2 = u.copy()
                    u2[phi] = 0
                    new_paths.append((pm + _pm_penalty(leaf, 0), u2))
                else:
                    for bit in (0, 1):
                        u2 = u.copy()
                        u2[phi] = bit
                        leaf = _leaf_llr(llr_ch, self.frozen_ind, u2, phi)
                        new_paths.append((pm + _pm_penalty(leaf, bit), u2))
            new_paths.sort(key=lambda x: x[0])
            paths = new_paths[: self.list_size]

        if self.crc_length > 0:
            valid = [(pm, u) for pm, u in paths if crc_check(u, self.crc_length)]
            if valid:
                paths = valid

        pm, u_hat = paths[0]
        return u_hat.astype(int), float(pm)
