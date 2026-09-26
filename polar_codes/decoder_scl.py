"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math

import numpy as np

from encoder import bit_reversal_permutation
from decoder_sc import f_operation, g_operation, _sc_decode_core


def _crc_poly(crc_length):
    if crc_length == 8:
        return 0x07
    if crc_length == 16:
        return 0x8005
    raise ValueError("crc_length must be 8 or 16")


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=np.int_)
    poly = _crc_poly(crc_length)
    mask = (1 << crc_length) - 1
    reg = 0
    for b in info_bits:
        reg ^= int(b) << (crc_length - 1)
        for _ in range(1):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & mask
            else:
                reg = (reg << 1) & mask
    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=np.int_
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。"""
    bits = np.asarray(bits, dtype=np.int_)
    if len(bits) < crc_length:
        return False
    expected = crc_encode(bits[:-crc_length], crc_length)
    return np.array_equal(expected, bits)


def _pm_penalty(llr, u):
    u_hard = 0 if llr >= 0 else 1
    return 0.0 if u == u_hard else abs(llr)


def _scl_decode_core(llr, frozen_bits, list_size):
    """基于 SC 树的列表译码（与 SC 核心一致的 u_up 更新）。"""
    llr = np.asarray(llr, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr)

    def decode_list(llr_node, frozen_node):
        n = len(llr_node)
        if n == 1:
            llr0 = llr_node[0]
            if frozen_node[0]:
                return [(np.array([0], dtype=np.int_), np.array([0.0]), 0.0)]
            u0 = 0 if llr0 >= 0 else 1
            return [
                (np.array([0], dtype=np.int_), np.array([0.0]), _pm_penalty(llr0, 0)),
                (np.array([1], dtype=np.int_), np.array([1.0]), _pm_penalty(llr0, 1)),
            ]

        half = n // 2
        llr1 = llr_node[:half]
        llr2 = llr_node[half:]
        f1 = frozen_node[:half]
        f2 = frozen_node[half:]

        llr_u = f_operation(llr1, llr2)
        left_paths = decode_list(llr_u, f1)

        merged = []
        for u1, u1_up, pm1 in left_paths:
            llr_l = g_operation(llr1, llr2, u1_up)
            right_paths = decode_list(llr_l, f2)
            for u2, u2_up, pm2 in right_paths:
                u = np.concatenate([u1, u2])
                u1_up_re = np.bitwise_xor(u1_up.astype(np.int_), u2_up.astype(np.int_)).astype(
                    np.float64
                )
                u_up = np.concatenate([u1_up_re, u2_up])
                merged.append((u, u_up, pm1 + pm2))

        if not frozen_node.any():
            merged.sort(key=lambda x: x[2])
            merged = merged[:list_size]
        else:
            merged.sort(key=lambda x: x[2])
            merged = merged[:list_size]
        return merged

    return decode_list(llr, frozen_bits)


class SCLDecoder:
    """SCL 译码器。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]
        self._rev = bit_reversal_permutation(N)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)[self._rev]
        if self.list_size == 1:
            u_hat = _sc_decode_core(llr_ch, self.frozen_bits)
            return u_hat, 0.0

        paths = _scl_decode_core(llr_ch, self.frozen_bits, self.list_size)
        candidates = []
        for u_hat, _, pm in paths:
            candidates.append((pm, u_hat))

        candidates.sort(key=lambda x: x[0])
        if self.crc_length > 0:
            valid = [
                (pm, u)
                for pm, u in candidates
                if crc_check(u[self.info_indices], self.crc_length)
            ]
            pm, u_hat = valid[0] if valid else candidates[0]
        else:
            pm, u_hat = candidates[0]

        return u_hat.copy(), pm
