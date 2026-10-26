"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np

from decoder_sc import f_operation, sc_decode_recursive, _prepare_llr


def _crc_poly(crc_length):
    if crc_length == 8:
        return 0x07
    if crc_length == 16:
        return 0x8005
    raise ValueError("crc_length must be 8 or 16")


def crc_encode(info_bits, crc_length=8):
    poly = _crc_poly(crc_length)
    reg = 0
    bits = np.asarray(info_bits, dtype=np.int8)
    for b in bits:
        reg ^= int(b) << (crc_length - 1)
        for _ in range(8):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)
    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=np.int8
    )
    return np.concatenate([bits, crc_bits])


def crc_check(bits, crc_length=8):
    if crc_length == 0:
        return True
    poly = _crc_poly(crc_length)
    reg = 0
    for b in np.asarray(bits, dtype=np.int8):
        reg ^= int(b) << (crc_length - 1)
        for _ in range(8):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)
    return reg == 0


def _leaf_llr(y, depth, node, target, u_prefix, n):
    """在已知 u_prefix[0:target] 时，求节点 target 的 LLR（标量）"""
    if depth == n - 1:
        return float(y[0])
    half = len(y) // 2
    l1, l2 = y[:half], y[half:]
    left = f_operation(l1, l2)
    if target < 2 * node + half:
        return _leaf_llr(left, depth + 1, 2 * node, target, u_prefix, n)
    u_left = u_prefix[2 * node : 2 * node + half]
    right = [
        l2[i] + (1 - 2 * int(u_left[i])) * l1[i] for i in range(half)
    ]
    return _leaf_llr(right, depth + 1, 2 * node + 1, target, u_prefix, n)


class SCLDecoder:
    """SCL 译码器（逐比特顺序列表，Lazy Copy u 向量）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N)) + 1
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.L = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]

    def _penalty(self, llr, u):
        hard = 0 if llr >= 0 else 1
        return 0.0 if int(u) == hard else abs(llr)

    def decode(self, llr_ch):
        llr_ch = _prepare_llr(llr_ch)
        if self.L <= 1:
            u = sc_decode_recursive(llr_ch, self.frozen_bits)
            return u.astype(int), 0.0

        paths = [(0.0, np.zeros(self.N, dtype=np.int8))]
        for phi in range(self.N):
            new_paths = []
            for pm, u in paths:
                llr = _leaf_llr(llr_ch, 0, 0, phi, u, self.n)
                if self.frozen_bits[phi]:
                    new_paths.append((pm + self._penalty(llr, 0), u.copy()))
                    new_paths[-1][1][phi] = 0
                else:
                    for bit in (0, 1):
                        u2 = u.copy()
                        u2[phi] = bit
                        new_paths.append((pm + self._penalty(llr, bit), u2))
            new_paths.sort(key=lambda p: p[0])
            paths = new_paths[: self.L]

        paths.sort(key=lambda p: p[0])
        best = paths[0]
        if self.crc_length > 0:
            for p in paths:
                if crc_check(p[1][self.info_indices], self.crc_length):
                    best = p
                    break
        return best[1].astype(int), float(best[0])
