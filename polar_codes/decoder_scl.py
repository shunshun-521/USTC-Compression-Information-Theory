"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np

from decoder_sc import (
    _cn_op,
    _polar_decode_sc_recursive,
    _vn_op,
    sc_decode,
)


def _crc_poly(crc_length):
    if crc_length == 8:
        return 0x07
    if crc_length == 16:
        return 0x8005
    raise ValueError("crc_length must be 8 or 16")


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=np.int8)
    poly = _crc_poly(crc_length)
    mask = (1 << crc_length) - 1
    reg = 0
    for b in info_bits:
        reg ^= int(b) << (crc_length - 1)
        if reg & (1 << (crc_length - 1)):
            reg = ((reg << 1) ^ poly) & mask
        else:
            reg = (reg << 1) & mask
    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=np.int8
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC。"""
    bits = np.asarray(bits, dtype=np.int8)
    if crc_length == 0:
        return True
    poly = _crc_poly(crc_length)
    mask = (1 << crc_length) - 1
    reg = 0
    for b in bits:
        reg ^= int(b) << (crc_length - 1)
        if reg & (1 << (crc_length - 1)):
            reg = ((reg << 1) ^ poly) & mask
        else:
            reg = (reg << 1) & mask
    return reg == 0


def _llr_penalty(llr, u):
    u_hard = 0 if llr >= 0 else 1
    return 0.0 if u == u_hard else abs(llr)


def _llr_at_bit(phi, llr_ch, u_prefix, frozen_bits):
    """在已知 u_prefix[0:phi] 时，计算比特 phi 的 LLR。"""

    def rec(llr_seg, frozen_seg, offset):
        n = len(llr_seg)
        if n == 1:
            return float(llr_seg[0])
        half = n // 2
        llr1 = llr_seg[:half]
        llr2 = llr_seg[half:]
        f1 = frozen_seg[:half]
        f2 = frozen_seg[half:]
        if phi < offset + half:
            return rec(_cn_op(llr1, llr2), f1, offset)
        u_left = u_prefix[offset : offset + half].astype(int)
        u_left_up = u_left.copy()
        if half > 1:
            _, u_left_up = _polar_decode_sc_recursive(_cn_op(llr1, llr2), f1.astype(int))
        return rec(_vn_op(llr1, llr2, u_left_up), f2, offset + half)

    return rec(np.asarray(llr_ch, dtype=np.float64), np.asarray(frozen_bits, dtype=bool), 0)


class SCLDecoder:
    """SCL 译码器（路径列表 + 路径度量）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        if self.list_size == 1 and self.crc_length == 0:
            u = sc_decode(llr_ch, self.frozen_bits)
            return u, 0.0

        paths = [(np.zeros(self.N, dtype=int), 0.0)]
        for phi in range(self.N):
            new_paths = []
            for u_hat, pm in paths:
                llr = _llr_at_bit(phi, llr_ch, u_hat, self.frozen_bits)
                if self.frozen_bits[phi]:
                    u_new = u_hat.copy()
                    u_new[phi] = 0
                    new_paths.append((u_new, pm + _llr_penalty(llr, 0)))
                else:
                    for bit in (0, 1):
                        u_new = u_hat.copy()
                        u_new[phi] = bit
                        new_paths.append((u_new, pm + _llr_penalty(llr, bit)))
            new_paths.sort(key=lambda x: x[1])
            paths = new_paths[: self.list_size]

        if self.crc_length > 0:
            passed = []
            for u_hat, pm in paths:
                info_bits = u_hat[self.info_indices]
                if crc_check(info_bits, self.crc_length):
                    passed.append((u_hat, pm))
            if passed:
                paths = passed

        best_u, best_pm = min(paths, key=lambda x: x[1])
        return best_u, best_pm
