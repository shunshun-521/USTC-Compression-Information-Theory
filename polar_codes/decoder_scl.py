"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np

from decoder_sc import (
    f_operation,
    g_operation,
    _align_channel_llr,
    partial_summation,
    sc_decode_recursive,
)


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_step(reg, bit, poly, crc_length):
    mask = (1 << crc_length) - 1
    msb = 1 << (crc_length - 1)
    reg ^= int(bit) << (crc_length - 1)
    reg &= mask
    if reg & msb:
        reg = ((reg << 1) ^ poly) & mask
    else:
        reg = (reg << 1) & mask
    return reg


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 并附加到信息比特后（系统型）"""
    info_bits = np.asarray(info_bits, dtype=int).ravel()
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    reg = 0
    for b in info_bits:
        reg = _crc_step(reg, b, poly, crc_length)
    for _ in range(crc_length):
        reg = _crc_step(reg, 0, poly, crc_length)
    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits 是否通过 CRC"""
    bits = np.asarray(bits, dtype=int).ravel()
    if len(bits) < crc_length:
        return False
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    reg = 0
    for b in bits:
        reg = _crc_step(reg, b, poly, crc_length)
    return reg == 0


def _llr_at_index(llr, u_hat, phi):
    """在已知 u_hat[0:phi] 时计算比特 phi 的 LLR"""

    def walk(node, bit_off, n):
        if n == 1:
            return float(node[0])
        half = n // 2
        if phi < bit_off + half:
            left = f_operation(node[:half], node[half:])
            return walk(left, bit_off, half)
        u_left = u_hat[bit_off : bit_off + half]
        c_left = partial_summation(u_left)
        right = g_operation(node[:half], node[half:], c_left)
        return walk(right, bit_off + half, half)

    return walk(np.asarray(llr, dtype=np.float64), 0, len(llr))


class _Path:
    __slots__ = ("pm", "u_hat")

    def __init__(self, N):
        self.pm = 0.0
        self.u_hat = np.zeros(N, dtype=np.int8)


class SCLDecoder:
    """SCL 译码器（路径列表 + 路径度量）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.m = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits).astype(bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]

    @staticmethod
    def _pm_penalty(llr_val, u_bit):
        hard = 0 if llr_val >= 0 else 1
        return 0.0 if u_bit == hard else abs(llr_val)

    def decode(self, llr_ch):
        if self.list_size == 1 and self.crc_length == 0:
            u_hat = sc_decode_recursive(llr_ch, self.frozen_bits)
            return u_hat.astype(int), 0.0

        llr = _align_channel_llr(llr_ch)
        paths = [_Path(self.N)]

        for phi in range(self.N):
            candidates = []
            for path in paths:
                llr_phi = _llr_at_index(llr, path.u_hat, phi)
                if self.frozen_bits[phi]:
                    new_p = _Path(self.N)
                    new_p.u_hat = path.u_hat.copy()
                    new_p.pm = path.pm + self._pm_penalty(llr_phi, 0)
                    new_p.u_hat[phi] = 0
                    candidates.append(new_p)
                else:
                    for u in (0, 1):
                        new_p = _Path(self.N)
                        new_p.u_hat = path.u_hat.copy()
                        new_p.pm = path.pm + self._pm_penalty(llr_phi, u)
                        new_p.u_hat[phi] = u
                        candidates.append(new_p)

            candidates.sort(key=lambda p: p.pm)
            paths = candidates[: self.list_size]

        if self.crc_length > 0:
            valid = [
                p
                for p in paths
                if crc_check(p.u_hat[self.info_indices], self.crc_length)
            ]
            best = min(valid, key=lambda p: p.pm) if valid else min(paths, key=lambda p: p.pm)
        else:
            best = min(paths, key=lambda p: p.pm)

        return best.u_hat.astype(int), best.pm
