"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from encoder import bit_reversal_permutation
from decoder_sc import f_operation, g_operation, _permute_llr, sc_decode

CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, crc_length):
    reg = 0
    for b in bits:
        reg ^= (int(b) << (crc_length - 1))
        for _ in range(crc_length):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)
    return reg


def crc_encode(info_bits, crc_length=8):
    info_bits = np.asarray(info_bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    rem = _crc_remainder(info_bits, poly, crc_length)
    crc_bits = np.array(
        [(rem >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=int)
    if len(bits) < crc_length:
        return False
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    rem = _crc_remainder(bits[:-crc_length], poly, crc_length)
    expected = np.array(
        [(rem >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int
    )
    return np.array_equal(bits[-crc_length:], expected)


def _pm_penalty(llr_val, u_val):
    hard = 0 if llr_val >= 0 else 1
    return 0.0 if u_val == hard else abs(llr_val)


def _partial_up(u_seg):
    """计算子段部分和向量（用于 g 运算）。"""
    n = len(u_seg)
    if n == 1:
        return u_seg.copy()
    half = n // 2
    left_up = _partial_up(u_seg[:half])
    right_up = _partial_up(u_seg[half:])
    return np.concatenate([left_up ^ right_up, right_up])


def _get_llr_phi(llr, frozen, u_hat, phi, offset=0):
    """已知 u_hat[0:phi] 时计算比特 phi 的 LLR。"""
    n = len(llr)
    if n == 1:
        return float(llr[0])
    half = n // 2
    if phi < offset + half:
        llr_up = f_operation(llr[:half], llr[half:])
        return _get_llr_phi(llr_up, frozen[:half], u_hat, phi, offset)
    u_left_up = _partial_up(u_hat[offset:offset + half])
    llr_down = g_operation(llr[:half], llr[half:], u_left_up)
    return _get_llr_phi(llr_down, frozen[half:], u_hat, phi, offset + half)


class SCLDecoder:
    """SCL 译码器（串行逐比特，轻量路径）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_mask = ~self.frozen_bits
        self._llr_perm = None

    def decode(self, llr_ch):
        if self.list_size == 1 and self.crc_length == 0:
            u_hat = sc_decode(llr_ch, self.frozen_bits)
            return u_hat, 0.0

        llr_perm = _permute_llr(llr_ch)
        frozen = self.frozen_bits
        N = self.N
        L = self.list_size

        paths = [(np.zeros(N, dtype=int), 0.0)]

        for phi in range(N):
            candidates = []
            for u_hat, pm in paths:
                llr_phi = _get_llr_phi(llr_perm, frozen, u_hat, phi)
                if frozen[phi]:
                    u_new = u_hat.copy()
                    u_new[phi] = 0
                    candidates.append((u_new, pm + _pm_penalty(llr_phi, 0)))
                else:
                    for u_val in (0, 1):
                        u_new = u_hat.copy()
                        u_new[phi] = u_val
                        candidates.append((u_new, pm + _pm_penalty(llr_phi, u_val)))

            candidates.sort(key=lambda x: x[1])
            paths = candidates[:L]

        if self.crc_length > 0:
            crc_ok = []
            for u_hat, pm in paths:
                info_bits = u_hat[self.info_mask]
                if crc_check(info_bits, self.crc_length):
                    crc_ok.append((u_hat, pm))
            if crc_ok:
                best = min(crc_ok, key=lambda x: x[1])
                return best[0].copy(), best[1]

        u_hat, pm = paths[0]
        return u_hat.copy(), pm
