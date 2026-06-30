"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np
from decoder_sc import (
    f_operation,
    g_operation,
    _prepare_channel_llr,
    _bit_reversed,
    _active_llr_level,
    _active_bit_level,
    _update_llrs,
    _update_bits,
)


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, crc_length):
    reg = 0
    for bit in bits:
        reg ^= int(bit) << (crc_length - 1)
        for _ in range(crc_length):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)
    return reg


def crc_encode(info_bits, crc_length=8):
    """
    计算 CRC 校验位并附加到信息比特后。
    """
    info_bits = np.asarray(info_bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    remainder = _crc_remainder(info_bits, poly, crc_length)
    crc_bits = np.array(
        [(remainder >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=int,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。"""
    bits = np.asarray(bits, dtype=int)
    if len(bits) < crc_length:
        return False
    expected = crc_encode(bits[:-crc_length], crc_length)
    return np.array_equal(bits, expected)


def _pm_update(pm, llr, u):
    hard = 0 if llr >= 0 else 1
    if u != hard:
        pm += abs(llr)
    return pm


class SCLDecoder:
    """SCL 译码器（含 Lazy Copy 优化）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]

    def decode(self, llr_ch):
        llr_ch = _prepare_channel_llr(llr_ch)
        N, n, L_size = self.N, self.n, self.list_size

        paths = [
            {
                "pm": 0.0,
                "L": np.zeros((N, n + 1), dtype=np.float64),
                "B": np.zeros((N, n + 1), dtype=np.int8),
            }
        ]
        paths[0]["L"][:, 0] = llr_ch

        for phi in range(N):
            l = _bit_reversed(phi, n)
            new_paths = []

            for path in paths:
                L = path["L"]
                B = path["B"]
                _update_llrs(L, B, l, n, N)
                llr0 = L[l, n]

                if self.frozen_bits[l]:
                    pm = _pm_update(path["pm"], llr0, 0)
                    B[l, n] = 0
                    _update_bits(B, l, n, N)
                    new_paths.append({"pm": pm, "L": L, "B": B})
                else:
                    for u_bit in (0, 1):
                        L_copy = L.copy()
                        B_copy = B.copy()
                        pm = _pm_update(path["pm"], llr0, u_bit)
                        B_copy[l, n] = u_bit
                        _update_bits(B_copy, l, n, N)
                        new_paths.append({"pm": pm, "L": L_copy, "B": B_copy})

            new_paths.sort(key=lambda p: p["pm"])
            paths = new_paths[:L_size]

        if self.crc_length > 0:
            valid = []
            for p in paths:
                info_bits = p["B"][:, n][self.info_indices]
                if crc_check(info_bits, self.crc_length):
                    valid.append(p)
            if valid:
                paths = valid

        best = min(paths, key=lambda p: p["pm"])
        return best["B"][:, n].astype(int), best["pm"]
