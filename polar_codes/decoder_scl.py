"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np
from encoder import bit_reversed
from decoder_sc import (
    f_operation,
    g_operation,
    _active_llr_level,
    _active_bit_level,
    _update_llrs,
    _update_bits,
)


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY

    reg = 0
    for b in info_bits:
        reg ^= int(b) << (crc_length - 1)
        for _ in range(crc_length):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)

    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC。"""
    bits = np.asarray(bits, dtype=int)
    if len(bits) < crc_length:
        return False
    expected = crc_encode(bits[:-crc_length], crc_length)
    return np.array_equal(bits[-crc_length:], expected[-crc_length:])


def _pm_penalty(llr, u):
    u_hard = 0 if llr >= 0 else 1
    return 0.0 if u == u_hard else abs(llr)


class SCLDecoder:
    """SCL 译码器（Lazy Copy：路径分裂时复制 LLR/比特数组）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        if 2 ** self.n != N:
            raise ValueError(f"N={N} must be a power of 2")
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_positions = np.where(~self.frozen_bits)[0]

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n, N = self.n, self.N

        L = np.full((N, n + 1), np.nan, dtype=np.float64)
        B = np.zeros((N, n + 1), dtype=np.int8)
        L[:, 0] = llr_ch

        paths = [{"pm": 0.0, "L": L, "B": B, "u": []}]

        for phi_nat in range(N):
            l = bit_reversed(phi_nat, n)
            candidates = []

            for path in paths:
                _update_llrs(path["L"], path["B"], l, n, N)
                llr = path["L"][l, n]

                if self.frozen_bits[l]:
                    pm = path["pm"] + _pm_penalty(llr, 0)
                    candidates.append((pm, path, 0))
                else:
                    for u_bit in (0, 1):
                        pm = path["pm"] + _pm_penalty(llr, u_bit)
                        candidates.append((pm, path, u_bit))

            candidates.sort(key=lambda x: x[0])
            candidates = candidates[: self.list_size]

            new_paths = []
            for pm, parent, u_bit in candidates:
                L_new = parent["L"].copy()
                B_new = parent["B"].copy()

                if self.frozen_bits[l]:
                    B_new[l, n] = 0
                else:
                    B_new[l, n] = u_bit

                _update_bits(B_new, l, n, N)
                new_paths.append({"pm": pm, "L": L_new, "B": B_new})

            paths = new_paths

        paths.sort(key=lambda p: p["pm"])

        if self.crc_length > 0:
            valid = []
            for p in paths:
                info_bits = p["B"][:, n][self.info_positions]
                if crc_check(info_bits, self.crc_length):
                    valid.append(p)
            if valid:
                paths = valid

        best = paths[0]
        return best["B"][:, n].astype(int), best["pm"]
