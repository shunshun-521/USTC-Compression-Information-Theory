"""
极化码 SCL（串行抵消列表）译码器，支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from decoder_sc_core import f_boxplus, g_boxplus


def crc_encode(info_bits, crc_length=8):
    info_bits = np.asarray(info_bits, dtype=np.int8)
    if crc_length == 8:
        poly = 0x07
    elif crc_length == 16:
        poly = 0x8005
    else:
        raise ValueError("crc_length must be 8 or 16")
    reg = 0
    for b in info_bits:
        reg ^= int(b) << (crc_length - 1)
        for _ in range(8):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)
    crc_bits = np.array([(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    if crc_length == 0:
        return True
    bits = np.asarray(bits, dtype=np.int8)
    return np.array_equal(bits[-crc_length:], crc_encode(bits[:-crc_length], crc_length)[-crc_length:])


def _llr_at_bit(llr, u_prefix, bit_pos):
    """给定 u[0:bit_pos] 已判决，计算 u[bit_pos] 的 LLR（与 sc_tree_decode 一致）。"""
    llr = np.asarray(llr, dtype=np.float64)
    N = len(llr)
    n = int(np.log2(N))

    def walk(L, depth, offset):
        if depth == 0:
            return float(L[0])
        half = 1 << (depth - 1)
        if bit_pos < offset + half:
            L_left = f_boxplus(L[:half], L[half:])
            return walk(L_left, depth - 1, offset)
        u_slice = u_prefix[offset : offset + half]
        L_right = g_boxplus(L[:half], L[half:], u_slice)
        return walk(L_right, depth - 1, offset + half)

    return walk(llr, n, 0)


class SCLDecoder:
    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.L = list_size
        self.crc_length = crc_length

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        if self.L == 1:
            from decoder_sc import sc_decode

            u = sc_decode(llr_ch, self.frozen_bits)
            return u, 0.0
        paths = [{"pm": 0.0, "u": np.zeros(self.N, dtype=int)}]

        for phi in range(self.N):
            expanded = []
            for st in paths:
                llr_phi = _llr_at_bit(llr_ch, st["u"], phi)
                if self.frozen_bits[phi]:
                    hard = 0 if llr_phi >= 0 else 1
                    pm = st["pm"] + (abs(llr_phi) if hard != 0 else 0.0)
                    nu = st["u"].copy()
                    nu[phi] = 0
                    expanded.append({"pm": pm, "u": nu})
                else:
                    for bit in (0, 1):
                        hard = 0 if llr_phi >= 0 else 1
                        pm = st["pm"] + (abs(llr_phi) if bit != hard else 0.0)
                        nu = st["u"].copy()
                        nu[phi] = bit
                        expanded.append({"pm": pm, "u": nu})
            expanded.sort(key=lambda x: x["pm"])
            paths = expanded[: self.L]

        if self.crc_length > 0:
            ok = [p for p in paths if crc_check(p["u"], self.crc_length)]
            if ok:
                paths = ok
        best = min(paths, key=lambda x: x["pm"])
        return best["u"], best["pm"]
